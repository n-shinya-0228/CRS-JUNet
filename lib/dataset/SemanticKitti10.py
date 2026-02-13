import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from ..utils.laserscan3 import LaserScan, SemLaserScan

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_EDGE = ['.png']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_edge(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_EDGE)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


def _interpolate_small_gaps_1d(arr, valid, max_gap=2, mode="linear"):
  """
  1次元の配列について、valid==Trueでない連続区間（穴）の長さがmax_gap以下の区間のみ補完する。
  先頭/末尾の穴は、両側が有効値で挟まれていないため補完しない。
  mode: "linear" or "nearest"
  """
  n = arr.shape[0]
  arr_out = arr.copy()
  i = 0
  while i < n:
    if valid[i]:
      i += 1
      continue
    # 無効区間 [i, j)
    j = i
    while j < n and not valid[j]:
      j += 1
    gap_len = j - i
    left = i - 1
    right = j
    # 両側に有効があり、かつ穴が閾値以下なら補完
    if left >= 0 and right < n and gap_len <= max_gap and valid[left] and valid[right]:
      if mode == "nearest":
        # 左右どちらに近いかで振り分け（等距離は左）
        mid = (left + right) / 2.0
        for k in range(i, j):
          arr_out[k] = arr[left] if k <= mid else arr[right]
      else:  # linear
        x0, y0 = left, arr[left]
        x1, y1 = right, arr[right]
        for k in range(i, j):
          t = (k - x0) / (x1 - x0)
          arr_out[k] = (1 - t) * y0 + t * y1
    # それ以外（長い穴 or 端の穴）はそのまま未補完
    i = j
  return arr_out


def inpaint_small_gaps_2d(range_img, mask_img, max_gap_row=2, max_gap_col=2, mode="linear"):
  """
  2Dのrange画像に対して、小さな穴のみを行方向＆列方向に1D補間する。
  - range_img: np.float32 [H,W], 無効は -1
  - mask_img : np.int32   [H,W], 1=観測あり, 0=なし
  戻り値: (range_filled, filled_mask)  ※filled_maskは「今回補完で新規に有効になった画素」
  """
  H, W = range_img.shape
  valid = mask_img.astype(bool) & (range_img > 0)
  range_out = range_img.copy()
  filled = np.zeros_like(valid, dtype=bool)

  # 行方向（横）での小穴補間
  for y in range(H):
    row = range_out[y, :]
    v = valid[y, :]
    row_filled = _interpolate_small_gaps_1d(row, v, max_gap=max_gap_row, mode=mode)
    filled_row = (~v) & (row_filled > 0)  # 新たに埋まった場所
    range_out[y, :] = np.where(filled_row, row_filled, row)
    valid[y, :] = v | filled_row
    filled[y, :] |= filled_row

  # 列方向（縦）での小穴補間（行方向で埋まらなかった箇所をさらに狙う）
  for x in range(W):
    col = range_out[:, x]
    v = valid[:, x]
    col_filled = _interpolate_small_gaps_1d(col, v, max_gap=max_gap_col, mode=mode)
    filled_col = (~v) & (col_filled > 0)
    range_out[:, x] = np.where(filled_col, col_filled, col)
    valid[:, x] = v | filled_col
    filled[:, x] |= filled_col

  # 無効は -1のまま
  range_out = np.where(valid, range_out, -1.0)
  return range_out.astype(np.float32), filled.astype(np.int32)


def median_filter_on_valid(range_img, valid_mask, ksize=3):
  """
  有効画素の局所ノイズ抑制のための軽い中央値フィルタ。
  - 無効領域に“にじませない”よう、無効はそのまま、近傍すべてが無効に近い場所は元値を保持。
  - 実装簡易化のため：一旦無効ピクセルのみ最近傍補完で埋めてからmedian、最後に元の無効を戻す。
  """
  if ksize <= 1:
    return range_img

  # 無効を最近傍で仮埋め（OpenCVのdistanceTransformを用いた最近傍充填）
  inv = (~valid_mask).astype(np.uint8)  # 無効=1
  # 距離変換のマスクは有効側を0/無効側を1にするが、最近傍のインデックス取得がないため、
  # 近似として無効を“近傍の有効値”で段階的に埋めるダイレーションを数回行う
  # （性能重視なら本格的な最近傍補間に差し替え可）
  tmp = range_img.copy()
  tmp2 = tmp.copy()
  for _ in range(2):  # 軽く2回だけ
    tmp2 = cv2.blur(tmp2, (3, 3))  # 近傍平均で暫定埋め
    tmp = np.where(valid_mask, tmp, tmp2)

  # 中央値フィルタ
  tmp_med = cv2.medianBlur(tmp.astype(np.float32), ksize)

  # 元の無効は戻す（-1）
  out = np.where(valid_mask, tmp_med, -1.0).astype(np.float32)
  return out


# LiDARの点群データとラベルを読み込み、前処理してテンソルに整形
class SemanticKitti(Dataset):
  def __init__(self, root,
               sequences,
               labels,
               color_map,
               learning_map,
               learning_map_inv,
               sensor,
               max_points=150000,
               gt=True, skip=0,
               # ★ 追加パラメータ（前処理制御）
               max_gap_row=2,            # 横方向の小穴閾値（ピクセル）
               max_gap_col=2,            # 縦方向の小穴閾値（ピクセル）
               inpaint_mode="linear",    # "linear" or "nearest"
               median_ksize=0):          # 0なら中央値フィルタ無効
    self.root = os.path.join(root, "sequences")
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]
    self.sensor_img_means = torch.tensor(sensor["img_means"], dtype=torch.float)  # 5ch: [range,x,y,z,remission]
    self.sensor_img_stds = torch.tensor(sensor["img_stds"], dtype=torch.float)    # 5ch: [range,x,y,z,remission]
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.max_points = max_points
    self.gt = gt

    self.max_gap_row = max_gap_row
    self.max_gap_col = max_gap_col
    self.inpaint_mode = inpaint_mode
    self.median_ksize = median_ksize

    self.nclasses = len(self.learning_map_inv)

    if os.path.isdir(self.root):
      print(f"Sequences folder exists! Using sequences from {self.root}")
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    assert isinstance(self.labels, dict)
    assert isinstance(self.color_map, dict)
    assert isinstance(self.learning_map, dict)
    assert isinstance(self.sequences, list)

    self.scan_files = []
    self.edge_files = []
    self.label_files = []

    for seq in self.sequences:
      seq = '{:02d}'.format(int(seq))
      print(f"parsing seq {seq}")

      scan_path = os.path.join(self.root, seq, "velodyne")
      edge_path = os.path.join(self.root, seq, "edge")
      label_path = os.path.join(self.root, seq, "labels")

      scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
      edge_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(edge_path)) for f in fn if is_edge(f)]
      label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_path)) for f in fn if is_label(f)]

      if self.gt:
        assert len(scan_files) == len(edge_files)
        assert len(scan_files) == len(label_files)

      self.scan_files.extend(scan_files)
      self.edge_files.extend(edge_files)
      self.label_files.extend(label_files)

    self.scan_files.sort()
    self.edge_files.sort()
    self.label_files.sort()

    if skip != 0:
      self.scan_files = self.scan_files[::skip]
      self.edge_files = self.edge_files[::skip]
      self.label_files = self.label_files[::skip]

    print("Using {} scans from sequences {}".format(len(self.scan_files), self.sequences))

  def __getitem__(self, index):
    scan_file = self.scan_files[index]
    edge_file = self.edge_files[index] if hasattr(self, 'edge_files') and len(self.edge_files) > index else None
    if self.gt:
      label_file = self.label_files[index]

    if self.gt:
      scan = SemLaserScan(self.color_map,
                          project=True,
                          H=self.sensor_img_H,
                          W=self.sensor_img_W,
                          fov_up=self.sensor_fov_up,
                          fov_down=self.sensor_fov_down)
    else:
      scan = LaserScan(project=True,
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down)

    if edge_file is not None and os.path.exists(edge_file):
      scan.open_edge(edge_file)
    else:
      scan.edge = np.zeros((scan.proj_H, scan.proj_W), dtype=np.uint8)

    scan.open_scan(scan_file)
    if self.gt:
      scan.open_label(label_file)
      scan.sem_label = self.map(scan.sem_label, self.learning_map)
      scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

    # ---------- ここから前処理：小さな欠損のみDepth Inpainting ----------
    proj_range_np = scan.proj_range.astype(np.float32).copy()            # [-1 or depth]
    proj_mask_np  = scan.proj_mask.astype(np.int32).copy()               # 1:観測, 0:無効

    # 小さな穴だけ埋める（行・列方向の1D補間、両側が観測で挟まれた短い穴のみ）
    proj_range_inp, filled_mask_np = inpaint_small_gaps_2d(
        proj_range_np, proj_mask_np,
        max_gap_row=self.max_gap_row,
        max_gap_col=self.max_gap_col,
        mode=self.inpaint_mode
    )

    # オプション：軽い中央値フィルタ（有効域に限定）
    if self.median_ksize and self.median_ksize > 1:
      valid_for_median = (proj_range_inp > 0)
      proj_range_inp = median_filter_on_valid(proj_range_inp, valid_for_median, ksize=int(self.median_ksize))
    

    # 「学習に使う有効域」= 観測 or 小穴補完
    valid_mask_np = (proj_mask_np.astype(bool) | (filled_mask_np.astype(bool))).astype(np.uint8)
    # ---------- ここまで ----------

    # unprojected tensors (padded)
    unproj_n_points = scan.points.shape[0]
    unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
    unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
    unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
    unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
    unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
    unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
    if self.gt:
      unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
      unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
    else:
      unproj_labels = []

    # projected tensors（Rangeだけ補完版を使用）
    proj_range = torch.from_numpy(proj_range_inp).clone()                # [H,W]（小穴のみ補完済み）
    proj_xyz = torch.from_numpy(scan.proj_xyz).clone()                   # [H,W,3]
    proj_remission = torch.from_numpy(scan.proj_remission).clone()       # [H,W]
    proj_obs_mask = torch.from_numpy(scan.proj_mask.astype(np.int32))    # 観測のみのマスク [H,W] {0,1}
    proj_valid_mask = torch.from_numpy(valid_mask_np.astype(np.int32))   # 観測 or 小穴補完 [H,W] {0,1}

    if self.gt:
      proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
      proj_labels = proj_labels * proj_obs_mask  # ラベルは観測点のみ
    else:
      proj_labels = []

    # ---------- 正規化とマスク適用の見直し ----------
    # チャンネル順: [range, x, y, z, remission] をそれぞれ個別に正規化
    range_ch = proj_range.unsqueeze(0)                     # [1,H,W]  補完を含む
    xyz_ch   = proj_xyz.permute(2, 0, 1)                  # [3,H,W]
    rem_ch   = proj_remission.unsqueeze(0)                # [1,H,W]

    img5 = torch.cat([range_ch, xyz_ch, rem_ch], dim=0)   # [5,H,W]

    # 正規化
    img5 = (img5 - self.sensor_img_means[:, None, None]) / (self.sensor_img_stds[:, None, None])

    # マスク適用
    # - Rangeは「観測 or 小穴補完」で活かす
    # - X,Y,Z,Remissionは観測のみ
    c_range = img5[0:1] * proj_valid_mask.float().unsqueeze(0)
    c_xyz   = img5[1:4] * proj_obs_mask.float().unsqueeze(0)
    c_rem   = img5[4:5] * proj_obs_mask.float().unsqueeze(0)

    img5_masked = torch.cat([c_range, c_xyz, c_rem], dim=0)  # [5,H,W]

# SemanticKitti11.py
# SemanticKitti10 をベースに「filled_mask方式」に修正

# --- 省略（上は SemanticKitti10 と同一） ---

    # 6ch目: 観測マスク
    # 7ch目: filled_mask（今回補完された画素のみ）
    proj_filled_mask = torch.from_numpy(filled_mask_np.astype(np.int32))

    proj = torch.cat([
        img5_masked,                               # [5,H,W]
        proj_obs_mask.float().unsqueeze(0),        # ch5: 観測
        proj_filled_mask.float().unsqueeze(0)      # ch6: 補完のみ ★
    ], dim=0)  # [7,H,W]

    # ---------- ここまで ----------

    # projection indices of original points
    proj_x = torch.full([self.max_points], -1, dtype=torch.long)
    proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x.squeeze())
    proj_y = torch.full([self.max_points], -1, dtype=torch.long)
    proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y.squeeze())

    path_norm = os.path.normpath(scan_file)
    path_split = path_norm.split(os.sep)
    path_seq = path_split[-3]
    path_name = path_split[-1].replace(".bin", ".label")

    return (proj,                  # [6,H,W]  (range=観測∪小穴補完 / xyz,rem=観測 / 最後に観測mask)
            proj_obs_mask,         # [H,W]    観測のみマスク（学習で使うならこちら推奨）
            proj_labels,           # [H,W]    long (masked by obs)
            unproj_labels,         # [M] or []
            path_seq, path_name,
            proj_x, proj_y,
            proj_range, unproj_range,
            proj_xyz, unproj_xyz,
            proj_remission, unproj_remissions,
            unproj_n_points,
            scan.edge)

  def __len__(self):
    return len(self.scan_files)

  def get_pointcloud(self, seq, name):
    base = os.path.splitext(name)[0]
    bin_path = os.path.join(self.root, f"{int(seq):02d}", "velodyne", base + ".bin")
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts

  @staticmethod
  def map(label, mapdict):
    maxkey = 0
    for key, data in mapdict.items():
      nel = len(data) if isinstance(data, list) else 1
      if key > maxkey:
        maxkey = key
    lut = (np.zeros((maxkey + 100, nel), dtype=np.int32)
           if nel > 1 else
           np.zeros((maxkey + 100), dtype=np.int32))
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    return lut[label]


