# import os
# import numpy as np
# import torch
# import cv2
# from torch.utils.data import Dataset
# from ..utils.laserscan3 import LaserScan, SemLaserScan

# EXTENSIONS_SCAN = ['.bin']
# EXTENSIONS_EDGE = ['.png']
# EXTENSIONS_LABEL = ['.label']


# def is_scan(filename):
#   return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

# def is_edge(filename):
#   return any(filename.endswith(ext) for ext in EXTENSIONS_EDGE)

# def is_label(filename):
#   return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


# def _interpolate_small_gaps_1d(arr, valid, max_gap=2, mode="linear"):
#   """
#   1次元の配列について、valid==Trueでない連続区間（穴）の長さがmax_gap以下の区間のみ補完する。
#   先頭/末尾の穴は、両側が有効値で挟まれていないため補完しない。
#   mode: "linear" or "nearest"
#   """
#   n = arr.shape[0]
#   arr_out = arr.copy()
#   i = 0
#   while i < n:
#     if valid[i]:
#       i += 1
#       continue
#     # 無効区間 [i, j)
#     j = i
#     while j < n and not valid[j]:
#       j += 1
#     gap_len = j - i
#     left = i - 1
#     right = j
#     # 両側に有効があり、かつ穴が閾値以下なら補完
#     if left >= 0 and right < n and gap_len <= max_gap and valid[left] and valid[right]:
#       if mode == "nearest":
#         # 左右どちらに近いかで振り分け（等距離は左）
#         mid = (left + right) / 2.0
#         for k in range(i, j):
#           arr_out[k] = arr[left] if k <= mid else arr[right]
#       else:  # linear
#         x0, y0 = left, arr[left]
#         x1, y1 = right, arr[right]
#         for k in range(i, j):
#           t = (k - x0) / (x1 - x0)
#           arr_out[k] = (1 - t) * y0 + t * y1
#     # それ以外（長い穴 or 端の穴）はそのまま未補完
#     i = j
#   return arr_out


# def inpaint_small_gaps_2d(range_img, mask_img, max_gap_row=2, max_gap_col=2, mode="linear"):
#   """
#   2Dのrange画像に対して、小さな穴のみを行方向＆列方向に1D補間する。
#   - range_img: np.float32 [H,W], 無効は -1
#   - mask_img : np.int32   [H,W], 1=観測あり, 0=なし
#   戻り値: (range_filled, filled_mask)  ※filled_maskは「今回補完で新規に有効になった画素」
#   """
#   H, W = range_img.shape
#   valid = mask_img.astype(bool) & (range_img > 0)
#   range_out = range_img.copy()
#   filled = np.zeros_like(valid, dtype=bool)

#   # 行方向（横）での小穴補間
#   for y in range(H):
#     row = range_out[y, :]
#     v = valid[y, :]
#     row_filled = _interpolate_small_gaps_1d(row, v, max_gap=max_gap_row, mode=mode)
#     filled_row = (~v) & (row_filled > 0)  # 新たに埋まった場所
#     range_out[y, :] = np.where(filled_row, row_filled, row)
#     valid[y, :] = v | filled_row
#     filled[y, :] |= filled_row

#   # 列方向（縦）での小穴補間（行方向で埋まらなかった箇所をさらに狙う）
#   for x in range(W):
#     col = range_out[:, x]
#     v = valid[:, x]
#     col_filled = _interpolate_small_gaps_1d(col, v, max_gap=max_gap_col, mode=mode)
#     filled_col = (~v) & (col_filled > 0)
#     range_out[:, x] = np.where(filled_col, col_filled, col)
#     valid[:, x] = v | filled_col
#     filled[:, x] |= filled_col

#   # 無効は -1のまま
#   range_out = np.where(valid, range_out, -1.0)
#   return range_out.astype(np.float32), filled.astype(np.int32)


# def median_filter_on_valid(range_img, valid_mask, ksize=3):
#   """
#   有効画素の局所ノイズ抑制のための軽い中央値フィルタ。
#   - 無効領域に“にじませない”よう、無効はそのまま、近傍すべてが無効に近い場所は元値を保持。
#   - 実装簡易化のため：一旦無効ピクセルのみ最近傍補完で埋めてからmedian、最後に元の無効を戻す。
#   """
#   if ksize <= 1:
#     return range_img

#   # 無効を最近傍で仮埋め（OpenCVのdistanceTransformを用いた最近傍充填）
#   inv = (~valid_mask).astype(np.uint8)  # 無効=1
#   # 距離変換のマスクは有効側を0/無効側を1にするが、最近傍のインデックス取得がないため、
#   # 近似として無効を“近傍の有効値”で段階的に埋めるダイレーションを数回行う
#   # （性能重視なら本格的な最近傍補間に差し替え可）
#   tmp = range_img.copy()
#   tmp2 = tmp.copy()
#   for _ in range(2):  # 軽く2回だけ
#     tmp2 = cv2.blur(tmp2, (3, 3))  # 近傍平均で暫定埋め
#     tmp = np.where(valid_mask, tmp, tmp2)

#   # 中央値フィルタ
#   tmp_med = cv2.medianBlur(tmp.astype(np.float32), ksize)

#   # 元の無効は戻す（-1）
#   out = np.where(valid_mask, tmp_med, -1.0).astype(np.float32)
#   return out


# # LiDARの点群データとラベルを読み込み、前処理してテンソルに整形
# class SemanticKitti(Dataset):
#   def __init__(self, root,
#                sequences,
#                labels,
#                color_map,
#                learning_map,
#                learning_map_inv,
#                sensor,
#                max_points=150000,
#                gt=True, skip=0,
#                # ★ 前処理制御
#                max_gap_row=2,            # 横方向の小穴閾値（ピクセル）
#                max_gap_col=2,            # 縦方向の小穴閾値（ピクセル）
#                inpaint_mode="linear",    # "linear" or "nearest"
#                median_ksize=0,           # 0なら中央値フィルタ無効
#                # ★ motorcyclist パッチ学習用
#                focus_motor=False,
#                focus_prob=0.5,
#                patch_h=64,
#                patch_w=256):
#     self.root = os.path.join(root, "sequences")
#     self.sequences = sequences
#     self.labels = labels
#     self.color_map = color_map
#     self.learning_map = learning_map
#     self.learning_map_inv = learning_map_inv
#     self.sensor = sensor
#     self.sensor_img_H = sensor["img_prop"]["height"]
#     self.sensor_img_W = sensor["img_prop"]["width"]
#     self.sensor_img_means = torch.tensor(sensor["img_means"], dtype=torch.float)  # 5ch: [range,x,y,z,remission]
#     self.sensor_img_stds = torch.tensor(sensor["img_stds"], dtype=torch.float)    # 5ch: [range,x,y,z,remission]
#     self.sensor_fov_up = sensor["fov_up"]
#     self.sensor_fov_down = sensor["fov_down"]
#     self.max_points = max_points
#     self.gt = gt

#     self.max_gap_row = max_gap_row
#     self.max_gap_col = max_gap_col
#     self.inpaint_mode = inpaint_mode
#     self.median_ksize = median_ksize

#     # motor パッチ学習パラメータ
#     self.focus_motor = focus_motor
#     self.focus_prob = focus_prob
#     self.patch_h = patch_h
#     self.patch_w = patch_w

#     self.nclasses = len(self.learning_map_inv)

#     if os.path.isdir(self.root):
#       print(f"Sequences folder exists! Using sequences from {self.root}")
#     else:
#       raise ValueError("Sequences folder doesn't exist! Exiting...")

#     assert isinstance(self.labels, dict)
#     assert isinstance(self.color_map, dict)
#     assert isinstance(self.learning_map, dict)
#     assert isinstance(self.sequences, list)

#     self.scan_files = []
#     self.label_files = []

#     for seq in self.sequences:
#         seq = '{:02d}'.format(int(seq))
#         print(f"parsing seq {seq}")

#         scan_path = os.path.join(self.root, seq, "velodyne")
#         label_path = os.path.join(self.root, seq, "labels")

#         scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
#         label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_path)) for f in fn if is_label(f)]

#         if self.gt:
#             assert len(scan_files) == len(label_files)

#         self.scan_files.extend(scan_files)
#         self.label_files.extend(label_files)


#     self.scan_files.sort()
#     self.label_files.sort()

#     if skip != 0:
#       self.scan_files = self.scan_files[::skip]
#     #   self.edge_files = self.edge_files[::skip]
#       self.label_files = self.label_files[::skip]

#     print("Using {} scans from sequences {}".format(len(self.scan_files), self.sequences))

#   def __getitem__(self, index):
#     scan_file = self.scan_files[index]
#     if self.gt:
#       label_file = self.label_files[index]

#     if self.gt:
#       scan = SemLaserScan(self.color_map,
#                           project=True,
#                           H=self.sensor_img_H,
#                           W=self.sensor_img_W,
#                           fov_up=self.sensor_fov_up,
#                           fov_down=self.sensor_fov_down)
#     else:
#       scan = LaserScan(project=True,
#                        H=self.sensor_img_H,
#                        W=self.sensor_img_W,
#                        fov_up=self.sensor_fov_up,
#                        fov_down=self.sensor_fov_down)

#     # ★ edge は常にセンサー解像度のゼロ画像にする
#     scan.edge = np.zeros((self.sensor_img_H, self.sensor_img_W), dtype=np.uint8)

#     # scan & label
#     scan.open_scan(scan_file)
#     if self.gt:
#       scan.open_label(label_file)
#       scan.sem_label = self.map(scan.sem_label, self.learning_map)
#       scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

#     # ---------- 小穴 inpaint ----------
#     proj_range_np = scan.proj_range.astype(np.float32).copy()            # [-1 or depth]
#     proj_mask_np  = scan.proj_mask.astype(np.int32).copy()               # 1:観測, 0:無効

#     proj_range_inp, filled_mask_np = inpaint_small_gaps_2d(
#         proj_range_np, proj_mask_np,
#         max_gap_row=self.max_gap_row,
#         max_gap_col=self.max_gap_col,
#         mode=self.inpaint_mode
#     )

#     # オプション：中央値フィルタ
#     if self.median_ksize and self.median_ksize > 1:
#       valid_for_median = (proj_range_inp > 0)
#       proj_range_inp = median_filter_on_valid(proj_range_inp, valid_for_median, ksize=int(self.median_ksize))

#     valid_mask_np = (proj_mask_np.astype(bool) | (filled_mask_np.astype(bool))).astype(np.uint8)

#     # ---------- ★ motor パッチ学習（train のみ） ----------
#     if self.gt and self.focus_motor and np.random.rand() < self.focus_prob:
#       MOTOR_ID = 8
#       H, W = proj_range_inp.shape

#       # 学習ラベル（xentropy ID）の2D
#       proj_labels_np = scan.proj_sem_label.astype(np.int32).copy()
#       motor_mask = (proj_labels_np == MOTOR_ID) & (proj_mask_np > 0)

#       if motor_mask.any():
#         ys, xs = np.where(motor_mask)
#         idx = np.random.randint(0, len(ys))
#         cy, cx = int(ys[idx]), int(xs[idx])

#         ph, pw = self.patch_h, self.patch_w
#         y0 = max(0, cy - ph // 2)
#         y1 = min(H, y0 + ph)
#         x0 = max(0, cx - pw // 2)
#         x1 = min(W, x0 + pw)
#         # 必要なら後ろ側を詰めてサイズを合わせる
#         y0 = max(0, y1 - ph)
#         x0 = max(0, x1 - pw)

#         # クロップ
#         proj_range_patch = proj_range_inp[y0:y1, x0:x1]
#         proj_xyz_patch   = scan.proj_xyz[y0:y1, x0:x1, :]
#         proj_rem_patch   = scan.proj_remission[y0:y1, x0:x1]
#         proj_mask_patch  = proj_mask_np[y0:y1, x0:x1]
#         valid_mask_patch = valid_mask_np[y0:y1, x0:x1]
#         labels_patch     = proj_labels_np[y0:y1, x0:x1]
#         edge_patch       = scan.edge[y0:y1, x0:x1]

#         # 元の解像度 (W,H) にリサイズして motor を相対的に拡大
#         target_size = (self.sensor_img_W, self.sensor_img_H)

#         proj_range_inp = cv2.resize(proj_range_patch, target_size, interpolation=cv2.INTER_LINEAR)
#         proj_rem_np    = cv2.resize(proj_rem_patch, target_size, interpolation=cv2.INTER_LINEAR)
#         proj_mask_np   = cv2.resize(proj_mask_patch.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
#         valid_mask_np  = cv2.resize(valid_mask_patch.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
#         proj_labels_np = cv2.resize(labels_patch.astype(np.int32), target_size, interpolation=cv2.INTER_NEAREST)
#         edge_np        = cv2.resize(edge_patch.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)

#         # xyz は ch ごとに resize
#         xyz_resized = []
#         for c in range(3):
#           ch = proj_xyz_patch[..., c]
#           ch_r = cv2.resize(ch, target_size, interpolation=cv2.INTER_LINEAR)
#           xyz_resized.append(ch_r)
#         proj_xyz_np = np.stack(xyz_resized, axis=-1)

#         # 書き戻し
#         scan.proj_xyz = proj_xyz_np.astype(np.float32)
#         scan.proj_remission = proj_rem_np.astype(np.float32)
#         scan.proj_sem_label = proj_labels_np.astype(np.int32)
#         scan.edge = edge_np.astype(np.uint8)
#       # motor がいないフレームはそのまま（通常サンプル）
#     # ---------- motor パッチここまで ----------

#     # unprojected tensors (padded)
#     unproj_n_points = scan.points.shape[0]
#     unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
#     unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points).clone()
#     unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
#     unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range).clone()
#     unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
#     unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions).clone()
#     if self.gt:
#       unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
#       unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label).clone()
#     else:
#       unproj_labels = []

#     # projected tensors（Rangeだけ補完版を使用）
#     proj_range = torch.from_numpy(proj_range_inp).clone()                # [H,W]
#     proj_xyz = torch.from_numpy(scan.proj_xyz.astype(np.float32)).clone()          # [H,W,3]
#     proj_remission = torch.from_numpy(scan.proj_remission.astype(np.float32)).clone()  # [H,W]
#     proj_obs_mask = torch.from_numpy(proj_mask_np.astype(np.int32)).clone()      # 観測のみ [H,W]
#     proj_valid_mask = torch.from_numpy(valid_mask_np.astype(np.int32)).clone()   # 観測 or 小穴補完 [H,W]

#     if self.gt:
#       proj_labels = torch.from_numpy(scan.proj_sem_label.astype(np.int64)).clone()
#       proj_labels = proj_labels * proj_obs_mask  # ラベルは観測点のみ
#     else:
#       proj_labels = []

#     # ---------- 正規化とマスク適用 ----------
#     range_ch = proj_range.unsqueeze(0)                     # [1,H,W]
#     xyz_ch   = proj_xyz.permute(2, 0, 1)                  # [3,H,W]
#     rem_ch   = proj_remission.unsqueeze(0)                # [1,H,W]

#     img5 = torch.cat([range_ch, xyz_ch, rem_ch], dim=0)   # [5,H,W]
#     img5 = (img5 - self.sensor_img_means[:, None, None]) / (self.sensor_img_stds[:, None, None])

#     c_range = img5[0:1] * proj_valid_mask.float().unsqueeze(0)
#     c_xyz   = img5[1:4] * proj_obs_mask.float().unsqueeze(0)
#     c_rem   = img5[4:5] * proj_obs_mask.float().unsqueeze(0)

#     img5_masked = torch.cat([c_range, c_xyz, c_rem], dim=0)  # [5,H,W]

#     # 6ch目に観測マスク
#     proj = torch.cat([img5_masked, proj_obs_mask.float().unsqueeze(0)], dim=0)  # [6,H,W]

#     # projection indices of original points
#     proj_x = torch.full([self.max_points], -1, dtype=torch.long)
#     proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x.squeeze().astype(np.int64)).clone()
#     proj_y = torch.full([self.max_points], -1, dtype=torch.long)
#     proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y.squeeze().astype(np.int64)).clone()

#     path_norm = os.path.normpath(scan_file)
#     path_split = path_norm.split(os.sep)
#     path_seq = path_split[-3]
#     path_name = path_split[-1].replace(".bin", ".label")

#     edge = torch.from_numpy(scan.edge.astype(np.uint8).copy())

#     return (proj,                  # [6,H,W]
#             proj_obs_mask,         # [H,W]
#             proj_labels,           # [H,W]
#             unproj_labels,         # [M] or []
#             path_seq, path_name,
#             proj_x, proj_y,
#             proj_range, unproj_range,
#             proj_xyz, unproj_xyz,
#             proj_remission, unproj_remissions,
#             unproj_n_points,
#             edge)

#   def __len__(self):
#     return len(self.scan_files)

#   def get_pointcloud(self, seq, name):
#     base = os.path.splitext(name)[0]
#     bin_path = os.path.join(self.root, f"{int(seq):02d}", "velodyne", base + ".bin")
#     pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
#     return pts

#   @staticmethod
#   def map(label, mapdict):
#     maxkey = 0
#     for key, data in mapdict.items():
#       nel = len(data) if isinstance(data, list) else 1
#       if key > maxkey:
#         maxkey = key
#     lut = (np.zeros((maxkey + 100, nel), dtype=np.int32)
#            if nel > 1 else
#            np.zeros((maxkey + 100), dtype=np.int32))
#     for key, data in mapdict.items():
#       try:
#         lut[key] = data
#       except IndexError:
#         print("Wrong key ", key)
#     return lut[label]


# Parser2.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils.laserscan3 import LaserScan, SemLaserScan
from .SemanticKitti7 import SemanticKitti


def kitti_collate_stack_only(batch):
    """
    SemanticKitti7.__getitem__ が返す 16 要素タプルを
    バッチ化するための collate 関数（train/valid/test 共通）。

    返り値も 16 要素で、Trainer 側の unpack と
    同じ並びになるように作る。
    """
    in_vol          = torch.stack([b[0] for b in batch], dim=0)   # [B,6,H,W]
    proj_mask       = torch.stack([b[1] for b in batch], dim=0)   # [B,H,W]
    proj_labels     = torch.stack([b[2] for b in batch], dim=0)   # [B,H,W]
    unproj_labels   = [b[3] for b in batch]                       # list

    path_seq        = [b[4] for b in batch]
    path_name       = [b[5] for b in batch]

    proj_x          = [b[6] for b in batch]                       # list of [M]
    proj_y          = [b[7] for b in batch]                       # list of [M]

    proj_range      = torch.stack([b[8] for b in batch], dim=0)   # [B,H,W]
    unproj_range    = [b[9] for b in batch]                       # list

    proj_xyz        = [b[10] for b in batch]                      # list of [H,W,3]
    unproj_xyz      = [b[11] for b in batch]                      # list of [M,3]

    proj_remission  = torch.stack([b[12] for b in batch], dim=0)  # [B,H,W]
    unproj_remiss   = [b[13] for b in batch]                      # list

    unproj_n_points = torch.tensor([b[14] for b in batch], dtype=torch.int32)  # [B]
    edge            = torch.stack([b[15] for b in batch], dim=0)  # [B,H,W]

    return (in_vol,
            proj_mask,
            proj_labels,
            unproj_labels,
            path_seq,
            path_name,
            proj_x,
            proj_y,
            proj_range,
            unproj_range,
            proj_xyz,
            unproj_xyz,
            proj_remission,
            unproj_remiss,
            unproj_n_points,
            edge)


class Parser:
    """
    SemanticKitti7 ベースのシンプル Parser
    - train  : 穴埋めあり, focus_motor=False, 通常の shuffle DataLoader
    - valid  : 穴埋めあり, focus_motor=False, shuffle=False
    - test   : 穴埋めあり, focus_motor=False, shuffle=False
    """

    def __init__(self, root, data_cfg, arch_cfg,
                 gt=True, shuffle_train=True):

        # 設定
        self.root = root
        self.data_cfg = data_cfg
        self.arch_cfg = arch_cfg
        self.gt = gt
        self.shuffle_train = shuffle_train

        # sensor / label 関連
        self.sensor = arch_cfg["dataset"]["sensor"]
        self.labels = data_cfg["labels"]
        self.color_map = data_cfg["color_map"]
        self.learning_map = data_cfg["learning_map"]
        self.learning_map_inv = data_cfg["learning_map_inv"]
        self.learning_ignore = data_cfg["learning_ignore"]

        self.max_points = arch_cfg["dataset"]["max_points"]
        self.batch_size = arch_cfg["train"]["batch_size"]
        self.workers = arch_cfg["train"]["workers"]
        self.DATASET_TYPE = SemanticKitti

        # クラス数（xentropy 用）
        self.nclasses = len(self.learning_map_inv)

        # シーケンス分割
        self.train_sequences = data_cfg["split"]["train"]
        self.valid_sequences = data_cfg["split"]["valid"]
        self.test_sequences  = data_cfg["split"]["test"]

        # ============================================================
        # 1. train dataset
        # ============================================================
        self.train_dataset = self.DATASET_TYPE(
            root=self.root,
            sequences=self.train_sequences,
            labels=self.labels,
            color_map=self.color_map,
            learning_map=self.learning_map,
            learning_map_inv=self.learning_map_inv,
            sensor=self.sensor,
            max_points=self.max_points,
            gt=self.gt,
            # 小さな穴を線形補完する inpaint 設定
            max_gap_row=2,
            max_gap_col=2,
            inpaint_mode="linear",
            median_ksize=0,
            # motor 用の局所拡大は使わない
            focus_motor=False,
            focus_prob=0.0,
            patch_h=64,
            patch_w=256,
        )

        self.trainloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=kitti_collate_stack_only,
        )

        # ============================================================
        # 2. valid dataset（augmentation なし, focus_motor=False）
        # ============================================================
        self.valid_dataset = self.DATASET_TYPE(
            root=self.root,
            sequences=self.valid_sequences,
            labels=self.labels,
            color_map=self.color_map,
            learning_map=self.learning_map,
            learning_map_inv=self.learning_map_inv,
            sensor=self.sensor,
            max_points=self.max_points,
            gt=self.gt,
            max_gap_row=2,
            max_gap_col=2,
            inpaint_mode="linear",
            median_ksize=0,
            focus_motor=False,
            focus_prob=0.0,
            patch_h=64,
            patch_w=256,
        )

        self.validloader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=kitti_collate_stack_only,
        )

        # ============================================================
        # 3. test dataset（augmentation なし, focus_motor=False）
        # ============================================================
        if self.test_sequences:
            self.test_dataset = self.DATASET_TYPE(
                root=self.root,
                sequences=self.test_sequences,
                labels=self.labels,
                color_map=self.color_map,
                learning_map=self.learning_map,
                learning_map_inv=self.learning_map_inv,
                sensor=self.sensor,
                max_points=self.max_points,
                gt=False,
                max_gap_row=2,
                max_gap_col=2,
                inpaint_mode="linear",
                median_ksize=0,
                focus_motor=False,
                focus_prob=0.0,
                patch_h=64,
                patch_w=256,
            )

            self.testloader = DataLoader(
                self.test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=kitti_collate_stack_only,
            )
        else:
            self.test_dataset = None
            self.testloader = None

    # ===== interface =====
    def get_train_set(self):
        return self.trainloader

    def get_valid_set(self):
        return self.validloader

    def get_test_set(self):
        return self.testloader

    def get_train_size(self):
        return len(self.trainloader)

    def get_valid_size(self):
        return len(self.validloader)

    def get_test_size(self):
        return len(self.testloader) if self.testloader is not None else 0

    def get_n_classes(self):
        return self.nclasses

    def get_xentropy_class_string(self, idx):
        orig_id = self.learning_map_inv[idx]
        return self.labels[orig_id]

    def to_original(self, label):
        return self.DATASET_TYPE.map(label, self.learning_map_inv)

    def to_xentropy(self, label):
        return self.DATASET_TYPE.map(label, self.learning_map)

    def to_color(self, label):
        label = self.DATASET_TYPE.map(label, self.learning_map_inv)
        return self.DATASET_TYPE.map(label, self.color_map)
