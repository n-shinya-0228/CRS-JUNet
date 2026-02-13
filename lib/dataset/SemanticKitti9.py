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

    # 無効を最近傍で仮埋め
    inv = (~valid_mask).astype(np.uint8)  # 無効=1
    tmp = range_img.copy()
    tmp2 = tmp.copy()
    for _ in range(2):  # 軽く2回だけ
        tmp2 = cv2.blur(tmp2, (3, 3))
        tmp = np.where(valid_mask, tmp, tmp2)

    # 中央値フィルタ
    tmp_med = cv2.medianBlur(tmp.astype(np.float32), ksize)

    # 元の無効は戻す（-1）
    out = np.where(valid_mask, tmp_med, -1.0).astype(np.float32)
    return out


class SemanticKitti(Dataset):
    def __init__(
        self,
        root,
        sequences,
        labels,
        color_map,
        learning_map,
        learning_map_inv,
        sensor,
        max_points=150000,
        gt=True,
        skip=0,
        # ★ 追加パラメータ（前処理制御）
        max_gap_row=2,          # 横方向の小穴閾値
        max_gap_col=2,          # 縦方向の小穴閾値
        inpaint_mode="linear",  # "linear" or "nearest"
        median_ksize=0,         # 0なら中央値フィルタ無効
    ):
        self.root = os.path.join(root, "sequences")
        self.max_points = max_points
        self.gt = gt
        self.sequences = sequences
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
        self.sensor_img_means = np.array(sensor["img_means"], dtype=np.float32)  # [5]
        self.sensor_img_stds = np.array(sensor["img_stds"], dtype=np.float32)    # [5]
        self.max_gap_row = max_gap_row
        self.max_gap_col = max_gap_col
        self.inpaint_mode = inpaint_mode
        self.median_ksize = median_ksize

        self.scan_files = []
        self.edge_files = []
        self.label_files = []

        for seq in self.sequences:
            seq = '{:02d}'.format(int(seq))
            print(f"parsing seq {seq}")

            scan_path = os.path.join(self.root, seq, "velodyne")
            edge_path = os.path.join(self.root, seq, "edge")
            label_path = os.path.join(self.root, seq, "labels")

            scan_files = [
                os.path.join(dp, f)
                for dp, dn, fn in os.walk(os.path.expanduser(scan_path))
                for f in fn if is_scan(f)
            ]
            edge_files = [
                os.path.join(dp, f)
                for dp, dn, fn in os.walk(os.path.expanduser(edge_path))
                for f in fn if is_scan(f.replace(".png", ".bin"))
            ]

            scan_files.sort()
            edge_files.sort()

            if self.gt:
                label_files = [
                    os.path.join(dp, f)
                    for dp, dn, fn in os.walk(os.path.expanduser(label_path))
                    for f in fn if f.endswith(".label")
                ]
                label_files.sort()
                if len(label_files) != len(scan_files):
                    raise RuntimeError(
                        f"Scan and label files count mismatch in seq {seq}"
                    )
                self.label_files.extend(label_files)

            self.scan_files.extend(scan_files)
            self.edge_files.extend(edge_files)

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, index):
        scan_file = self.scan_files[index]
        edge_file = self.edge_files[index] if hasattr(self, 'edge_files') and len(self.edge_files) > index else None
        if self.gt:
            label_file = self.label_files[index]

        # scan インスタンス
        if self.gt:
            scan = SemLaserScan(
                self.color_map,
                project=True,
                H=self.sensor_img_H,
                W=self.sensor_img_W,
                fov_up=self.sensor_fov_up,
                fov_down=self.sensor_fov_down,
            )
        else:
            scan = LaserScan(
                project=True,
                H=self.sensor_img_H,
                W=self.sensor_img_W,
                fov_up=self.sensor_fov_up,
                fov_down=self.sensor_fov_down,
            )

        scan.open_scan(scan_file)
        if self.gt:
            scan.open_label(label_file)
        if edge_file is not None and hasattr(scan, "open_edge"):
            scan.open_edge(edge_file)

        proj_range = scan.proj_range.copy().astype(np.float32)         # [H,W]
        proj_xyz = scan.proj_xyz.copy().astype(np.float32)             # [H,W,3]
        proj_remission = scan.proj_remission.copy().astype(np.float32) # [H,W]
        proj_mask_np = scan.proj_mask.astype(np.int32)                 # [H,W] {0,1}

        # ---------- range の小穴補完 ----------
        proj_range_inp, filled_mask_np = self._inpaint_range_small_holes(
            proj_range,
            proj_mask_np,
            max_gap_row=self.max_gap_row,
            max_gap_col=self.max_gap_col,
            mode=self.inpaint_mode,
            median_ksize=self.median_ksize,
        )

        # 「学習に使う有効域」= 観測 or 小穴補完
        valid_mask_np = (proj_mask_np.astype(bool) | (filled_mask_np.astype(bool))).astype(np.uint8)

        # unprojected tensors (padded)
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)

        # ========= ★ ラベルの learning_map 変換をここで行う =========
        if self.gt:
            # 生ラベル (SemanticKITTI ID) -> 学習用ラベル ID (0〜19)
            sem_label_mapped = self.map(scan.sem_label, self.learning_map).astype(np.int32)
            proj_sem_label_mapped = self.map(scan.proj_sem_label, self.learning_map).astype(np.int32)

            unproj_labels = torch.full([self.max_points], -1, dtype=torch.int32)
            unproj_labels[:unproj_n_points] = torch.from_numpy(sem_label_mapped)

        else:
            unproj_labels = []

        # projected tensors（Rangeだけ補完版を使用）
        proj_range_t = torch.from_numpy(proj_range_inp).clone()             # [H,W]
        proj_xyz_t = torch.from_numpy(proj_xyz).clone()                     # [H,W,3]
        proj_remission_t = torch.from_numpy(proj_remission).clone()         # [H,W]
        proj_obs_mask = torch.from_numpy(proj_mask_np.astype(np.int32))     # [H,W]
        proj_valid_mask = torch.from_numpy(valid_mask_np.astype(np.int32))  # [H,W]

        if self.gt:
            proj_labels = torch.from_numpy(proj_sem_label_mapped).clone()   # [H,W]
            proj_labels = proj_labels * proj_obs_mask                       # 観測点のみ
        else:
            proj_labels = []

        # ---------- 正規化 + 8ch 構築 ----------
        range_ch = proj_range_t.unsqueeze(0)               # [1,H,W]
        xyz_ch   = proj_xyz_t.permute(2, 0, 1)             # [3,H,W]
        rem_ch   = proj_remission_t.unsqueeze(0)           # [1,H,W]

        img5 = torch.cat([range_ch, xyz_ch, rem_ch], dim=0)  # [5,H,W]
        img5 = (img5 - self.sensor_img_means[:, None, None]) / (
            self.sensor_img_stds[:, None, None]
        )

        # マスク適用
        c_range = img5[0:1] * proj_valid_mask.float().unsqueeze(0)
        c_xyz   = img5[1:4] * proj_obs_mask.float().unsqueeze(0)
        c_rem   = img5[4:5] * proj_obs_mask.float().unsqueeze(0)
        img5_masked = torch.cat([c_range, c_xyz, c_rem], dim=0)  # [5,H,W]

        # 追加 ch1: height（z の平均からの相対高さ）
        z_np = proj_xyz[..., 2].astype(np.float32)
        valid_np = proj_valid_mask.numpy().astype(bool)

        if valid_np.any():
            z_valid = z_np[valid_np]
            z_mean = float(z_valid.mean())
            height_np = z_np - z_mean

            h_valid = height_np[valid_np]
            h_std = float(h_valid.std() + 1e-3)
            height_np = (height_np - float(h_valid.mean())) / h_std
            height_np[~valid_np] = 0.0
        else:
            height_np = np.zeros_like(z_np, dtype=np.float32)
        height_ch = torch.from_numpy(height_np).unsqueeze(0)  # [1,H,W]

        # 追加 ch2: normal-like（z勾配ノルム）
        if valid_np.any():
            z_for_grad = z_np.copy()
            z_for_grad[~valid_np] = 0.0
            gy, gx = np.gradient(z_for_grad)
            normal_like = np.sqrt(gx * gx + gy * gy).astype(np.float32)
            n_valid = normal_like[valid_np]
            n_std = float(n_valid.std() + 1e-3)
            normal_like = (normal_like - float(n_valid.mean())) / n_std
            normal_like[~valid_np] = 0.0
        else:
            normal_like = np.zeros_like(z_np, dtype=np.float32)
        normal_ch = torch.from_numpy(normal_like).unsqueeze(0)  # [1,H,W]

        # 6ch目: 観測マスク, 7-8ch目: height, normal-like
        mask_ch = proj_obs_mask.float().unsqueeze(0)  # [1,H,W]
        proj = torch.cat([img5_masked, mask_ch, height_ch, normal_ch], dim=0)  # [8,H,W]

        # projection indices
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x.squeeze())
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y.squeeze())

        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")

        return (
            proj,                 # [8,H,W]
            proj_obs_mask,        # [H,W]
            proj_labels,          # [H,W]  (0〜19, 観測のみ)
            unproj_labels,        # [M]
            path_seq, path_name,
            proj_x, proj_y,
            proj_range_t, unproj_range,
            proj_xyz_t, unproj_xyz,
            proj_remission_t, unproj_remissions,
            unproj_n_points,
            scan.edge,
        )

    def _inpaint_range_small_holes(
        self,
        range_img,
        mask_img,
        max_gap_row=2,
        max_gap_col=2,
        mode="linear",
        median_ksize=0,
    ):
        """
        range_img: np.float32 [H,W], 無効は -1
        mask_img : np.int32   [H,W], 1=観測あり, 0=なし
        1) 横方向 / 縦方向に長さ <= max_gap の小さな穴だけ補完
        2) 必要なら median filter で平滑化
        戻り値: (range_filled, filled_mask)
        """
        H, W = range_img.shape
        valid = mask_img.astype(bool) & (range_img > 0)
        range_out = range_img.copy()
        filled = np.zeros_like(valid, dtype=bool)

        # 行方向
        for y in range(H):
            row = range_out[y]
            v = valid[y]
            self._interp_line_small_gaps(row, v, max_gap_row, filled[y])

        # 列方向
        for x in range(W):
            col = range_out[:, x]
            v = valid[:, x]
            self._interp_line_small_gaps(col, v, max_gap_col, filled[:, x])

        if median_ksize > 1:
            range_med = range_out.copy()
            invalid_mask = ~valid
            range_med[invalid_mask] = 0.0
            range_med = cv2.medianBlur(range_med.astype(np.float32), median_ksize)
            range_out[valid] = range_med[valid]

        filled_mask = (~valid) & (range_out > 0)
        return range_out, filled_mask.astype(np.uint8)

    def _interp_line_small_gaps(self, arr, valid, max_gap, filled_mask_row):
        """
        1次元配列 arr と有効フラグ valid について、
        長さ <= max_gap の穴だけ線形で埋める。
        """
        n = len(arr)
        i = 0
        while i < n:
            if valid[i]:
                i += 1
                continue
            left = i - 1
            while i < n and not valid[i]:
                i += 1
            right = i
            if left < 0 or right >= n:
                continue
            gap = right - left - 1
            if gap <= 0 or gap > max_gap:
                continue
            y0 = arr[left]
            y1 = arr[right]
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                arr[left + k] = (1.0 - t) * y0 + t * y1
                filled_mask_row[left + k] = True

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
        lut = (
            np.zeros((maxkey + 100, nel), dtype=np.int32)
            if nel > 1 else
            np.zeros((maxkey + 100), dtype=np.int32)
        )
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        return lut[label]
