import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from ..utils.laserscan3 import LaserScan, SemLaserScan


# ============ ユーティリティ ============

def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    vmin, vmax = np.min(x), np.max(x)
    if vmax - vmin < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - vmin) / (vmax - vmin)

def _build_lut(mapdict: dict) -> np.ndarray:
    maxkey = max(mapdict.keys()) if len(mapdict) > 0 else 0
    lut = np.zeros((maxkey + 100), dtype=np.int32)
    for k, v in mapdict.items():
        lut[k] = v
    return lut


# ============ DILATE 前処理 ============

def dilate_range_fill(range_img: np.ndarray,
                      valid_mask: np.ndarray,
                      kernel_size: int = 11,
                      max_blend: float = 0.6) -> np.ndarray:
    """
    無効画素のみ、近傍平均(blur)と近傍最大(dilate)のブレンドで埋める。
    """
    base = range_img.copy()
    base[~valid_mask] = 0.0

    k = int(kernel_size)
    if k % 2 == 0:
        k += 1

    # 有効画素数と合計
    num = cv2.blur(valid_mask.astype(np.float32), (k, k))
    den = cv2.blur(base.astype(np.float32), (k, k))
    mean_prop = np.where(num > 1e-6, den / (num + 1e-6), 0.0)

    max_prop = cv2.dilate(base.astype(np.float32),
                          np.ones((k, k), np.uint8),
                          iterations=1)
    fill = (1.0 - max_blend) * mean_prop + max_blend * max_prop

    out = range_img.copy()
    out[~valid_mask] = fill[~valid_mask]
    out[~np.isfinite(out)] = -1.0
    return out.astype(np.float32)


def dilate_fill_iterative(range_img: np.ndarray,
                          valid_mask: np.ndarray,
                          passes: int = 3,
                          kernel_size: int = 7,
                          max_blend: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """
    反復膨張：1回埋めたらその画素も有効扱いにして次周回で母集団に入れる。
    """
    r = range_img.copy()
    m = valid_mask.copy()
    for _ in range(max(1, int(passes))):
        r = dilate_range_fill(r, m, kernel_size=kernel_size, max_blend=max_blend)
        m[:] = True   # 埋めた画素も次反復で近傍として使う
    return r, m


def dilate_label_fill_majority(label_img: np.ndarray,
                               valid_mask: np.ndarray,
                               kernel_size: int = 7) -> np.ndarray:
    """
    欠損ラベルを近傍の多数決で埋める（単純・高速）。
    """
    H, W = label_img.shape
    out = label_img.copy()
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    half = k // 2
    invalid = ~valid_mask
    ys, xs = np.where(invalid)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - half), min(H, y + half + 1)
        x0, x1 = max(0, x - half), min(W, x + half + 1)
        patch_l = label_img[y0:y1, x0:x1]
        patch_m = valid_mask[y0:y1, x0:x1]
        if not np.any(patch_m):  # 周囲すべて無効
            continue
        vals = patch_l[patch_m]
        uniq, cnt = np.unique(vals, return_counts=True)
        out[y, x] = int(uniq[np.argmax(cnt)])
    return out


# ============ Dataset 本体（膨張版） ============

# class SemanticKitti(Dataset):
#     """
#     - 投影：SemLaserScan (レンジ画像作成)
#     - 前処理：DILATE（反復 + 平均/最大ブレンド）※ train のみ
#     - ラベル：多数決で欠損を埋めるかどうかは fill_label で制御
#     - 出力：6ch（range/xyz/rem + mask）と学習ラベル

#       train:  fill_label=True  → range 穴埋め + ラベル穴埋め + マスクも埋めた部分を有効扱い
#       valid:  fill_label=False → range 穴埋めなし / ラベル穴埋めなし / マスクは元の投影点のみ
#     """
#     def __init__(self, root, sequences, labels, color_map,
#                  learning_map, learning_map_inv, sensor,
#                  max_points=150000, gt=True, skip=0,
#                  dilate_passes=3, kernel_size=11, max_blend=0.6,
#                  fill_label: bool = True):
#         super().__init__()
#         self.root = os.path.join(root, "sequences")
#         self.sequences = [f"{int(s):02d}" for s in sequences]
#         self.labels = labels
#         self.color_map = color_map
#         self.learning_map = learning_map
#         self.learning_map_inv = learning_map_inv
#         self.sensor = sensor
#         self.H = sensor["img_prop"]["height"]
#         self.W = sensor["img_prop"]["width"]
#         self.fov_up = sensor["fov_up"]
#         self.fov_down = sensor["fov_down"]
#         self.img_means = torch.tensor(sensor["img_means"], dtype=torch.float32)
#         self.img_stds  = torch.tensor(sensor["img_stds"], dtype=torch.float32)
#         self.max_points = max_points
#         self.gt = gt

#         self.dilate_passes = dilate_passes
#         self.kernel_size = kernel_size
#         self.max_blend = max_blend

#         # ★ train/valid の挙動切り替えフラグ
#         self.fill_label = fill_label

#         # LUT
#         self.lut_map = _build_lut(self.learning_map)

#         # ファイル列挙
#         self.scan_files, self.label_files = [], []
#         for seq in self.sequences:
#             vpath = os.path.join(self.root, seq, "velodyne")
#             lpath = os.path.join(self.root, seq, "labels")
#             scans = [os.path.join(vpath, f) for f in sorted(os.listdir(vpath)) if f.endswith(".bin")]
#             self.scan_files += scans
#             if self.gt:
#                 labels = [os.path.join(lpath, f) for f in sorted(os.listdir(lpath)) if f.endswith(".label")]
#                 self.label_files += labels
#         if skip:
#             self.scan_files = self.scan_files[::skip]
#             if self.gt:
#                 self.label_files = self.label_files[::skip]

#     def __len__(self):
#         return len(self.scan_files)

#     def __getitem__(self, index):
#         scan_file = self.scan_files[index]
#         label_file = self.label_files[index] if self.gt else None

#         scan = SemLaserScan(self.color_map, project=True,
#                             H=self.H, W=self.W,
#                             fov_up=self.fov_up, fov_down=self.fov_down)
#         scan.open_scan(scan_file)
#         if self.gt:
#             scan.open_label(label_file)
#             # 学習マップに変換
#             scan.sem_label = self.lut_map[scan.sem_label]
#             scan.proj_sem_label = self.lut_map[scan.proj_sem_label]

#         proj_range = scan.proj_range.astype(np.float32)
#         proj_mask  = scan.proj_mask.astype(bool)
#         proj_xyz   = scan.proj_xyz.astype(np.float32)
#         proj_rem   = scan.proj_remission.astype(np.float32)
#         proj_label = scan.proj_sem_label.astype(np.int32) if self.gt else None

#         # ===========================
#         # ① range の穴埋め (train のみ)
#         # ===========================
#         if self.fill_label:
#             # train: DILATE で range を埋める
#             proj_range_used, proj_mask_used = dilate_fill_iterative(
#                 proj_range, proj_mask,
#                 passes=self.dilate_passes,
#                 kernel_size=self.kernel_size,
#                 max_blend=self.max_blend
#             )
#         else:
#             # valid / test: 穴埋めしない（元の投影をそのまま使用）
#             proj_range_used = proj_range
#             proj_mask_used  = proj_mask

#         # ===========================
#         # ② ラベル穴埋め (train のみ)
#         # ===========================
#         if self.gt:
#             if self.fill_label:
#                 # train: 欠損ラベルも多数決で埋める
#                 proj_label_used = dilate_label_fill_majority(
#                     proj_label, proj_mask, kernel_size=7
#                 )
#             else:
#                 # valid: ラベル穴埋め無し → 投影された点だけ評価
#                 proj_label_used = proj_label
#         else:
#             proj_label_used = None

#         # ===========================
#         # ③ 学習入力 6ch
#         # ===========================
#         # NOTE: 埋めているのは range だけ。xyz, rem は元のまま（穴あり）
#         ch0 = _normalize01(proj_range_used)     # range (train: 埋め済 / valid: 生)
#         ch1 = _normalize01(proj_xyz[..., 0])    # x
#         ch2 = _normalize01(proj_xyz[..., 1])    # y
#         ch3 = _normalize01(proj_xyz[..., 2])    # z
#         ch4 = _normalize01(proj_rem)           # remission
#         ch5 = proj_mask_used.astype(np.float32)  # マスクも train/valid で切り替え

#         img6 = np.stack([ch0, ch1, ch2, ch3, ch4, ch5], axis=0)
#         # 統計正規化（最初の5ch）
#         img6[:5] = (torch.from_numpy(img6[:5]) - self.img_means[:, None, None]) / self.img_stds[:, None, None]
#         proj = torch.from_numpy(img6).float()

#         mask_t = torch.from_numpy(ch5).unsqueeze(0).float()

#         if self.gt:
#             labels = torch.from_numpy(proj_label_used.astype(np.int64))
#             return proj, mask_t, labels
#         else:
#             return proj, mask_t, None

#     @staticmethod
#     def map(label, mapdict):
#         """
#         Parser.to_xentropy から呼ばれるクラスメソッド。
#         元の SemanticKitti3.py と同じ実装。
#         """
#         maxkey = 0
#         for key, data in mapdict.items():
#             nel = len(data) if isinstance(data, list) else 1
#             if key > maxkey:
#                 maxkey = key
#         lut = (np.zeros((maxkey + 100, nel), dtype=np.int32)
#                if nel > 1 else
#                np.zeros((maxkey + 100), dtype=np.int32))
#         for key, data in mapdict.items():
#             try:
#                 lut[key] = data
#             except IndexError:
#                 print("Wrong key ", key)
#         return lut[label]

class SemanticKitti(Dataset):
    """
    - 投影：SemLaserScan (レンジ画像作成)
    - 前処理：DILATE（反復 + 平均/最大ブレンド）→ train / valid 両方で実施
    - ラベル：
        train: 多数決で欠損を埋める（擬似ラベル）
        valid: 欠損は埋めず、投影された点だけを評価対象にする
    - 出力：6ch（range/xyz/rem + mask）と学習ラベル

      train:  fill_label=True  → range 穴埋め + ラベル穴埋め + マスクも埋めた部分を有効扱い
      valid:  fill_label=False → range 穴埋めあり / ラベル穴埋めなし / マスクは元の投影点のみ
    """
    def __init__(self, root, sequences, labels, color_map,
                 learning_map, learning_map_inv, sensor,
                 max_points=150000, gt=True, skip=0,
                 dilate_passes=3, kernel_size=11, max_blend=0.6,
                 fill_label: bool = True):
        super().__init__()
        self.root = os.path.join(root, "sequences")
        self.sequences = [f"{int(s):02d}" for s in sequences]
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.H = sensor["img_prop"]["height"]
        self.W = sensor["img_prop"]["width"]
        self.fov_up = sensor["fov_up"]
        self.fov_down = sensor["fov_down"]
        self.img_means = torch.tensor(sensor["img_means"], dtype=torch.float32)
        self.img_stds  = torch.tensor(sensor["img_stds"], dtype=torch.float32)
        self.max_points = max_points
        self.gt = gt

        # DILATE のパラメータ
        self.dilate_passes = dilate_passes
        self.kernel_size = kernel_size
        self.max_blend = max_blend

        # ★ train / valid の挙動切り替えフラグ
        #   - train  : fill_label=True
        #   - valid  : fill_label=False
        self.fill_label = fill_label

        # LUT
        self.lut_map = _build_lut(self.learning_map)

        # ファイル列挙
        self.scan_files, self.label_files = [], []
        for seq in self.sequences:
            vpath = os.path.join(self.root, seq, "velodyne")
            lpath = os.path.join(self.root, seq, "labels")
            scans = [os.path.join(vpath, f) for f in sorted(os.listdir(vpath)) if f.endswith(".bin")]
            self.scan_files += scans
            if self.gt:
                labels = [os.path.join(lpath, f) for f in sorted(os.listdir(lpath)) if f.endswith(".label")]
                self.label_files += labels
        if skip:
            self.scan_files = self.scan_files[::skip]
            if self.gt:
                self.label_files = self.label_files[::skip]

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, index):
        scan_file = self.scan_files[index]
        label_file = self.label_files[index] if self.gt else None

        scan = SemLaserScan(self.color_map, project=True,
                            H=self.H, W=self.W,
                            fov_up=self.fov_up, fov_down=self.fov_down)
        scan.open_scan(scan_file)
        if self.gt:
            scan.open_label(label_file)
            # 学習ラベルにマッピング
            scan.sem_label = self.lut_map[scan.sem_label]
            scan.proj_sem_label = self.lut_map[scan.proj_sem_label]

        proj_range = scan.proj_range.astype(np.float32)
        proj_mask  = scan.proj_mask.astype(bool)
        proj_xyz   = scan.proj_xyz.astype(np.float32)
        proj_rem   = scan.proj_remission.astype(np.float32)
        proj_label = scan.proj_sem_label.astype(np.int32) if self.gt else None

        # ===========================
        # ① range の穴埋め（train / valid 共通）
        # ===========================
        proj_range_filled, proj_mask_filled = dilate_fill_iterative(
            proj_range, proj_mask,
            passes=self.dilate_passes,
            kernel_size=self.kernel_size,
            max_blend=self.max_blend
        )

        # ===========================
        # ② ラベル穴埋め（train のみ）
        # ===========================
        if self.gt:
            if self.fill_label:
                # train: 欠損ラベルも多数決で埋める（擬似ラベル込みで学習）
                proj_label_used = dilate_label_fill_majority(
                    proj_label, proj_mask, kernel_size=7
                )
                # マスクは DILATE 後のもの（擬似ラベルも有効扱い）
                mask_used = proj_mask_filled
            else:
                # valid: ラベルは元の投影ラベルのみを使う（穴は unlabeled のまま）
                proj_label_used = proj_label
                # マスクは元の投影点だけを有効にする → 評価・loss は元点のみ
                mask_used = proj_mask
        else:
            proj_label_used = None
            # 推論のみのときは inpaint 後マスクをそのまま使う
            mask_used = proj_mask_filled

        # ===========================
        # ③ 学習入力 6ch
        # ===========================
        # NOTE: 埋めているのは range だけ。xyz / rem は元のまま（穴あり）
        ch0 = _normalize01(proj_range_filled)       # range（train/valid: 穴埋め済み）
        ch1 = _normalize01(proj_xyz[..., 0])        # x
        ch2 = _normalize01(proj_xyz[..., 1])        # y
        ch3 = _normalize01(proj_xyz[..., 2])        # z
        ch4 = _normalize01(proj_rem)               # remission
        ch5 = mask_used.astype(np.float32)         # ★ train / valid でマスクを切り替え

        img6 = np.stack([ch0, ch1, ch2, ch3, ch4, ch5], axis=0)
        # 統計正規化（最初の5chのみ）
        img6[:5] = (torch.from_numpy(img6[:5]) - self.img_means[:, None, None]) / self.img_stds[:, None, None]
        proj = torch.from_numpy(img6).float()

        mask_t = torch.from_numpy(ch5).unsqueeze(0).float()

        if self.gt:
            labels = torch.from_numpy(proj_label_used.astype(np.int64))
            return proj, mask_t, labels
        else:
            return proj, mask_t, None

    @staticmethod
    def map(label, mapdict):
        """
        Parser.to_xentropy から呼ばれるクラスメソッド。
        元の SemanticKitti3.py と同じ実装。
        """
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
