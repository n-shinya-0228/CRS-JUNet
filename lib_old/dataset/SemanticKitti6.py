# import os
# import numpy as np
# import torch
# import cv2
# from torch.utils.data import Dataset
# from ..utils.laserscan3 import SemLaserScan


# # ============ 共通ユーティリティ ============

# def _normalize01(x: np.ndarray) -> np.ndarray:
#     x = x.astype(np.float32)
#     vmin, vmax = np.min(x), np.max(x)
#     if vmax - vmin < 1e-6:
#         return np.zeros_like(x, dtype=np.float32)
#     return (x - vmin) / (vmax - vmin)


# def _build_lut(mapdict: dict) -> np.ndarray:
#     maxkey = max(mapdict.keys()) if len(mapdict) > 0 else 0
#     lut = np.zeros((maxkey + 100), dtype=np.int32)
#     for k, v in mapdict.items():
#         lut[k] = v
#     return lut


# # ============ ラベル穴埋め：3×3・6票 majority ============

# def fill_label_majority_3x3(labels: np.ndarray,
#                             valid_mask: np.ndarray,
#                             min_votes: int = 6) -> np.ndarray:
#     """
#     欠損ラベルを「3×3 近傍 + 6票以上」の多数決で埋める保守的版。

#     - labels: (H,W) int, 0=unlabeled
#     - valid_mask: (H,W) bool, True=元の投影点が存在
#     - min_votes: 近傍8ピクセル中この数以上同じラベルならそのラベルを採用
#     """
#     H, W = labels.shape
#     out = labels.copy()

#     half = 1  # 3×3
#     invalid = ~valid_mask
#     ys, xs = np.where(invalid)

#     for y, x in zip(ys, xs):
#         y0, y1 = max(0, y - half), min(H, y + half + 1)
#         x0, x1 = max(0, x - half), min(W, x + half + 1)

#         patch_l = labels[y0:y1, x0:x1]
#         patch_m = valid_mask[y0:y1, x0:x1]

#         if not np.any(patch_m):
#             continue

#         vals = patch_l[patch_m]
#         # unlabeled(0) は投票に使わない
#         vals = vals[vals != 0]
#         if vals.size < min_votes:
#             continue

#         uniq, cnt = np.unique(vals, return_counts=True)
#         j = np.argmax(cnt)
#         if cnt[j] >= min_votes:
#             out[y, x] = int(uniq[j])

#     return out


# # ============ Label-aware inpaint: クラスごとに range 膨張 ============

# def label_aware_inpaint_range(range_img: np.ndarray,
#                               labels_filled: np.ndarray,
#                               max_iters: int = 3) -> np.ndarray:
#     """
#     ラベルをガイドに range を埋める label-aware inpaint。

#       - range_img: (H,W) float32, 0 or <=0 を欠損とみなす
#       - labels_filled: (H,W) int, 0=unlabeled (ガイドに使わない)
#       - max_iters: 膨張の反復回数（小さめ 2〜3 推奨）

#     アルゴリズム：
#       各クラス c について
#         1. valid_c = (labels==c) & (range>0)
#         2. 3x3 カーネルで dilation を行い、同クラス領域内の“穴”を埋める
#         3. これを max_iters 回繰り返す
#     """
#     H, W = range_img.shape
#     range_out = range_img.copy()

#     kernel = np.ones((3, 3), np.uint8)

#     # 0 は unlabeled なのでスキップ
#     classes = np.unique(labels_filled)
#     for c in classes:
#         if c == 0:
#             continue

#         # クラス c の領域
#         mask_c = (labels_filled == c)

#         # その中で range が有効な画素
#         valid_c = mask_c & (range_out > 0)

#         if not np.any(valid_c):
#             continue

#         # クラス c のみを対象に膨張
#         range_c = range_out.copy()

#         for _ in range(max_iters):
#             # current 有効値だけをソースにして dilation
#             src = range_c.copy()
#             src[~valid_c] = 0.0  # 有効画素以外は 0 にしておく
#             neigh_max = cv2.dilate(src.astype(np.float32), kernel, iterations=1)

#             # target: クラス c の中でまだ range が無効だが、近傍に有効値がある画素
#             target = mask_c & (~valid_c) & (neigh_max > 0)

#             if not np.any(target):
#                 break

#             range_c[target] = neigh_max[target]
#             valid_c[target] = True

#         # 最終的に埋まった値を range_out に反映
#         newly = mask_c & (range_out <= 0) & (range_c > 0)
#         range_out[newly] = range_c[newly]

#     return range_out.astype(np.float32)


# # ============ Dataset 本体（label-aware 版） ============

# class SemanticKitti(Dataset):
#     """
#     SemanticKITTI spherical projection dataset with label-aware inpaint.

#     train 時 (fill_label=True):
#       - range:
#           label_filled に基づき、クラスごとに 3×3 dilation で inpaint
#       - labels:
#           3×3・6票 majority で一部の穴に擬似ラベルを付与 (label_filled)
#       - mask(6ch目):
#           元の投影点           -> 1.0
#           擬似ラベルで埋めた点 -> alpha_pseudo (例 0.3)
#           その他               -> 0.0

#     valid 時 (fill_label=False):
#       - range:
#           元の proj_label をガイドに label-aware inpaint（ラベルは評価用のまま）
#       - labels:
#           proj_label をそのまま使用（穴は0のまま）
#       - mask(6ch目):
#           元の投影点のみ 1.0、それ以外 0.0
#     """
#     def __init__(self, root, sequences, labels, color_map,
#                  learning_map, learning_map_inv, sensor,
#                  max_points=150000, gt=True, skip=0,
#                  fill_label: bool = True,
#                  majority_min_votes: int = 6,
#                  inpaint_iters: int = 3):
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

#         self.fill_label = fill_label
#         self.majority_min_votes = majority_min_votes
#         self.inpaint_iters = inpaint_iters

#         # LUT
#         self.lut_map = _build_lut(self.learning_map)

#         # ファイル列挙
#         self.scan_files, self.label_files = [], []
#         for seq in self.sequences:
#             vpath = os.path.join(self.root, seq, "velodyne")
#             lpath = os.path.join(self.root, seq, "labels")
#             scans = [os.path.join(vpath, f)
#                      for f in sorted(os.listdir(vpath))
#                      if f.endswith(".bin")]
#             self.scan_files += scans
#             if self.gt:
#                 labels = [os.path.join(lpath, f)
#                           for f in sorted(os.listdir(lpath))
#                           if f.endswith(".label")]
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
#             # 学習ラベルにマッピング
#             scan.sem_label = self.lut_map[scan.sem_label]
#             scan.proj_sem_label = self.lut_map[scan.proj_sem_label]

#         proj_range = scan.proj_range.astype(np.float32)
#         proj_mask  = scan.proj_mask.astype(bool)          # 元の投影点マスク
#         proj_xyz   = scan.proj_xyz.astype(np.float32)
#         proj_rem   = scan.proj_remission.astype(np.float32)
#         proj_label = scan.proj_sem_label.astype(np.int32) if self.gt else None

#         # ---------------------------
#         # 1) ラベルのガイドを作る
#         # ---------------------------
#         if self.gt:
#             if self.fill_label:
#                 # train: 3×3・6票 majority で一部の穴にラベル付与
#                 label_filled = fill_label_majority_3x3(
#                     proj_label, proj_mask, min_votes=self.majority_min_votes
#                 )
#             else:
#                 # valid: ラベルは元の投影ラベルのまま（穴=0）
#                 label_filled = proj_label
#         else:
#             label_filled = np.zeros_like(proj_range, dtype=np.int32)

#         # ---------------------------
#         # 2) label-aware inpaint で range を埋める
#         # ---------------------------
#         proj_range_filled = label_aware_inpaint_range(
#             proj_range, label_filled, max_iters=self.inpaint_iters
#         )

#         # ---------------------------
#         # 3) loss 用マスク（6ch目）を作る
#         # ---------------------------
#         alpha_pseudo = 0.3  # 擬似ラベルの重み

#         if self.gt:
#             # 元の投影点: proj_mask = True
#             orig_mask = proj_mask.astype(np.float32)

#             if self.fill_label:
#                 # 擬似ラベル: label_filled != 0 かつ 元マスク=0
#                 pseudo_mask = ((label_filled != 0) & (~proj_mask)).astype(np.float32)
#                 weight_map = orig_mask + alpha_pseudo * pseudo_mask
#                 labels_used = label_filled
#             else:
#                 # valid: ラベル穴埋め無し, 元マスクのみ重み1
#                 pseudo_mask = np.zeros_like(orig_mask)
#                 weight_map = orig_mask
#                 labels_used = proj_label
#         else:
#             # テスト: マスク = proj_mask（元投影点のみ1）
#             orig_mask = proj_mask.astype(np.float32)
#             pseudo_mask = np.zeros_like(orig_mask)
#             weight_map = orig_mask
#             labels_used = None

#         # ---------------------------
#         # 4) 学習入力 6ch を構築
#         # ---------------------------
#         ch0 = _normalize01(proj_range_filled)     # range（label-aware inpaint 済み）
#         ch1 = _normalize01(proj_xyz[..., 0])      # x
#         ch2 = _normalize01(proj_xyz[..., 1])      # y
#         ch3 = _normalize01(proj_xyz[..., 2])      # z
#         ch4 = _normalize01(proj_rem)              # remission
#         ch5 = weight_map.astype(np.float32)       # ★ loss weight: 元=1.0, 擬似=0.3, その他=0

#         img6 = np.stack([ch0, ch1, ch2, ch3, ch4, ch5], axis=0)
#         # 統計正規化（最初の5chのみ）
#         img6[:5] = (torch.from_numpy(img6[:5]) - self.img_means[:, None, None]) / self.img_stds[:, None, None]
#         proj = torch.from_numpy(img6).float()

#         mask_t = torch.from_numpy(ch5).unsqueeze(0).float()

#         if self.gt:
#             labels_t = torch.from_numpy(labels_used.astype(np.int64))
#             return proj, mask_t, labels_t
#         else:
#             return proj, mask_t, None

#     @staticmethod
#     def map(label, mapdict):
#         """
#         Parser.to_xentropy から呼ばれるクラスメソッド。
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


import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from ..utils.laserscan3 import SemLaserScan

# ================== 小物体強調用の定数 ==================

# SemanticKITTI xentropy ID での小物体クラス
# ここでは特に motorcyclist (= 8) を最重要クラスとして扱う
SMALL_OBJ_CLASSES = {8}

# ラベル多数決時に小物体クラスの票数に掛ける係数
SMALL_OBJ_VOTE_BOOST = 2.0  # 例: 2倍

# range inpaint の膨張回数を小物体クラスだけ増やす
SMALL_OBJ_EXTRA_ITERS = 2   # 例: 2 iter 追加

# 擬似ラベルの loss weight（小物体だけ高くする）
ALPHA_PSEUDO_OTHER = 0.3    # 通常クラスの擬似ラベル重み
ALPHA_PSEUDO_SMALL = 0.7    # 小物体クラスの擬似ラベル重み


# ============ 共通ユーティリティ ============

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


# ============ ラベル穴埋め：3×3 majority（小物体優遇版） ============

def fill_label_majority_3x3(labels: np.ndarray,
                            valid_mask: np.ndarray,
                            min_votes: int = 6) -> np.ndarray:
    """
    欠損ラベルを「3×3 近傍 + min_votes以上」の多数決で埋める。

    - labels: (H,W) int, 0=unlabeled
    - valid_mask: (H,W) bool, True=元の投影点が存在
    - min_votes: 近傍8ピクセル中この数以上同じラベルならそのラベルを採用

    ★改良点
      - motorcyclist など SMALL_OBJ_CLASSES に含まれるクラスの票数を
        SMALL_OBJ_VOTE_BOOST 倍して、多数決で勝ちやすくする。
        ただし「min_votes 以上」という条件は元の票数 cnt で判定するので、
        1〜2ピクセルからの暴走拡大は防ぐ。
    """
    H, W = labels.shape
    out = labels.copy()

    half = 1  # 3×3
    invalid = ~valid_mask
    ys, xs = np.where(invalid)

    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - half), min(H, y + half + 1)
        x0, x1 = max(0, x - half), min(W, x + half + 1)

        patch_l = labels[y0:y1, x0:x1]
        patch_m = valid_mask[y0:y1, x0:x1]

        if not np.any(patch_m):
            continue

        vals = patch_l[patch_m]
        # unlabeled(0) は投票に使わない
        vals = vals[vals != 0]
        if vals.size < min_votes:
            continue

        uniq, cnt = np.unique(vals, return_counts=True)

        # --- 小物体クラスの票数だけブースト ---
        scores = cnt.astype(np.float32)
        for idx, cls in enumerate(uniq):
            if cls in SMALL_OBJ_CLASSES:
                scores[idx] *= SMALL_OBJ_VOTE_BOOST

        j = np.argmax(scores)

        # ★ min_votes 判定は元の cnt で行う（ブースト後ではない）
        if cnt[j] >= min_votes:
            out[y, x] = int(uniq[j])

    return out


# ============ Label-aware inpaint: クラスごとに range 膨張（小物体は余分に拡張） ============

def label_aware_inpaint_range(range_img: np.ndarray,
                              labels_filled: np.ndarray,
                              max_iters: int = 3) -> np.ndarray:
    """
    ラベルをガイドに range を埋める label-aware inpaint。

      - range_img: (H,W) float32, 0 or <=0 を欠損とみなす
      - labels_filled: (H,W) int, 0=unlabeled (ガイドに使わない)
      - max_iters: 膨張の基本回数（例: 3）

    アルゴリズム：
      各クラス c について
        1. valid_c = (labels==c) & (range>0)
        2. 3x3 カーネルで dilation を行い、同クラス領域内の“穴”を埋める
        3. これを max_iters 回繰り返す
        4. ただし small-object クラス(c ∈ SMALL_OBJ_CLASSES) だけは
           max_iters + SMALL_OBJ_EXTRA_ITERS 回実行し、少し厚めに拡張する。
    """
    H, W = range_img.shape
    range_out = range_img.copy()

    kernel = np.ones((3, 3), np.uint8)

    # 0 は unlabeled なのでスキップ
    classes = np.unique(labels_filled)
    for c in classes:
        if c == 0:
            continue

        # クラス c の領域
        mask_c = (labels_filled == c)

        # その中で range が有効な画素
        valid_c = mask_c & (range_out > 0)

        if not np.any(valid_c):
            continue

        # クラス c のみを対象に膨張
        range_c = range_out.copy()

        # 小物体クラスだけ膨張回数を増やす
        local_iters = max_iters + \
            SMALL_OBJ_EXTRA_ITERS if c in SMALL_OBJ_CLASSES else max_iters

        for _ in range(local_iters):
            # current 有効値だけをソースにして dilation
            src = range_c.copy()
            src[~valid_c] = 0.0  # 有効画素以外は 0 にしておく
            neigh_max = cv2.dilate(src.astype(np.float32), kernel, iterations=1)

            # target: クラス c の中でまだ range が無効だが、近傍に有効値がある画素
            target = mask_c & (~valid_c) & (neigh_max > 0)

            if not np.any(target):
                break

            range_c[target] = neigh_max[target]
            valid_c[target] = True

        # 最終的に埋まった値を range_out に反映
        newly = mask_c & (range_out <= 0) & (range_c > 0)
        range_out[newly] = range_c[newly]

    return range_out.astype(np.float32)


# ============ Dataset 本体（label-aware 版） ============

class SemanticKitti(Dataset):
    """
    SemanticKITTI spherical projection dataset with label-aware inpaint.

    ★改良版の方針
      - train / valid どちらも「クラス穴埋め＋range の inpaint」を同じロジックで実行
      - 小物体クラス（特に motorcyclist=8）を多数決と inpaint 回数で優遇して、
        2D 画像上で少し“厚め”に表現しやすくする
      - 擬似ラベルで埋めた画素にはマスクで重みを付けるが、
        小物体クラスの擬似ラベルは他クラスより高い重みで学習させる
      - test(gt=False) のときはラベル穴埋めなし、マスクは元の投影点だけ 1.0
    """
    def __init__(self, root, sequences, labels, color_map,
                 learning_map, learning_map_inv, sensor,
                 max_points=150000, gt=True, skip=0,
                 fill_label: bool = True,      # 互換性のため引数だけ残す
                 majority_min_votes: int = 6,
                 inpaint_iters: int = 3):
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

        # fill_label はもう分岐には使わない（train/valid 共通で穴埋め）
        self.fill_label = True
        self.majority_min_votes = majority_min_votes
        self.inpaint_iters = inpaint_iters

        # LUT
        self.lut_map = _build_lut(self.learning_map)

        # ファイル列挙
        self.scan_files, self.label_files = [], []
        for seq in self.sequences:
            vpath = os.path.join(self.root, seq, "velodyne")
            lpath = os.path.join(self.root, seq, "labels")
            scans = [os.path.join(vpath, f)
                     for f in sorted(os.listdir(vpath))
                     if f.endswith(".bin")]
            self.scan_files += scans
            if self.gt:
                labels = [os.path.join(lpath, f)
                          for f in sorted(os.listdir(lpath))
                          if f.endswith(".label")]
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
        proj_mask  = scan.proj_mask.astype(bool)          # 元の投影点マスク
        proj_xyz   = scan.proj_xyz.astype(np.float32)
        proj_rem   = scan.proj_remission.astype(np.float32)
        proj_label = scan.proj_sem_label.astype(np.int32) if self.gt else None

        # ---------------------------
        # 1) ラベルのガイドを作る（小物体優遇版 majority）
        # ---------------------------
        if self.gt:
            label_filled = fill_label_majority_3x3(
                proj_label, proj_mask, min_votes=self.majority_min_votes
            )
        else:
            label_filled = np.zeros_like(proj_range, dtype=np.int32)

        # ---------------------------
        # 2) label-aware inpaint で range を埋める（小物体だけ extra iters）
        # ---------------------------
        proj_range_filled = label_aware_inpaint_range(
            proj_range, label_filled, max_iters=self.inpaint_iters
        )

        # ---------------------------
        # 3) loss 用マスク（6ch目）を作る
        # ---------------------------
        if self.gt:
            # 元の投影点: proj_mask = True
            orig_mask = proj_mask.astype(np.float32)

            # 擬似ラベル: label_filled != 0 かつ 元マスク=0
            pseudo_mask_all = ((label_filled != 0) & (~proj_mask)).astype(np.float32)

            # motorcyclist など小物体クラスの擬似ラベルだけ重みを高くする
            small_mask = pseudo_mask_all * np.isin(
                label_filled, list(SMALL_OBJ_CLASSES)
            ).astype(np.float32)
            other_mask = pseudo_mask_all - small_mask  # 残り

            weight_map = (
                orig_mask
                + ALPHA_PSEUDO_SMALL * small_mask
                + ALPHA_PSEUDO_OTHER * other_mask
            )

            # train/valid 共通で穴埋め後ラベルを loss / IoU に渡す
            labels_used = label_filled
        else:
            # テスト: マスク = proj_mask（元投影点のみ1）
            orig_mask = proj_mask.astype(np.float32)
            weight_map = orig_mask
            labels_used = None

        # ---------------------------
        # 4) 学習入力 6ch を構築
        # ---------------------------
        ch0 = _normalize01(proj_range_filled)     # range（label-aware inpaint 済み）
        ch1 = _normalize01(proj_xyz[..., 0])      # x
        ch2 = _normalize01(proj_xyz[..., 1])      # y
        ch3 = _normalize01(proj_xyz[..., 2])      # z
        ch4 = _normalize01(proj_rem)              # remission
        ch5 = weight_map.astype(np.float32)       # ★ loss weight: 元=1.0, 擬似(小物体)=0.7, 擬似(その他)=0.3

        img6 = np.stack([ch0, ch1, ch2, ch3, ch4, ch5], axis=0)
        # 統計正規化（最初の5chのみ）
        img6[:5] = (torch.from_numpy(img6[:5]) - self.img_means[:, None, None]) / self.img_stds[:, None, None]
        proj = torch.from_numpy(img6).float()

        mask_t = torch.from_numpy(ch5).unsqueeze(0).float()

        if self.gt:
            labels_t = torch.from_numpy(labels_used.astype(np.int64))
            return proj, mask_t, labels_t
        else:
            return proj, mask_t, None

    @staticmethod
    def map(label, mapdict):
        """
        Parser.to_xentropy から呼ばれるクラスメソッド。
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
