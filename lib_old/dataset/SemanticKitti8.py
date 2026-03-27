import os
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils.laserscan3 import SemLaserScan


def _normalize01(x: np.ndarray) -> np.ndarray:
    """
    Min-max 正規化 (0〜1)。全て同じ値の場合は 0 配列を返す。
    """
    x = x.astype(np.float32)
    vmin, vmax = float(np.min(x)), float(np.max(x))
    if vmax - vmin < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - vmin) / (vmax - vmin)


def _build_lut(mapdict: dict) -> np.ndarray:
    """
    semantic-kitti.yaml の learning_map などから LUT を構築。
    """
    maxkey = max(mapdict.keys()) if len(mapdict) > 0 else 0
    lut = np.zeros((maxkey + 100), dtype=np.int32)
    for k, v in mapdict.items():
        lut[k] = v
    return lut


class SemanticKitti(Dataset):
    """
    ラベル穴埋め無し + マスク付き学習 用のシンプルな SemanticKITTI データセット。

    - ラベル:
        * 元の投影ラベル proj_sem_label をそのまま使用
        * 穴 (proj_mask=False) や unlabeled は 0 のまま
        * loss / IoU 側では ignore_index=0 として完全無視する想定
    - マスク (6ch目 & Trainer に渡す proj_mask):
        * 元の投影マスク proj_mask をそのまま float で使用 (1.0 or 0.0)
        * boundary loss などでも「有効画素だけ平均」するのに使用
    - 入力 6ch:
        0: range         (0〜1 正規化後、さらに img_means / img_stds で標準化)
        1: x
        2: y
        3: z
        4: remission
        5: LiDAR マスク (proj_mask: 1.0=点あり, 0.0=点なし)

    train / valid / test すべて同じロジックで処理し、
    「ラベルの穴埋め」は一切行わない。
    """

    def __init__(self,
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
                 fill_label: bool = True):
        super().__init__()

        # ルートとシーケンス
        self.root = os.path.join(root, "sequences")
        self.sequences = [f"{int(s):02d}" for s in sequences]

        # ラベル関係
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv

        # センサー設定
        self.sensor = sensor
        self.H = sensor["img_prop"]["height"]
        self.W = sensor["img_prop"]["width"]
        self.fov_up = sensor["fov_up"]
        self.fov_down = sensor["fov_down"]
        self.img_means = torch.tensor(sensor["img_means"], dtype=torch.float32)
        self.img_stds = torch.tensor(sensor["img_stds"], dtype=torch.float32)

        self.max_points = max_points
        self.gt = gt

        # 互換性のため引数として受けるが、この版では一切使用しない
        self.fill_label = False

        # ラベルマッピング LUT
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

        # ---- LaserScan の読み込み & 投影 ----
        scan = SemLaserScan(self.color_map,
                            project=True,
                            H=self.H, W=self.W,
                            fov_up=self.fov_up,
                            fov_down=self.fov_down)
        scan.open_scan(scan_file)

        if self.gt:
            scan.open_label(label_file)
            # 学習用ラベル ID に変換 (learning_map)
            scan.sem_label = self.lut_map[scan.sem_label]
            scan.proj_sem_label = self.lut_map[scan.proj_sem_label]

        proj_range = scan.proj_range.astype(np.float32)       # (H,W)
        proj_xyz = scan.proj_xyz.astype(np.float32)           # (H,W,3)
        proj_rem = scan.proj_remission.astype(np.float32)     # (H,W)
        proj_mask = scan.proj_mask.astype(bool)               # (H,W)

        if self.gt:
            proj_label = scan.proj_sem_label.astype(np.int32)  # (H,W)
        else:
            proj_label = None

        # ---------------------------
        # 1) range / xyz / rem をそのまま使用（値の穴埋めなし）
        # ---------------------------
        # ここでは range の inpaint は一切行わず、投影そのままを使う。
        # 欠損部分 (proj_mask=False) は 0 のまま。
        ch0 = _normalize01(proj_range)
        ch1 = _normalize01(proj_xyz[..., 0])
        ch2 = _normalize01(proj_xyz[..., 1])
        ch3 = _normalize01(proj_xyz[..., 2])
        ch4 = _normalize01(proj_rem)

        # 6ch 目: LiDAR マスク (1.0 or 0.0)
        weight_map = proj_mask.astype(np.float32)
        ch5 = weight_map

        img = np.stack([ch0, ch1, ch2, ch3, ch4, ch5], axis=0).astype(np.float32)
        img_t = torch.from_numpy(img)
        # 最初の 5ch を img_means / img_stds で標準化
        img_t[:5] = (img_t[:5] - self.img_means[:, None, None]) / self.img_stds[:, None, None]
        proj_tensor = img_t.float()

        # proj_mask (loss / boundary 用)
        mask_t = torch.from_numpy(weight_map).unsqueeze(0).float()

        if self.gt:
            labels_t = torch.from_numpy(proj_label.astype(np.int64))
            return proj_tensor, mask_t, labels_t
        else:
            # テスト時などラベル無し
            return proj_tensor, mask_t, None

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
