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


# ============ INPAINT（joint bilateral + 反復） ============

def median_filter_on_valid(img: np.ndarray, valid_mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    if ksize <= 1:
        return img.copy()
    tmp = img.copy()
    tmp[~valid_mask] = 0.0
    med = cv2.medianBlur(tmp.astype(np.float32), int(ksize))
    out = np.where(valid_mask, med, -1.0).astype(np.float32)
    return out

def joint_bilateral_inpaint(range_img: np.ndarray,
                            rem_img: np.ndarray,
                            valid_mask: np.ndarray,
                            window: int = 5,
                            sigma_spatial: float = 2.0,
                            sigma_guide: float = 0.2,
                            iters: int = 2,
                            median_ksize: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    反復 joint-bilateral による深度補完。
    ガイド：|∇range| と remission（0..1に正規化）
    """
    H, W = range_img.shape
    out = range_img.copy()

    # ガイド成分
    r = range_img.copy()
    r[r <= 0] = 0.0
    gx = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    rem_norm = cv2.normalize(rem_img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    guide = np.stack([grad, rem_norm], axis=-1)

    half = window // 2
    invalid = ~valid_mask
    ys, xs = np.where(invalid)

    for _ in range(max(1, int(iters))):
        updates = []
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - half), min(H, y + half + 1)
            x0, x1 = max(0, x - half), min(W, x + half + 1)
            patch_r = out[y0:y1, x0:x1]
            patch_g = guide[y0:y1, x0:x1, :]
            patch_m = valid_mask[y0:y1, x0:x1]
            if not np.any(patch_m):
                continue

            yp, xp = np.where(patch_m)
            dy = (yp + y0 - y).astype(np.float32)
            dx = (xp + x0 - x).astype(np.float32)
            w_sp = np.exp(-(dx * dx + dy * dy) / (2 * sigma_spatial * sigma_spatial))

            gp = guide[y, x, :]
            gq = patch_g[yp, xp, :]
            d2 = ((gq - gp[None, :]) ** 2).sum(axis=1)
            w_g = np.exp(-d2 / (2 * sigma_guide * sigma_guide))

            w = w_sp * w_g
            vals = patch_r[yp, xp]
            ok = vals > 0
            if not np.any(ok):
                continue
            w = w * ok.astype(np.float32)

            s = float(np.sum(w))
            if s <= 1e-6:
                continue
            v = float(np.sum(w * vals) / (s + 1e-6))
            updates.append((y, x, v))

        if not updates:
            break
        for y, x, v in updates:
            out[y, x] = v
            valid_mask[y, x] = True   # 埋めた画素も次反復に使う

    out = median_filter_on_valid(out, valid_mask, ksize=median_ksize)
    return out.astype(np.float32), valid_mask


def label_propagate_by_depth(range_filled: np.ndarray,
                             label_img: np.ndarray,
                             valid_mask: np.ndarray,
                             radius: int = 3) -> np.ndarray:
    """
    欠損ラベルを “深度が最も近い” 近傍のラベルで埋める（robust）。
    """
    H, W = label_img.shape
    out = label_img.copy()
    invalid = ~valid_mask
    ys, xs = np.where(invalid)
    r2 = radius * radius
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        x0, x1 = max(0, x - radius), min(W, x + radius + 1)
        patch_r = range_filled[y0:y1, x0:x1]
        patch_l = label_img[y0:y1, x0:x1]
        patch_m = valid_mask[y0:y1, x0:x1]
        if not np.any(patch_m):  # 周囲に有効がない
            continue
        rr = patch_r[patch_m]
        ll = patch_l[patch_m]
        med = np.median(rr)
        idx = int(np.argmin(np.abs(rr - med)))
        out[y, x] = int(ll[idx])
    return out


# ============ Dataset 本体（inpaint版） ============

class SemanticKitti(Dataset):
    """
    - 投影：SemLaserScan
    - 前処理：depth inpainting（joint bilateral + 反復）
    - ラベル：深度最近傍で欠損埋め（オプション）
    - 出力：6ch（range/xyz/rem + mask）と学習ラベル
    """
    def __init__(self, root, sequences, labels, color_map,
                 learning_map, learning_map_inv, sensor,
                 max_points=150000, gt=True, skip=0,
                 inpaint_window=5, inpaint_iters=2,
                 sigma_spatial=2.0, sigma_guide=0.2, median_ksize=3,
                 fill_label: bool = True):
        """
        fill_label:
            True  -> ラベル穴埋めを行う（train 用）
            False -> ラベル穴埋めを行わない（valid/test 用）
        """
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

        # ★ ラベルを埋めるかどうか
        self.fill_label = fill_label

        self.inpaint_window = inpaint_window
        self.inpaint_iters = inpaint_iters
        self.sigma_spatial = sigma_spatial
        self.sigma_guide = sigma_guide
        self.median_ksize = median_ksize

        self.lut_map = _build_lut(self.learning_map)

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
                            H=self.H, W=self.W, fov_up=self.fov_up, fov_down=self.fov_down)
        scan.open_scan(scan_file)
        if self.gt:
            scan.open_label(label_file)
            # 学習マップでラベル変換
            scan.sem_label = self.lut_map[scan.sem_label]
            scan.proj_sem_label = self.lut_map[scan.proj_sem_label]

        proj_range = scan.proj_range.astype(np.float32)
        # ★ 元のマスク（穴あり）
        proj_mask_orig = scan.proj_mask.astype(bool)
        proj_xyz   = scan.proj_xyz.astype(np.float32)
        proj_rem   = scan.proj_remission.astype(np.float32)
        proj_label = scan.proj_sem_label.astype(np.int32) if self.gt else None

        # --- INPAINT（反復）で depth の穴を埋める ---
        # inpaint 用にはコピーを渡す（中で書き換えられる）
        proj_mask_for_inpaint = proj_mask_orig.copy()
        proj_range_filled, proj_mask_filled = joint_bilateral_inpaint(
            proj_range, proj_rem, proj_mask_for_inpaint,
            window=self.inpaint_window, sigma_spatial=self.sigma_spatial,
            sigma_guide=self.sigma_guide, iters=self.inpaint_iters, median_ksize=self.median_ksize
        )

        # --- ラベル穴埋めはオプション ---
        if self.gt:
            if self.fill_label:
                # train: 元の穴（proj_mask_orig が False の位置）だけを埋める
                proj_label_used = label_propagate_by_depth(
                    proj_range_filled, proj_label, proj_mask_orig, radius=3
                )
                # 擬似ラベルを付けた位置も「有効」として扱う
                mask_used = proj_mask_filled
            else:
                # valid/test: ラベルは元のまま、穴は unlabeled のまま
                proj_label_used = proj_label
                # マスクは「元の有効画素のみ」有効にする（穴は loss/IoU から除外）
                mask_used = proj_mask_orig
        else:
            proj_label_used = None
            # 推論だけのときは inpaint 後マスクをそのまま使う
            mask_used = proj_mask_filled

        # --- 学習入力 6ch ---
        ch0 = _normalize01(proj_range_filled)
        ch1 = _normalize01(proj_xyz[..., 0])
        ch2 = _normalize01(proj_xyz[..., 1])
        ch3 = _normalize01(proj_xyz[..., 2])
        ch4 = _normalize01(proj_rem)
        ch5 = mask_used.astype(np.float32)  # ★ マスクも train/valid で切り替え

        img6 = np.stack([ch0, ch1, ch2, ch3, ch4, ch5], axis=0)
        # 統計正規化（5ch分）
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
