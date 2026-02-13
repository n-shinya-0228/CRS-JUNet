# check_point3.py
# 深度の穴埋め (DILATE vs INPAINT) と、ラベルの穴埋め結果を可視化して保存するスクリプト

import os
import sys
import json
import numpy as np
import cv2
from datetime import datetime

# ---- Projection utils ----
from lib.utils.laserscan3 import SemLaserScan  # labels も扱える版

# ====================== CONFIG ======================
# 対象の .bin ファイル 1枚
BIN_PATH = "/home/jun/src/SemanticKitti/sequences/01/velodyne/000001.bin"

# .label のパスは BIN_PATH から自動生成 (/velodyne/ → /labels/ , .bin → .label)
def guess_label_path(bin_path: str) -> str:
    seq_dir = os.path.dirname(bin_path)               # .../sequences/08/velodyne
    base = os.path.basename(bin_path).replace(".bin", ".label")
    labels_dir = os.path.join(os.path.dirname(seq_dir), "labels")
    return os.path.join(labels_dir, base)

LABEL_PATH = guess_label_path(BIN_PATH)

# 出力ディレクトリ
SAVE_DIR = f"output_compare_full/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# 投影パラメータ
H = 64
W = 512
FOV_UP = 3.0
FOV_DOWN = -23.0

# Dilation 用パラメータ（range）
DILATE_KERNEL = 7
DILATE_PASSES = 3
DILATE_MAX_BLEND = 0.8

# Inpaint 用パラメータ（depth joint-bilateral）
WIN = 5
SIGMA_SPATIAL = 2.0
SIGMA_GUIDE = 0.2
INPAINT_ITERS = 1

# 軽いノイズ除去 (0 なら無効)
MEDIAN_KSIZE = 0

# 可視化
ZERO_GRAY = 180
# ====================================================


# ---------------- 共通ヘルパー (range 可視化) ----------------
def normalize_to_uint8(arr, valid_mask):
    """正の depth を 0..255 に正規化, invalid は 0 のまま"""
    arr = arr.astype(np.float32)
    pos = np.where(arr > 0, arr, 0.0)
    vmax = float(np.max(pos)) if np.any(pos > 0) else 0.0
    out = np.zeros_like(pos, dtype=np.uint8)
    if vmax > 0:
        out = (pos / vmax * 255.0).astype(np.uint8)
    out[~valid_mask] = 0
    return out


def colorize_range(range_u8, zero_gray=ZERO_GRAY):
    """COLORMAP_JET で可視化しつつ, 0 はグレーに塗る"""
    color = cv2.applyColorMap(range_u8, cv2.COLORMAP_JET)
    zero_mask = (range_u8 == 0)
    color[zero_mask] = (zero_gray, zero_gray, zero_gray)
    return color


def compute_zero_stats(range_img, valid_mask):
    Hh, Ww = range_img.shape
    zero_or_neg = (range_img <= 0.0)
    return {
        "overall_zero_or_neg_ratio": float(np.mean(zero_or_neg)),
        "bottom_half_zero_or_neg_ratio": float(np.mean(zero_or_neg[Hh//2:])),
        "pixels_total": int(Hh * Ww),
        "pixels_zero_or_neg": int(np.sum(zero_or_neg)),
    }


def concat_h(images, pad=6, pad_color=(32, 32, 32)):
    """画像を横に並べて 1 枚にする"""
    h = max(img.shape[0] for img in images)
    out = []
    for img in images:
        if img.shape[0] != h:
            img = cv2.resize(
                img,
                (int(img.shape[1] * h / img.shape[0]), h),
                interpolation=cv2.INTER_NEAREST,
            )
        out.append(img)
        out.append(np.full((h, pad, 3), pad_color, dtype=np.uint8))
    return np.concatenate(out[:-1], axis=1)


# ---------------- range の DILATE 穴埋め ----------------
def dilate_range_fill(range_img, valid_mask, kernel_size=5, max_blend=0.8):
    """
    SemanticKitti4.py の dilate_range_fill と同じ思想:
    - 有効画素の近傍平均＋近傍最大をブレンドして invalid を埋める
    """
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1

    base = range_img.copy()
    base[~valid_mask] = 0.0

    # 有効画素数と合計
    num = cv2.blur(valid_mask.astype(np.float32), (k, k))
    den = cv2.blur(base.astype(np.float32), (k, k))
    mean_prop = np.where(num > 1e-6, den / (num + 1e-6), 0.0)

    max_prop = cv2.dilate(base.astype(np.float32), np.ones((k, k), np.uint8), iterations=1)
    fill = (1.0 - max_blend) * mean_prop + max_blend * max_prop

    out = range_img.copy()
    out[~valid_mask] = fill[~valid_mask]
    out[~np.isfinite(out)] = -1.0
    return out.astype(np.float32)


# ---------------- range の inpaint (joint-bilateral) ----------------
def joint_bilateral_inpaint(range_img, rem_img, valid_mask,
                            window=5, sigma_spatial=2.0,
                            sigma_guide=0.2, iters=1):
    """
    SemanticKitti5.py に倣った joint-bilateral inpaint:
      w = exp(-||Δxy||^2 / 2σs^2) * exp(-||Δguide||^2 / 2σg^2)
    guide: [|∇range|, remission]
    """
    Hh, Ww = range_img.shape
    out = range_img.copy()

    r = range_img.copy()
    r[r <= 0] = 0.0
    gx = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    def to01(x):
        pos = x[x > 0]
        if pos.size == 0:
            return np.zeros_like(x, dtype=np.float32)
        p1, p99 = np.percentile(pos, 1), np.percentile(pos, 99)
        if p99 - p1 < 1e-6:
            return np.zeros_like(x, dtype=np.float32)
        y = (x - p1) / (p99 - p1)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    g1 = to01(grad)
    g2 = to01(rem_img)
    guide = np.stack([g1, g2], axis=-1)

    half = window // 2
    invalid = ~valid_mask
    ys, xs = np.where(invalid)

    for _ in range(max(1, int(iters))):
        updates = []
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - half), min(Hh, y + half + 1)
            x0, x1 = max(0, x - half), min(Ww, x + half + 1)
            pv = valid_mask[y0:y1, x0:x1]
            if not np.any(pv):
                continue
            pr = out[y0:y1, x0:x1]
            pg = guide[y0:y1, x0:x1, :]
            yy, xx = np.where(pv)
            dy = (yy + y0 - y).astype(np.float32)
            dx = (xx + x0 - x).astype(np.float32)
            w_sp = np.exp(-(dy * dy + dx * dx) / (2.0 * sigma_spatial * sigma_spatial))

            gp = guide[y, x, :]
            gq = pg[yy, xx, :]
            diff = gq - gp[None, :]
            d2 = (diff * diff).sum(axis=1)
            w_g = np.exp(-d2 / (2.0 * sigma_guide * sigma_guide))

            w = w_sp * w_g
            vals = pr[yy, xx]
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
            valid_mask[y, x] = True

    out[~valid_mask] = -1.0
    return out.astype(np.float32)


def median_filter_on_valid(range_img, valid_mask, ksize=3):
    if not ksize or ksize <= 1:
        return range_img.astype(np.float32)
    tmp = range_img.copy()
    tmp[~valid_mask] = 0.0
    med = cv2.medianBlur(tmp.astype(np.float32), int(ksize))
    out = np.where(valid_mask, med, -1.0).astype(np.float32)
    return out


# ---------------- ラベルの穴埋め (SemanticKitti4/5 と同じ思想) ----------------
def dilate_label_fill_majority(label_img, valid_mask, kernel_size=7):
    """
    SemanticKitti4.py の dilate_label_fill_majority とほぼ同じ:
    欠損ラベルを近傍の多数決で埋める
    """
    Hh, Ww = label_img.shape
    out = label_img.copy()
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    half = k // 2
    invalid = ~valid_mask
    ys, xs = np.where(invalid)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - half), min(Hh, y + half + 1)
        x0, x1 = max(0, x - half), min(Ww, x + half + 1)
        patch_l = label_img[y0:y1, x0:x1]
        patch_m = valid_mask[y0:y1, x0:x1]
        if not np.any(patch_m):
            continue
        vals = patch_l[patch_m]
        uniq, cnt = np.unique(vals, return_counts=True)
        out[y, x] = int(uniq[np.argmax(cnt)])
    return out


def label_propagate_by_depth(range_filled, label_img, valid_mask, radius=3):
    """
    SemanticKitti5.py の label_propagate_by_depth と同じ:
    欠損ラベルを「深度が中央値に一番近い近傍」のラベルで埋める
    """
    Hh, Ww = label_img.shape
    out = label_img.copy()
    invalid = ~valid_mask
    ys, xs = np.where(invalid)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - radius), min(Hh, y + radius + 1)
        x0, x1 = max(0, x - radius), min(Ww, x + radius + 1)
        patch_r = range_filled[y0:y1, x0:x1]
        patch_l = label_img[y0:y1, x0:x1]
        patch_v = valid_mask[y0:y1, x0:x1]
        if not np.any(patch_v):
            continue
        rr = patch_r[patch_v]
        ll = patch_l[patch_v]
        med = np.median(rr)
        idx = int(np.argmin(np.abs(rr - med)))
        out[y, x] = int(ll[idx])
    return out


# ---------------- ラベル画像のカラー可視化 ----------------
def labels_to_color_image(label_img, valid_mask=None):
    """
    ラベル ID ごとに固定の疑似カラーを割り当てて BGR 画像に変換する
    （クラス意味は気にせず穴埋めの挙動だけ確認したい用途）
    """
    label_img = label_img.astype(np.int32)
    max_id = int(label_img.max())
    if max_id < 0:
        max_id = 0

    # 0..max_id までのカラー表を作る（適当な hash 色）
    colors = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for i in range(max_id + 1):
        # 適当だが再現性のある色
        colors[i] = [
            (i * 37) % 256,
            (i * 17) % 256,
            (i * 97) % 256,
        ]

    color_img = colors[label_img]  # (H,W,3) BGR

    # 無効画素はグレーに
    if valid_mask is not None:
        inv = ~valid_mask
        color_img[inv] = (ZERO_GRAY, ZERO_GRAY, ZERO_GRAY)

    return color_img


def compute_boundary_map(label_img, valid_mask, ksize=3, thresh=0.1):
    """
    ラベル画像から境界マップ(連続値)と境界マスク(0/1)を作る
    - Sobel でラベルの勾配を取り、勾配の大きいところ＝境界とみなす
    - thresh: 境界とみなす閾値
    """
    lab = label_img.astype(np.float32)

    # ラベル勾配 (x,y)
    gx = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx * gx + gy * gy)  # 勾配の大きさ

    # 無効画素は 0 に
    mag[~valid_mask] = 0.0

    # 閾値以上を境界とみなす
    boundary_mask = mag > thresh

    return mag, boundary_mask


# ---------------- main ----------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    if not os.path.isfile(BIN_PATH):
        print(f"[ERROR] BIN_PATH not found: {BIN_PATH}")
        sys.exit(1)
    if not os.path.isfile(LABEL_PATH):
        print(f"[ERROR] LABEL_PATH not found: {LABEL_PATH}")
        sys.exit(1)

    # ---- SemLaserScan で projection + label 読み込み ----
    # 色は使わないので sem_color_dict=None & max_classes を大きめにしておく
    scan = SemLaserScan(
        sem_color_dict=None,
        project=True,
        H=H,
        W=W,
        fov_up=FOV_UP,
        fov_down=FOV_DOWN,
        max_classes=3000,  # 254 など大きめのラベルIDにも対応
    )
    scan.open_scan(BIN_PATH)
    scan.open_label(LABEL_PATH)

    # ---- 生ラベルを learning_map で学習用ID(0-19)へ圧縮 ----
    import yaml

    with open("config/labels/semantic-kitti.yaml", "r") as f:
        label_cfg = yaml.safe_load(f)

    learning_map = label_cfg["learning_map"]

    proj_label_raw = scan.proj_sem_label.copy()

    max_key = max(learning_map.keys())
    lookup = np.zeros(max_key + 1, dtype=np.int32)
    for k, v in learning_map.items():
        lookup[k] = v

    proj_label_train = lookup[np.clip(proj_label_raw, 0, max_key)]

    scan.proj_sem_label = proj_label_train

    raw_range = scan.proj_range.astype(np.float32)          # [-1 or depth]
    raw_mask = scan.proj_mask.astype(bool)                  # 投影された点だけ True
    raw_rem  = scan.proj_remission.astype(np.float32)
    raw_label = scan.proj_sem_label.astype(np.int32)        # 投影ラベル（穴は 0 が多い）

    # ---- range 可視化 (Raw) ----
    raw_u8 = normalize_to_uint8(raw_range, raw_mask)
    raw_color = colorize_range(raw_u8)
    cv2.imwrite(os.path.join(SAVE_DIR, "raw_range_colormap.png"), raw_color)
    cv2.imwrite(os.path.join(SAVE_DIR, "raw_mask.png"), (raw_mask.astype(np.uint8) * 255))
    stats_raw = compute_zero_stats(raw_range, raw_mask)

    # ---- Method A: DILATE range ----
    valid_a = raw_mask.copy()
    range_a = raw_range.copy()
    for _ in range(DILATE_PASSES):
        range_a = dilate_range_fill(range_a, valid_a, kernel_size=DILATE_KERNEL, max_blend=DILATE_MAX_BLEND)
        valid_a[:] = True

    if MEDIAN_KSIZE and MEDIAN_KSIZE > 1:
        range_a = median_filter_on_valid(range_a, valid_a, ksize=MEDIAN_KSIZE)

    u8_a = normalize_to_uint8(range_a, valid_a)
    color_a = colorize_range(u8_a)
    cv2.imwrite(os.path.join(SAVE_DIR, "dilate_range_colormap.png"), color_a)
    cv2.imwrite(os.path.join(SAVE_DIR, "dilate_valid_mask.png"), (valid_a.astype(np.uint8) * 255))
    stats_a = compute_zero_stats(range_a, valid_a)

    # ---- Method B: INPAINT range ----
    valid_b = raw_mask.copy()
    range_b = joint_bilateral_inpaint(
        raw_range,
        raw_rem,
        valid_b,
        window=WIN,
        sigma_spatial=SIGMA_SPATIAL,
        sigma_guide=SIGMA_GUIDE,
        iters=INPAINT_ITERS,
    )
    if MEDIAN_KSIZE and MEDIAN_KSIZE > 1:
        range_b = median_filter_on_valid(range_b, valid_b, ksize=MEDIAN_KSIZE)

    u8_b = normalize_to_uint8(range_b, valid_b)
    color_b = colorize_range(u8_b)
    cv2.imwrite(os.path.join(SAVE_DIR, "inpaint_range_colormap.png"), color_b)
    cv2.imwrite(os.path.join(SAVE_DIR, "inpaint_valid_mask.png"), (valid_b.astype(np.uint8) * 255))
    stats_b = compute_zero_stats(range_b, valid_b)

    # ---- range panel (RAW | DILATE | INPAINT) ----
    panel_range = concat_h([
        cv2.putText(raw_color.copy(),   "RAW",     (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2),
        cv2.putText(color_a.copy(),     "DILATE",  (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2),
        cv2.putText(color_b.copy(),     "INPAINT", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2),
    ])
    cv2.imwrite(os.path.join(SAVE_DIR, "panel_range_raw_dilate_inpaint.png"), panel_range)

    # =========================================================
    # ラベルの穴埋め & 可視化
    # =========================================================

    # RAW ラベル（穴は 0 のまま）
    label_raw_vis = labels_to_color_image(raw_label, valid_mask=raw_mask)
    cv2.imwrite(os.path.join(SAVE_DIR, "raw_label_color.png"), label_raw_vis)

    # ---- RAW ラベルの境界マップ（重み）を計算して保存 ----
    boundary_mag, boundary_mask = compute_boundary_map(raw_label, raw_mask)

    # 連続値の境界強度を 0-255 に正規化して保存（重み画像）
    if boundary_mag.max() > 0:
        bmag_u8 = (boundary_mag / (boundary_mag.max() + 1e-6) * 255.0).astype(np.uint8)
    else:
        bmag_u8 = np.zeros_like(boundary_mag, dtype=np.uint8)
    cv2.imwrite(os.path.join(SAVE_DIR, "raw_boundary_mag.png"), bmag_u8)

    # 0/1 の境界マスクを保存
    bmask_u8 = (boundary_mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(SAVE_DIR, "raw_boundary_mask.png"), bmask_u8)

    # ラベル画像に境界を重ねた可視化も作る（境界を赤で塗る）
    overlay = label_raw_vis.copy()
    overlay[boundary_mask] = [0, 0, 255]  # BGR で赤
    cv2.imwrite(os.path.join(SAVE_DIR, "raw_label_with_boundary.png"), overlay)

    # DILATE ラベル（多数決で補完）
    label_dilate = dilate_label_fill_majority(raw_label, raw_mask, kernel_size=7)
    label_dilate_vis = labels_to_color_image(label_dilate, valid_mask=valid_a)
    cv2.imwrite(os.path.join(SAVE_DIR, "dilate_label_color.png"), label_dilate_vis)

    # INPAINT ラベル（深度最近傍で補完）
    label_inpaint = label_propagate_by_depth(range_b, raw_label, raw_mask, radius=3)
    label_inpaint_vis = labels_to_color_image(label_inpaint, valid_mask=valid_b)
    cv2.imwrite(os.path.join(SAVE_DIR, "inpaint_label_color.png"), label_inpaint_vis)

    # ラベルの比較パネル
    panel_label = concat_h([
        cv2.putText(label_raw_vis.copy(),     "RAW LABEL",     (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2),
        cv2.putText(label_dilate_vis.copy(),  "DILATE LABEL",  (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2),
        cv2.putText(label_inpaint_vis.copy(), "INPAINT LABEL", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2),
    ])
    cv2.imwrite(os.path.join(SAVE_DIR, "panel_label_raw_dilate_inpaint.png"), panel_label)

    # ---- 統計情報も JSON で保存 ----
    out_stats = {
        "raw": stats_raw,
        "dilate": stats_a,
        "inpaint": stats_b,
        "config": {
            "H": H, "W": W,
            "FOV_UP": FOV_UP, "FOV_DOWN": FOV_DOWN,
            "DILATE_KERNEL": DILATE_KERNEL,
            "DILATE_PASSES": DILATE_PASSES,
            "DILATE_MAX_BLEND": DILATE_MAX_BLEND,
            "WIN": WIN,
            "SIGMA_SPATIAL": SIGMA_SPATIAL,
            "SIGMA_GUIDE": SIGMA_GUIDE,
            "INPAINT_ITERS": INPAINT_ITERS,
            "MEDIAN_KSIZE": MEDIAN_KSIZE,
        },
    }
    with open(os.path.join(SAVE_DIR, "compare_zero_stats.json"), "w", encoding="utf-8") as f:
        json.dump(out_stats, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "raw_overall": out_stats["raw"]["overall_zero_or_neg_ratio"],
        "dilate_overall": out_stats["dilate"]["overall_zero_or_neg_ratio"],
        "inpaint_overall": out_stats["inpaint"]["overall_zero_or_neg_ratio"],
    }, indent=2, ensure_ascii=False))
    print(f"Saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
