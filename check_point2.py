# compare_fill_visualize.py
# Visualize and compare two hole-filling methods (Dilation vs Depth Inpainting)
# Built on top of user's check_point.py projection/normalization flow.

import os
import sys
import json
import numpy as np
import cv2
from datetime import datetime

# ---- Projection utils (same interface as user's code) ----
from lib.utils.laserscan3 import LaserScan  # projection (H,W,FOV)  :contentReference[oaicite:1]{index=1}

# ====================== CONFIG ======================
# Path to one SemanticKITTI .bin file
BIN_PATH = "/home/jun/src/SemanticKitti/sequences/08/velodyne/000001.bin"

# Output directory
SAVE_DIR = f"output_compare_full/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Projection parameters (same as your good FOVs)
H = 64
W = 512
FOV_UP = 3.0
FOV_DOWN = -23.0

# Dilation params
DILATE_KERNEL = 7  # odd, e.g., 3/5/7
DILATE_PASSES = 3       # 追加：膨張を何回もかける
DILATE_MAX_BLEND = 0.8  # 追加：最大値寄りに（侵食を起こしやすく）

# Depth inpainting params (joint-bilateral style)
WIN = 5            # odd window size
SIGMA_SPATIAL = 2.0
SIGMA_GUIDE = 0.2
INPAINT_ITERS = 1  # 1~2 is usually enough

# Optional light denoise on valid area only
MEDIAN_KSIZE = 0   # 0=off; try 3 if speckle visible

# Visualization
ZERO_GRAY = 180    # gray for zeros in color map
# ====================================================


# ---------------- Common helpers (from check_point style) ----------------
def normalize_to_uint8(arr, valid_mask):
    """Normalize positive depths to 0..255; keep invalid at 0."""
    arr = arr.astype(np.float32)
    pos = np.where(arr > 0, arr, 0.0)
    vmax = float(np.max(pos)) if np.any(pos > 0) else 0.0
    out = np.zeros_like(pos, dtype=np.uint8)
    if vmax > 0:
        out = (pos / vmax * 255.0).astype(np.uint8)
    out[~valid_mask] = 0
    return out


def colorize_range(range_u8, zero_gray=ZERO_GRAY):
    """Apply COLORMAP_JET; paint zeros gray for clarity."""
    color = cv2.applyColorMap(range_u8, cv2.COLORMAP_JET)
    zero_mask = (range_u8 == 0)
    color[zero_mask] = (zero_gray, zero_gray, zero_gray)
    return color


def compute_zero_stats(range_img, valid_mask):
    """Zero/invalid ratios for diagnostics."""
    Hh, Ww = range_img.shape
    zero_or_neg = (range_img <= 0.0)
    return {
        "overall_zero_or_neg_ratio": float(np.mean(zero_or_neg)),
        "bottom_half_zero_or_neg_ratio": float(np.mean(zero_or_neg[Hh//2:])),
        "pixels_total": int(Hh * Ww),
        "pixels_zero_or_neg": int(np.sum(zero_or_neg))
    }


def concat_h(images, pad=6, pad_color=(32, 32, 32)):
    """Horizontally concatenate BGR images with padding."""
    h = max(img.shape[0] for img in images)
    out = []
    for img in images:
        if img.shape[0] != h:
            img = cv2.resize(img, (int(img.shape[1]*h/img.shape[0]), h), interpolation=cv2.INTER_NEAREST)
        out.append(img)
        out.append(np.full((h, pad, 3), pad_color, dtype=np.uint8))
    return np.concatenate(out[:-1], axis=1)


# ---------------- Method A: Dilation-based filling ----------------
def dilate_range_fill(range_img, valid_mask, kernel_size=5):
    """
    Fill invalid with nearby valid (morphological-inspired, but for float):
    - compute a quick neighbor average within a kernel for invalids
    - here we use a dilate-like approach by taking local max as seed
      and mix with mean to avoid overgrowth.
    """
    k = int(kernel_size)
    k = k if k % 2 == 1 else k + 1
    # Prepare two guided proposals
    # (1) local mean of valid neighbors
    v = valid_mask.astype(np.uint8)
    # to avoid biasing, blur valid-weighted range / valid count
    r = range_img.copy()
    r[~valid_mask] = 0.0
    num = cv2.blur(v.astype(np.float32), (k, k))
    den = cv2.blur(r.astype(np.float32), (k, k))
    mean_prop = np.where(num > 1e-6, den / (num + 1e-6), 0.0)

    # (2) local max as a strong seed (dilation-like)
    max_prop = cv2.dilate(r.astype(np.float32), np.ones((k, k), np.uint8), iterations=1)

    # blend (heuristic): mostly mean, a little max to push through thin gaps
    fill = 0.8 * mean_prop + 0.2 * max_prop
    out = range_img.copy()
    out[~valid_mask] = fill[~valid_mask]
    out[~np.isfinite(out)] = -1.0
    return out.astype(np.float32)


def dilate_label_fill(label_img, valid_mask, kernel_size=5):
    """
    Classic binary dilation for labels: copy majority class within kernel.
    If ties/empty, leave as is.
    """
    k = int(kernel_size)
    k = k if k % 2 == 1 else k + 1
    Hh, Ww = label_img.shape
    out = label_img.copy()
    invalid = ~valid_mask
    ys, xs = np.where(invalid)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - k//2), min(Hh, y + k//2 + 1)
        x0, x1 = max(0, x - k//2), min(Ww, x + k//2 + 1)
        patch_l = label_img[y0:y1, x0:x1]
        patch_v = valid_mask[y0:y1, x0:x1]
        if not np.any(patch_v):
            continue
        vals = patch_l[patch_v]
        # majority vote
        uniq, cnt = np.unique(vals, return_counts=True)
        out[y, x] = uniq[np.argmax(cnt)]
    return out


# ---------------- Method B: Depth inpainting (joint-bilateral style) ----------------
def joint_bilateral_inpaint(range_img, rem_img, valid_mask,
                            window=5, sigma_spatial=2.0, sigma_guide=0.2, iters=1):
    """
    Fill invalid pixels using neighbors' weighted average:
      w = exp(-||Δxy||^2 / 2σs^2) * exp(-||Δguide||^2 / 2σg^2)
    guide: [|∇range|, remission]
    """
    Hh, Ww = range_img.shape
    out = range_img.copy()

    # build guide
    r = range_img.copy()
    r[r <= 0] = 0.0
    gx = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    # normalize to 0..1
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


def label_propagate_by_depth(range_filled, label_img, valid_mask, radius=3):
    """
    After depth fill, assign labels to previously invalid pixels
    by picking the neighbor whose depth is closest to local median (robust).
    """
    Hh, Ww = range_filled.shape
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


# ---------------- Optional denoise on valid pixels only ----------------
def median_filter_on_valid(range_img, valid_mask, ksize=3):
    if not ksize or ksize <= 1:
        return range_img.astype(np.float32)
    tmp = range_img.copy()
    tmp[~valid_mask] = 0.0
    med = cv2.medianBlur(tmp.astype(np.float32), int(ksize))
    out = np.where(valid_mask, med, -1.0).astype(np.float32)
    return out


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    if not os.path.isfile(BIN_PATH):
        print(f"[ERROR] BIN_PATH not found: {BIN_PATH}")
        sys.exit(1)

    # ---- Raw projection (as in your script) ----
    scan = LaserScan(project=True, H=H, W=W, fov_up=FOV_UP, fov_down=FOV_DOWN)
    scan.open_scan(BIN_PATH)

    raw_range = scan.proj_range.astype(np.float32)            # [-1 or depth]
    raw_mask = (scan.proj_mask.astype(np.uint8) > 0)          # bool
    raw_rem  = scan.proj_remission.astype(np.float32)
    raw_label = None  # labels not loaded here; visualization focuses on depth/mask

    # Base visuals
    raw_u8 = normalize_to_uint8(raw_range, raw_mask)
    raw_color = colorize_range(raw_u8)
    cv2.imwrite(os.path.join(SAVE_DIR, "raw_range_colormap.png"), raw_color)
    cv2.imwrite(os.path.join(SAVE_DIR, "raw_mask.png"), (raw_mask.astype(np.uint8) * 255))
    stats_raw = compute_zero_stats(raw_range, raw_mask)

    # ---- Method A: Dilation-based filling ----
    valid_a = raw_mask.copy()
    range_a = raw_range.copy()
    for _ in range(DILATE_PASSES):
        range_a = dilate_range_fill(range_a, valid_a, kernel_size=DILATE_KERNEL)
    # ★1回埋めたら“そこも有効”として次回の母集団に入れる
    valid_a[:] = True
    # range_a = dilate_range_fill(raw_range, valid_a, kernel_size=DILATE_KERNEL)
    # if MEDIAN_KSIZE and MEDIAN_KSIZE > 1:
    #     range_a = median_filter_on_valid(range_a, valid_a, ksize=MEDIAN_KSIZE)
    u8_a = normalize_to_uint8(range_a, valid_a)
    color_a = colorize_range(u8_a)
    cv2.imwrite(os.path.join(SAVE_DIR, "dilate_range_colormap.png"), color_a)
    cv2.imwrite(os.path.join(SAVE_DIR, "dilate_valid_mask.png"), (valid_a.astype(np.uint8) * 255))
    stats_a = compute_zero_stats(range_a, valid_a)

    # ---- Method B: Depth inpainting (joint-bilateral) ----
    valid_b = raw_mask.copy()
    range_b = joint_bilateral_inpaint(raw_range, raw_rem, valid_b,
                                      window=WIN, sigma_spatial=SIGMA_SPATIAL,
                                      sigma_guide=SIGMA_GUIDE, iters=INPAINT_ITERS)
    if MEDIAN_KSIZE and MEDIAN_KSIZE > 1:
        range_b = median_filter_on_valid(range_b, valid_b, ksize=MEDIAN_KSIZE)
    u8_b = normalize_to_uint8(range_b, valid_b)
    color_b = colorize_range(u8_b)
    cv2.imwrite(os.path.join(SAVE_DIR, "inpaint_range_colormap.png"), color_b)
    cv2.imwrite(os.path.join(SAVE_DIR, "inpaint_valid_mask.png"), (valid_b.astype(np.uint8) * 255))
    stats_b = compute_zero_stats(range_b, valid_b)

    # ---- Panel (Raw | Dilate | Inpaint) ----
    panel = concat_h([
        cv2.putText(raw_color.copy(),    "RAW",      (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2),
        cv2.putText(color_a.copy(),      "DILATE",   (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2),
        cv2.putText(color_b.copy(),      "INPAINT",  (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2),
    ])
    cv2.imwrite(os.path.join(SAVE_DIR, "panel_raw_dilate_inpaint.png"), panel)

    # ---- Save stats JSON ----
    out_stats = {
        "raw":      stats_raw,
        "dilate":   stats_a,
        "inpaint":  stats_b,
        "config": {
            "H": H, "W": W, "FOV_UP": FOV_UP, "FOV_DOWN": FOV_DOWN,
            "DILATE_KERNEL": DILATE_KERNEL,
            "WIN": WIN, "SIGMA_SPATIAL": SIGMA_SPATIAL, "SIGMA_GUIDE": SIGMA_GUIDE, "INPAINT_ITERS": INPAINT_ITERS,
            "MEDIAN_KSIZE": MEDIAN_KSIZE
        }
    }
    with open(os.path.join(SAVE_DIR, "compare_zero_stats.json"), "w", encoding="utf-8") as f:
        json.dump(out_stats, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "raw_overall": out_stats["raw"]["overall_zero_or_neg_ratio"],
        "dilate_overall": out_stats["dilate"]["overall_zero_or_neg_ratio"],
        "inpaint_overall": out_stats["inpaint"]["overall_zero_or_neg_ratio"]
    }, indent=2, ensure_ascii=False))
    print(f"Saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
