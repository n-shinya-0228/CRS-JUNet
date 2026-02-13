
import os
import sys
import json
import numpy as np
import cv2
from lib.utils.laserscan3 import LaserScan
from datetime import datetime

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ====================== CONFIG ======================
# Path to one SemanticKITTI .bin file
BIN_PATH = "/home/jun/src/SemanticKitti/sequences/01/velodyne/000001.bin"

# Output directory
SAVE_DIR = f"UnpNet/output_compare_simple/{now}"

# Projection parameters
H = 64
W = 512
FOV_UP = 3.0
FOV_DOWN = -23.0

# Inpainting params (same idea as SemanticKitti2)
MAX_GAP_ROW = 2          # small-gap threshold (horizontal)
MAX_GAP_COL = 2          # small-gap threshold (vertical)
INPAINT_MODE = "linear"  # "linear" or "nearest"
MEDIAN_KSIZE = 0         # 0=off; small odd (e.g., 3) to enable

# Visualization
ZERO_GRAY = 180          # gray for zeros in color map
# ====================================================


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


def compute_zero_stats(range_img, mask_bool):
    """Compute overall + bottom-half ratios for zeros/invalids."""
    H, W = range_img.shape
    zero_or_neg = (range_img <= 0.0)  # includes -1 and 0
    overall_ratio = float(np.mean(zero_or_neg))
    bottom_ratio = float(np.mean(zero_or_neg[H//2:]))
    return {
        "overall_zero_or_neg_ratio": overall_ratio,
        "bottom_half_zero_or_neg_ratio": bottom_ratio,
        "pixels_total": int(H * W),
        "pixels_zero_or_neg": int(np.sum(zero_or_neg))
    }


def _interp_small_gaps_1d(arr, valid, max_gap=2, mode="linear"):
    """
    Fill only short invalid runs (length <= max_gap) that are bracketed by valid samples.
    mode: 'linear' or 'nearest'.
    """
    n = arr.shape[0]
    out = arr.copy()
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        gap_len = j - i
        left = i - 1
        right = j
        if left >= 0 and right < n and gap_len <= max_gap and valid[left] and valid[right]:
            if mode == "nearest":
                mid = (left + right) / 2.0
                for k in range(i, j):
                    out[k] = arr[left] if k <= mid else arr[right]
            else:  # linear
                x0, y0 = left, arr[left]
                x1, y1 = right, arr[right]
                for k in range(i, j):
                    t = (k - x0) / (x1 - x0)
                    out[k] = (1 - t) * y0 + t * y1
        i = j
    return out


def inpaint_small_gaps_2d(range_img, mask_img, max_gap_row=2, max_gap_col=2, mode="linear"):
    """
    2D range image: fill only short gaps along rows, then columns.
    - range_img: float32 [H,W], invalid as <=0 (or -1)
    - mask_img : int/bool [H,W], 1 if observed
    Returns: (range_filled, filled_mask_int)
    """
    H, W = range_img.shape
    valid = (mask_img.astype(bool) & (range_img > 0))
    out = range_img.copy()
    filled = np.zeros_like(valid, dtype=bool)

    # Row-wise small-gap interpolation
    for y in range(H):
        row = out[y, :]
        v = valid[y, :].copy()
        row_filled = _interp_small_gaps_1d(row, v, max_gap=max_gap_row, mode=mode)
        new_ok = (~v) & (row_filled > 0)
        out[y, :] = np.where(new_ok, row_filled, row)
        valid[y, :] = v | new_ok
        filled[y, :] |= new_ok

    # Column-wise small-gap interpolation
    for x in range(W):
        col = out[:, x]
        v = valid[:, x].copy()
        col_filled = _interp_small_gaps_1d(col, v, max_gap=max_gap_col, mode=mode)
        new_ok = (~v) & (col_filled > 0)
        out[:, x] = np.where(new_ok, col_filled, col)
        valid[:, x] = v | new_ok
        filled[:, x] |= new_ok

    out = np.where(valid, out, -1.0).astype(np.float32)
    return out, filled.astype(np.int32)


def median_filter_on_valid(range_img, valid_mask, ksize=3):
    """
    Light median denoise on valid pixels only (invalid kept as -1).
    To avoid bleeding into invalid, we do a quick fill-then-median-then-restore-invalid.
    """
    if ksize is None or ksize <= 1:
        return range_img.astype(np.float32)

    # Quick local averaging fill for invalids (2 passes)
    tmp = range_img.copy()
    filled = tmp.copy()
    for _ in range(2):
        # average in 3x3 neighborhood to estimate invalids
        avg = cv2.blur(filled.astype(np.float32), (3, 3))
        filled = np.where(valid_mask, tmp, avg)

    med = cv2.medianBlur(filled.astype(np.float32), int(ksize))
    out = np.where(valid_mask, med, -1.0).astype(np.float32)
    return out


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.isfile(BIN_PATH):
        print(f"[ERROR] BIN_PATH not found: {BIN_PATH}")
        sys.exit(1)

    # ---- Raw 2D projection ----
    scan = LaserScan(project=True, H=H, W=W, fov_up=FOV_UP, fov_down=FOV_DOWN)
    scan.open_scan(BIN_PATH)

    raw_range = scan.proj_range.astype(np.float32)      # [-1 or depth]
    raw_mask = (scan.proj_mask.astype(np.uint8) > 0)    # bool

    raw_u8 = normalize_to_uint8(raw_range, raw_mask)
    raw_color = colorize_range(raw_u8, ZERO_GRAY)

    cv2.imwrite(os.path.join(SAVE_DIR, "raw_range_uint8.png"), raw_u8)
    cv2.imwrite(os.path.join(SAVE_DIR, "raw_range_colormap.png"), raw_color)
    cv2.imwrite(os.path.join(SAVE_DIR, "raw_mask.png"), (raw_mask.astype(np.uint8) * 255))

    raw_stats = compute_zero_stats(raw_range, raw_mask)

    # ---- SemanticKitti2-like ops (in-script) ----
    proc_range, filled_mask = inpaint_small_gaps_2d(
        raw_range, raw_mask.astype(np.int32),
        max_gap_row=MAX_GAP_ROW, max_gap_col=MAX_GAP_COL, mode=INPAINT_MODE
    )

    if MEDIAN_KSIZE and MEDIAN_KSIZE > 1:
        valid_for_median = (proc_range > 0)
        proc_range = median_filter_on_valid(proc_range, valid_for_median, ksize=int(MEDIAN_KSIZE))

    proc_valid = (raw_mask | filled_mask.astype(bool)).astype(bool)

    proc_u8 = normalize_to_uint8(proc_range, proc_valid)
    proc_color = colorize_range(proc_u8, ZERO_GRAY)

    cv2.imwrite(os.path.join(SAVE_DIR, "proc_range_uint8.png"), proc_u8)
    cv2.imwrite(os.path.join(SAVE_DIR, "proc_range_colormap.png"), proc_color)
    cv2.imwrite(os.path.join(SAVE_DIR, "proc_valid_mask.png"), (proc_valid.astype(np.uint8) * 255))

    proc_stats = compute_zero_stats(proc_range, proc_valid)

    # ---- Save stats ----
    compare = {
        "raw": raw_stats,
        "processed": proc_stats,
        "config": {
            "H": H, "W": W, "FOV_UP": FOV_UP, "FOV_DOWN": FOV_DOWN,
            "MAX_GAP_ROW": MAX_GAP_ROW, "MAX_GAP_COL": MAX_GAP_COL,
            "INPAINT_MODE": INPAINT_MODE, "MEDIAN_KSIZE": MEDIAN_KSIZE
        }
    }
    stats_path = os.path.join(SAVE_DIR, "compare_zero_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(compare, f, ensure_ascii=False, indent=2)

    print("=== Zero ratios ===")
    print(json.dumps({
        "raw_overall": compare["raw"]["overall_zero_or_neg_ratio"],
        "proc_overall": compare["processed"]["overall_zero_or_neg_ratio"],
        "raw_bottom_half": compare["raw"]["bottom_half_zero_or_neg_ratio"],
        "proc_bottom_half": compare["processed"]["bottom_half_zero_or_neg_ratio"],
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {SAVE_DIR}")
    print(f"Stats JSON: {stats_path}")


if __name__ == "__main__":
    main()






