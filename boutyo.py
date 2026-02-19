import os
import json
import numpy as np
import cv2
from datetime import datetime

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ===================== USER SETTINGS (edit here) =====================
BIN_PATH = "/home/shiny/SemanticKitti/sequences/08/velodyne/000001.bin"   # ★ここを書き換える
SAVE_DIR  = f"UnpNet/output_compare_simple/{now}"  # None の場合: ./UnpNet/output_compare_modes/<timestamp>/ に保存

H, W = 64, 512
FOV_UP, FOV_DOWN = 3.0, -23.0

# linear small-gap params
MAX_GAP_ROW = 2
MAX_GAP_COL = 2
INPAINT_MODE = "linear"   # or "nearest"

# conditional morphology params
MORPH_ITERS = 1
DEPTH_TAU_ABS = 0.5
DEPTH_TAU_REL = 0.05

ZERO_GRAY = 180  # visualization: 0深度を灰色描画
# =====================================================================

# Try to import user's LaserScan class (as used in check_point.py)
try:
    from lib.utils.laserscan3 import LaserScan
except Exception as e:
    print("[WARN] Could not import lib.utils.laserscan3.LaserScan. "
          "Run from your repo root or adjust PYTHONPATH. Error:", e)

def normalize_to_uint8(arr, valid_mask):
    arr = arr.astype(np.float32)
    pos = np.where(arr > 0, arr, 0.0)
    vmax = float(np.max(pos)) if np.any(pos > 0) else 0.0
    out = np.zeros_like(pos, dtype=np.uint8)
    if vmax > 0:
        out = (pos / vmax * 255.0).astype(np.uint8)
    out[~valid_mask] = 0
    return out

def colorize_range(range_u8, zero_gray=180):
    color = cv2.applyColorMap(range_u8, cv2.COLORMAP_JET)
    zero_mask = (range_u8 == 0)
    color[zero_mask] = (zero_gray, zero_gray, zero_gray)
    return color

def compute_zero_stats(range_img, mask_bool):
    H_, W_ = range_img.shape
    zero_or_neg = (range_img <= 0.0)
    overall_ratio = float(np.mean(zero_or_neg))
    bottom_ratio = float(np.mean(zero_or_neg[H_//2:]))
    return {
        "overall_zero_or_neg_ratio": overall_ratio,
        "bottom_half_zero_or_neg_ratio": bottom_ratio,
        "pixels_total": int(H_ * W_),
        "pixels_zero_or_neg": int(np.sum(zero_or_neg))
    }

def _interp_small_gaps_1d(arr, valid, max_gap=2, mode="linear"):
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
            else:
                x0, y0 = left, arr[left]
                x1, y1 = right, arr[right]
                for k in range(i, j):
                    t = (k - x0) / (x1 - x0)
                    out[k] = (1 - t) * y0 + t * y1
        i = j
    return out

def inpaint_small_gaps_2d(range_img, mask_img, max_gap_row=2, max_gap_col=2, mode="linear"):
    H_, W_ = range_img.shape
    valid = (mask_img.astype(bool) & (range_img > 0))
    out = range_img.copy()
    filled = np.zeros_like(valid, dtype=bool)

    for y in range(H_):
        row = out[y, :]
        v = valid[y, :].copy()
        row_filled = _interp_small_gaps_1d(row, v, max_gap=max_gap_row, mode=mode)
        new_ok = (~v) & (row_filled > 0)
        out[y, :] = np.where(new_ok, row_filled, row)
        valid[y, :] = v | new_ok
        filled[y, :] |= new_ok

    for x in range(W_):
        col = out[:, x]
        v = valid[:, x].copy()
        col_filled = _interp_small_gaps_1d(col, v, max_gap=max_gap_col, mode=mode)
        new_ok = (~v) & (col_filled > 0)
        out[:, x] = np.where(new_ok, col_filled, col)
        valid[:, x] = v | new_ok
        filled[:, x] |= new_ok

    out = np.where(valid, out, -1.0).astype(np.float32)
    return out, filled

def conditional_morph_fill(range_img, obs_mask, smallgap_mask,
                           edge_mask=None, iters=1,
                           depth_tau_abs=0.5, depth_tau_rel=0.05):
    H_, W_ = range_img.shape
    valid = (obs_mask.astype(bool)) & (range_img > 0)
    if edge_mask is None:
        edge_mask = np.zeros_like(valid, dtype=bool)

    def shift(arr, dy, dx, fill_val):
        out = np.full_like(arr, fill_val)
        y0 = max(0,  dy); y1 = min(H_, H_+dy)
        x0 = max(0,  dx); x1 = min(W_, W_+dx)
        out[y0:y1, x0:x1] = arr[y0-dy:y1-dy, x0-dx:x1-dx]
        return out

    filled_any = np.zeros_like(valid, dtype=bool)
    out = range_img.copy()
    target = (~valid) & smallgap_mask & (~edge_mask)

    for _ in range(iters):
        if not np.any(target):
            break

        nbrs = []
        masks = []
        for (dy, dx) in [(-1,0), (1,0), (0,-1), (0,1)]:
            v = shift(valid.astype(np.uint8), dy, dx, 0).astype(bool)
            r = shift(out, dy, dx, -1.0)
            masks.append(v & (r > 0))
            nbrs.append(r)

        nbr_stack = np.stack([np.where(m, r, np.nan) for r, m in zip(nbrs, masks)], axis=0)
        nbr_med = np.nanmedian(nbr_stack, axis=0)

        diff = np.abs(nbr_stack - np.expand_dims(nbr_med, 0))
        all_nan = np.all(np.isnan(diff), axis=0)
        safe_diff = np.where(np.isnan(diff), np.inf, diff)
        idx = np.argmin(safe_diff, axis=0)
        gather = np.take_along_axis(nbr_stack, np.expand_dims(idx, 0), axis=0)[0]

        tau = np.maximum(depth_tau_abs, depth_tau_rel * np.maximum(nbr_med, 1e-6))
        pass_jump = np.abs(gather - nbr_med) <= tau

        can_fill = target & (~edge_mask) & (~all_nan) & pass_jump & (~np.isnan(gather))

        out[can_fill] = gather[can_fill]
        valid[can_fill] = True
        filled_any |= can_fill

        target = (~valid) & smallgap_mask & (~edge_mask)

    out[~valid] = -1.0
    return out.astype(np.float32), filled_any

def has_obs_neighbor(mask):
    up    = np.zeros_like(mask, bool); up[1:]     = mask[:-1]
    down  = np.zeros_like(mask, bool); down[:-1]  = mask[1:]
    left  = np.zeros_like(mask, bool); left[:,1:] = mask[:,:-1]
    right = np.zeros_like(mask, bool); right[:,:-1]= mask[:,1:]
    return up | down | left | right

def run_compare(bin_path, save_root,
                H=64, W=512, fov_up=3.0, fov_down=-23.0,
                max_gap_row=2, max_gap_col=2, inpaint_mode="linear",
                morph_iters=1, depth_tau_abs=0.5, depth_tau_rel=0.05,
                zero_gray=180):

    os.makedirs(save_root, exist_ok=True)

    from lib.utils.laserscan3 import LaserScan

    scan = LaserScan(project=True, H=H, W=W, fov_up=fov_up, fov_down=fov_down)
    scan.open_scan(bin_path)
    raw_range = scan.proj_range.astype(np.float32)
    raw_mask = (scan.proj_mask.astype(np.uint8) > 0)

    raw_u8 = normalize_to_uint8(raw_range, raw_mask)
    cv2.imwrite(os.path.join(save_root, "raw_range_uint8.png"), raw_u8)
    cv2.imwrite(os.path.join(save_root, "raw_range_colormap.png"), colorize_range(raw_u8, zero_gray))
    cv2.imwrite(os.path.join(save_root, "raw_mask.png"), (raw_mask.astype(np.uint8) * 255))

    raw_stats = compute_zero_stats(raw_range, raw_mask)

    lin_range, lin_filled = inpaint_small_gaps_2d(
        raw_range, raw_mask.astype(np.int32),
        max_gap_row=max_gap_row, max_gap_col=max_gap_col, mode=inpaint_mode
    )
    lin_valid = (raw_mask | lin_filled).astype(bool)
    lin_u8 = normalize_to_uint8(lin_range, lin_valid)
    cv2.imwrite(os.path.join(save_root, "A_linear_range_uint8.png"), lin_u8)
    cv2.imwrite(os.path.join(save_root, "A_linear_range_colormap.png"), colorize_range(lin_u8, zero_gray))
    cv2.imwrite(os.path.join(save_root, "A_linear_valid_mask.png"), (lin_valid.astype(np.uint8) * 255))
    lin_stats = compute_zero_stats(lin_range, lin_valid)

    raw_invalid = (raw_range <= 0)
    smallgap_mask = raw_invalid & has_obs_neighbor(raw_mask)

    morph_only_range, morph_only_filled = conditional_morph_fill(
        raw_range, raw_mask, smallgap_mask,
        edge_mask=None, iters=morph_iters,
        depth_tau_abs=depth_tau_abs, depth_tau_rel=depth_tau_rel
    )
    morph_only_valid = (raw_mask | morph_only_filled).astype(bool)
    mo_u8 = normalize_to_uint8(morph_only_range, morph_only_valid)
    cv2.imwrite(os.path.join(save_root, "B_morph_only_range_uint8.png"), mo_u8)
    cv2.imwrite(os.path.join(save_root, "B_morph_only_range_colormap.png"), colorize_range(mo_u8, zero_gray))
    cv2.imwrite(os.path.join(save_root, "B_morph_only_valid_mask.png"), (morph_only_valid.astype(np.uint8) * 255))
    morph_only_stats = compute_zero_stats(morph_only_range, morph_only_valid)

    still_invalid = (lin_range <= 0)
    smallgap_mask2 = still_invalid & has_obs_neighbor(lin_valid.astype(bool))

    hybrid_range, hybrid_filled = conditional_morph_fill(
        lin_range, lin_valid, smallgap_mask2,
        edge_mask=None, iters=morph_iters,
        depth_tau_abs=depth_tau_abs, depth_tau_rel=depth_tau_rel
    )
    hybrid_valid = (lin_valid | hybrid_filled).astype(bool)
    hy_u8 = normalize_to_uint8(hybrid_range, hybrid_valid)
    cv2.imwrite(os.path.join(save_root, "C_hybrid_range_uint8.png"), hy_u8)
    cv2.imwrite(os.path.join(save_root, "C_hybrid_range_colormap.png"), colorize_range(hy_u8, zero_gray))
    cv2.imwrite(os.path.join(save_root, "C_hybrid_valid_mask.png"), (hybrid_valid.astype(np.uint8) * 255))
    hybrid_stats = compute_zero_stats(hybrid_range, hybrid_valid)

    compare = {
        "raw": raw_stats,
        "A_linear_only": lin_stats,
        "B_morph_only": morph_only_stats,
        "C_hybrid": hybrid_stats,
        "config": {
            "H": H, "W": W, "FOV_UP": fov_up, "FOV_DOWN": fov_down,
            "MAX_GAP_ROW": max_gap_row, "MAX_GAP_COL": max_gap_col,
            "INPAINT_MODE": inpaint_mode,
            "MORPH_ITERS": morph_iters, "DEPTH_TAU_ABS": depth_tau_abs, "DEPTH_TAU_REL": depth_tau_rel
        }
    }
    stats_path = os.path.join(save_root, "compare_zero_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(compare, f, ensure_ascii=False, indent=2)

    print("=== Zero ratios (lower is better) ===")
    print(json.dumps({
        "raw_overall": compare["raw"]["overall_zero_or_neg_ratio"],
        "A_linear_overall": compare["A_linear_only"]["overall_zero_or_neg_ratio"],
        "B_morph_overall": compare["B_morph_only"]["overall_zero_or_neg_ratio"],
        "C_hybrid_overall": compare["C_hybrid"]["overall_zero_or_neg_ratio"],
        "raw_bottom_half": compare["raw"]["bottom_half_zero_or_neg_ratio"],
        "A_linear_bottom_half": compare["A_linear_only"]["bottom_half_zero_or_neg_ratio"],
        "B_morph_bottom_half": compare["B_morph_only"]["bottom_half_zero_or_neg_ratio"],
        "C_hybrid_bottom_half": compare["C_hybrid"]["bottom_half_zero_or_neg_ratio"],
    }, indent=2, ensure_ascii=False))
    print("\nSaved images & stats to:", save_root)
    print("Stats JSON:", stats_path)

if __name__ == "__main__":
    if not os.path.isfile(BIN_PATH):
        raise FileNotFoundError(f"BIN_PATH not found: {BIN_PATH}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_dir = SAVE_DIR 
    run_compare(
        bin_path=BIN_PATH,
        save_root=out_dir,
        H=H, W=W, fov_up=FOV_UP, fov_down=FOV_DOWN,
        max_gap_row=MAX_GAP_ROW, max_gap_col=MAX_GAP_COL, inpaint_mode=INPAINT_MODE,
        morph_iters=MORPH_ITERS, depth_tau_abs=DEPTH_TAU_ABS, depth_tau_rel=DEPTH_TAU_REL,
        zero_gray=ZERO_GRAY
    )
