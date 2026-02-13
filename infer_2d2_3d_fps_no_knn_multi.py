#JunNetも使える
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_2d2_3d_fps_no_knn_multi.py
- 2D range-view inference -> unproject back to 3D point labels
- Supports ChatNet4 / JunNet13
- NO KNN filling: unprojected points remain 0 (unlabeled)
- Save SemanticKITTI .label predictions (uint32)
- Report FPS (model-only) and FPS (end-to-end)

Usage (JunNet13):
python infer_2d2_3d_fps_no_knn_multi.py \
  --model junnet13 \
  --dataset ~/src/SemanticKitti \
  --arch_cfg config/arch/UnpNet.yaml \
  --data_cfg config/labels/semantic-kitti.yaml \
  --p logs/xxx/best_val.path \
  --split valid \
  -s logs/pred_junnet13_no_knn

Usage (ChatNet4):
python infer_2d2_3d_fps_no_knn_multi.py \
  --model chatnet4 \
  --dataset ~/src/SemanticKitti \
  --arch_cfg config/arch/UnpNet.yaml \
  --data_cfg config/labels/semantic-kitti.yaml \
  --p logs/xxx/best_val.path \
  --split valid \
  -s logs/pred_chatnet4_no_knn
"""

import argparse
import os
import os.path as osp
import yaml
import numpy as np
from tqdm import tqdm
import torch
import shutil
import time

from lib.dataset.Parser import Parser


# ------------------------------
# Model import helpers
# ------------------------------
def import_model(model_name: str):
    model_name = model_name.lower()
    if model_name == "chatnet4":
        try:
            from lib.models.ChatNet4 import ChatNet4
        except Exception:
            # fallback if you placed file locally
            from lib.models.ChatNet4 import ChatNet4
        return ChatNet4
    elif model_name == "junnet13":
        try:
            from lib.models.JunNet13 import JunNet13
        except Exception:
            # fallback for the uploaded file name
            from lib.models.JunNet13 import JunNet13
        return JunNet13
    else:
        raise ValueError(f"Unknown --model {model_name}. Choose from: chatnet4, junnet13")


# ------------------------------
# Helpers
# ------------------------------
def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def build_inv_map(DATA: dict):
    inv = DATA.get("learning_map_inv", None)
    if inv is None:
        return None
    max_k = max(int(k) for k in inv.keys())
    lut = np.zeros(max_k + 1, dtype=np.int32)
    for k, v in inv.items():
        lut[int(k)] = int(v)
    return lut


def normalize_seq_str(s) -> str:
    s = str(s)
    s = osp.basename(s)
    return s.zfill(2)


def _as_gt_label_name(path_name_str: str) -> str:
    base = os.path.basename(path_name_str).replace(".bin", ".label")
    if not base.endswith(".label"):
        base += ".label"
    stem = base[:-6]
    if stem.isdigit():
        base = f"{int(stem):06d}.label"
    return base


def _load_points_xyz(dataset_root: str, seq: str, name: str):
    bin_name = name
    if bin_name.endswith(".label"):
        bin_name = bin_name.replace(".label", ".bin")
    elif not bin_name.endswith(".bin"):
        bin_name = bin_name + ".bin"
    bin_path = osp.join(dataset_root, "sequences", str(seq).zfill(2), "velodyne", bin_name)
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    return pts


def insert_or_append_mask(in_vol: torch.Tensor, expected_in: int, mask_idx=None):
    """
    in_vol: (B,C,H,W). If expected_in != C and differs by +1, insert/append a valid-pixel mask.
    """
    B, C, H, W = in_vol.shape
    if expected_in is None or C == expected_in:
        return in_vol
    if C + 1 == expected_in:
        mask = (in_vol.abs().sum(dim=1, keepdim=True) > 0).float()
        if mask_idx is not None and 0 <= mask_idx <= C:
            left = in_vol[:, :mask_idx, :, :]
            right = in_vol[:, mask_idx:, :, :]
            in_vol = torch.cat([left, mask, right], dim=1)
        else:
            in_vol = torch.cat([in_vol, mask], dim=1)
    else:
        print(f"[WARN] Channel mismatch: got {C}, expected {expected_in}. Only +/-1 is auto-handled.")
    return in_vol


def argmax_logits(outs):
    """
    ChatNet4/JunNet13 both return dict with 'logits' (as in your files).
    """
    if isinstance(outs, dict):
        logits = outs.get("logits", None)
        if logits is None:
            for v in outs.values():
                if isinstance(v, torch.Tensor) and v.dim() >= 4:
                    logits = v
                    break
        if logits is None:
            raise RuntimeError("Model output dict does not contain 'logits' tensor.")
    elif isinstance(outs, torch.Tensor):
        logits = outs
    else:
        raise RuntimeError("Unexpected model output type.")
    return logits, logits.argmax(dim=1)


def find_batch_fields(batch):
    """
    Find:
      - path_seq (list[str])
      - path_name (list[str])
      - proj_idx (Tensor int, maybe pix2pt or pt2pix)
    """
    path_seq = None
    path_name = None
    proj_idx = None

    for item in batch:
        if isinstance(item, (list, tuple)) and len(item) >= 1 and isinstance(item[0], str):
            if path_seq is None:
                path_seq = item
            elif path_name is None:
                path_name = item

    tensor_like = [t for t in batch if isinstance(t, torch.Tensor)]
    idx_cand = []
    for t in tensor_like:
        if t.dtype in (torch.int16, torch.int32, torch.int64):
            if (t.min() <= -1) or (t.max() > 10000):
                idx_cand.append(t)
    idx_cand = sorted(idx_cand, key=lambda tt: -tt.numel())
    if idx_cand:
        proj_idx = idx_cand[0]

    return path_seq, path_name, proj_idx


def load_ckpt_into_model(model, ckpt_path: str, strict: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if (missing or unexpected) and strict is False:
        print(f"[WARN] load_state_dict(strict=False): missing={len(missing)}, unexpected={len(unexpected)}")
    return ckpt


# ------------------------------
# Args
# ------------------------------
def build_argparser():
    p = argparse.ArgumentParser("2D->3D inference (NO KNN) with FPS (ChatNet4/JunNet13)")
    p.add_argument("--model", required=True, choices=["chatnet4", "junnet13"],
                   help="Select model architecture")
    p.add_argument("--dataset", "-d", required=True, type=str,
                   help="SemanticKITTI dataset root (folder containing 'sequences')")
    p.add_argument("--arch_cfg", "-ac", required=True, type=str,
                   help="Architecture yaml cfg file")
    p.add_argument("--data_cfg", "-dc", required=True, type=str,
                   help="Data/labels yaml cfg file")
    p.add_argument("--p", "--pretrained", dest="pretrained", required=True, type=str,
                   help="checkpoint path (*.path)")
    p.add_argument("--strict", action="store_true",
                   help="Use strict=True when loading ckpt (default False). If you know ckpt matches exactly, enable.")
    p.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    p.add_argument("--save_path", "-s", required=True, type=str)
    p.add_argument("--warmup", type=int, default=20, help="GPU warmup iterations for stable FPS")
    p.add_argument("--limit", type=int, default=-1, help="limit number of scans (-1=all)")
    return p


# ------------------------------
# Main
# ------------------------------
def main():
    args = build_argparser().parse_args()

    ARCH = load_yaml(args.arch_cfg)
    DATA = load_yaml(args.data_cfg)
    inv_map = build_inv_map(DATA)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # clean save_path
    if os.path.exists(args.save_path):
        print(f"[INFO] Removing existing save_path: {args.save_path}")
        shutil.rmtree(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)

    # Parser / loader
    parser = Parser(root=args.dataset, data_cfg=DATA, arch_cfg=ARCH, gt=False, shuffle_train=False)
    if args.split == "train":
        loader = parser.get_train_set()
    elif args.split == "valid":
        loader = parser.get_valid_set()
    else:
        loader = parser.get_test_set()

    nclasses = parser.get_n_classes()

    # Build model
    ModelCls = import_model(args.model)
    in_ch = ARCH.get("model", {}).get("in_channels", 6)
    model = ModelCls(in_channels=in_ch, nclasses=nclasses).to(device)

    print("Loading checkpoint:", args.pretrained)
    load_ckpt_into_model(model, args.pretrained, strict=args.strict)
    model.eval()

    expected_in = ARCH.get("model", {}).get("in_channels", None)
    mask_idx    = ARCH.get("model", {}).get("mask_idx", None)

    # timing accumulators
    model_time_sum = 0.0
    e2e_time_sum   = 0.0
    scan_count     = 0

    # Warmup
    if device.type == "cuda":
        print(f"[INFO] Warmup {args.warmup} iters...")
        it = iter(loader)
        with torch.no_grad():
            for _ in range(args.warmup):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                in_vol = batch[0].to(device, non_blocking=True)
                in_vol = insert_or_append_mask(in_vol, expected_in or in_vol.shape[1], mask_idx=mask_idx)
                torch.cuda.synchronize()
                _ = model(in_vol)
                torch.cuda.synchronize()
        print("[INFO] Warmup done.")

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"infer(no-knn:{args.model})")
        for batch in pbar:
            if args.limit > 0 and scan_count >= args.limit:
                break

            t_e2e0 = time.perf_counter()

            in_vol = batch[0].to(device, non_blocking=True)  # (B,C,H,W)
            B, C, H, W = in_vol.shape
            if expected_in is None:
                expected_in = C
            in_vol = insert_or_append_mask(in_vol, expected_in, mask_idx=mask_idx)

            # forward timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = model(in_vol)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            model_time_sum += (t1 - t0)

            _, pred2d = argmax_logits(outs)  # (B,H,W)
            pred2d = pred2d.detach().cpu().numpy().astype(np.int32)

            path_seq, path_name, proj_idx = find_batch_fields(batch)
            if path_seq is None or path_name is None or proj_idx is None:
                raise RuntimeError("Batch is missing required fields: path_seq/path_name/proj_idx")

            # per item: unproject & save
            for bi in range(B):
                seq_str  = normalize_seq_str(path_seq[bi])
                out_name = _as_gt_label_name(str(path_name[bi]))

                pred_hw   = pred2d[bi]          # (H,W)
                H_, W_    = pred_hw.shape
                pred_flat = pred_hw.reshape(-1)

                proj_idx_b    = proj_idx[bi].cpu().numpy()
                proj_idx_flat = proj_idx_b.reshape(-1).astype(np.int64)

                pts_xyz  = _load_points_xyz(args.dataset, seq_str, out_name)
                N_points = pts_xyz.shape[0]

                # final labels (learning IDs)
                final_pred = np.zeros(N_points, dtype=np.int32)  # unprojected remain 0 unlabeled
                known_mask = np.zeros(N_points, dtype=bool)

                # --- writeback (auto-detect pt2pix or pix2pt) ---
                if proj_idx_flat.size == N_points:
                    # Case A: pt2pix (per-point -> pixel index)
                    pt2pix = proj_idx_flat
                    mask_pt = pt2pix >= 0
                    if mask_pt.any():
                        max_pix = int(pt2pix[mask_pt].max(initial=-1))
                        W0 = max(1, int((max_pix + 1 + H_ - 1) // H_))
                        y = pt2pix[mask_pt] // W0
                        x0 = pt2pix[mask_pt] %  W0
                        x = (x0 * W_) // W0
                        y = np.clip(y, 0, H_-1)
                        x = np.clip(x, 0, W_-1)
                        pix_id = y * W_ + x
                        final_pred[mask_pt] = pred_flat[pix_id]
                        known_mask[mask_pt] = True
                else:
                    # Case B: pix2pt (per-pixel -> point index)
                    pix2pt = proj_idx_flat
                    M = pix2pt.size
                    if M % H_ == 0:
                        H0, W0 = H_, M // H_
                    else:
                        H_candidates = [H_, 64, 32, 16, 128]
                        H0 = next((h for h in H_candidates if M % h == 0), H_)
                        W0 = M // H0

                    mask_px = pix2pt >= 0
                    if mask_px.any():
                        j  = np.arange(M, dtype=np.int64)
                        y0 = j // W0
                        x0 = j %  W0
                        x  = (x0 * W_) // W0
                        y0 = np.clip(y0, 0, H_-1)
                        x  = np.clip(x,  0, W_-1)
                        j2 = y0 * W_ + x
                        vals = pred_flat[j2]
                        pts  = pix2pt[mask_px]
                        final_pred[pts] = vals[mask_px]
                        known_mask[pts] = True

                # NO KNN: leave unknown as 0 (unlabeled)

                # map learning IDs -> original IDs
                if inv_map is not None:
                    max_k = inv_map.shape[0] - 1
                    final_pred = np.clip(final_pred, 0, max_k)
                    final_pred = inv_map[final_pred]

                # save
                out_dir  = osp.join(args.save_path, "sequences", seq_str, "predictions")
                ensure_dir(out_dir)
                tmp_path = osp.join(out_dir, out_name + ".tmp")
                out_path = osp.join(out_dir, out_name)
                final_pred.astype(np.uint32).tofile(tmp_path)
                os.replace(tmp_path, out_path)

                scan_count += 1

            t_e2e1 = time.perf_counter()
            e2e_time_sum += (t_e2e1 - t_e2e0)

            # live display
            if scan_count > 0:
                fps_model = scan_count / max(model_time_sum, 1e-12)
                fps_e2e   = scan_count / max(e2e_time_sum, 1e-12)
                pbar.set_postfix({
                    "scans": scan_count,
                    "FPS(model)": f"{fps_model:.2f}",
                    "FPS(E2E)": f"{fps_e2e:.2f}",
                    "ms/model": f"{1000.0 * model_time_sum / scan_count:.2f}",
                    "ms/E2E": f"{1000.0 * e2e_time_sum / scan_count:.2f}",
                })

    if scan_count == 0:
        print("[WARN] No scans processed.")
        return

    fps_model = scan_count / model_time_sum
    fps_e2e   = scan_count / e2e_time_sum

    print("\n========== FPS SUMMARY (NO KNN) ==========")
    print(f"Model                : {args.model}")
    print(f"Processed scans       : {scan_count}")
    print(f"Model-only avg (ms)   : {1000.0 * model_time_sum / scan_count:.3f} ms/scan")
    print(f"End-to-end avg (ms)   : {1000.0 * e2e_time_sum / scan_count:.3f} ms/scan")
    print(f"FPS (model-only)      : {fps_model:.3f}")
    print(f"FPS (end-to-end)      : {fps_e2e:.3f}")
    print("=========================================\n")
    print("Done.")


if __name__ == "__main__":
    main()
