import argparse
import gzip
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lib.utils.laserscan_Pointnet import (
    PolarPointEncoder,
    build_polar_point_features,
)


EXPECTED_H = 512
EXPECTED_W = 512
EXPECTED_MAX_RADIUS = 51.2
EXPECTED_FEATURE_DIM = 16
POINT_FEATURE_DIM = 7


def is_tensor_state_dict(value):
    return (
        isinstance(value, Mapping)
        and len(value) > 0
        and all(isinstance(key, str) for key in value)
        and all(torch.is_tensor(tensor) for tensor in value.values())
    )


def find_state_dict(checkpoint):
    if is_tensor_state_dict(checkpoint):
        return checkpoint, "encoder-only state_dict"

    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(
            "Checkpoint must be a state_dict or a mapping containing one"
        )

    for key in ("encoder", "model", "state_dict"):
        if key not in checkpoint:
            continue

        value = checkpoint[key]
        if is_tensor_state_dict(value):
            return value, f"checkpoint['{key}']"

        if isinstance(value, Mapping):
            nested_encoder = value.get("encoder")
            if is_tensor_state_dict(nested_encoder):
                return nested_encoder, f"checkpoint['{key}']['encoder']"

    raise RuntimeError(
        "No supported encoder state_dict was found. Expected an encoder-only "
        "state_dict or one under 'encoder', 'model', or 'state_dict'."
    )


def normalize_encoder_state_dict(state_dict, expected_keys):
    state_dict = dict(state_dict)
    expected_keys = set(expected_keys)

    if set(state_dict) == expected_keys:
        return state_dict, "direct"

    prefixes = (
        "encoder.",
        "module.encoder.",
        "model.encoder.",
        "module.",
        "model.",
    )
    for prefix in prefixes:
        candidate = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if set(candidate) == expected_keys:
            return candidate, f"stripped '{prefix}'"

    missing = sorted(expected_keys - set(state_dict))
    unexpected = sorted(set(state_dict) - expected_keys)
    raise RuntimeError(
        "Encoder state_dict keys do not match the current architecture. "
        f"Missing={missing[:8]}, unexpected={unexpected[:8]}"
    )


def load_encoder(checkpoint_path, feature_dim, device):
    encoder = PolarPointEncoder(
        in_channels=POINT_FEATURE_DIM,
        out_channels=feature_dim,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    raw_state, checkpoint_format = find_state_dict(checkpoint)
    state, normalization = normalize_encoder_state_dict(
        raw_state,
        encoder.state_dict().keys(),
    )

    incompatible = encoder.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict encoder load returned incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    loaded_state = encoder.state_dict()
    if not all(
        torch.equal(loaded_state[key].cpu(), state[key].cpu())
        for key in loaded_state
    ):
        raise RuntimeError("Loaded encoder weights differ from the checkpoint")

    encoder.requires_grad_(False)
    encoder.eval()
    encoder.to(device)

    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise RuntimeError("Encoder freeze/eval configuration failed")

    return encoder, {
        "checkpoint_format": checkpoint_format,
        "normalization": normalization,
        "state_tensors": len(state),
        "parameters": sum(p.numel() for p in encoder.parameters()),
        "strict": True,
    }


def normalize_sequence(sequence):
    try:
        sequence_id = int(sequence)
    except ValueError as exc:
        raise ValueError(f"Invalid sequence ID: {sequence}") from exc

    if sequence_id < 0 or sequence_id > 99:
        raise ValueError(f"Sequence ID is out of range: {sequence}")
    return f"{sequence_id:02d}"


def resolve_sequences(dataset_root, requested_sequences):
    sequences_root = dataset_root / "sequences"
    if not sequences_root.is_dir():
        raise RuntimeError(f"Sequence directory not found: {sequences_root}")

    if requested_sequences:
        sequences = []
        for value in requested_sequences:
            sequence = normalize_sequence(value)
            if sequence not in sequences:
                sequences.append(sequence)
    else:
        sequences = sorted(
            path.name
            for path in sequences_root.iterdir()
            if path.is_dir() and (path / "velodyne").is_dir()
        )

    if not sequences:
        raise RuntimeError("No sequences with velodyne scans were found")

    for sequence in sequences:
        velodyne_dir = sequences_root / sequence / "velodyne"
        if not velodyne_dir.is_dir():
            raise RuntimeError(
                f"Velodyne directory not found for sequence {sequence}: "
                f"{velodyne_dir}"
            )

    return sequences


def validate_feature_shapes(
    point_features,
    unique_cells,
    inverse_indices,
    cell_features,
    feature_dim,
    H,
    W,
):
    if point_features.ndim != 2 or point_features.shape[1] != POINT_FEATURE_DIM:
        raise RuntimeError(
            f"Unexpected point feature shape: {tuple(point_features.shape)}"
        )
    if unique_cells.ndim != 1:
        raise RuntimeError(
            f"Unexpected unique_cells shape: {tuple(unique_cells.shape)}"
        )
    if inverse_indices.shape != (point_features.shape[0],):
        raise RuntimeError(
            "inverse_indices does not match the valid point count: "
            f"{tuple(inverse_indices.shape)} vs {point_features.shape[0]}"
        )
    if cell_features.shape != (unique_cells.numel(), feature_dim):
        raise RuntimeError(
            "Unexpected cell feature shape: "
            f"{tuple(cell_features.shape)}, expected "
            f"({unique_cells.numel()}, {feature_dim})"
        )

    if unique_cells.numel() > 0:
        cell_min = int(unique_cells.min().item())
        cell_max = int(unique_cells.max().item())
        if cell_min < 0 or cell_max >= H * W:
            raise RuntimeError(
                f"Cell ID out of range: min={cell_min}, max={cell_max}"
            )
        if unique_cells.numel() > 1 and not torch.all(
            unique_cells[1:] > unique_cells[:-1]
        ):
            raise RuntimeError("Cell IDs are not sorted and unique")

    if torch.isnan(cell_features).any() or torch.isinf(cell_features).any():
        raise RuntimeError("Generated cell features contain NaN or Inf")


def save_sparse_features(data, output_path, overwrite):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(fd)
    temporary_path = Path(temporary_name)

    try:
        with gzip.open(temporary_path, "wb") as f:
            torch.save(data, f)

        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")

        temporary_path.chmod(0o644)
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def load_gzip_torch(path):
    with gzip.open(path, "rb") as f:
        return torch.load(
            f,
            map_location="cpu",
            weights_only=True,
        )


def print_generation_sanity(
    sequence,
    frame,
    raw_points,
    point_features,
    unique_cells,
    cell_features,
):
    finite_features = cell_features.float()
    has_nan = bool(torch.isnan(finite_features).any().item())
    has_inf = bool(torch.isinf(finite_features).any().item())

    print("\nFirst-scan sanity check")
    print(f"  sequence/frame     : {sequence}/{frame}")
    print(f"  raw points shape   : {tuple(raw_points.shape)}")
    print(f"  point_features     : {tuple(point_features.shape)}")
    print(f"  unique_cells       : {tuple(unique_cells.shape)}")
    print(f"  cell_features      : {tuple(cell_features.shape)}")

    if unique_cells.numel() > 0:
        print(
            "  cell id min/max    : "
            f"{int(unique_cells.min().item())} / "
            f"{int(unique_cells.max().item())}"
        )
        print(f"  feature min        : {finite_features.min().item():.8f}")
        print(f"  feature max        : {finite_features.max().item():.8f}")
        print(f"  feature mean       : {finite_features.mean().item():.8f}")
        print(
            "  feature std        : "
            f"{finite_features.std(unbiased=False).item():.8f}"
        )
    else:
        print("  cell id min/max    : N/A")
        print("  feature statistics : N/A (no occupied cells)")

    print(f"  has NaN            : {has_nan}")
    print(f"  has Inf            : {has_inf}")


def compare_polar_mask(polar_path, generated_mask, H, W):
    if not polar_path.exists():
        return {
            "available": False,
            "reason": f"Polar file not found: {polar_path}",
        }

    polar_data = load_gzip_torch(polar_path)
    if not isinstance(polar_data, Mapping) or "mask_t" not in polar_data:
        raise RuntimeError(
            f"Expected a gzip dict with 'mask_t' in {polar_path}"
        )

    polar_mask = polar_data["mask_t"]
    if polar_mask.shape == (1, H, W):
        polar_mask = polar_mask.squeeze(0)
    if polar_mask.shape != (H, W):
        raise RuntimeError(
            f"Unexpected Polar mask shape: {tuple(polar_mask.shape)}"
        )
    polar_mask = polar_mask.bool().cpu()

    different = int(torch.count_nonzero(polar_mask ^ generated_mask).item())
    agreement = float((polar_mask == generated_mask).float().mean().item())
    return {
        "available": True,
        "pointnet_occupied": int(generated_mask.sum().item()),
        "polar_occupied": int(polar_mask.sum().item()),
        "agreement": agreement,
        "different_cells": different,
    }


def verify_saved_features(
    output_path,
    expected_cell_ids,
    expected_features,
    save_dtype,
    feature_dim,
    H,
    W,
    max_radius,
    polar_path,
):
    saved = load_gzip_torch(output_path)
    required_keys = {
        "cell_ids",
        "features",
        "feature_dim",
        "H",
        "W",
        "max_radius",
    }
    if not isinstance(saved, Mapping) or not required_keys.issubset(saved):
        raise RuntimeError(
            f"Saved feature file is missing keys: {required_keys - set(saved)}"
        )

    cell_ids = saved["cell_ids"]
    features = saved["features"]
    expected_saved_features = expected_features.cpu().to(save_dtype)

    if cell_ids.shape != expected_cell_ids.shape:
        raise RuntimeError("Reloaded cell_ids shape does not match")
    if features.shape != expected_saved_features.shape:
        raise RuntimeError("Reloaded features shape does not match")
    if cell_ids.dtype != torch.long:
        raise RuntimeError(f"Unexpected cell_ids dtype: {cell_ids.dtype}")
    if features.dtype != save_dtype:
        raise RuntimeError(f"Unexpected saved feature dtype: {features.dtype}")
    if not torch.equal(cell_ids, expected_cell_ids.cpu()):
        raise RuntimeError("Reloaded cell_ids differ from generated cell IDs")
    if not torch.equal(features, expected_saved_features):
        raise RuntimeError("Reloaded features differ from the saved tensor")
    if torch.isnan(features).any() or torch.isinf(features).any():
        raise RuntimeError("Reloaded features contain NaN or Inf")

    if (
        int(saved["feature_dim"]) != feature_dim
        or int(saved["H"]) != H
        or int(saved["W"]) != W
        or not math.isclose(float(saved["max_radius"]), max_radius)
    ):
        raise RuntimeError("Reloaded feature metadata does not match")

    if save_dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-6, 1e-7
    features_allclose = torch.allclose(
        features.float(),
        expected_features.cpu().float(),
        rtol=rtol,
        atol=atol,
    )
    if not features_allclose:
        raise RuntimeError("Reloaded features failed the allclose check")

    max_abs_error = (
        float(
            (features.float() - expected_features.cpu().float())
            .abs()
            .max()
            .item()
        )
        if features.numel() > 0
        else 0.0
    )

    dense = torch.zeros(feature_dim, H * W, dtype=torch.float32)
    dense[:, cell_ids] = features.float().T
    dense = dense.view(feature_dim, H, W)

    occupied_mask = torch.zeros(H * W, dtype=torch.bool)
    occupied_mask[cell_ids] = True
    occupied_mask = occupied_mask.view(H, W)
    if int(occupied_mask.sum().item()) != cell_ids.numel():
        raise RuntimeError("Dense occupied-mask reconstruction failed")

    polar_comparison = compare_polar_mask(
        polar_path=polar_path,
        generated_mask=occupied_mask,
        H=H,
        W=W,
    )

    print("\nSaved-file reload test")
    print(f"  path               : {output_path}")
    print(f"  cell_ids shape     : {tuple(cell_ids.shape)}")
    print(f"  features shape     : {tuple(features.shape)}")
    print(f"  features dtype     : {features.dtype}")
    print("  cell_ids exact     : True")
    print("  features allclose  : True")
    print(f"  max abs error      : {max_abs_error:.8f}")
    print("  NaN / Inf          : False / False")

    print("\nDense reconstruction test")
    print(f"  dense shape        : {tuple(dense.shape)}")
    print(f"  mask shape         : {tuple(occupied_mask.shape)}")
    print(f"  mask occupied      : {int(occupied_mask.sum().item())}")

    print("\nPolar observation-mask comparison")
    if polar_comparison["available"]:
        print(
            "  PointNet occupied  : "
            f"{polar_comparison['pointnet_occupied']}"
        )
        print(
            "  Polar occupied     : "
            f"{polar_comparison['polar_occupied']}"
        )
        print(
            "  mask agreement     : "
            f"{polar_comparison['agreement']:.8f}"
        )
        print(
            "  different cells    : "
            f"{polar_comparison['different_cells']}"
        )
    else:
        print(f"  unavailable        : {polar_comparison['reason']}")

    return {
        "reload_cell_ids_exact": True,
        "reload_features_allclose": features_allclose,
        "max_abs_error": max_abs_error,
        "dense_shape": tuple(dense.shape),
        "mask_occupied": int(occupied_mask.sum().item()),
        "polar_comparison": polar_comparison,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        "Generate frozen Polar Point Encoder cell features"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--checkpoint",
        default="point_encoder_runs/best_point_encoder.pth",
    )
    parser.add_argument("--output_folder", default="pointnet_16ch")
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--max_radius", type=float, default=51.2)
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--debug_samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save_dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    args = parser.parse_args()

    if args.feature_dim != EXPECTED_FEATURE_DIM:
        parser.error(
            f"Step B requires --feature_dim {EXPECTED_FEATURE_DIM}"
        )
    if args.H != EXPECTED_H or args.W != EXPECTED_W:
        parser.error(f"Step B requires --H {EXPECTED_H} --W {EXPECTED_W}")
    if not math.isclose(args.max_radius, EXPECTED_MAX_RADIUS):
        parser.error(
            f"Step B requires --max_radius {EXPECTED_MAX_RADIUS}"
        )
    if args.debug_samples < 0:
        parser.error("--debug_samples must be zero or greater")

    output_folder = Path(args.output_folder)
    if (
        output_folder.is_absolute()
        or len(output_folder.parts) != 1
        or output_folder.name in ("", ".", "..")
    ):
        parser.error("--output_folder must be a single relative folder name")

    return args


def main():
    args = parse_args()
    dataset_root = Path(args.dataset)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")

    sequences = resolve_sequences(dataset_root, args.sequences)
    save_dtype = (
        torch.float16 if args.save_dtype == "float16" else torch.float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, load_info = load_encoder(
        checkpoint_path=checkpoint_path,
        feature_dim=args.feature_dim,
        device=device,
    )

    print("================================")
    print("Polar Point Feature Generation")
    print("================================")
    print(f"checkpoint         : {checkpoint_path}")
    print(f"checkpoint format  : {load_info['checkpoint_format']}")
    print(f"key normalization  : {load_info['normalization']}")
    print(f"strict load        : {load_info['strict']}")
    print(f"state tensors      : {load_info['state_tensors']}")
    print(f"encoder parameters : {load_info['parameters']}")
    print("encoder mode       : eval, frozen")
    print(f"device             : {device}")
    print(f"sequences          : {sequences}")
    print(f"output folder      : {args.output_folder}")
    print(f"save dtype         : {save_dtype}")
    print(f"debug samples      : {args.debug_samples or 'all'}")
    print("================================")

    processed = 0
    skipped = 0
    failed = 0
    total_occupied_cells = 0
    considered = 0
    first_scan_checked = False

    for sequence in sequences:
        sequence_dir = dataset_root / "sequences" / sequence
        scan_files = sorted((sequence_dir / "velodyne").glob("*.bin"))

        if args.debug_samples > 0:
            remaining = args.debug_samples - considered
            if remaining <= 0:
                break
            scan_files = scan_files[:remaining]

        output_dir = sequence_dir / args.output_folder
        output_dir.mkdir(parents=True, exist_ok=True)

        progress = tqdm(
            scan_files,
            desc=f"Sequence {sequence}",
            unit="frame",
        )
        for scan_path in progress:
            considered += 1
            frame = scan_path.stem
            output_path = output_dir / f"{frame}.pt"

            if output_path.exists() and not args.overwrite:
                skipped += 1
                progress.set_postfix(frame=frame, status="skipped")
                continue

            try:
                raw_points = np.fromfile(
                    scan_path,
                    dtype=np.float32,
                ).reshape(-1, 4)
                points = torch.from_numpy(raw_points).to(device)

                with torch.inference_mode():
                    point_features, unique_cells, inverse_indices = (
                        build_polar_point_features(
                            points,
                            H=args.H,
                            W=args.W,
                            max_radius=args.max_radius,
                        )
                    )
                    cell_features = encoder(
                        point_features,
                        inverse_indices,
                    )

                validate_feature_shapes(
                    point_features=point_features,
                    unique_cells=unique_cells,
                    inverse_indices=inverse_indices,
                    cell_features=cell_features,
                    feature_dim=args.feature_dim,
                    H=args.H,
                    W=args.W,
                )

                cell_ids_cpu = unique_cells.detach().cpu().long()
                cell_features_cpu = cell_features.detach().cpu().float()

                if not first_scan_checked:
                    print_generation_sanity(
                        sequence=sequence,
                        frame=frame,
                        raw_points=raw_points,
                        point_features=point_features,
                        unique_cells=unique_cells,
                        cell_features=cell_features,
                    )

                save_data = {
                    "cell_ids": cell_ids_cpu,
                    "features": cell_features_cpu.to(save_dtype),
                    "feature_dim": args.feature_dim,
                    "H": args.H,
                    "W": args.W,
                    "max_radius": args.max_radius,
                }
                save_sparse_features(
                    data=save_data,
                    output_path=output_path,
                    overwrite=args.overwrite,
                )

                occupied_cells = int(cell_ids_cpu.numel())
                processed += 1
                total_occupied_cells += occupied_cells
                progress.set_postfix(
                    frame=frame,
                    occupied=occupied_cells,
                )

                if not first_scan_checked:
                    polar_path = (
                        sequence_dir
                        / "polar_512_prlb"
                        / f"{frame}.pt"
                    )
                    verify_saved_features(
                        output_path=output_path,
                        expected_cell_ids=cell_ids_cpu,
                        expected_features=cell_features_cpu,
                        save_dtype=save_dtype,
                        feature_dim=args.feature_dim,
                        H=args.H,
                        W=args.W,
                        max_radius=args.max_radius,
                        polar_path=polar_path,
                    )
                    print(
                        f"\nSaved file size     : "
                        f"{output_path.stat().st_size} bytes"
                    )
                    first_scan_checked = True

            except Exception as exc:
                failed += 1
                tqdm.write(
                    f"[FAILED] {sequence}/{frame}: "
                    f"{type(exc).__name__}: {exc}"
                )

    average_occupied = (
        total_occupied_cells / processed
        if processed > 0
        else 0.0
    )

    print("\n================================")
    print("Feature generation summary")
    print("================================")
    print(f"processed            : {processed}")
    print(f"skipped              : {skipped}")
    print(f"failed               : {failed}")
    print(f"total occupied cells : {total_occupied_cells}")
    print(f"average occupied     : {average_occupied:.2f}")
    print("================================")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
