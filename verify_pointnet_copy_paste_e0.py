"""Step E.0: read-only integrity audit, never a training entrypoint."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch
import yaml

from generate_point_features import load_encoder
from lib.dataset.SemanticKitti_PointNet import SemanticKitti
from lib.utils.laserscan_Pointnet import build_polar_point_features
from lib.utils.laserscan_Polar4 import SemLaserScan
from lib.utils.pointnet_copy_paste_e0 import (
    CandidateTrace, ambiguous_priority_cells, cell_ids_numpy, differences, normalize_handcrafted,
    off_reference, project_handcrafted, read_bev, read_raw, seed_all, sha256,
    transplant_cell_points,
)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def dataset_for(root, data, sequences, train=False, probability=0):
    return SemanticKitti(
        root=str(root), sequences=sequences, labels=data["labels"],
        color_map=data["color_map"], learning_map=data["learning_map"],
        learning_map_inv=data["learning_map_inv"],
        sensor={"img_prop": {"height": 512, "width": 512}},
        is_train=train, copy_paste_prob=probability,
    )


def regenerate(encoder, points, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        features, cells, inverse = build_polar_point_features(torch.from_numpy(points).to(device))
        values = encoder(features, inverse)
        cells, values = cells.cpu(), values.cpu()
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    dense = torch.zeros(16, 512 * 512)
    dense[:, cells] = values.T
    occupied = torch.zeros(512 * 512, dtype=torch.bool)
    occupied[cells] = True
    return dense.reshape(16, 512, 512), occupied.reshape(512, 512), {
        "seconds_including_transfer": seconds, "peak_allocated_bytes": peak,
        "occupied_cells": len(cells),
    }


def cache_reproduction(dataset, paths, encoder, device, data, log):
    rows = []
    for number, path in enumerate(paths):
        points, labels = read_raw(path)
        cached_hand, cached_mask, cached_label = read_bev(path)
        start = time.perf_counter()
        hand, mask, label = project_handcrafted(points, labels, data["learning_map"])
        projection_seconds = time.perf_counter() - start
        point, occupied, timing = regenerate(encoder, points, device)
        cached_point, cache_occupied = dataset.load_pointnet_feature(path)
        with gzip.open(dataset.get_pointnet_file(path), "rb") as stream:
            sparse = torch.load(stream, map_location="cpu", weights_only=True)
        row = {"scan": str(path), "handcrafted_projection_seconds": projection_seconds,
               "handcrafted_fp16": differences(hand.half().float(), cached_hand),
               "mask": differences(mask, cached_mask), "labels": differences(label, cached_label),
               "pointnet_fp16": differences(point.half().float(), cached_point, cache_occupied),
               "pointnet_fp32_vs_saved_fp16": differences(point, cached_point, cache_occupied),
               "pointnet_fp16_with_rounding_tolerance": differences(
                   point.half().float(), cached_point, cache_occupied, atol=1e-5, rtol=0.002),
               "handcrafted_fp16_with_rounding_tolerance": differences(
                   hand.half().float(), cached_hand, atol=1e-5, rtol=0.002),
               "pointnet_numpy_occupancy_discrepancy_cells": int((occupied != mask[0]).sum()),
               "pointnet_regenerated_vs_cached_occupancy_cells": int((occupied != cache_occupied).sum()),
               "cached_pointnet_vs_polar_occupancy_cells": int((cache_occupied != cached_mask[0].bool()).sum()),
               "duplicate_cached_cell_ids": int(sparse["cell_ids"].numel() - sparse["cell_ids"].unique().numel()),
               "timing": timing}
        if number == 0:
            rng = np.random.get_state()
            scan = SemLaserScan(sem_color_dict=data["color_map"], project=False)
            np.random.set_state(rng)
            scan.set_points(points[:, :3], points[:, 3])
            scan.project = True
            start = time.perf_counter()
            scan.set_label(labels.astype(np.int32))
            lut = np.zeros(max(data["learning_map"]) + 100, dtype=np.int64)
            for key, value in data["learning_map"].items():
                lut[key] = value
            actual_hand = torch.from_numpy(scan.pseudo_image.transpose(2, 0, 1))
            actual_label = torch.from_numpy(lut[scan.proj_sem_label])
            row["original_projector_seconds"] = time.perf_counter() - start
            row["reference_vs_original_projector"] = {
                "hand_fp32": differences(hand, actual_hand),
                "hand_fp16": differences(hand.half(), actual_hand.half()),
                "mask": differences(mask[0], torch.from_numpy(scan.proj_mask)),
                "label": differences(label, actual_label),
                "original_vs_cache_fp16": differences(actual_hand.half(), cached_hand),
            }
        rows.append(row)
        log(f"Original reproduction {number + 1}/{len(paths)}: {path}; "
            f"PN max error {row['pointnet_fp16']['max_absolute_error']:.6g}")
    return rows


def paste_invariants(dataset, before, after, paste):
    src_hand, src_mask, src_label = read_bev(paste["src_file"])
    src_point, occupied = dataset.load_pointnet_feature(paste["src_file"])
    src_point = dataset.align_pointnet_mask(paste["src_file"], src_point, occupied, src_mask)
    y, x, h, w = paste["ty"], paste["tx"], paste["h"], paste["w"]
    expected = [value.clone() for value in before]
    m = paste["m"][0]
    for dst, src in zip(expected, (src_hand, src_point, src_mask, src_label)):
        region = dst[..., y:y+h, x:x+w]
        source = src[..., paste["y1"]:paste["y2"], paste["x1"]:paste["x2"]]
        region[..., m] = source[..., m]
    pmask = after[4][0].bool()
    return {
        "hand7_mismatch_values": differences(after[0], expected[0])["different_values"],
        "point16_mismatch_values": differences(after[1], expected[1])["different_values"],
        "mask_mismatch_cells": differences(after[2], expected[2])["different_values"],
        "label_mismatch_cells": differences(after[3], expected[3])["different_values"],
        "outside_paste_changed_values": sum(differences(a, b, ~pmask)["different_values"]
                                           for a, b in zip(after[:4], before)),
        "paste_cells": int(pmask.sum()),
        "new_observed_cells": int((pmask & ~before[2][0].bool()).sum()),
        "overwritten_observed_cells": int((pmask & before[2][0].bool()).sum()),
        "same_label_as_before_cells": int((pmask & (after[3] == before[3])).sum()),
        "all_16_features_unchanged_cells": int((pmask & (after[1] == before[1]).all(0)).sum()),
        "paste_mask_zero_observation_cells": int((pmask & ~after[2][0].bool()).sum()),
        "donor_missing_pointnet_paste_cells": int((m & ~occupied[
            paste["y1"]:paste["y2"], paste["x1"]:paste["x2"]]).sum()),
        "zero_pointnet_paste_cells": int((pmask & (after[1] == 0).all(0)).sum()),
    }


def augmentation_test(dataset, index, before, after):
    calls = 0

    def inject(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return tuple(v.clone() for v in after)
        return (*args[:4], torch.zeros_like(args[2]))

    code = SemanticKitti.__getitem__.__code__
    source = Path(code.co_filename).read_text().splitlines()
    observed = {"flip": False}

    def trace(frame, event, arg):
        if frame.f_code is not code:
            return None
        if event == "line" and "proj_tensor = torch.flip" in source[frame.f_lineno - 1]:
            observed["flip"] = True
        if event == "return":
            observed["shift"] = frame.f_locals["roll_shift"]
            observed["drop_mask"] = frame.f_locals.get("drop_mask", torch.ones_like(after[2])).clone()
        return trace

    dataset.copy_paste_prob = 1.0
    dataset.is_train = True
    seed_all(7)
    old = sys.gettrace()
    try:
        sys.settrace(trace)
        with patch.object(dataset, "apply_bev_copy_paste", side_effect=inject):
            actual = dataset[index]
    finally:
        sys.settrace(old)
        dataset.copy_paste_prob = 0.0
    expected = [normalize_handcrafted(after[0]), after[1], after[2], after[3], after[4]]
    for i, value in enumerate(expected):
        dim = 1 if i == 3 else 2
        value = torch.roll(value, observed["shift"], dim)
        if observed["flip"]:
            value = value.flip([dim])
        if i != 3:
            value = value * observed["drop_mask"]
        expected[i] = value
    hybrid = torch.cat(expected[:2])
    errors = [differences(a, b)["different_values"] for a, b in zip(
        (actual[0], actual[1], actual[2], actual[14]),
        (hybrid, expected[2], expected[3], expected[4]))]
    dropped = observed["drop_mask"][0] == 0
    if not observed["flip"] or not dropped.any():
        raise AssertionError("Augmentation diagnostic must exercise both flip and dropout")
    return {"method": "real successful paste injected into unchanged Dataset; actual augmentation traced",
            "shift": observed["shift"], "flip": observed["flip"],
            "dropout_cells": int(dropped.sum()), "array_transform_mismatch_values": errors,
            "dropped_nonignore_labels": int((dropped & (actual[2] != 0)).sum()),
            "zero_mask_nonignore_labels": int(((actual[1][0] == 0) & (actual[2] != 0)).sum()),
            "labels_cleared_by_dropout": False,
            "geometric_raw_equivalence": False}


def diagnostic_png(path, before, after, regenerated, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    palette[0] = (0, 0, 0, 1)
    fig, axes = plt.subplots(3, 4, figsize=(17, 12))
    images = [(before[3], "Before label", "label"), (after[3], "Feature CP label", "label"),
              (after[4][0], "Paste mask", "gray"),
              ((after[1] - before[1]).norm(dim=0), "PointNet16 change L2", "magma"),
              (before[2][0], "Before observation", "gray"), (after[2][0], "After observation", "gray"),
              (before[0][0], "Before max_z (raw)", "viridis"), (after[0][0], "After max_z (raw)", "viridis"),
              (regenerated[3], "Raw alternative label", "label"),
              (regenerated[2][0], "Raw alternative occupancy", "gray"),
              ((after[1] - regenerated[1]).norm(dim=0), "CP vs raw-regenerated PN16 L2", "magma"),
              ((after[0] - regenerated[0]).abs().sum(0), "CP vs raw-regenerated 7ch L1", "magma")]
    for axis, (value, name, color) in zip(axes.flat, images):
        kwargs = {"cmap": ListedColormap(palette), "vmin": 0, "vmax": 19} if color == "label" else {"cmap": color}
        axis.imshow(value.numpy(), origin="lower", interpolation="nearest", **kwargs)
        axis.set_title(name, fontsize=10)
        axis.set_xlabel("Azimuth column")
        axis.set_ylabel("Radius row")
    fig.suptitle(title + "\nRaw alternative is diagnostic cell replacement, NOT legacy CP raw reconstruction", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    # Zoom all panels to the paste support, using common bounds.
    y, x = torch.where(after[4][0] > 0)
    for axis in axes.flat:
        axis.set_xlim(max(int(x.min()) - 8, 0), min(int(x.max()) + 8, 511))
        axis.set_ylim(max(int(y.min()) - 8, 0), min(int(y.max()) + 8, 511))
    fig.savefig(path.with_name(path.stem + "_zoom.png"), dpi=140)
    plt.close(fig)


def run(args):
    root = Path.cwd().resolve()
    output = (root / args.output).resolve()
    if output.parent != root / "logs" or "stepE0" not in output.name:
        raise ValueError("Output must be a dedicated logs/*stepE0* directory")
    output.mkdir(exist_ok=False)
    (output / "before_after_examples").mkdir()
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib_cache")

    # Deny Python-level writes anywhere except this fresh E.0 output directory.
    def readonly_guard(event, arguments):
        if event == "open":
            name, mode, flags = arguments
            if isinstance(name, (str, bytes, os.PathLike)):
                writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                    isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
                if writing:
                    path = Path(os.fsdecode(name)).resolve()
                    if not path.is_relative_to(output) and str(path) not in ("/dev/null",):
                        raise PermissionError(f"E.0 read-only audit denied write: {path}")
    sys.addaudithook(readonly_guard)

    def log(message):
        line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        with (output / "execution_log.txt").open("a") as stream:
            stream.write(line + "\n")

    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    log("E.0 integrity verification only. NO TRAINING, NO OPTIMIZER, NO TRAINING CHECKPOINT.")
    unit = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests",
                           "-p", "test_pointnet_copy_paste_integrity_e0.py", "-v"],
                          capture_output=True, text=True)
    (output / "unit_tests.log").write_text(unit.stdout + unit.stderr)
    if unit.returncode:
        raise RuntimeError("E.0 unit tests failed; see unit_tests.log")
    source_paths = list((root / "lib").rglob("*.py")) + list((root / "config").rglob("*.yaml"))
    source_paths += [root / "generate_point_features.py", root / "prepare_bev2.py", root / "train_pointnet_hybrid.py"]
    source_hashes = {str(p.relative_to(root)): sha256(p) for p in source_paths}
    old_artifacts = {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns]
                     for p in (root / "logs").rglob("*") if p.is_file() and not p.is_relative_to(output)}
    data = yaml.safe_load((root / "config/labels/semantic-kitti.yaml").read_text())
    dataset = dataset_for(root / "SemanticKitti", data, data["split"]["train"], True)
    allowed = {f"{int(s):02d}" for s in data["split"]["train"]}
    donor_sequences = Counter(dataset.get_seq_frame(p)[0] for p in dataset.bev_files)
    if set(donor_sequences) - allowed or "08" in donor_sequences:
        raise RuntimeError("Train donor split violation")
    (output / "donor_files.txt").write_text("\n".join(dataset.bev_files) + "\n")
    save_json(output / "donor_split_audit.json", {
        "allowed_train_sequences": sorted(allowed), "donor_sequence_counts": donor_sequences,
        "donor_files": "donor_files.txt", "donor_list_sha256": sha256(output / "donor_files.txt"),
        "validation_or_test_donors": 0, "validation_or_test_targets": 0,
        "parser_train_valid_test_cp_probability": 0.0,
        "dataset_itself_enforces_split": False,
        "scope": "isolated audit train pool plus production Parser source inspection; no valid/test samples used"})
    checkpoint = root / args.encoder
    encoder_hash = sha256(checkpoint)
    encoder, load_info = load_encoder(str(checkpoint), 16, device)
    log(f"Fixed encoder {checkpoint}; sha256={encoder_hash}; eval/frozen={not encoder.training}")
    paths = [dataset.scan_files[0], dataset.scan_files[len(dataset) // 3], dataset.scan_files[2 * len(dataset) // 3]]
    consistency = {"encoder_checkpoint": str(checkpoint), "encoder_sha256": encoder_hash,
                   "load_info": load_info, "cache_embeds_encoder_provenance": False,
                   "cache_generation_precision_metadata": "not stored; tested FP32 computation and FP16 storage",
                   "original_reproduction": cache_reproduction(dataset, paths, encoder, device, data, log),
                   "raw_alternative_comparisons": []}
    save_json(output / "feature_consistency.json", consistency)
    seed_all(args.seed)
    statistics, successes, cp_seconds, cached_seconds, online_seconds = [], [], [], [], []
    invocations = 0
    first_case = None
    read_paths = set(paths)
    while len(statistics) < args.min_candidates or len(successes) < args.min_successes:
        if len(statistics) >= args.max_candidates:
            log("Candidate ceiling reached; unmet sample target will be reported, not hidden.")
            break
        index = random.randrange(len(dataset))
        path = dataset.scan_files[index]
        hand, mask, label = read_bev(path)
        point, occupied = dataset.load_pointnet_feature(path)
        point = dataset.align_pointnet_mask(path, point, occupied, mask)
        before = (hand, point, mask, label)
        prev = dataset.load_neighbor_label(dataset.get_neighbor_file(path, -1))
        nxt = dataset.load_neighbor_label(dataset.get_neighbor_file(path, 1))
        trace = CandidateTrace()
        after, seconds = trace.call(dataset, [v.clone() for v in before],
                                    prev_labels_t=prev, next_labels_t=nxt)
        invocations += 1
        cp_seconds.append(seconds)
        for row in trace.rows:
            row.update(invocation=invocations, target=path, call_seconds=seconds)
        if trace.success is None:
            if any(not torch.equal(a, b) for a, b in zip(before, after[:4])) or after[4].any():
                raise AssertionError("Failed paste mutated its inputs")
        else:
            paste = trace.success
            invariant = paste_invariants(dataset, before, after, paste)
            invariant.update(target=path, donor=paste["src_file"], class_id=paste["obj_cls"],
                             radial_shift_cells=paste["ty"] - paste["y1"],
                             visible_ratio=paste["visible_ratio"], call_seconds=seconds)
            trace.rows[-1].update(invariant)
            successes.append(invariant)
            read_paths.update([path, paste["src_file"]])
            target_points, target_labels = read_raw(path)
            donor_points, donor_labels = read_raw(paste["src_file"])
            start = time.perf_counter()
            raw_points, raw_labels, raw_info = transplant_cell_points(
                target_points, target_labels, donor_points, donor_labels, paste)
            raw_hand, raw_mask, raw_label = project_handcrafted(raw_points, raw_labels, data["learning_map"])
            raw_point, pn_occupied, timing = regenerate(encoder, raw_points, device)
            online_total = time.perf_counter() - start
            online_seconds.append(online_total)
            pmask = after[4][0].bool()
            comparison = {"success_index": len(successes), "target": path, "donor": paste["src_file"],
                          "raw_policy": "diagnostic non-rigid polar cell replacement, no occlusion guarantee",
                          "raw": raw_info, "online_seconds_excluding_raw_disk_reads": online_total,
                          "encoder_timing": timing,
                          "pointnet_vs_numpy_occupancy_cells": int((pn_occupied != raw_mask[0]).sum()),
                          "regions": {}}
            ambiguous = ambiguous_priority_cells(raw_points, raw_labels, data["learning_map"])
            mismatch = after[3] != raw_label
            comparison["label_priority_tie_diagnostic"] = {
                "outside_paste_label_mismatches": int((mismatch & ~pmask).sum()),
                "outside_mismatches_with_multiple_top_priority_learning_ids": int((mismatch & ~pmask & ambiguous).sum()),
                "outside_mismatches_without_top_priority_ties": int((mismatch & ~pmask & ~ambiguous).sum()),
                "cause": "original np.argsort(priority) is unstable in tied priorities; changing point count changes tie ordering"}
            for name, region in (("paste", pmask), ("outside", ~pmask)):
                comparison["regions"][name] = {
                    "hand7": differences(after[0], raw_hand.half().float(), region, atol=1e-3, rtol=1e-3),
                    "point16": differences(after[1], raw_point.half().float(), region, atol=1e-3, rtol=1e-3),
                    "label": differences(after[3], raw_label, region),
                    "mask": differences(after[2], raw_mask, region),
                    "normalized_hand7": differences(normalize_handcrafted(after[0]),
                        normalize_handcrafted(raw_hand.half().float()), region, atol=1e-3, rtol=1e-3)}
            consistency["raw_alternative_comparisons"].append(comparison)
            if len(successes) <= 3:
                diagnostic_png(output / "before_after_examples" / f"case_{len(successes):02d}.png",
                               before, after, (raw_hand, raw_point, raw_mask, raw_label),
                               f"Train {dataset.get_seq_frame(path)} / donor {dataset.get_seq_frame(paste['src_file'])}; class {paste['obj_cls']}")
            if first_case is None:
                first_case = (index, before, after)
                start = time.perf_counter()
                cache_file = output / "diagnostic_augmented_cache.npz"
                np.savez_compressed(cache_file, points=raw_points, packed_labels=raw_labels,
                                    hand7=raw_hand.numpy().astype(np.float16),
                                    point16=raw_point.numpy().astype(np.float16),
                                    mask=raw_mask.numpy(), semantic=raw_label.numpy())
                write_time = time.perf_counter() - start
                for _ in range(5):
                    start = time.perf_counter()
                    with np.load(cache_file) as stored:
                        _ = {key: stored[key] for key in stored.files}
                    cached_seconds.append(time.perf_counter() - start)
                consistency["precomputed_augmented_cache"] = {
                    "path": str(cache_file.relative_to(root)), "bytes": cache_file.stat().st_size,
                    "write_seconds": write_time, "read_seconds_warm_filesystem": cached_seconds,
                    "scope": "one diagnostic alternative, not a production cache format or full E.1 benchmark"}
            log(f"Success {len(successes)}: class={paste['obj_cls']}, cells={invariant['paste_cells']}, "
                f"raw PN mismatch cells={comparison['regions']['paste']['point16']['different_cells']}")
        statistics.extend(trace.rows)
        if invocations % 20 == 0:
            log(f"Progress: calls={invocations}, candidates={len(statistics)}, successes={len(successes)}")
    fields = sorted(set().union(*(row.keys() for row in statistics)))
    with (output / "copy_paste_statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(statistics)
    save_json(output / "feature_consistency.json", consistency)
    regressions = []
    for index in (0, len(dataset) // 3, 2 * len(dataset) // 3):
        for seed in (0, 17, 42):
            expected = off_reference(dataset, index, seed)
            seed_all(seed)
            actual = dataset[index]
            errors = [differences(a, b)["different_values"] for a, b in zip(
                (actual[0], actual[1], actual[2], actual[14]), expected)]
            regressions.append({"index": index, "seed": seed, "mismatch_values": errors})
    augmentation = augmentation_test(dataset, *first_case) if first_case else {"not_run": "no successful paste"}
    tests = {"cp_off_regression": regressions, "post_cp_augmentation": augmentation,
             "unit_tests": {"returncode": unit.returncode, "log": "unit_tests.log"},
             "successful_pastes": successes,
             "sample_targets_met": len(statistics) >= args.min_candidates and len(successes) >= args.min_successes}
    save_json(output / "test_results.json", tests)
    unchanged_sources = all(sha256(root / p) == digest for p, digest in source_hashes.items())
    unchanged_old_artifacts = all((root / p).is_file() and [(root / p).stat().st_size,
        (root / p).stat().st_mtime_ns] == metadata for p, metadata in old_artifacts.items())
    manifests = {}
    for path in sorted(read_paths):
        p = Path(path)
        for candidate in [p, Path(dataset.get_pointnet_file(path)),
                          p.parent.parent / "velodyne" / (p.stem + ".bin"),
                          p.parent.parent / "labels" / (p.stem + ".label")]:
            manifests[str(candidate)] = sha256(candidate)
    save_json(output / "read_data_sha256.json", manifests)
    save_json(output / "source_sha256.json", source_hashes)
    reasons = Counter(row["reason"] for row in statistics if not row["success"])
    verification = {
        "verdict": "FAIL", "formal_training_started": False, "optimizer_created": False,
        "method": "existing feature-level CP; independent raw alternative is NOT legacy raw reconstruction",
        "calls": invocations, "candidates": len(statistics), "successful_pastes": len(successes),
        "candidate_success_rate": len(successes) / max(len(statistics), 1),
        "call_success_rate": len(successes) / max(invocations, 1), "skip_reasons": reasons,
        "seed": args.seed, "sample_targets_met": tests["sample_targets_met"],
        "source_hashes_unchanged": unchanged_sources,
        "existing_log_artifact_size_mtime_unchanged": unchanged_old_artifacts,
        "read_only_write_guard": "Python open writes limited to new E.0 output; no data/source writes allowed",
        "encoder_hash_unchanged": sha256(checkpoint) == encoder_hash,
        "encoder_eval_frozen": not encoder.training and not any(p.requires_grad for p in encoder.parameters()),
        "raw_original_cache_hashes": "read_data_sha256.json (post-read manifest; no full-dataset hash claim)",
        "cp_off_regression_pass": all(not any(row["mismatch_values"]) for row in regressions),
        "original_cache_reproduction_with_rounding_tolerance_pass": all(
            not row["pointnet_fp16_with_rounding_tolerance"]["different_values"]
            and not row["handcrafted_fp16_with_rounding_tolerance"]["different_values"]
            and not row["mask"]["different_values"] and not row["labels"]["different_values"]
            and not row["pointnet_regenerated_vs_cached_occupancy_cells"] for row in consistency["original_reproduction"]),
        "original_reproduction_tolerance": "atol=1e-5, rtol=0.002 (about two FP16 relative ULPs); exact differences retained",
        "feature_copy_invariants_pass": all(not any(row[k] for k in (
            "hand7_mismatch_values", "point16_mismatch_values", "mask_mismatch_cells",
            "label_mismatch_cells", "outside_paste_changed_values", "paste_mask_zero_observation_cells")) for row in successes) and bool(successes),
        "timing": {"feature_cp_call_seconds_mean": float(np.mean(cp_seconds)),
                   "feature_cp_call_seconds_total": sum(cp_seconds),
                   "online_raw_prototype_seconds_mean": float(np.mean(online_seconds)) if online_seconds else None,
                   "precomputed_read_seconds_mean": float(np.mean(cached_seconds)) if cached_seconds else None,
                   "verification_wall_seconds": time.perf_counter() - started,
                   "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                   "scope": "sequential isolated audit with tracing; not training throughput; warm cache; no backward"},
        "gpu": {"name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU only",
                "encoder_peak_allocated_bytes": max((r["encoder_timing"]["peak_allocated_bytes"]
                    for r in consistency["raw_alternative_comparisons"]), default=0)},
        "blockers": ["Legacy CP has no modified raw points; moving cached PointNet does not preserve absolute xy inputs",
                     "Dropout keeps nonignore labels at zero-mask cells; main/aux Focal and Lovasz are not observation-masked",
                     "Legacy CP crops rather than wraps angular boundaries and may fragment objects",
                     "Encoder provenance not embedded in old caches; see measured original reproduction errors"],
    }
    severe = (not verification["feature_copy_invariants_pass"] or
              not verification["cp_off_regression_pass"] or
              not verification["original_cache_reproduction_with_rounding_tolerance_pass"] or
              augmentation.get("dropped_nonignore_labels", 0) > 0 or not tests["sample_targets_met"])
    verification["verdict"] = "FAIL" if severe else "CONDITIONAL PASS"
    save_json(output / "verification.json", verification)
    (output / "integrity_report.md").write_text(build_report(verification, consistency, tests))
    log(f"Finished: {verification['verdict']}; {len(successes)}/{len(statistics)} candidates successful. NO TRAINING.")


def build_report(v, c, tests):
    originals = c["original_reproduction"]
    changed = [r["regions"]["paste"]["point16"] for r in c["raw_alternative_comparisons"]]
    successes = tests["successful_pastes"]
    class_counts = dict(Counter(row["class_id"] for row in successes))
    return f"""# Step E.0 Integrity Report

## Verdict: {v['verdict']}

No formal training, optimizer, or training checkpoint was created. Existing Dataset,
Trainer, models, configs, source checkpoints and SemanticKITTI data remain read-only.

The current 24ch Dataset already copies **both 7ch and 16ch**, along with semantic
labels and observation/paste masks. This is **feature-level Copy-Paste**, after
PointNet cache generation. It is not raw-point Copy-Paste and has no post-paste raw
point cloud to reconstruct. The old 8ch implementation alone cannot handle 16ch.

## Measured Results

- Seed: {v['seed']}; production-function calls: {v['calls']}.
- Candidate selections: {v['candidates']}; successful pastes: {v['successful_pastes']}.
- Candidate success rate: {v['candidate_success_rate']:.4%}; call success rate: {v['call_success_rate']:.4%}.
- Successful class counts (learning IDs): {class_counts}. Other CP classes were
  not successfully observed in this real-data sample; no all-class guarantee.
- Pasted cells: {sum(r['paste_cells'] for r in successes)}; new observations:
  {sum(r['new_observed_cells'] for r in successes)}; overwritten observations:
  {sum(r['overwritten_observed_cells'] for r in successes)}.
- Sample targets met: {v['sample_targets_met']}.
- Exact donor feature/label/mask correspondence: {v['feature_copy_invariants_pass']}.
- Donor-correspondence mismatch totals (values/cells): hand7
  {sum(r['hand7_mismatch_values'] for r in successes)}, PointNet16
  {sum(r['point16_mismatch_values'] for r in successes)}, label
  {sum(r['label_mismatch_cells'] for r in successes)}, mask
  {sum(r['mask_mismatch_cells'] for r in successes)}. Missing donor PointNet cells
  inside paste: {sum(r['donor_missing_pointnet_paste_cells'] for r in successes)}.
- CP-OFF regression (three scans x three fixed seeds): {v['cp_off_regression_pass']}.
- Original regeneration within documented FP16 tolerance: {v['original_cache_reproduction_with_rounding_tolerance_pass']}.
- Actual transformed arrays compared after roll/flip/dropout: {tests['post_cp_augmentation']}.
- Original regeneration scans: {len(originals)}. Full errors and occupancy differences:
  `feature_consistency.json`; no unmeasured whole-dataset guarantee.
- Raw alternative comparisons: {len(changed)}; PointNet mismatch cells in paste regions:
  {sum(r['different_cells'] for r in changed)}; maximum absolute PointNet error:
  {max((r['max_absolute_error'] for r in changed), default=0):.8g} (atol=rtol=0.001).
- Skip counts: `{dict(v['skip_reasons'])}`. Per-candidate and per-success CSV records
  include class, cells, new/overwritten observations and all array mismatch counts.
- Training/validation/test leakage in audited donor pool: zero; exact list in
  `donor_files.txt`, split counts and SHA256 in `donor_split_audit.json`.

Unchanged semantic *values* are not necessarily failed pastes: two different
objects can have the same class. We require equality to the selected donor, not
inequality to the original class. All-16-unchanged cells are counted separately.

## Raw Regeneration And Geometry

The independent diagnostic alternative selects **all points in selected donor
cells**, removes all points in corresponding target cells, shifts radius and
angle preserving within-cell offsets, and preserves z/remission/packed labels.
It regenerates 7ch with the original priority-based label rule and 16ch with the
existing `generate_point_features.load_encoder` and `build_polar_point_features`.
It does not zero-fill missing PointNet features or invent random features.
Zeros outside occupied cells are the normal sparse-to-dense representation.

This is a **non-rigid polar cell-replacement prototype**, not physical object
insertion, a 3D visibility model, nor a recovery of current CP's raw state.
Even this closest reference changes absolute x/y and x_diff/y_diff. The PointNet
inputs include absolute x/51.2 and y/51.2, so copied features need not equal
features regenerated after movement. Array-wise roll/flip preserves correspondence
but is likewise not equivalent to raw-point rotation/reflection with recomputation.

Original regeneration max FP16 errors: hand7
{max(r['handcrafted_fp16']['max_absolute_error'] for r in originals):.8g}, PointNet16
{max(r['pointnet_fp16']['max_absolute_error'] for r in originals):.8g}. We retain exact
error counts and also test atol1e-5/rtol0.002 (approximately two FP16 relative ULPs).
The vectorized reference differs from the actual Polar4 projector by up to float32
mean-reduction rounding; its first-scan labels/mask agree and the original projector
matches the original cached7 bitwise after FP16 storage.

Outside-paste regenerated-label mismatches total
{sum(r['label_priority_tie_diagnostic']['outside_paste_label_mismatches'] for r in c['raw_alternative_comparisons'])}.
Of these, {sum(r['label_priority_tie_diagnostic']['outside_mismatches_with_multiple_top_priority_learning_ids'] for r in c['raw_alternative_comparisons'])}
are in cells with multiple learning IDs at equal highest priority. The legacy
priority sort is not stable: changing point count changes tie ordering even in
unmodified cells. This is **not** a normalization effect or evidence that feature
CP changed those cells. It is an additional raw-regeneration label-rule constraint.
NumPy/Torch occupancy disagreements are recorded separately and must not be hidden
by the production loader's 1% tolerance when designing strict raw CP.

The existing 64-column crop clips at the angular seam rather than wrapping.
Destination starts exclude the last possible column; objects can be cropped or
partially hidden by existing foreground. Temporal checks are not ego-motion
registered. No claim of object completeness or physical occlusion correctness.

## Mask And Loss Blocker

Dropout zeros 7ch,16ch,observation,paste mask but leaves nonzero labels.
`trainer_PointNet.py::_mix_losses` uses ignore label 0 for main/aux Focal and
Lovasz, not observation-mask exclusion; only boundary loss is mask-weighted.
This is inherited CP-OFF behavior, not changed in E.0. It must be explicitly
resolved or scientifically justified in a separate new E.1 pipeline, not silently
patched into the baseline. Empty normalized z channels have baseline value -0.2
before dropout, which is not stale PointNet data. max_z postprocessing reconstructs
`input[:,0]*5+1`; changing feature meaning would change postprocessing semantics.

## A / B / C Cost Comparison

Measured isolated audit values (not a training throughput prediction):

| Mode | Time | Consistency / Constraints |
| --- | --- | --- |
| A raw CP + regeneration | {v['timing']['online_raw_prototype_seconds_mean']} s/scan mean | Preferred; fixed encoder inference, raw geometry policy still needs production design |
| B precomputed augmented cache | {v['timing']['precomputed_read_seconds_mean']} s/read mean | One warm-cache example, same regenerated features; finite augmentation bank, storage cost |
| C feature-level CP | {v['timing']['feature_cp_call_seconds_mean']} s/call mean | Production function traced, success and skips; not point-level equivalent |

GPU: {v['gpu']}; CPU peak RSS: {v['timing']['cpu_peak_rss_bytes']} bytes.
A timing includes in-memory raw transform + vectorized 7ch + PointNet preprocessing,
forward and host transfers; excludes initial raw disk reads. C includes donor I/O
and trace overhead, excludes initial target load. B is not the production sparse
cache format. These timings have different scopes and cannot be treated as an
end-to-end training speed ranking. No segmentation forward/backward was benchmarked,
so full-model peak memory, worker/CUDA integration and epoch duration are unverified.

## E.1 Decision

Do **not** start formal E.1 training yet. Prefer A in a new isolated train-only
Dataset/Parser: rigid object transformation, explicit point-label selection,
occlusion/merge policy, periodic geometry, regenerated 7+16, fixed eval encoder,
cache provenance and a tested label/mask/loss contract. If runtime or full-model
memory is unsuitable, use B with immutable train-only variants and provenance.
C can only be a separately named feature augmentation ablation, with its geometry
limitations and the mask/loss issue explicitly addressed; it is not point-level PASS.

## Artifacts And Safety

- `copy_paste_statistics.csv`, `test_results.json`, `feature_consistency.json`.
- `donor_split_audit.json`, `donor_files.txt`, `read_data_sha256.json`.
- `before_after_examples/`: three full and three zoomed 12-panel diagnostics.
- `diagnostic_augmented_cache.npz`: one regenerated diagnostic example, never read by training.
- `execution_log.txt`, `verification.json`, `source_sha256.json`.
- `dataset_pipeline_audit.md`: source-backed pipeline, boundaries and implementation limitations.
- `memo/copy_paste_off_experiments_summary.md`: separate historical experiment review.

Source hashes unchanged: {v['source_hashes_unchanged']}; old log artifact size/mtime
unchanged: {v['existing_log_artifact_size_mtime_unchanged']}; encoder SHA256 unchanged:
{v['encoder_hash_unchanged']}. Python write guard limited all writes to this fresh
E.0 output directory. Read-data hashes are a post-read manifest, not a before/after
hash of the full dataset. Unit-test results are recorded separately.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="logs/SJNet3_PointNet_stepE0_copy_paste_integrity")
    parser.add_argument("--encoder", default="point_encoder_runs/best_point_encoder.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--min-candidates", type=int, default=100)
    parser.add_argument("--min-successes", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=2400)
    arguments = parser.parse_args()
    output_already_existed = Path(arguments.output).exists()
    try:
        run(arguments)
    except Exception:
        destination = Path(arguments.output)
        if not output_already_existed and destination.is_dir() and "stepE0" in destination.name and destination.parent.resolve() == Path.cwd() / "logs":
            with (destination / "execution_log.txt").open("a") as stream:
                stream.write(traceback.format_exc())
        raise
