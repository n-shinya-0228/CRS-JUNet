#!/usr/bin/env python3
"""Step D.1: fixed C.2 best/EMA weights, validation-only residual-alpha sweep."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml
from loguru import logger

from lib.dataset.Parser_PointNet import Parser
from lib.models.SJNet3_PointNet_Gated_Local3 import SJNet3_PointNet_Gated_Local3
from lib.models.SJNet3_PointNet_Gated_Local3_FixedAlpha import SJNet3_PointNet_Gated_Local3_FixedAlpha
from lib.utils.ioueval import iouEval

EXPECTED_MIOU = 0.508253
REPRODUCTION_TOLERANCE = 0.0001
TIE_TOLERANCE = 1e-12


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def normalize_state(state):
    if state and all(name.startswith("module.") for name in state):
        return {name.removeprefix("module."): value for name, value in state.items()}
    return state


def select_evaluation_state(checkpoint, arch, data):
    if checkpoint.get("experiment") != "Step C.2" or checkpoint.get("model_name") != "SJNet3_PointNet_Gated_Local3":
        raise ValueError("D.1 requires the Step C.2 Local3 checkpoint")
    if checkpoint.get("epoch") != checkpoint.get("best_val_epoch"):
        raise ValueError("The saved epoch must be the best validation epoch")
    if checkpoint.get("epoch") != 18:
        raise ValueError("D.1 expects the C.2 best checkpoint from epoch 18")
    if checkpoint.get("arch_cfg") != arch or checkpoint.get("data_cfg") != data:
        raise ValueError("Requested config/label mapping differs from the C.2 checkpoint")
    if not math.isfinite(checkpoint.get("best_val_iou", float("nan"))):
        raise ValueError("Missing/non-finite checkpoint best mIoU")
    if arch["train"].get("use_ema", True):
        if "model_ema" not in checkpoint or "model_train" not in checkpoint:
            raise ValueError("C.2 EMA best checkpoint must contain model_ema and model_train")
        state = normalize_state(checkpoint["model_ema"])
        saved = normalize_state(checkpoint["model"])
        if set(state) != set(saved) or any(not torch.equal(state[k], saved[k]) for k in state):
            raise ValueError("Best inference model does not match the saved EMA weights")
        return state, "EMA (model_ema; equal to best checkpoint model)"
    return normalize_state(checkpoint["model"]), "normal model (EMA disabled in C.2 config)"


def original_predictions(logits, features):
    # Identical height-based correction from C.2 Trainer.validate(), after argmax.
    predictions = logits.argmax(dim=1)
    max_z = features[:, 0] * 5.0 + 1.0
    predictions[(predictions == 7) & (max_z < -0.8)] = 2
    predictions[(predictions == 8) & (max_z < -0.8)] = 3
    return predictions


def reproduction_passed(result, checkpoint_miou):
    return (abs(result["miou"] - EXPECTED_MIOU) <= REPRODUCTION_TOLERANCE and
            abs(result["miou"] - checkpoint_miou) <= REPRODUCTION_TOLERANCE)


def best_alphas(values):
    best = max(values.values())
    winners = [alpha for alpha, value in values.items() if abs(value - best) <= TIE_TOLERANCE]
    status = "indistinguishable_all_zero" if all(value == 0 for value in values.values()) else (
        "tie" if len(winners) > 1 else "unique")
    return winners, best, status


def dataset_manifest(dataset, root):
    root = Path(root).resolve()
    entries, ids = [], []
    for bev in dataset.scan_files:
        files = []
        for path in (Path(bev), Path(dataset.get_pointnet_file(bev))):
            info = path.stat()
            files.append(dict(path=str(path.resolve().relative_to(root)),
                              size=info.st_size, mtime_ns=info.st_mtime_ns))
        entries.append(files)
        seq, _ = dataset.get_seq_frame(bev)
        ids.append(f"{seq}/{Path(bev).stem}.label")
    encoded = json.dumps(entries, sort_keys=True).encode()
    return entries, ids, hashlib.sha256(encoded).hexdigest()


@torch.no_grad()
def preflight(model, state, batch, device):
    features, mask = batch[0].to(device), batch[1].to(device)
    assert batch[14].count_nonzero().item() == 0
    x = torch.cat([features, mask.float()], dim=1)
    assert x.shape == (3, 24, 512, 512), tuple(x.shape)
    original = SJNet3_PointNet_Gated_Local3(in_channels=24, nclasses=20, drop=.3)
    original.load_state_dict(state, strict=True)
    original.requires_grad_(False).to(device).eval()
    reference_hash = state_digest(model.state_dict())
    observed = {}

    def residual(module, inputs, output):
        observed["base"], observed["residual"] = inputs[0].detach(), output.detach()

    def local(module, inputs, output):
        expected = observed["base"] + model.fixed_alpha * observed["residual"]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        observed["local"] = output.detach().clone()

    handles = [model.point_context.local.register_forward_hook(residual),
               model.point_context.register_forward_hook(local)]
    try:
        expected = original(x)
        model.set_alpha(1)
        actual = model(x)
        differences = {}
        for name in expected:
            if not torch.isfinite(actual[name]).all() or not torch.isfinite(expected[name]).all():
                raise RuntimeError(f"Non-finite preflight output: {name}")
            differences[name] = (actual[name] - expected[name]).abs().max().item()
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(observed["local"], observed["base"] + observed["residual"], rtol=0, atol=0)
        at_one = observed["local"]
        del expected, actual
        model.set_alpha(.4)
        changed = model(x)
        assert not torch.equal(at_one, observed["local"])
        assert all(torch.isfinite(value).all().item() for value in changed.values())
        del changed
        assert state_digest(model.state_dict()) == reference_hash
        assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
        return dict(status="PASS", shape=list(x.shape), state_dict_exact=True,
                    alpha_one_output_max_abs_diff=differences, alpha_changes_local=True,
                    all_outputs_finite=True, no_gradients=True)
    finally:
        for handle in handles:
            handle.remove()
        model.set_alpha(1)
        del original


def _progress_record(alpha, evaluator, next_batch, seen, forward_seconds, elapsed, peak, metadata_id):
    return dict(alpha=alpha, next_batch=next_batch, seen_scans=seen,
                confusion_matrix=evaluator.conf_matrix.cpu().tolist(),
                forward_cuda_seconds=forward_seconds, wall_seconds=elapsed,
                peak_allocated_GiB=peak, metadata_id=metadata_id)


@torch.no_grad()
def evaluate_alpha(model, loader, alpha, device, ignore, class_names, scan_ids, output, metadata_id, report_every):
    model.set_alpha(alpha)
    model.eval()
    expected_hash = state_digest(model.state_dict())
    evaluator = iouEval(len(class_names), device, ignore)
    path = output / f"progress_alpha_{alpha:g}.json"
    start_batch, seen, forward_seconds, prior_wall, peak = 0, 0, 0.0, 0.0, 0.0
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["metadata_id"] != metadata_id or saved["alpha"] != alpha:
            raise ValueError("Partial result belongs to a different evaluation")
        matrix = torch.tensor(saved["confusion_matrix"], dtype=torch.long, device=device)
        start_batch, seen = saved["next_batch"], saved["seen_scans"]
        if (matrix.shape != evaluator.conf_matrix.shape or (matrix < 0).any() or
                not 0 <= start_batch <= len(loader) or seen != min(start_batch * loader.batch_size, len(scan_ids))):
            raise ValueError("Invalid partial confusion matrix or data offset")
        evaluator.conf_matrix.copy_(matrix)
        forward_seconds, prior_wall = saved["forward_cuda_seconds"], saved["wall_seconds"]
        peak = saved["peak_allocated_GiB"]
        logger.info(f"Resuming alpha={alpha:g} after {start_batch} batches / {seen} scans")
    begin = time.perf_counter()
    next_batch = start_batch
    event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats(device)

    def persist():
        allocated = max(peak, torch.cuda.max_memory_allocated(device) / 2**30)
        atomic_json(path, _progress_record(alpha, evaluator, next_batch, seen, forward_seconds,
                    prior_wall + time.perf_counter() - begin, allocated, metadata_id))

    try:
        for index, batch in enumerate(loader):
            features, mask, labels = batch[:3]
            ids = [f"{seq}/{name}" for seq, name in zip(batch[4], batch[5])]
            offset = index * loader.batch_size
            if ids != scan_ids[offset:offset + len(ids)]:
                raise RuntimeError("Validation scan ordering differs from the recorded manifest")
            if index < start_batch:
                continue
            if batch[14].count_nonzero().item() != 0:
                raise RuntimeError("Copy-Paste OFF requires an empty paste mask")
            features, mask = features.to(device), mask.to(device)
            labels = labels.to(device, non_blocking=True).long()
            mask = mask.unsqueeze(1) if mask.ndim == 3 else mask
            x = torch.cat([features, mask.float()], dim=1)
            if x.shape[1] != 24:
                raise RuntimeError("Expected 23 features followed by one observation mask")
            event_start.record()
            outputs = model(x)
            event_end.record()
            event_end.synchronize()
            forward_seconds += event_start.elapsed_time(event_end) / 1000
            if not torch.isfinite(outputs["logits"]).all():
                raise RuntimeError(f"Non-finite logits at alpha={alpha}, batch={index}")
            predictions = original_predictions(outputs["logits"], features)
            evaluator.addBatch(predictions, labels)
            seen += features.shape[0]
            next_batch = index + 1
            del outputs, predictions, x, features, mask, labels
            if next_batch % report_every == 0 or next_batch == len(loader):
                persist()
                partial, _ = evaluator.getIoU()
                logger.info(f"alpha={alpha:g} | batches={next_batch}/{len(loader)} | scans={seen}/{len(scan_ids)} "
                            f"| partial mIoU={partial.item():.6f} | seconds={prior_wall + time.perf_counter() - begin:.1f}")
        if seen != len(scan_ids):
            raise RuntimeError("Evaluation did not cover the entire validation set")
        if state_digest(model.state_dict()) != expected_hash:
            raise RuntimeError("Model state changed during evaluation")
        miou, per_class = evaluator.getIoU()
        accuracy = evaluator.getacc()
        result = dict(alpha=alpha, miou=miou.item(), accuracy=accuracy.item(),
                      per_class_iou={class_names[cid]: per_class[cid].item() for cid in range(len(class_names)) if cid not in ignore},
                      confusion_matrix=evaluator.conf_matrix.cpu().tolist(),
                      scans=seen, batches=next_batch, model_state_sha256=expected_hash,
                      wall_seconds=prior_wall + time.perf_counter() - begin, forward_cuda_seconds=forward_seconds,
                      peak_allocated_GiB=max(peak, torch.cuda.max_memory_allocated(device) / 2**30))
        if not all(math.isfinite(value) for value in [result["miou"], result["accuracy"], *result["per_class_iou"].values()]):
            raise RuntimeError("Non-finite evaluation metrics")
        persist()
        return result
    except BaseException:
        persist()
        raise


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_results(output, run):
    results = run["results"]
    if "1.0" not in results:
        return
    reference = results["1.0"]
    alphas = [alpha for alpha in run["metadata"]["alphas"] if str(alpha) in results]
    class_names = list(reference["per_class_iou"])
    columns = [f"alpha_{alpha:g}" for alpha in alphas]
    write_csv(output / "results.csv", ["alpha", "miou", "accuracy", "delta_miou", "wall_seconds", "forward_cuda_seconds", "peak_allocated_GiB"],
              [dict(alpha=alpha, **{key: results[str(alpha)][key] for key in
                    ("miou", "accuracy", "wall_seconds", "forward_cuda_seconds", "peak_allocated_GiB")},
                    delta_miou=results[str(alpha)]["miou"] - reference["miou"]) for alpha in alphas])
    rows, deltas, winners = [], [], []
    table = ["| Class | " + " | ".join(f"alpha={alpha:g}" for alpha in alphas) + " | Best alpha |",
             "|---|" + "---|" * (len(alphas) + 1)]
    for name in [*class_names, "mIoU"]:
        values = {alpha: results[str(alpha)]["miou"] if name == "mIoU" else results[str(alpha)]["per_class_iou"][name] for alpha in alphas}
        best, score, status = best_alphas(values)
        label = ", ".join(f"{alpha:g}" for alpha in best)
        if status == "indistinguishable_all_zero":
            label = "indistinguishable (all zero)"
        elif status == "tie":
            label += " (tie)"
        table.append("| " + name + " | " + " | ".join(f"{values[alpha]:.6f}" for alpha in alphas) + " | " + label + " |")
        if name != "mIoU":
            rows.append(dict(Class=name, **{column: values[alpha] for column, alpha in zip(columns, alphas)}))
            deltas.append(dict(Class=name, **{column: values[alpha] - reference["per_class_iou"][name] for column, alpha in zip(columns, alphas)}))
            winners.append(dict(Class=name, best_alphas=";".join(map(str, best)), best_iou=score, status=status,
                                delta_from_alpha_one=score - reference["per_class_iou"][name]))
    write_csv(output / "per_class_iou.csv", ["Class", *columns], rows)
    write_csv(output / "per_class_delta.csv", ["Class", *columns], deltas)
    write_csv(output / "best_alpha_by_class.csv", ["Class", "best_alphas", "best_iou", "status", "delta_from_alpha_one"], winners)
    lines = ["# Step D.1: Fixed-alpha Sensitivity", "", f"Status: {run['status']}", "",
             f"Checkpoint: {run['metadata']['checkpoint']} (epoch 18).", f"Weights: {run['metadata']['weight_source']}.",
             f"Alpha=1 mIoU: {reference['miou']:.9f}; original: {EXPECTED_MIOU:.6f}; difference: {reference['miou'] - EXPECTED_MIOU:+.9f}.",
             f"Reproduction gate: {run.get('baseline_passed', False)} (tolerance {REPRODUCTION_TOLERANCE}).", "", *table, "",
             "## Detailed Class Changes", "", "| Class | Best tested alpha | Best delta from alpha=1 |", "|---|---|---|"]
    focus = ["bicycle", "motorcycle", "person", "bicyclist", "pole", "traffic-sign", "car", "truck", "other-vehicle", "road", "parking", "sidewalk", "terrain", "vegetation"]
    by_name = {row["Class"]: row for row in winners}
    for name in focus:
        row = by_name[name]
        lines.append(f"| {name} | {row['best_alphas']} | {row['delta_from_alpha_one']:+.6f} |")
    lower, higher = [], []
    for name in class_names:
        base = reference["per_class_iou"][name]
        low = max((results[str(a)]["per_class_iou"][name] - base for a in alphas if a < 1), default=0)
        high = max((results[str(a)]["per_class_iou"][name] - base for a in alphas if a > 1), default=0)
        if low > TIE_TOLERANCE:
            lower.append(f"{name} ({low:+.6f})")
        if high > TIE_TOLERANCE:
            higher.append(f"{name} ({high:+.6f})")
    overall, score, status = best_alphas({a: results[str(a)]["miou"] for a in alphas})
    lines += ["", "## Observations", "", f"Highest mIoU among completed conditions: {score:.6f}, alpha={overall} ({status}).",
              "Classes improved by at least one lower alpha: " + (", ".join(lower) or "none") + ".",
              "Classes improved by at least one higher alpha: " + (", ".join(higher) or "none") + ".",
              "", "These observations concern one global fixed alpha at a time. Class-specific preferred values do not establish that combining them into a spatial map will improve performance.",
              "Alpha is selected on this validation split, so the selected score includes selection bias. Use separate selection/final-evaluation data for D.2 when possible.",
              "No ground truth is used to assign inference alpha. No retraining, optimizer, backward or weight updates occur.",
              "Ties use an absolute IoU tolerance of 1e-12; an all-zero class has no distinguishable best alpha.",
              "", "## Timing", "", "| alpha | Evaluation wall seconds | CUDA forward seconds |", "|---|---|---|"]
    for alpha in alphas:
        result = results[str(alpha)]
        lines.append(f"| {alpha:g} | {result['wall_seconds']:.2f} | {result['forward_cuda_seconds']:.2f} |")
    lines += ["", "Wall time includes data loading, metrics, state checks and partial-result saves; CUDA forward time covers model inference only.",
              "Validation uses the original accumulated confusion-matrix implementation, excludes class 0 and preserves the C.2 height-based prediction correction.",
              "Input remains 23 feature channels plus observation mask, batch size 3, FP32/no autocast, eval/no_grad.",
              "Results saved after each condition; resume verifies checkpoint/config/data identity before using completed or partial results."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    if len(alphas) == len(run["metadata"]["alphas"]):
        plot_results(output, results, alphas, class_names)


def plot_results(output, results, alphas, class_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    with plt.rc_context({"font.size": 10}):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(alphas, [results[str(a)]["miou"] for a in alphas], marker="o")
        ax.set(xlabel="Fixed alpha", ylabel="Validation mIoU", xticks=alphas)
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(output / "miou_vs_alpha.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name in ("bicycle", "motorcycle", "person", "truck", "traffic-sign"):
            ax.plot(alphas, [results[str(a)]["per_class_iou"][name] for a in alphas], marker="o", label=name)
        ax.set(xlabel="Fixed alpha", ylabel="Class IoU", xticks=alphas)
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(output / "selected_class_iou_vs_alpha.png", dpi=180)
        plt.close(fig)
        delta = np.array([[results[str(a)]["per_class_iou"][name] - results["1.0"]["per_class_iou"][name] for a in alphas] for name in class_names])
        limit = max(float(np.abs(delta).max()), 1e-6)
        fig, ax = plt.subplots(figsize=(8, 9))
        im = ax.imshow(delta, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
        ax.set_xticks(range(len(alphas)), [f"{a:g}" for a in alphas])
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("Fixed alpha (reference = 1.0)")
        for row in range(len(class_names)):
            for column in range(len(alphas)):
                ax.text(column, row, f"{delta[row, column]:+.3f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(delta[row, column]) > limit * .55 else "black")
        fig.colorbar(im, ax=ax, label="Class IoU difference")
        fig.tight_layout()
        fig.savefig(output / "per_class_delta_heatmap.png", dpi=180)
        plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", required=True)
    cli.add_argument("--arch_cfg", default="config/arch/sjnet_pointnet_gated_local3_stablelr.yaml")
    cli.add_argument("--data_cfg", default="config/labels/semantic-kitti.yaml")
    cli.add_argument("--checkpoint", default="logs/SJNet3_PointNet_stepC2_local3_gated_lr2e3_bs3/best_val.path")
    cli.add_argument("--alphas", nargs="+", type=float, default=[.4, .6, .8, 1., 1.2])
    cli.add_argument("--output", default="logs/SJNet3_PointNet_stepD1_fixed_alpha_sensitivity")
    cli.add_argument("--resume", action="store_true")
    cli.add_argument("--report-every", type=int, default=100)
    cli.add_argument("--preflight-only", action="store_true", help="Run the real-data preflight only, never the full sweep")
    args = cli.parse_args()
    if (not args.alphas or 1.0 not in args.alphas or len(set(args.alphas)) != len(args.alphas)
            or not all(math.isfinite(alpha) for alpha in args.alphas) or args.report_every < 1):
        cli.error("Use unique finite alphas including 1.0, and report-every >= 1")
    if Path(args.checkpoint).name != "best_val.path":
        cli.error("D.1 requires best_val.path, never latest.path")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce C.2 inference conditions")
    arch = yaml.safe_load(Path(args.arch_cfg).read_text())
    data = yaml.safe_load(Path(args.data_cfg).read_text())
    if arch["model"]["name"] != "SJNet3_PointNet_Gated_Local3" or arch["train"]["batch_size"] != 3:
        raise ValueError("D.1 requires the original C.2 model/config and batch size 3")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    started = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state, source = select_evaluation_state(checkpoint, arch, data)
    expected_hash = state_digest(state)
    checkpoint_score = float(checkpoint["best_val_iou"])
    parser = Parser(root=args.dataset, data_cfg=data, arch_cfg=arch, gt=True, shuffle_train=False)
    loader = parser.get_valid_set()
    if not len(loader) or parser.valid_dataset.is_train or parser.valid_dataset.copy_paste_prob != 0:
        raise ValueError("Require a nonempty, augmentation/Copy-Paste OFF validation dataset")
    entries, scan_ids, manifest_hash = dataset_manifest(parser.valid_dataset, args.dataset)
    classes = {cid: parser.get_xentropy_class_string(cid) for cid in range(parser.get_n_classes())}
    ignore = [cid for cid, ignored in data["learning_ignore"].items() if ignored]
    if parser.get_n_classes() != 20 or ignore != [0]:
        raise ValueError("D.1 expects 19 evaluated classes, ignoring unlabeled class 0")
    metadata = dict(
        experiment="Step D.1", dataset_root=str(Path(args.dataset).resolve()),
        checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=file_sha256(args.checkpoint),
        checkpoint_epoch=checkpoint["epoch"], checkpoint_best_miou=checkpoint_score, weight_source=source,
        model_state_sha256=expected_hash, arch_cfg=arch, data_cfg=data,
        arch_cfg_sha256=file_sha256(args.arch_cfg), data_cfg_sha256=file_sha256(args.data_cfg),
        validation_sequences=data["split"]["valid"], validation_scans=len(scan_ids), validation_batches=len(loader),
        validation_manifest_sha256=manifest_hash, alphas=args.alphas,
        batch_size=loader.batch_size, workers=loader.num_workers, pin_memory=loader.pin_memory,
        persistent_workers=loader.persistent_workers, prefetch_factor=loader.prefetch_factor,
        shuffle=False, drop_last=loader.drop_last, copy_paste_prob=0, amp=False, dtype="float32",
        metrics="original iouEval; accumulated confusion matrix; class 0 ignored",
        postprocess="C.2 max_z<-0.8: prediction 7->2, 8->3; no additional mask/KNN",
        torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(),
        inference_only=True, optimizer_created=False, backward=False,
        evaluation_script_sha256=file_sha256(__file__),
        fixed_alpha_model_sha256=file_sha256("lib/models/SJNet3_PointNet_Gated_Local3_FixedAlpha.py"),
    )
    metadata_id = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    output = Path(args.output)
    if output.exists():
        if not args.resume:
            raise ValueError(f"Output already exists: {output}; use --resume for this same experiment")
        run = json.loads((output / "results.json").read_text())
        if run["metadata_id"] != metadata_id:
            raise ValueError("Resume checkpoint/config/dataset/environment identity differs")
    else:
        output.mkdir(parents=True)
        run = dict(metadata=metadata, metadata_id=metadata_id, status="preflight", results={}, baseline_passed=False)
        atomic_json(output / "results.json", run)
        atomic_json(output / "validation_manifest.json", dict(entries=entries, scan_ids=scan_ids))
        (output / "arch_cfg.yaml").write_text(yaml.safe_dump(arch, sort_keys=False))
        (output / "data_cfg.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    logger.add(output / "log.txt")
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"Checkpoint={args.checkpoint} | epoch=18 | weights={source} | original best mIoU={checkpoint_score:.9f}")
    raw = normalize_state(checkpoint.get("model_train", checkpoint["model"]))
    logger.info(f"EMA/raw tensor differences: {sum(not torch.equal(raw[k], state[k]) for k in state)}")
    logger.info(f"Validation: sequences={data['split']['valid']} | scans={len(scan_ids)} | batches={len(loader)} | batch=3 | AMP=False")
    model = SJNet3_PointNet_Gated_Local3_FixedAlpha(in_channels=24, nclasses=parser.get_n_classes(), drop=arch["model"]["dropout"])
    model.load_state_dict(state, strict=True)
    if state_digest(model.state_dict()) != expected_hash:
        raise RuntimeError("Strictly loaded common weights differ from the C.2 checkpoint")
    device = torch.device("cuda")
    model.requires_grad_(False).to(device).eval()
    try:
        if "preflight" not in run:
            batch = next(iter(loader))
            run["preflight"] = preflight(model, state, batch, device)
            run["preflight"]["peak_allocated_GiB"] = torch.cuda.max_memory_allocated(device) / 2**30
            atomic_json(output / "results.json", run)
            logger.info(f"Preflight: {run['preflight']}")
            del batch
        del state, raw, checkpoint
        torch.cuda.empty_cache()
        if args.preflight_only:
            logger.info("Preflight complete; full evaluation has not started")
            return
        for alpha in [1.0, *[value for value in args.alphas if value != 1.0]]:
            key = str(alpha)
            if key in run["results"]:
                result = run["results"][key]
                if result["model_state_sha256"] != expected_hash or result["scans"] != len(scan_ids):
                    raise ValueError("Completed result does not match this model/validation set")
                logger.info(f"Using completed alpha={alpha:g}: mIoU={result['miou']:.9f}")
            else:
                run.update(status="evaluating", current_alpha=alpha)
                atomic_json(output / "results.json", run)
                logger.info(f"Starting full validation: alpha={alpha:g}")
                result = evaluate_alpha(model, loader, alpha, device, ignore, classes, scan_ids, output, metadata_id, args.report_every)
                run["results"][key] = result
            if alpha == 1:
                run["baseline_passed"] = reproduction_passed(result, checkpoint_score)
                if not run["baseline_passed"]:
                    run["status"] = "reproduction_failed"
                    atomic_json(output / "results.json", run)
                    export_results(output, run)
                    raise RuntimeError(f"Alpha=1 reproduction failed: {result['miou']:.9f}; remaining alphas NOT evaluated")
                logger.info(f"Alpha=1 reproduction PASS: mIoU={result['miou']:.9f}, diff={result['miou']-EXPECTED_MIOU:+.9f}")
            atomic_json(output / "results.json", run)
            export_results(output, run)
            logger.info(f"Completed alpha={alpha:g} | mIoU={result['miou']:.9f} | acc={result['accuracy']:.9f} | wall={result['wall_seconds']:.1f}s")
        if file_sha256(args.checkpoint) != metadata["checkpoint_sha256"]:
            raise RuntimeError("Source checkpoint changed during evaluation")
        run.update(status="complete", current_alpha=None, invocation_seconds=time.perf_counter() - started)
        atomic_json(output / "results.json", run)
        export_results(output, run)
        logger.info(f"D.1 complete: {output}")
    except BaseException as error:
        if run["status"] != "reproduction_failed":
            run["status"] = "interrupted"
        run["last_error"] = f"{type(error).__name__}: {error}"
        atomic_json(output / "results.json", run)
        export_results(output, run)
        logger.exception("D.1 stopped; completed/partial results retained for --resume")
        raise


if __name__ == "__main__":
    main()
