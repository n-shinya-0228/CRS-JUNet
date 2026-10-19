#!/usr/bin/env python3
"""Step D.2: frozen C.2 EMA, shared-stage1 Baseline/Hard/Soft inference only."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml
from loguru import logger

import evaluate_pointnet_fixed_alpha_sensitivity as d1
import evaluate_pointnet_high_alpha_sensitivity as d1b
from lib.models.SJNet3_PointNet_Gated_Local3_ClassAlpha import (
    ALPHA_TABLE, CLASS_NAMES, SJNet3_PointNet_Gated_Local3_ClassAlpha, build_alpha_maps,
)
from lib.utils.pointnet_class_alpha_reporting import export_results
from lib.utils.pointnet_class_alpha_statistics import AlphaStatistics, HIST_WIDTH, MODES

ROOT = Path(__file__).resolve().parent
REFERENCE_COMMIT = "396385a5051d3358b9d4c471268ca8be4e09ff3d"


def validate_alpha_table(source, classes):
    if set(classes) != set(range(20)) or tuple(classes[cid] for cid in range(20)) != CLASS_NAMES:
        raise ValueError("Current learning IDs do not match the fixed D.2 table")
    with (source / "best_alpha_by_class.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    by_name = {row["Class"]: row for row in rows}
    if len(rows) != 19 or set(by_name) != set(CLASS_NAMES[1:]):
        raise ValueError("D.1b best-alpha CSV class names/coverage differ")
    for cid, name in enumerate(CLASS_NAMES[1:], 1):
        row = by_name[name]
        if cid == 8:
            if row["best_alphas"] != "N/A" or float(row["best_iou"]) != 0:
                raise ValueError("Motorcyclist must use the specified all-zero fallback")
        elif row["status"] != "unique" or float(row["best_alphas"]) != ALPHA_TABLE[cid]:
            raise ValueError("D.2 fixed lookup disagrees with D.1b: " + name)


def capture_protected(source):
    paths = {p for folder in ("lib", "config") for p in (ROOT/folder).rglob("*") if p.suffix in (".py", ".yaml")}
    for directory in (source, ROOT/"logs/SJNet3_PointNet_stepD1_fixed_alpha_sensitivity"):
        paths.update(p for p in directory.rglob("*") if p.is_file())
    paths.update((ROOT/"evaluate_pointnet_fixed_alpha_sensitivity.py", ROOT/"evaluate_pointnet_high_alpha_sensitivity.py"))
    return {str(p.resolve()): d1.file_sha256(p) for p in sorted(paths)}


@torch.no_grad()
def smoke_test(model, state, batch, device):
    checks = d1.preflight(model, state, batch, device)
    model.set_alpha(1.2)
    x = torch.cat((batch[0].to(device), batch[1].to(device).float()), dim=1)
    reference = d1.SJNet3_PointNet_Gated_Local3_FixedAlpha(in_channels=24, nclasses=20, drop=.3, alpha=1.2)
    reference.load_state_dict(state, strict=True)
    reference.requires_grad_(False).to(device).eval()
    expected = reference(x)
    actual = model(x, alpha_map=x.new_full((3, 1, 512, 512), 1.2))
    differences = {}
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], atol=1e-5, rtol=1e-4)
        differences[name] = (expected[name]-actual[name]).abs().max().item()
    del expected, actual, reference
    stage1 = model(x)
    assert stage1["logits"].shape == (3, 20, 512, 512)
    hard, soft = build_alpha_maps(stage1["logits"], x[:, 23:24])
    checks.update(constant_alpha12_all_output_max_abs_diff=differences, same_raw_stage1_logits=True,
                  input_shape=list(x.shape), stage1_shape=list(stage1["logits"].shape),
                  hard_shape=list(hard.shape), soft_shape=list(soft.shape), ground_truth_used_for_maps=False)
    del stage1
    for name, alpha in (("Hard", hard), ("Soft", soft)):
        assert alpha.shape == (3, 1, 512, 512)
        assert torch.isfinite(alpha).all() and alpha.min() >= .6-1e-6 and alpha.max() <= 1.6+1e-6
        torch.testing.assert_close(alpha[x[:, 23:24] <= 0], torch.full_like(alpha[x[:, 23:24] <= 0], 1.2), atol=0, rtol=0)
        output = model(x, alpha_map=alpha)
        assert all(torch.isfinite(value).all().item() for value in output.values())
        assert model.point_context._alpha_map is None
        checks[name+"_range"] = [alpha.min().item(), alpha.max().item()]
        del output
    assert d1.state_digest(model.state_dict()) == d1.state_digest(state)
    assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
    checks.update(status="PASS", strict_state_hash=True, no_alpha_parameters_or_buffers=True,
                  peak_allocated_GiB=torch.cuda.max_memory_allocated(device)/2**30)
    return checks


@torch.no_grad()
def evaluate_three_modes(model, loader, device, classes, scan_ids, output, metadata_id, report_every=50):
    model.eval()
    model.set_alpha(1.2)
    initial_hash = d1.state_digest(model.state_dict())
    evaluators = {mode:d1.iouEval(20, device, [0]) for mode in MODES}
    scratch = {mode:d1.iouEval(20, device, [0]) for mode in MODES}
    statistics = AlphaStatistics()
    path = output / "progress_d2.json"
    next_batch, seen, prior_wall, peak = 0, 0, 0., 0.
    pixels_per_scan = None
    timings = dict.fromkeys(MODES, 0.)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["metadata_id"] != metadata_id:
            raise ValueError("D.2 progress identity mismatch")
        next_batch, seen = saved["next_batch"], saved["seen_scans"]
        if not 0 <= next_batch <= len(loader) or seen != min(next_batch*loader.batch_size, len(scan_ids)):
            raise ValueError("Invalid D.2 progress scan offset")
        statistics = AlphaStatistics.restore(saved["statistics"])
        pixels_per_scan = saved["pixels_per_scan"]
        pixels = int(statistics.count[0]+statistics.unobserved)
        if pixels_per_scan is not None and pixels != seen*pixels_per_scan:
            raise ValueError("Saved alpha statistics do not match completed scan coverage")
        for mode, evaluator in evaluators.items():
            matrix = torch.tensor(saved["confusion_matrices"][mode], dtype=torch.long, device=device)
            if matrix.shape != (20, 20) or (matrix < 0).any() or matrix.sum().item() != pixels:
                raise ValueError("Invalid saved D.2 confusion matrix/counts")
            evaluator.conf_matrix.copy_(matrix)
        prior_wall, peak, timings = saved["wall_seconds"], saved["peak_allocated_GiB"], saved["forward_seconds"]
        logger.info(f"Resuming D.2 at batch={next_batch}, scans={seen}; all three modes committed together")
    start_batch = next_batch
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def persist():
        d1.atomic_json(path, dict(metadata_id=metadata_id, next_batch=next_batch, seen_scans=seen, pixels_per_scan=pixels_per_scan,
            confusion_matrices={mode:e.conf_matrix.cpu().tolist() for mode, e in evaluators.items()},
            statistics=statistics.snapshot(), wall_seconds=prior_wall+time.perf_counter()-started,
            peak_allocated_GiB=max(peak, torch.cuda.max_memory_allocated(device)/2**30), forward_seconds=timings))

    def forward(x, alpha_map=None):
        event_start.record()
        outputs = model(x, alpha_map=alpha_map)
        event_end.record()
        event_end.synchronize()
        elapsed = event_start.elapsed_time(event_end)/1000
        if not all(torch.isfinite(value).all().item() for value in outputs.values()):
            raise RuntimeError("Non-finite D.2 forward output")
        if model.point_context._alpha_map is not None:
            raise RuntimeError("External alpha map leaked beyond forward")
        return outputs, elapsed

    try:
        for index, batch in enumerate(loader):
            batch_ids = [f"{seq}/{name}" for seq, name in zip(batch[4], batch[5])]
            offset = index*loader.batch_size
            if batch_ids != scan_ids[offset:offset+len(batch_ids)]:
                raise ValueError("D.2 scan order differs from the D.1b manifest")
            if index < start_batch:
                continue
            if batch[14].count_nonzero().item():
                raise ValueError("Copy-Paste must remain OFF")
            features, mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device).long()
            actual_pixels = int(labels.shape[-2]*labels.shape[-1])
            if pixels_per_scan is not None and pixels_per_scan != actual_pixels:
                raise ValueError("Validation image size changed between batches")
            pixels_per_scan = actual_pixels
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            x = torch.cat((features, mask.float()), dim=1)
            if x.shape[1] != 24 or x.dtype != torch.float32:
                raise ValueError("Require 23 features + mask, float32")
            stage1, stage1_seconds = forward(x)
            hard, soft = build_alpha_maps(stage1["logits"], mask)
            observed = mask[:, 0] > 0
            observed_cpu = observed.cpu().numpy()
            target_cpu = batch[2].numpy()[observed_cpu]
            raw_prediction_cpu = stage1["logits"].argmax(1)[observed].cpu().numpy()
            final_predictions = []
            pending_timings = {"Baseline":stage1_seconds}
            for mode, alpha in zip(MODES, (None, hard, soft)):
                if mode == "Baseline":
                    outputs = stage1
                else:
                    outputs, pending_timings[mode] = forward(x, alpha)
                predictions = d1.original_predictions(outputs["logits"], features)
                scratch[mode].reset()
                scratch[mode].addBatch(predictions, labels)
                final_predictions.append(predictions[observed].cpu().numpy())
                del outputs, predictions
                if mode == "Baseline":
                    del stage1
            # Build the complete batch delta before touching any committed state.
            pending = AlphaStatistics()
            pending.update(hard[:, 0][observed].cpu().numpy(), soft[:, 0][observed].cpu().numpy(),
                           target_cpu, raw_prediction_cpu, final_predictions,
                           int(observed.numel()-observed.count_nonzero().item()))
            for example_index in (0, len(scan_ids)//2, len(scan_ids)-1):
                local = example_index-offset
                if 0 <= local < len(batch_ids):
                    example = output/f"alpha_example_{example_index:06d}.npz"
                    temporary = example.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, scan_id=batch_ids[local],
                        mask=observed_cpu[local], hard=hard[local, 0].cpu().numpy(), soft=soft[local, 0].cpu().numpy())
                    os.replace(temporary, example)
            for mode in MODES:
                evaluators[mode].conf_matrix.add_(scratch[mode].conf_matrix)
                timings[mode] += pending_timings[mode]
            statistics.merge(pending)
            seen += len(batch_ids)
            next_batch = index+1
            del x, features, mask, labels, hard, soft, observed, pending
            if next_batch % report_every == 0 or next_batch == len(loader):
                persist()
                scores = {mode:round(e.getIoU()[0].item(), 6) for mode, e in evaluators.items()}
                logger.info(f"D.2 batches={next_batch}/{len(loader)} | scans={seen}/{len(scan_ids)} | partial={scores} | seconds={prior_wall+time.perf_counter()-started:.1f}")
        if seen != len(scan_ids) or d1.state_digest(model.state_dict()) != initial_hash:
            raise RuntimeError("Incomplete D.2 validation or changed model weights")
        results = {}
        for mode, evaluator in evaluators.items():
            miou, per_class = evaluator.getIoU()
            results[mode] = dict(miou=miou.item(), accuracy=evaluator.getacc().item(),
                per_class_iou={classes[cid]:per_class[cid].item() for cid in range(1, 20)},
                confusion_matrix=evaluator.conf_matrix.cpu().tolist(), scans=seen, batches=next_batch,
                model_state_sha256=initial_hash)
        persist()
        return dict(results=results, forward_seconds=timings, evaluation_wall_seconds=prior_wall+time.perf_counter()-started,
                    peak_allocated_GiB=max(peak, torch.cuda.max_memory_allocated(device)/2**30)), statistics
    except BaseException:
        persist()
        raise


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", required=True)
    cli.add_argument("--arch_cfg", default="config/arch/sjnet_pointnet_gated_local3_stablelr.yaml")
    cli.add_argument("--data_cfg", default="config/labels/semantic-kitti.yaml")
    cli.add_argument("--checkpoint", default="logs/SJNet3_PointNet_stepC2_local3_gated_lr2e3_bs3/best_val.path")
    cli.add_argument("--source", default="logs/SJNet3_PointNet_stepD1b_high_alpha_sensitivity")
    cli.add_argument("--output", default="logs/SJNet3_PointNet_stepD2_class_fixed_alpha")
    cli.add_argument("--report-every", type=int, default=50)
    cli.add_argument("--resume", action="store_true")
    cli.add_argument("--preflight-only", action="store_true")
    args = cli.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    protected_dirs = (source, ROOT/"logs/SJNet3_PointNet_stepD1_fixed_alpha_sensitivity", Path(args.checkpoint).resolve().parent)
    if any(output == p or p in output.parents or output in p.parents for p in protected_dirs):
        cli.error("D.2 output must not overlap original experiment directories")
    if output.exists() and not args.resume:
        cli.error("Output exists; use --resume only for the identical D.2 experiment")
    if args.report_every < 1 or Path(args.checkpoint).name != "best_val.path":
        cli.error("Require report-every >= 1 and C.2 best_val.path")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for D.1b-compatible inference")
    started = time.perf_counter()
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    arch, data = [yaml.safe_load(Path(p).read_text()) for p in (args.arch_cfg, args.data_cfg)]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state, weight_source = d1.select_evaluation_state(checkpoint, arch, data)
    if not weight_source.startswith("EMA"):
        raise ValueError("Require C.2 EMA weights")
    parser = d1.Parser(root=args.dataset, data_cfg=data, arch_cfg=arch, gt=True, shuffle_train=False)
    loader = parser.get_valid_set()
    entries, scan_ids, manifest_hash = d1.dataset_manifest(parser.valid_dataset, args.dataset)
    classes = {cid:parser.get_xentropy_class_string(cid) for cid in range(parser.get_n_classes())}
    ignore = [cid for cid, ignored in data["learning_ignore"].items() if ignored]
    if (len(scan_ids) != 4071 or len(loader) != 1357 or loader.batch_size != 3 or
            data["split"]["valid"] != [8] or ignore != [0] or parser.valid_dataset.is_train or parser.valid_dataset.copy_paste_prob != 0):
        raise ValueError("Require original sequence 08, 4071 scans, batch 3, ignore 0, no augmentation/Copy-Paste")
    metadata = dict(experiment="Step D.2", dataset_root=str(Path(args.dataset).resolve()),
        checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=d1.file_sha256(args.checkpoint),
        checkpoint_epoch=checkpoint["epoch"], checkpoint_best_miou=checkpoint["best_val_iou"], weight_source=weight_source,
        model_state_sha256=d1.state_digest(state), arch_cfg=arch, data_cfg=data,
        arch_cfg_sha256=d1.file_sha256(args.arch_cfg), data_cfg_sha256=d1.file_sha256(args.data_cfg),
        validation_sequences=data["split"]["valid"], validation_scans=len(scan_ids), validation_batches=len(loader),
        validation_manifest_sha256=manifest_hash, alphas=[1.2], batch_size=loader.batch_size,
        workers=loader.num_workers, pin_memory=loader.pin_memory, persistent_workers=loader.persistent_workers,
        prefetch_factor=loader.prefetch_factor, shuffle=False, drop_last=loader.drop_last, copy_paste_prob=0,
        amp=False, dtype="float32", metrics="original iouEval; accumulated confusion matrix; class 0 ignored",
        postprocess="C.2 max_z<-0.8: prediction 7->2, 8->3; no additional mask/KNN",
        torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(), inference_only=True,
        optimizer_created=False, backward=False, evaluation_script_sha256=d1.file_sha256(d1.__file__),
        fixed_alpha_model_sha256=d1.file_sha256(ROOT/"lib/models/SJNet3_PointNet_Gated_Local3_FixedAlpha.py"))
    old = json.loads((ROOT/"logs/SJNet3_PointNet_stepD1_fixed_alpha_sensitivity/results.json").read_text())
    imported = d1b.validate_source(old, metadata,
        json.loads((source/"validation_manifest.json").read_text()), entries, scan_ids, classes)
    previous = json.loads((source/"results.json").read_text())
    if previous["status"] != "complete" or any(previous["results"].get(k) != value for k, value in imported.items()):
        raise ValueError("D.1b is incomplete or does not exactly preserve D.1 references")
    for key in old["metadata"]:
        if key not in ("experiment", "alphas") and d1b.json_normalized(previous["metadata"][key]) != d1b.json_normalized(metadata[key]):
            raise ValueError("D.1b compatibility mismatch: " + key)
    for alpha in d1b.ALL_ALPHAS:
        d1b.validate_result(previous["results"][str(alpha)], alpha, metadata, classes)
    validate_alpha_table(source, classes)
    protected = capture_protected(source)
    protected[str(Path(args.checkpoint).resolve())] = metadata["checkpoint_sha256"]
    # Historical inference dependencies present in this checkout must be exact.
    for path, expected in previous["metadata"]["protected_files"].items():
        relative = Path(path).relative_to(ROOT)
        if relative.parts[0] == "lib" and Path(path).exists() and d1.file_sha256(path) != expected:
            raise ValueError("Historical inference dependency differs: " + path)
    metadata.update(alpha_table=list(ALPHA_TABLE), class_names=list(CLASS_NAMES), modes=list(MODES),
        stage1_alpha=1.2, temperature=1., maps_from_raw_logits=True, ground_truth_for_assignment=False,
        unobserved_alpha=1.2, soft_quantile_histogram_width=HIST_WIDTH,
        exploratory_development_evaluation=True, alpha_table_retuned=False,
        source_results_sha256=d1.file_sha256(source/"results.json"), source_dir=str(source),
        source_csv_sha256=d1.file_sha256(source/"best_alpha_by_class.csv"),
        script_sha256=d1.file_sha256(__file__), protected_files=protected,
        recovered_reference_commit=REFERENCE_COMMIT, cudnn_deterministic=True, cudnn_benchmark=True, seed=0)
    metadata = d1b.json_normalized(metadata)
    metadata_id = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    if output.exists():
        run = json.loads((output/"results.json").read_text())
        if run["metadata_id"] != metadata_id:
            raise ValueError("D.2 resume metadata differs")
    else:
        output.mkdir(parents=True)
        run = dict(metadata=metadata, metadata_id=metadata_id, status="preflight", baseline_passed=False,
                   reference=previous["results"]["1.2"], results={})
        d1.atomic_json(output/"metadata.json", metadata)
        d1.atomic_json(output/"validation_manifest.json", dict(entries=entries, scan_ids=scan_ids))
        d1.write_csv(output/"class_alpha_table.csv", ["learning_id", "class_name", "alpha"],
                     [dict(learning_id=i, class_name=name, alpha=ALPHA_TABLE[i]) for i, name in enumerate(CLASS_NAMES)])
        for src, name in ((args.arch_cfg, "arch_cfg.yaml"), (args.data_cfg, "data_cfg.yaml")):
            (output/name).write_bytes(Path(src).read_bytes())
        d1.atomic_json(output/"results.json", run)
    logger.add(output/"log.txt")
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info("C.2 EMA strict load; raw shared stage1 logits for Hard/Soft; GT never used for assignment; height correction only on final predictions; unobserved alpha=1.2")
    logger.info(f"Fixed learning-ID alpha table (not retuned): {dict(zip(CLASS_NAMES, ALPHA_TABLE))}")
    model = SJNet3_PointNet_Gated_Local3_ClassAlpha(in_channels=24, nclasses=20, drop=arch["model"]["dropout"])
    model.load_state_dict(state, strict=True)
    if sum(p.numel() for p in model.parameters()) != 28876586 or d1.state_digest(model.state_dict()) != metadata["model_state_sha256"]:
        raise ValueError("C.2 parameter count or strict EMA state differs")
    device = torch.device("cuda")
    model.requires_grad_(False).to(device).eval()
    try:
        if "smoke" not in run:
            run["smoke"] = smoke_test(model, state, next(iter(loader)), device)
            d1.atomic_json(output/"results.json", run)
            logger.info(f"Smoke PASS: {run['smoke']}")
        del state, checkpoint
        torch.cuda.empty_cache()
        if args.preflight_only:
            logger.info("Smoke complete; full Baseline/Hard/Soft evaluation has not started")
            return
        if "baseline_reproduction" not in run:
            run["status"] = "baseline_reproduction"
            d1.atomic_json(output/"results.json", run)
            logger.info("Starting required full alpha=1.2 baseline reproduction BEFORE D.2 comparison")
            run["baseline_reproduction"] = d1.evaluate_alpha(model, loader, 1.2, device, ignore, classes, scan_ids, output, metadata_id, args.report_every)
        baseline = run["baseline_reproduction"]
        d1b.validate_result(baseline, 1.2, metadata, classes)
        run["baseline_passed"] = abs(baseline["miou"]-run["reference"]["miou"]) <= 1e-4
        d1.atomic_json(output/"results.json", run)
        if not run["baseline_passed"]:
            raise RuntimeError("Alpha=1.2 baseline reproduction failed; D.2 results NOT finalized")
        logger.info(f"Baseline reproduction PASS: {baseline['miou']:.12f}, diff={baseline['miou']-run['reference']['miou']:+.12f}")
        run["status"] = "evaluating"
        d1.atomic_json(output/"results.json", run)
        if run.get("results") and set(run["results"]) == set(MODES):
            saved = json.loads((output/"progress_d2.json").read_text())
            statistics = AlphaStatistics.restore(saved["statistics"])
        else:
            evaluated, statistics = evaluate_three_modes(model, loader, device, classes, scan_ids, output, metadata_id, args.report_every)
            run.update(evaluated)
        if abs(run["results"]["Baseline"]["miou"]-run["reference"]["miou"]) > 1e-4:
            raise RuntimeError("Shared stage1 baseline differs from D.1b beyond tolerance")
        for result in run["results"].values():
            d1b.validate_result(dict(result, alpha=1.2), 1.2, metadata, classes)
        if statistics.count[0]+statistics.unobserved != 4071*512*512:
            raise ValueError("Alpha statistics do not cover every validation cell")
        if d1.state_digest(model.state_dict()) != metadata["model_state_sha256"]:
            raise RuntimeError("Model weights changed")
        d1b.verify_protected(protected)
        run.update(status="complete", current_alpha=None, invocation_seconds=time.perf_counter()-started,
                   protected_files_unchanged=True)
        export_results(output, run, statistics)
        d1.atomic_json(output/"results.json", run)
        verification = dict(status="PASS", baseline_reproduced=True, state_hash_unchanged=True,
            strict_C2_keys=True, parameters=28876586, smoke=run["smoke"],
            fixed_table_matches_D1b=True, ground_truth_used_for_assignment=False,
            all_three_modes_scans=4071, all_three_modes_batches=1357, alpha_statistics_cover_all_pixels=True,
            protected_files_unchanged=True, inference_only=True)
        d1.atomic_json(output/"verification.json", verification)
        logger.info(f"D.2 complete: { {mode:r['miou'] for mode,r in run['results'].items()} }")
    except BaseException as error:
        run.update(status="interrupted", last_error=f"{type(error).__name__}: {error}")
        d1.atomic_json(output/"results.json", run)
        logger.exception("D.2 stopped; completed/partial results retained for safe --resume")
        raise


if __name__ == "__main__":
    main()
