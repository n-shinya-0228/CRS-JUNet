#!/usr/bin/env python3
"""Step D.1b: import compatible D.1 measurements and evaluate four new alphas."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import yaml
from loguru import logger

import evaluate_pointnet_fixed_alpha_sensitivity as d1

OLD_ALPHAS = [.4, .6, .8, 1., 1.2]
NEW_ALPHAS = [1.1, 1.3, 1.4, 1.6]
ALL_ALPHAS = sorted(OLD_ALPHAS + NEW_ALPHAS)
ROOT = Path(__file__).resolve().parent


def json_normalized(value):
    return json.loads(json.dumps(value))


def validate_result(result, alpha, metadata, classes):
    if (result["alpha"] != alpha or result["scans"] != metadata["validation_scans"] or
            result["batches"] != metadata["validation_batches"] or
            result["model_state_sha256"] != metadata["model_state_sha256"]):
        raise ValueError(f"Result identity mismatch at alpha={alpha:g}")
    expected_names = [classes[cid] for cid in range(1, 20)]
    if list(result["per_class_iou"]) != expected_names:
        raise ValueError(f"Class names/order mismatch at alpha={alpha:g}")
    matrix = np.asarray(result["confusion_matrix"])
    if (matrix.shape != (20, 20) or not np.issubdtype(matrix.dtype, np.integer) or
            (matrix < 0).any() or matrix.sum() != metadata["validation_scans"] * 512 * 512):
        raise ValueError(f"Incomplete/invalid confusion matrix at alpha={alpha:g}")
    evaluator = d1.iouEval(20, torch.device("cpu"), [0])
    evaluator.conf_matrix.copy_(torch.from_numpy(matrix))
    miou, per_class = evaluator.getIoU()
    np.testing.assert_allclose([result["miou"], result["accuracy"], *result["per_class_iou"].values()],
                               [miou.item(), evaluator.getacc().item(), *per_class[1:].tolist()],
                               atol=1e-12, rtol=0)


def validate_source(old, current, saved_manifest, entries, scan_ids, classes):
    if (old["status"] != "complete" or not old.get("baseline_passed") or
            old.get("preflight", {}).get("status") != "PASS"):
        raise ValueError("D.1 must have completed its reproduction gate and preflight")
    metadata = old["metadata"]
    if metadata["experiment"] != "Step D.1" or metadata["alphas"] != OLD_ALPHAS:
        raise ValueError("Unexpected source experiment/alpha conditions")
    mismatches = [key for key in metadata if key not in ("experiment", "alphas") and
                  (key not in current or json_normalized(metadata[key]) != json_normalized(current[key]))]
    if mismatches:
        raise ValueError("D.1 compatibility mismatch: " + ", ".join(mismatches))
    if saved_manifest != dict(entries=entries, scan_ids=scan_ids):
        raise ValueError("D.1 feature metadata or scan order changed")
    if set(old["results"]) != {str(a) for a in OLD_ALPHAS}:
        raise ValueError("D.1 is missing one or more measured conditions")
    for alpha in OLD_ALPHAS:
        validate_result(old["results"][str(alpha)], alpha, current, classes)
    if not d1.reproduction_passed(old["results"]["1.0"], current["checkpoint_best_miou"]):
        raise ValueError("Imported D.1 alpha=1 result fails the C.2 reproduction gate")
    return copy.deepcopy(old["results"])


def protected_hashes(source):
    # Preserve both experiment artifacts and the original implementation files.
    paths = {path for path in source.rglob("*") if path.is_file()}
    for folder in ("lib", "config", "tests", "memo"):
        paths.update(path for path in (ROOT / folder).rglob("*") if path.suffix in (".py", ".yaml", ".md", ".txt"))
    paths.update(ROOT.glob("train*.py"))
    paths.add(Path(d1.__file__).resolve())
    reference = source / "protected_files.sha256"
    for line in reference.read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        path = (ROOT / filename).resolve()
        if d1.file_sha256(path) != expected:
            raise ValueError(f"Original D.1 dependency changed: {filename}")
        paths.add(path)
    new_files = {ROOT / "tests/test_pointnet_high_alpha.py", ROOT / "memo/stepD1b_high_alpha_sensitivity.md"}
    return {str(path.resolve()): d1.file_sha256(path) for path in sorted(paths - new_files)}


def verify_protected(hashes):
    changed = [path for path, expected in hashes.items() if d1.file_sha256(path) != expected]
    if changed:
        raise ValueError("Protected files changed: " + ", ".join(changed))


@torch.no_grad()
def smoke_high_alpha(model, batch, device):
    x = torch.cat([batch[0].to(device), batch[1].to(device).float()], dim=1)
    assert tuple(x.shape) == (3, 24, 512, 512)
    assert batch[14].count_nonzero().item() == 0
    before = d1.state_digest(model.state_dict())
    observed = {}

    def capture(module, inputs, output):
        observed["local"] = output.detach().clone()

    handle = model.point_context.register_forward_hook(capture)
    try:
        model.set_alpha(1.)
        reference = model(x)
        at_one = observed["local"]
        del reference
        model.set_alpha(1.1)
        outputs = model(x)
        if torch.equal(at_one, observed["local"]):
            raise RuntimeError("Alpha=1.1 did not change point_local")
        if not all(torch.isfinite(value).all().item() for value in outputs.values()):
            raise RuntimeError("Non-finite alpha=1.1 smoke output")
        if before != d1.state_digest(model.state_dict()) or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError("Smoke test changed weights or produced gradients")
        return dict(status="PASS", alpha=1.1, shape=list(x.shape), alpha_changes_local=True,
                    all_outputs_finite=True, no_gradients=True, model_state_sha256=before)
    finally:
        handle.remove()


def class_best(results, alphas, name):
    values = {a: results[str(a)]["per_class_iou"][name] for a in alphas}
    winners, score, status = d1.best_alphas(values)
    label = "N/A" if status == "indistinguishable_all_zero" else ";".join(f"{a:g}" for a in winners)
    return label, score, status


def export_results(output, run):
    # Reuse D.1's CSV, full 19-class Markdown table, and plotting machinery.
    d1.export_results(output, run)
    results = run["results"]
    alphas = [a for a in ALL_ALPHAS if str(a) in results]
    names = list(results["1.0"]["per_class_iou"])
    columns = [f"alpha_{a:g}" for a in alphas]
    fields = ["alpha", "source", "miou", "accuracy", "delta_vs_alpha1", "delta_vs_alpha12",
              "wall_seconds", "forward_cuda_seconds", "peak_allocated_GiB"]
    d1.write_csv(output / "results.csv", fields, [dict(alpha=a,
        source="D.1 imported" if a in OLD_ALPHAS else "D.1b measured",
        **{k: results[str(a)][k] for k in ("miou", "accuracy", "wall_seconds", "forward_cuda_seconds", "peak_allocated_GiB")},
        delta_vs_alpha1=results[str(a)]["miou"] - results["1.0"]["miou"],
        delta_vs_alpha12=results[str(a)]["miou"] - results["1.2"]["miou"]) for a in alphas])
    shutil.copyfile(output / "per_class_delta.csv", output / "per_class_delta_vs_alpha1.csv")
    d1.write_csv(output / "per_class_delta_vs_alpha12.csv", ["Class", *columns], [dict(Class=name,
        **{column: results[str(a)]["per_class_iou"][name] - results["1.2"]["per_class_iou"][name]
           for column, a in zip(columns, alphas)}) for name in names])
    winners = []
    for name in names:
        label, score, status = class_best(results, alphas, name)
        winners.append(dict(Class=name, best_alphas=label, best_iou=score, status=status,
                            delta_vs_alpha1=score - results["1.0"]["per_class_iou"][name],
                            delta_vs_alpha12=score - results["1.2"]["per_class_iou"][name]))
    d1.write_csv(output / "best_alpha_by_class.csv", list(winners[0]), winners)
    overall, score, status = d1.best_alphas({a: results[str(a)]["miou"] for a in alphas})
    report = (output / "comparison.md").read_text().replace(
        "# Step D.1: Fixed-alpha Sensitivity", "# Step D.1b: High-alpha Sensitivity").replace(
        "indistinguishable (all zero)", "N/A (all zero)")
    lines = ["", "## D.1b Provenance And References", "",
             "D.1 compatibility checks passed for checkpoint/EMA state, complete configs and label mapping, dataset root, validation feature manifest/order, environment, original evaluation code and protected dependency hashes.",
             "D.1 measurements are imported without changes or re-evaluation; only alpha=1.1, 1.3, 1.4, 1.6 are newly evaluated.",
             "CSV deltas are in raw IoU units; heatmaps and the following deltas use percentage points (100 times IoU difference).",
             "", "| Alpha | Source | mIoU | Accuracy | Delta vs 1.0 (pp) | Delta vs 1.2 (pp) |",
             "|---|---|---|---|---|---|"]
    for a in alphas:
        r = results[str(a)]
        lines.append(f"| {a:g} | {'D.1' if a in OLD_ALPHAS else 'D.1b'} | {r['miou']:.9f} | {r['accuracy']:.9f} | "
                     f"{100*(r['miou']-results['1.0']['miou']):+.6f} | {100*(r['miou']-results['1.2']['miou']):+.6f} |")
    lines += ["", "## Class Best Values And Changes Above 1.2", "",
              "| Class | Best alpha (all nine) | Best IoU | 1.3 delta (pp) | 1.4 delta (pp) | 1.6 delta (pp) |",
              "|---|---|---|---|---|---|"]
    for row in winners:
        name = row["Class"]
        deltas = [f"{100*(results[str(a)]['per_class_iou'][name]-results['1.2']['per_class_iou'][name]):+.6f}"
                  if str(a) in results else "pending" for a in (1.3, 1.4, 1.6)]
        label = row["best_alphas"] + (" (tie)" if row["status"] == "tie" else "")
        lines.append(f"| {name} | {label} | {row['best_iou']:.9f} | " + " | ".join(deltas) + " |")
    lines += ["", "## Hypotheses And Search Range", "",
              f"Best tested mIoU: {score:.9f}, alpha={overall} ({status}); delta vs 1.2: {100*(score-results['1.2']['miou']):+.6f} pp."]
    for hypothesis, focus in (("A", ("bicycle", "motorcycle")), ("B", ("truck", "traffic-sign")), ("C", ("person", "parking"))):
        changes = []
        for name in focus:
            values = [f"alpha={a:g}: {100*(results[str(a)]['per_class_iou'][name]-results['1.2']['per_class_iou'][name]):+.6f} pp"
                      for a in (1.3, 1.4, 1.6) if str(a) in results]
            changes.append(name + " (" + ", ".join(values) + ")")
        lines.append(f"Hypothesis {hypothesis}, measured changes versus 1.2: " + "; ".join(changes) + ".")
    lines.append("Hypothesis D: class-wise best tested alphas are listed above; these are retrospective global-alpha comparisons, not a ground-truth-guided inference policy.")
    if len(alphas) == 9:
        lines.append("The best mIoU reaches the upper tested boundary 1.6; consider 1.8 or 2.0 separately. Neither is evaluated here."
                     if 1.6 in overall else "The mIoU optimum among tested points is inside the tested range; no additional alphas are evaluated.")
        old_best_names = [n for n in names if d1.best_alphas({a: results[str(a)]["per_class_iou"][n] for a in OLD_ALPHAS})[0] == [1.2]]
        improved = [n for n in old_best_names if max(results[str(a)]["per_class_iou"][n] for a in (1.3, 1.4, 1.6)) > results["1.2"]["per_class_iou"][n] + d1.TIE_TOLERANCE]
        lines.append("Classes previously best at 1.2 that improve above 1.2: " + (", ".join(improved) or "none") + ".")
        selected = results[str(overall[0])]
        better = [n for n in names if selected["per_class_iou"][n] > results["1.2"]["per_class_iou"][n] + d1.TIE_TOLERANCE]
        worse = [n for n in names if selected["per_class_iou"][n] < results["1.2"]["per_class_iou"][n] - d1.TIE_TOLERANCE]
        lines += [f"At overall-best alpha={overall[0]:g}, improved versus 1.2: " + (", ".join(better) or "none") + ".",
                  "Worsened versus 1.2: " + (", ".join(worse) or "none") + "."]
        plot_results(output, results, names)
    measured = [results[str(a)] for a in NEW_ALPHAS if str(a) in results]
    lines += ["", "## New Evaluation Timing And Limitations", "",
              f"New conditions wall time: {sum(r['wall_seconds'] for r in measured):.2f} s; CUDA forward time: {sum(r['forward_cuda_seconds'] for r in measured):.2f} s.",
              "D.1 timings in the combined table are historical, not additional work in D.1b.",
              "One checkpoint and one validation split only; small gains have no repeated-run significance estimate. Choosing alpha on this split introduces selection bias.",
              "Feature compatibility is checked using paths, sizes and nanosecond mtimes (D.1's recorded manifest), not a full hash of every feature file.",
              "No training or feature generation occurs. Original observation-mask handling, auxiliary/boundary heads and height correction remain unchanged."]
    (output / "comparison.md").write_text(report + "\n".join(lines) + "\n")


def plot_results(output, results, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    def decorate(ax):
        ax.axvline(1., color="gray", linestyle=":", label="D.1 reference 1.0")
        ax.axvline(1.2, color="black", linestyle="--", label="D.1 best 1.2")
        ax.set_xticks(ALL_ALPHAS)
        ax.set_xlabel("Fixed alpha")
        ax.grid(alpha=.2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ALL_ALPHAS, [results[str(a)]["miou"] for a in ALL_ALPHAS], color="steelblue", label="Combined mIoU")
    ax.scatter(OLD_ALPHAS, [results[str(a)]["miou"] for a in OLD_ALPHAS], color="steelblue", marker="o", label="D.1 imported")
    ax.scatter(NEW_ALPHAS, [results[str(a)]["miou"] for a in NEW_ALPHAS], color="crimson", marker="D", label="D.1b measured")
    decorate(ax)
    ax.set_ylabel("Validation mIoU")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "miou_curve.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ("bicycle", "motorcycle", "person", "truck", "traffic-sign"):
        line, = ax.plot(ALL_ALPHAS, [results[str(a)]["per_class_iou"][name] for a in ALL_ALPHAS], marker="o", label=name)
        ax.scatter(NEW_ALPHAS, [results[str(a)]["per_class_iou"][name] for a in NEW_ALPHAS], marker="D", s=40, color=line.get_color(), zorder=3)
    decorate(ax)
    ax.plot([], [], color="gray", marker="D", linestyle="none", label="New measurements")
    ax.set_ylabel("Class IoU")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "class_iou_curves.png", dpi=180)
    plt.close(fig)
    for reference, filename in (("1.0", "class_delta_heatmap.png"), ("1.2", "class_delta_heatmap_vs_alpha12.png")):
        delta = 100 * np.array([[results[str(a)]["per_class_iou"][n] - results[reference]["per_class_iou"][n] for a in ALL_ALPHAS] for n in names])
        limit = max(float(np.abs(delta).max()), 1e-6)
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(delta, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
        ax.set_xticks(range(9), [f"{a:g}" for a in ALL_ALPHAS])
        ax.set_yticks(range(19), names)
        ax.set_xlabel(f"Fixed alpha (reference = {reference})")
        for row in range(19):
            for col in range(9):
                ax.text(col, row, f"{delta[row, col]:+.3f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(delta[row, col]) > .55*limit else "black")
        fig.colorbar(im, ax=ax, label="Class IoU difference (percentage points)")
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", required=True)
    cli.add_argument("--arch_cfg", default="config/arch/sjnet_pointnet_gated_local3_stablelr.yaml")
    cli.add_argument("--data_cfg", default="config/labels/semantic-kitti.yaml")
    cli.add_argument("--checkpoint", default="logs/SJNet3_PointNet_stepC2_local3_gated_lr2e3_bs3/best_val.path")
    cli.add_argument("--source", default="logs/SJNet3_PointNet_stepD1_fixed_alpha_sensitivity")
    cli.add_argument("--output", default="logs/SJNet3_PointNet_stepD1b_high_alpha_sensitivity")
    cli.add_argument("--resume", action="store_true")
    cli.add_argument("--preflight-only", action="store_true")
    cli.add_argument("--report-every", type=int, default=100)
    args = cli.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if source == output or source in output.parents or output in source.parents:
        cli.error("D.1b output must be separate from the read-only D.1 source")
    if Path(args.checkpoint).name != "best_val.path" or args.report_every < 1:
        cli.error("Require best_val.path and report-every >= 1")
    if output.exists() and not args.resume:
        cli.error("Output exists; use --resume for the same experiment")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for identical D.1 conditions")
    started = time.perf_counter()
    old = json.loads((source / "results.json").read_text())
    protected = protected_hashes(source)
    arch, data = [yaml.safe_load(Path(p).read_text()) for p in (args.arch_cfg, args.data_cfg)]
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state, weight_source = d1.select_evaluation_state(checkpoint, arch, data)
    if not weight_source.startswith("EMA"):
        raise ValueError("D.1b requires the same EMA weights as D.1")
    parser = d1.Parser(root=args.dataset, data_cfg=data, arch_cfg=arch, gt=True, shuffle_train=False)
    loader = parser.get_valid_set()
    entries, scan_ids, manifest_hash = d1.dataset_manifest(parser.valid_dataset, args.dataset)
    classes = {cid: parser.get_xentropy_class_string(cid) for cid in range(parser.get_n_classes())}
    ignore = [cid for cid, ignored in data["learning_ignore"].items() if ignored]
    if (parser.valid_dataset.is_train or parser.valid_dataset.copy_paste_prob != 0 or
            len(scan_ids) != 4071 or len(loader) != 1357 or loader.batch_size != 3 or
            data["split"]["valid"] != [8] or len(classes) != 20 or ignore != [0]):
        raise ValueError("Require unaugmented sequence 08, 4071 scans, batch 3, class 0 ignored")
    metadata = dict(experiment="Step D.1b", dataset_root=str(Path(args.dataset).resolve()),
        checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=d1.file_sha256(args.checkpoint),
        checkpoint_epoch=checkpoint["epoch"], checkpoint_best_miou=checkpoint["best_val_iou"], weight_source=weight_source,
        model_state_sha256=d1.state_digest(state), arch_cfg=arch, data_cfg=data,
        arch_cfg_sha256=d1.file_sha256(args.arch_cfg), data_cfg_sha256=d1.file_sha256(args.data_cfg),
        validation_sequences=data["split"]["valid"], validation_scans=len(scan_ids), validation_batches=len(loader),
        validation_manifest_sha256=manifest_hash, alphas=ALL_ALPHAS,
        batch_size=loader.batch_size, workers=loader.num_workers, pin_memory=loader.pin_memory,
        persistent_workers=loader.persistent_workers, prefetch_factor=loader.prefetch_factor,
        shuffle=False, drop_last=loader.drop_last, copy_paste_prob=0, amp=False, dtype="float32",
        metrics="original iouEval; accumulated confusion matrix; class 0 ignored",
        postprocess="C.2 max_z<-0.8: prediction 7->2, 8->3; no additional mask/KNN",
        torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(),
        inference_only=True, optimizer_created=False, backward=False,
        evaluation_script_sha256=d1.file_sha256(d1.__file__),
        fixed_alpha_model_sha256=d1.file_sha256(ROOT / "lib/models/SJNet3_PointNet_Gated_Local3_FixedAlpha.py"))
    imported = validate_source(old, metadata, json.loads((source / "validation_manifest.json").read_text()), entries, scan_ids, classes)
    metadata.update(new_alphas=NEW_ALPHAS, imported_alphas=OLD_ALPHAS, source_dir=str(source),
                    source_results_sha256=d1.file_sha256(source / "results.json"),
                    wrapper_sha256=d1.file_sha256(__file__), protected_files=protected,
                    class_ids_and_names=classes, cudnn_deterministic=True, cudnn_benchmark=True,
                    matmul_precision="high", seed=0)
    metadata = json_normalized(metadata)
    metadata_id = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    if output.exists():
        run = json.loads((output / "results.json").read_text())
        if run["metadata_id"] != metadata_id:
            raise ValueError("Resume identity differs: checkpoint/config/data/code/environment")
        if any(run["results"].get(key) != value for key, value in imported.items()):
            raise ValueError("Imported D.1 results changed in resumed D.1b run")
    else:
        output.mkdir(parents=True)
        run = dict(metadata=metadata, metadata_id=metadata_id, status="preflight", results=imported,
                   baseline_passed=True, compatibility=dict(status="PASS", imported_alphas=OLD_ALPHAS))
        d1.atomic_json(output / "results.json", run)
        d1.atomic_json(output / "validation_manifest.json", dict(entries=entries, scan_ids=scan_ids))
        d1.atomic_json(output / "metadata.json", metadata)
        for name, value in (("arch_cfg.yaml", arch), ("data_cfg.yaml", data)):
            (output / name).write_text(yaml.safe_dump(value, sort_keys=False))
    logger.add(output / "log.txt")
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"D.1 compatibility PASS | imported={OLD_ALPHAS} | new={NEW_ALPHAS} | weights={weight_source}")
    model = d1.SJNet3_PointNet_Gated_Local3_FixedAlpha(in_channels=24, nclasses=20, drop=arch["model"]["dropout"])
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda")
    model.requires_grad_(False).to(device).eval()
    if d1.state_digest(model.state_dict()) != metadata["model_state_sha256"]:
        raise RuntimeError("Strictly loaded model does not match the EMA state hash")
    try:
        if "smoke" not in run:
            batch = next(iter(loader))
            run["preflight"] = d1.preflight(model, state, batch, device)
            run["smoke"] = smoke_high_alpha(model, batch, device)
            run["smoke"]["peak_allocated_GiB"] = torch.cuda.max_memory_allocated(device) / 2**30
            d1.atomic_json(output / "results.json", run)
            logger.info(f"C.2 equivalence preflight: {run['preflight']}")
            logger.info(f"Alpha=1.1 smoke: {run['smoke']}")
            del batch
        del state, checkpoint
        torch.cuda.empty_cache()
        export_results(output, run)
        if args.preflight_only:
            logger.info("Preflight/smoke complete; no full new condition evaluated")
            return
        for alpha in NEW_ALPHAS:
            key = str(alpha)
            if key in run["results"]:
                validate_result(run["results"][key], alpha, metadata, classes)
                logger.info(f"Reusing completed new alpha={alpha:g}")
                continue
            run.update(status="evaluating", current_alpha=alpha)
            d1.atomic_json(output / "results.json", run)
            logger.info(f"Starting full validation alpha={alpha:g} | 4071 scans | FP32 | frozen EMA")
            result = d1.evaluate_alpha(model, loader, alpha, device, ignore, classes, scan_ids, output, metadata_id, args.report_every)
            validate_result(result, alpha, metadata, classes)
            run["results"][key] = result
            d1.atomic_json(output / "results.json", run)
            export_results(output, run)
            logger.info(f"Completed alpha={alpha:g} | mIoU={result['miou']:.9f} | accuracy={result['accuracy']:.9f} | wall={result['wall_seconds']:.1f}s")
        verify_protected(protected)
        run.update(status="complete", current_alpha=None, protected_files_unchanged=True,
                   invocation_seconds=time.perf_counter() - started)
        run.pop("last_error", None)
        d1.atomic_json(output / "results.json", run)
        export_results(output, run)
        logger.info(f"D.1b complete; original files unchanged: {output}")
    except BaseException as error:
        run.update(status="interrupted", last_error=f"{type(error).__name__}: {error}")
        d1.atomic_json(output / "results.json", run)
        export_results(output, run)
        logger.exception("D.1b stopped; completed and partial measurements retained for --resume")
        raise


if __name__ == "__main__":
    main()
