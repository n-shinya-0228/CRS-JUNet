"""D.2 derived tables and plots; all writes target the new experiment only."""

from pathlib import Path

import numpy as np

import evaluate_pointnet_fixed_alpha_sensitivity as d1
from lib.models.SJNet3_PointNet_Gated_Local3_ClassAlpha import ALPHA_TABLE, CLASS_NAMES
from .pointnet_class_alpha_statistics import HIST_WIDTH, MODES


def alpha_rows(statistics):
    rows = []
    scopes = [("observed", 0, "observed", "", ""), ("unobserved", 0, "unobserved", "", ""),
              ("all", 0, "all", "", "")]
    scopes += [("GT_observed", 1+cid, "observed", cid, name) for cid, name in enumerate(CLASS_NAMES)]
    scopes += [("raw_prediction_observed", 21+cid, "observed", cid, name) for cid, name in enumerate(CLASS_NAMES)]
    scopes += [("stage1_correct_observed_labeled", 41, "observed", "", ""),
               ("stage1_wrong_observed_labeled", 42, "observed", "", "")]
    for mode, name in enumerate(MODES[1:]):
        for scope, row, population, cid, label in scopes:
            rows.append(dict(mode=name, scope=scope, class_id=cid, class_name=label,
                             **statistics.distribution(mode, row, population)))
    return rows


def best_modes(values):
    if all(value == 0 for value in values):
        return "N/A (all zero)"
    winners = [mode for mode, value in zip(MODES, values) if abs(value-max(values)) <= 1e-12]
    return ";".join(winners) + (" (tie)" if len(winners) > 1 else "")


def export_results(output, run, statistics=None):
    results = run.get("results", {})
    reference = run["reference"]
    if not results:
        return
    measured_baseline = results["Baseline"]
    rows = [dict(mode=mode, miou=result["miou"], accuracy=result["accuracy"],
                 delta_vs_D1b_alpha12=result["miou"]-reference["miou"],
                 delta_vs_measured_baseline=result["miou"]-measured_baseline["miou"],
                 shared_stage1_forward_seconds=run.get("forward_seconds", {}).get("Baseline"),
                 own_forward_seconds=run.get("forward_seconds", {}).get(mode))
            for mode, result in results.items()]
    d1.write_csv(output / "results.csv", list(rows[0]), rows)
    if any(mode not in results for mode in MODES):
        return
    names = CLASS_NAMES[1:]
    ious, deltas = [], []
    table = ["| Class | Baseline alpha=1.2 | D.2a Hard | D.2b Soft | Best |", "|---|---|---|---|---|"]
    for name in [*names, "mIoU"]:
        values = [results[mode]["miou"] if name == "mIoU" else results[mode]["per_class_iou"][name] for mode in MODES]
        table.append(f"| {name} | " + " | ".join(f"{v:.9f}" for v in values) + " | " + best_modes(values) + " |")
        if name != "mIoU":
            ious.append(dict(Class=name, **dict(zip(MODES, values)), Best=best_modes(values),
                             D1b_alpha12=reference["per_class_iou"][name]))
            deltas.append(dict(Class=name, **{mode:values[i]-reference["per_class_iou"][name] for i, mode in enumerate(MODES)},
                               Soft_minus_Hard=values[2]-values[1]))
    d1.write_csv(output / "per_class_iou.csv", list(ious[0]), ious)
    d1.write_csv(output / "per_class_delta.csv", list(deltas[0]), deltas)
    lines = ["# Step D.2: Class-fixed Alpha, Two-stage Inference", "", f"Status: {run['status']}",
             "", "Exploratory/development evaluation only. Sequence 08 was already used for checkpoint and alpha selection; this is NOT an independent test score. A retrospective split of sequence 08 does not remove that leakage.",
             "No retraining, optimizer, scheduler, backward, learned alpha predictor, feature generation or alpha-table retuning occurs.",
             "", f"Checkpoint: {run['metadata']['checkpoint']}, C.2 epoch 18, EMA, 28,876,586 parameters.",
             f"D.1b alpha=1.2 reference mIoU: {reference['miou']:.12f}.",
             f"Full pre-comparison baseline reproduction: {run['baseline_reproduction']['miou']:.12f}; gate passed: {run['baseline_passed']} (absolute tolerance 1e-4).",
             f"Shared-stage1 baseline mIoU: {measured_baseline['miou']:.12f}.", "", *table,
             "", "CSV deltas are raw IoU units; graphs/paragraph deltas in pp are 100 times the IoU difference.",
             "", "## Fixed Learning-ID Lookup", "", "| ID | Class | Alpha |", "|---|---|---|"]
    lines += [f"| {cid} | {name} | {ALPHA_TABLE[cid]:g} |" for cid, name in enumerate(CLASS_NAMES)]
    lines += ["", "The fixed table was checked against D.1b best_alpha_by_class.csv. Unlabeled and all-zero motorcyclist use the specified fallback 1.2.",
              "Hard uses raw stage1 argmax; Soft uses raw stage1 softmax at temperature 1.0. Both maps come from the SAME main logits; auxiliary/boundary outputs and GT are not used.",
              "Unobserved cells use alpha=1.2 in both methods. Existing input/skip masks are unchanged. Only final predictions receive the original max_z < -.8 correction, 7->2 and 8->3.",
              "", "## Measured Changes And Hypotheses", ""]
    for mode in MODES[1:]:
        improved = [n for n in names if results[mode]["per_class_iou"][n]-reference["per_class_iou"][n] > 1e-12]
        worsened = [n for n in names if results[mode]["per_class_iou"][n]-reference["per_class_iou"][n] < -1e-12]
        lines += [f"{mode}: mIoU {results[mode]['miou']:.9f}; delta vs D.1b alpha=1.2 {100*(results[mode]['miou']-reference['miou']):+.6f} pp.",
                  "Improved classes: " + (", ".join(improved) or "none") + ".",
                  "Worsened classes: " + (", ".join(worsened) or "none") + "."]
    lines.append(f"Soft minus Hard mIoU: {results['D2_Soft']['miou']-results['D2_Hard']['miou']:+.9f}.")
    for label, focus in (("A: small objects", ("person", "bicycle", "motorcycle")), ("B: truck", ("truck",))):
        lines.append(f"Hypothesis {label}, measured deltas versus D.1b alpha=1.2:")
        for name in focus:
            lines.append(name + ": " + ", ".join(f"{mode} {100*(results[mode]['per_class_iou'][name]-reference['per_class_iou'][name]):+.6f} pp" for mode in MODES[1:]) + ".")
    if statistics is not None:
        rows = alpha_rows(statistics)
        d1.write_csv(output / "alpha_statistics.csv", list(rows[0]), rows)
        diagnostics = statistics.diagnostics()
        d1.atomic_json(output / "stage1_error_diagnostics.json", diagnostics)
        run["alpha_statistics"] = [row for row in rows if row["scope"] in ("observed", "unobserved", "all")]
        run["error_diagnostics"] = diagnostics
        lines += ["", "## Alpha Maps And Stage1 Errors", "",
                  "| Mode | Population | Count | Mean | Std | Min | Max | p05 | p50 | p95 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for row in run["alpha_statistics"]:
            values = ["N/A" if row[k] is None else f"{row[k]:.6f}" for k in ("mean", "std", "min", "max", "p05", "p50", "p95")]
            lines.append(f"| {row['mode']} | {row['scope']} | {row['count']} | " + " | ".join(values) + " |")
        lines += ["", "Population std uses streaming float64 sums/squared sums. Hard quantiles use exact fixed-value histogram ranks; Soft quantiles use 0.0001-wide bins (absolute quantile approximation error at most 0.0001).",
                  "The per-class CSV includes distributions conditioned on observed GT classes and raw predicted classes, plus stage1-correct/wrong groups. GT appears only in post-forward diagnostics.",
                  f"Hard/Soft observed mean absolute difference: {diagnostics['observed_hard_soft_mean_absolute_difference']}.",
                  f"Hard/Soft all-cell mean absolute difference: {diagnostics['all_hard_soft_mean_absolute_difference']}.",
                  f"Raw stage1 correct/wrong: {diagnostics['stage1_correct_count']}/{diagnostics['stage1_wrong_count']} among {diagnostics['count']} observed labeled cells; correct fraction {diagnostics['stage1_correct_fraction']}, wrong fraction {diagnostics['stage1_wrong_fraction']}.",
                  "", "| Stage1 Group | Final Mode | Correct Cells | Group Count | Final Correct Fraction |", "|---|---|---|---|---|"]
        for mode in MODES:
            for label, result in diagnostics["by_mode"][mode].items():
                lines.append(f"| {label} | {mode} | {result['final_correct']} | {result['count']} | {result['final_correct_fraction']} |")
        wrong = [diagnostics["by_mode"][mode]["stage1_wrong"]["final_correct_fraction"] for mode in MODES[1:]]
        lines += [f"Hypothesis C, wrong-stage1 group final correct fractions: Hard={wrong[0]}, Soft={wrong[1]}. These conditional diagnostics compare changed predictions, not a causal experiment isolating only assignment errors.",
                  "Conditional confusion matrices use original ignore-0 metrics. Direct final-correct fractions count every observed labeled cell, including predictions of class 0 as errors; denominators therefore differ from original ignore-0 accuracy."]
        plot_results(output, results, reference, statistics)
    lines += ["", "## Timing And Safeguards", "",
              f"Three-condition streaming evaluation wall seconds: {run.get('evaluation_wall_seconds')}.",
              f"CUDA forward seconds by mode: {run.get('forward_seconds')}; Hard/Soft share the measured Baseline stage1 forward, so their two-pass inference costs include that additional forward.",
              f"Peak allocated GPU GiB: {run.get('peak_allocated_GiB')}; preflight peak is reported separately in results.json.",
              "Results use 4071 scans in manifest order, batch 3, float32/no autocast, eval/no_grad, the same saved PointNet features and accumulated original iouEval confusion matrices.",
              "Progress commits all three conditions and diagnostics together at completed-batch boundaries. Resume checks identity and never re-adds committed scans.",
              "No independent-data generalization or statistical significance claim is made; no alpha values were adjusted after observing D.2 results.",
              "", "## Files And Command", "",
              "results.csv/results.json, per_class_iou.csv, per_class_delta.csv, alpha_statistics.csv, stage1_error_diagnostics.json, class_alpha_table.csv, comparison.md, four PNG figures, metadata.json, validation_manifest.json, verification.json, log.txt, progress_d2.json and selected alpha_example_*.npz files.",
              "```bash", "MPLCONFIGDIR=/tmp/stepd2_mpl .venv/bin/python -B -u evaluate_pointnet_class_fixed_alpha.py --dataset SemanticKitti", "```",
              "Add --resume for the same experiment. --preflight-only does not run full validation."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def plot_results(output, results, reference, statistics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = CLASS_NAMES[1:]
    x = np.arange(19)
    colors = ("#777777", "#267da0", "#cd7047")
    for delta, filename in ((False, "class_iou_comparison.png"), (True, "class_delta_comparison.png")):
        fig, ax = plt.subplots(figsize=(13, 5))
        modes = MODES[1:] if delta else MODES
        width = .8/len(modes)
        for index, mode in enumerate(modes):
            values = np.array([results[mode]["per_class_iou"][n] for n in names])
            if delta:
                values = 100*(values-np.array([reference["per_class_iou"][n] for n in names]))
            ax.bar(x+(index-(len(modes)-1)/2)*width, values, width, label=mode, color=colors[MODES.index(mode)])
        ax.set_xticks(x, names, rotation=60, ha="right")
        ax.set_ylabel("Delta vs D.1b alpha=1.2 (pp)" if delta else "Class IoU")
        ax.axhline(0, color="black", linewidth=.6)
        ax.legend()
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        fig.savefig(output/filename, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode, name in enumerate(MODES[1:]):
        histogram = statistics.hist[mode, 0]
        positions = .6+np.arange(len(histogram))*HIST_WIDTH
        bins = np.linspace(.59, 1.61, 52)
        counts, _ = np.histogram(positions, bins=bins, weights=histogram)
        ax.stairs(counts/max(1, histogram.sum()), bins, label=name, linewidth=2)
    ax.set(xlabel="Observed-cell alpha", ylabel="Fraction of observed cells per bin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/"alpha_distribution.png", dpi=180)
    plt.close(fig)
    examples = sorted(Path(output).glob("alpha_example_*.npz"))
    if examples:
        fig, axes = plt.subplots(len(examples), 3, figsize=(12, 4*len(examples)), squeeze=False)
        for row, path in enumerate(examples):
            with np.load(path, allow_pickle=False) as sample:
                axes[row, 0].imshow(sample["mask"], cmap="gray", vmin=0, vmax=1)
                axes[row, 0].set_title(str(sample["scan_id"]) + " observation mask")
                for col, name in enumerate(("hard", "soft"), 1):
                    im = axes[row, col].imshow(sample[name], cmap="viridis", vmin=.6, vmax=1.6)
                    axes[row, col].set_title("D2_"+name.title()+" alpha")
                    fig.colorbar(im, ax=axes[row, col], fraction=.046, pad=.04)
                for ax in axes[row]:
                    ax.set_xticks([])
                    ax.set_yticks([])
        fig.suptitle("Inference information only; no ground-truth overlay")
        fig.tight_layout()
        fig.savefig(output/"alpha_map_examples.png", dpi=160)
        plt.close(fig)
