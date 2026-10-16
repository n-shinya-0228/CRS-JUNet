# Step D.1b: High-alpha Sensitivity

## Completed Results (2026-10-14)

All four requested new conditions completed: 4071 scans / 1357 batches each.
C.2 epoch-18 best EMA checkpoint and all weights remained fixed. D.1's five
original result dictionaries were imported exactly, without re-evaluation.

| Alpha | mIoU | Accuracy | Delta From 1.2 (IoU Units) |
| --- | --- | --- | --- |
| .4 | .503000788 | .890520888 | -.005626211 |
| .6 | .505682624 | .891923156 | -.002944376 |
| .8 | .507405620 | .892888917 | -.001221380 |
| 1.0 | .508255775 | .893508374 | -.000371225 |
| 1.1 | .508502136 | .893700969 | -.000124863 |
| 1.2 | .508627000 | .893835308 | 0 |
| 1.3 | .508590489 | .893903245 | -.000036510 |
| 1.4 | .508528906 | .893929140 | -.000098093 |
| 1.6 | .508127709 | .893880272 | -.000499290 |

Overall best remains alpha=1.2, inside the tested range. No higher-alpha
condition improves overall mIoU. Alpha=1.6 is .049929 percentage points below
1.2. Neither 1.8 nor 2.0 is evaluated.

Hypothesis A is supported for bicycle, not motorcycle. Bicycle improves
monotonically above 1.2 to .207102 at 1.6; motorcycle's best remains 1.2.
Hypothesis B is supported for truck (.681002 at 1.6), while traffic-sign
improves only at 1.3 (.483619) and then declines. Person and parking decline
above 1.2, supporting hypothesis C. At 1.6 their changes from 1.2 are
-.819602 and -.383458 percentage points, respectively. Preferred class-wise
alphas differ, supporting hypothesis D on this validation split only.

Best tested alpha by class:

- .6: person, parking.
- .8: bicyclist, road, sidewalk.
- 1.0: car, trunk.
- 1.1: building.
- 1.2: motorcycle.
- 1.3: pole, traffic-sign.
- 1.6: bicycle, truck, other-vehicle, other-ground, fence, vegetation, terrain.
- N/A: motorcyclist (all nine IoUs are zero).

New evaluation wall time: 1752.282 s (29 min 12.282 s).
New CUDA forward time: 1643.375 s (27 min 23.375 s).
Invocation including setup/preflight/verification: 1767.113 s.
Preflight peak allocated GPU memory: 10.436 GiB; full-evaluation peak: 2.869 GiB.
No NaN, Inf, OOM or evaluation interruption occurred.

Syntax check and 27 related tests passed, including original D.1 partial
restart and new compatibility, import, both-reference exports, ties/N/A and
plot checks. Tests emitted existing Matplotlib/pyparsing deprecation warnings
and the expected CUDA-disabled warning from synthetic trainer tests; these
do not affect the CUDA inference experiment. The real alpha=1 preflight
exactly matched original C.2 for all five outputs; the alpha=1.1 smoke passed.

Independent NumPy reconstruction matches every saved mIoU, accuracy and all
19 IoUs within 1e-12. All nine raw matrices contain 4071 * 512 * 512 counts,
and all model-state hashes match the checkpoint EMA hash. Protected original
source/checkpoint/D.1 artifact hashes are unchanged. Existing user edits in
memo/kangae.txt are preserved. Curves and both heatmaps were visually checked.

## Saved Files

Under logs/SJNet3_PointNet_stepD1b_high_alpha_sensitivity/:

- results.csv, results.json: nine conditions, metrics/timing and provenance.
- per_class_iou.csv: all 19 classes across nine alphas.
- per_class_delta_vs_alpha1.csv, per_class_delta_vs_alpha12.csv: both references.
- best_alpha_by_class.csv: best IoU/alpha, explicit ties and all-zero N/A.
- comparison.md: complete tables, hypotheses, timing and limitations.
- miou_curve.png, class_iou_curves.png: imported/new conditions distinguished.
- class_delta_heatmap.png, class_delta_heatmap_vs_alpha12.png: percentage points.
- metadata.json, validation_manifest.json, arch_cfg.yaml, data_cfg.yaml.
- log.txt, progress_alpha_1.1/1.3/1.4/1.6.json: execution and restart records.
- verification.json: independent final audit summary.

The reused D.1 exporter also creates per_class_delta.csv and three legacy-name
plots inside D.1b only. No original D.1 output is overwritten.

## Implementation And Reproduction

Evaluation only. Import D.1 alpha=.4, .6, .8, 1.0, 1.2 unchanged; evaluate
only 1.1, 1.3, 1.4, 1.6 with the original D.1 evaluate_alpha function.
No model, dataset, trainer, config or D.1 artifact is edited.

The wrapper checks checkpoint/config hashes, EMA state, label mapping,
validation feature-file metadata and scan order, original source hashes,
GPU/PyTorch, loader settings and every imported confusion matrix/metric.
Incompatible experiments stop before import. Full evaluation remains
sequence 08, 4071 scans, batch size 3, float32/no autocast, frozen eval/no_grad,
class 0 ignored, with the original 7->2 / 8->3 height correction.

The real-data preflight reuses D.1's original-C.2 equivalence check, followed
by a short alpha=1.1 smoke test. No full alpha=1 evaluation is repeated.
Every completed/partial new condition is saved with D.1's restart helper.
Resume requires the same experiment identity and preserves imported data.

CSV deltas use raw IoU units. Heatmap deltas are percentage points.
Combined exports compare both alpha=1 and alpha=1.2, with all 19 classes,
explicit ties and N/A for all-zero classes. D.1 timing rows are historical.

```bash
MPLCONFIGDIR=/tmp/stepd1b_mpl .venv/bin/python -B -u evaluate_pointnet_high_alpha_sensitivity.py \
  --dataset SemanticKitti \
  --output logs/SJNet3_PointNet_stepD1b_high_alpha_sensitivity
```

Add --resume to continue the same experiment. --preflight-only performs
compatibility/preflight/smoke without any full new-alpha validation.
No alphas beyond the four requested values are automatically run.

Reports and artifacts: logs/SJNet3_PointNet_stepD1b_high_alpha_sensitivity/.
Results select alpha on one validation split, with selection bias and no
repeated-run significance estimate. Class-specific global-alpha preferences
do not prove that a learned/spatially varying alpha would improve accuracy.
Feature identity uses D.1's recorded paths, sizes and mtimes, not content hashes.
