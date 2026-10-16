# Step D.1: Fixed-alpha Sensitivity

Evaluation-only experiment using the trained C.2 best checkpoint. No training
code, optimizer, scheduler, backward pass or weight update is used.

## Completed Results (2026-10-14)

All five full-validation conditions completed successfully. Alpha=1 reproduced
C.2 with mIoU=.5082557747585773, difference +.0000027747585773 from the rounded
reference .508253, comfortably within the required .0001 tolerance.

| Fixed Alpha | mIoU | Accuracy | mIoU Delta From Alpha=1 |
| --- | --- | --- | --- |
| .4 | .503000788 | .890520888 | -.005254987 |
| .6 | .505682624 | .891923156 | -.002573151 |
| .8 | .507405620 | .892888917 | -.000850155 |
| 1.0 | .508255775 | .893508374 | 0 |
| 1.2 | .508627000 | .893835308 | +.000371225 |

The highest mIoU among the five tested values is alpha=1.2; the gain is small.
Class-wise best values differ: person/parking prefer .6, bicyclist/road/sidewalk
prefer .8, car/trunk prefer 1.0, and the remaining nonzero classes prefer 1.2.
Motorcyclist has IoU=0 in all five conditions and no distinguishable best.
Bicycle and motorcycle worsened at lower alpha in this frozen-checkpoint test.

Full evaluation wall time: 2,186.94 seconds (36 min 26.94 sec).
CUDA model-forward time: 2,050.57 seconds (34 min 10.57 sec).
Each condition took approximately 7 min 18 sec including loading/metrics.
All five raw confusion matrices contain exactly 4,071 * 512 * 512 entries;
independent NumPy reconstruction reproduces all saved mIoUs, accuracies and
19 class IoUs within 1e-12. Model-state hashes are identical across all alpha
conditions and match the checkpoint EMA state. Protected original-file and
checkpoint hashes also match. No NaN/Inf/OOM occurred.

All required CSV/JSON/Markdown outputs and three PNG graphs are saved in
logs/SJNet3_PointNet_stepD1_fixed_alpha_sensitivity/. See comparison.md there
for the complete 19-class table and per-class deltas. The mIoU curve and
delta heatmap were visually checked.

## Files

New source files:

- lib/models/SJNet3_PointNet_Gated_Local3_FixedAlpha.py
- evaluate_pointnet_fixed_alpha_sensitivity.py
- tests/test_pointnet_fixed_alpha.py
- memo/stepD1_fixed_alpha_sensitivity.md

Existing C.2/C.3/C.4 model/config/training files are not edited. The unrelated
user edits in memo/kangae.txt are preserved.

## Model And Checkpoint

The model inherits C.2's complete forward and replaces only this coefficient:

```python
point_local = point_base + fixed_alpha * local_residual
```

fixed_alpha is a plain Python float, not a parameter or buffer. State keys,
parameter count, stems, Local3, mask modulation, fusion and all five outputs
are identical to C.2. The checkpoint loads with strict=True, and the loaded
state hash must equal the selected checkpoint state's hash. The same hash is
verified after every complete alpha condition.

Source: logs/SJNet3_PointNet_stepC2_local3_gated_lr2e3_bs3/best_val.path.
Saved/best epoch: 18. Exact recorded best mIoU: 0.508252934833837.
The checkpoint's model equals model_ema exactly; 356 tensors differ between
model_train and model_ema. D.1 selects model_ema, matching C.2 validation.
latest.path and checkpoints from other experiments are rejected.

## Evaluation Conditions

- Alpha values: .4, .6, .8, 1.0, 1.2. Alpha=1 is evaluated first and reused
  as the reference row rather than evaluated a second time.
- Sequence 08, all 4,071 validation scans, 1,357 batches of size 3.
- Existing Parser_PointNet/SemanticKitti_PointNet, label mapping, fixed saved
  PointNet features, workers=4, pin_memory, persistent workers and prefetch=4.
- Sequential loader, drop_last=False, no validation augmentation/Copy-Paste.
- model.eval(), frozen parameters, torch.no_grad(), float32/no autocast.
- C.2 torch/numpy seed 0 and backend precision/determinism/benchmark settings.
- Original height correction after argmax: max_z < -.8 maps predicted 7->2
  and 8->3. No extra KNN, mask filtering or per-scan metric averaging.
- Original iouEval accumulates the confusion matrix over the entire split,
  then computes accuracy and IoUs with class 0 ignored.
- Ground truth is used only in evaluation metrics, never to select inference
  alpha by class or cell.

Each condition checks the exact scan order against a saved manifest. Metadata
records checkpoint/config hashes, feature-file paths/sizes/mtimes, model-state
hash, source-code hashes, dataset root, settings, GPU and torch version.

## Verification And Reproduction Gate

Seven focused tests passed: strict C.2 state compatibility/all-output/mask
equivalence, local residual formula, EMA selection and wrong-checkpoint
rejection, metric parity, reproduction gate, ties/all-zero classes, partial
restart and exports/plots. py_compile passed for all new Python files.

Real GPU preflight used [3,24,512,512] input. At alpha=1 all five output tensors
exactly matched original C.2 (max absolute difference 0). Alpha=.4 changed
point_local; outputs stayed finite, weights stayed identical and no gradients
were present. Observed preflight peak allocated memory: 10.436165 GiB.
Preflight artifacts: logs/StepD1_preflight_20261014/.

Before any other alpha, the script requires the full alpha=1 mIoU to be within
.0001 of BOTH .508253 and the exact checkpoint best mIoU. On failure, it saves
the reference result and stops without evaluating the other conditions.
No metric adjustment is performed to force agreement.

## Results And Restart

Output: logs/SJNet3_PointNet_stepD1_fixed_alpha_sensitivity/.
Live/completed results are recorded in results.json and log.txt.

After each condition, results.csv, per_class_iou.csv, per_class_delta.csv,
best_alpha_by_class.csv and comparison.md are regenerated from measured data.
After all five conditions, three PNGs are written: mIoU curve, five-class
IoU curves and the 19-class delta heatmap.

Progress JSON files atomically save the confusion matrix and completed batch
offset every 100 batches, at completion, and on handled interruption. Resume
requires exactly matching checkpoint/config/data/source/environment metadata,
reuses completed conditions and continues partial conditions without counting
the same scans twice. Previous batches may be reloaded for alignment but their
predictions are not reevaluated or re-added to the confusion matrix.

Best-alpha ties use an absolute IoU tolerance of 1e-12. All-zero classes are
explicitly indistinguishable. The selected score is measured on the selection
validation split and therefore includes selection bias. Preferred class-wise
fixed alphas do not prove that combining them into a spatial map will improve
performance; use separate parameter-selection and final-evaluation data for
D.2 where possible.

CUDA forward time covers model execution including all existing heads. Wall
time also includes data loading, metrics, hashing and progress saves. Timing
is saved per alpha in results.csv/results.json.

## Command

```bash
MPLCONFIGDIR=/tmp/stepd1_mpl .venv/bin/python -B -u evaluate_pointnet_fixed_alpha_sensitivity.py \
  --dataset SemanticKitti \
  --arch_cfg config/arch/sjnet_pointnet_gated_local3_stablelr.yaml \
  --checkpoint logs/SJNet3_PointNet_stepC2_local3_gated_lr2e3_bs3/best_val.path \
  --alphas 0.4 0.6 0.8 1.0 1.2 \
  --output logs/SJNet3_PointNet_stepD1_fixed_alpha_sensitivity
```

Add --resume to the same command to continue the same experiment. Use
--preflight-only with a separate output directory to check inference without
running the full validation sweep.
