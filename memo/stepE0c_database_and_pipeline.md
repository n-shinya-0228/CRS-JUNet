# Step E.0c / E.1

## Boundaries

All original sources, databases, caches, checkpoints and experiments remain read-only.
New donors: `copy_paste_database_e0c_trainonly/`.
Preparation evidence: `logs/SJNet3_PointNet_stepE0c_database_training_setup/`.
Formal OFF: `logs/SJNet3_PointNet_stepE1a_raw_matched_cp0/`.
Formal ON: `logs/SJNet3_PointNet_stepE1b_raw_cp_on/`.

## Instance database

Only sequences 00-07,09,10. Nonzero instance IDs are scoped to one scan.
Whole observed instances are kept; mixed raw semantics, instance zero, malformed,
fragmented and implausibly sized instances are rejected, never trimmed to pass.
E.0b class-specific point/PCA-size/duplicate/remission checks are reused.
Additional fixed support tests: 0.6m voxels for small objects, 1.2m for vehicles,
largest 26-connected component >=90% of points, <=5% isolated nearest neighbors.
These are conservative geometric screens, not proof of unseen physical completeness.
No validation scores are used to tune donor selection. Rider/vehicle pairs are
excluded because proximity/instance IDs do not establish reliable association.
Each donor includes original coordinates, packed labels, source point indices,
and complete source identity, extraction-code/config and donor SHA256 metadata.

## Matched raw control

Both arms read raw points/packed labels and regenerate 7+16+1 channels on one
authoritative GPU PointNet grid. Handcrafted and PointNet values follow the old
FP16 cache quantization before normalization/input. Labels use priority followed
by highest raw semantic ID for ties. Original cached validation stays unchanged.
Legacy C.2 is a reference, not the matched control.

ON uses probability .5, one object, at most 10 candidate placements, balanced
eligible-class trials, rigid yaw/translation, E.0b ground/OBB collision checks and
2048x64 angular depth-buffer visibility. This is an approximation, not a calibrated
LiDAR simulator. Failed scans use their unmodified raw input and the same generation.
Synthetic pasted instance IDs avoid target collisions; true donor IDs remain in provenance.

CPU workers load/augment raw clouds. The frozen, eval/no_grad PointNet runs only
in the main CUDA process. `CUBLAS_WORKSPACE_CONFIG=:4096:8`; deterministic algorithms
are scoped to raw feature generation, not segmentation backward. C.2 segmentation
uses BF16+GradScaler, validation FP32. Existing model mask/skip/head behavior is unchanged.

The C.2 post-feature azimuth roll/flip and .5-probability 10% cell dropout are kept
in BOTH arms. This retains legacy feature-permutation augmentation, not re-encoding
the rotated absolute-coordinate PointNet inputs. Separate per-scan random streams
control scan order, ordinary augmentation and CP. Dropout retains semantic labels
(intentional denoising); boundary BCE is restricted to the post-dropout mask.

## C.2 recipe evidence

Git commit `c0a0716c8e0b270b5ed98162bfe80874635bec0a` contains the missing StableLR
trainer. Its complete source is saved under the NEW setup directory. Core loss,
EMA, Sobel, focal, scheduler and early-stop methods were extracted into the new
`lib/utils/pointnet_c2_recipe_e1.py`, and AST equality is tested by the gate.
This is a Reconstructed C.2 Recipe with recovered core methods, not a claim that
the entire original training entry/random stream was exactly restored.

AdamW all segmentation parameters: lr .002, decay .005; 5 warmup + 65 cosine,
minimum 1e-5; native batch3; EMA .99; gradient clip5. Focal gamma1, content-normalized
clamped class weights, Lovasz .5, aux .30/.15/.10, boundary .20, paste0. Only aux2/4
decay linearly at epoch50..60. Max70; ES patience12/min_delta.001/min_epoch30.
Existing BatchNorm remains in segmentation train mode. Encoder is always eval.

## Verification / Running

```
.venv/bin/python -B -u build_copy_paste_database_e0c.py
.venv/bin/python -B -u verify_raw_copy_paste_e0c.py
.venv/bin/python -B -u run_pointnet_raw_copy_paste_e1_sequence.py
```

The two formal configs differ ONLY in `copy_paste.enabled`:
`config/arch/sjnet_pointnet_local3_raw_cp_e1a_cp0.yaml` (false),
`config/arch/sjnet_pointnet_local3_raw_cp_e1b_cp_on.yaml` (true).
Disabled CP has effective probability zero; the dormant configured candidate
probability remains .5 to keep other settings identical.

Formal entry refuses to run without a complete PASS gate bound to source/config,
encoder, old C.2 checkpoint and new DB manifests. Verification includes every donor
against all original instance points, independent new-CP projections/pooling,
legacy cache regression, real native-batch3 updates in both arms, checkpoint/resume,
identical seed0 initial hashes, and all4071 legacy-validation scans for alpha1,
alpha1.2 and the frozen D.2b soft table. Model smoke weights are discarded.

The supervisor runs OFF to completion, then ON, then fixed post-training evaluation.
It holds an exclusive lock; never runs both trainers simultaneously. Resume with
`run_pointnet_raw_copy_paste_e1_sequence.py --resume`. Direct training entry supports
`--resume logs/<run>/latest.path`. Logs append. SIGINT/SIGTERM saves an optimizer-boundary
checkpoint with raw/EMA weights, optimizer/scheduler/scaler, RNG and scan offset.
Validation/epoch-boundary interruption may finish that validation before stopping.
The first update and progress boundaries 3000/6000 also save resumable checkpoints.
Preparation can reuse completed full validation with `verify_raw_copy_paste_e0c.py
--reuse-validation` ONLY when model/evaluation code, labels, checkpoint and full
cache manifest signatures are identical. All DB/raw/smoke/recipe tests run again.

Best selection is ONLY alpha1 EMA validation mIoU. All19 class IoUs are logged each
epoch; CP success/rejections/added/removed/new-cells/raw-label histograms are saved
in `cp_stats.csv`. Post-evaluation alpha1.2/D.2b never reselects best checkpoints.
The reused D.2b table was selected on validation: results remain exploratory, not
independent held-out test evidence.
