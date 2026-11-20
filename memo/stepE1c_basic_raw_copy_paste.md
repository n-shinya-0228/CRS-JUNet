# Step E.1c: Basic Raw Point Copy-Paste, Maximum 3 Objects

Experiment: `logs/SJNet3_PointNet_stepE1c_raw_cp_basic3/`.
Created on branch `lab_desk`. Existing E.1a/E.1b files are read-only references.

## Placement

- Probability 0.5 per scan, at most three independently selected donors.
- Existing train-only database, balanced classes 2..6: bicycle, motorcycle,
  truck, other-vehicle, person. Donors retain the existing database quality audit.
- Uniform yaw and area-uniform polar XY center. At most 10 placement attempts
  per selected donor. Duplicate donors are skipped within a scan.
- Align donor bottom to the local 10th-percentile Z plus 0.03 m, using points
  within 2 m. If no nearby points exist, use the scan's in-range Z percentile.
- Require the whole object to be within radius (0, 51.2) m and sensor elevation
  [-23, 3] degrees. Do not crop an object to pass these checks.
- No ground class/support requirement, plane fitting, slope test, OBB collision
  test, depth buffer, occlusion filtering, or 80% visibility threshold.
- Append only: original XYZ, remission, and packed labels are unchanged.
  Each pasted donor receives an unused synthetic instance ID; raw semantics
  and remission are retained. All pasted points contribute to a union mask.
- Recompute handcrafted7, frozen PointNet16, labels, and mask from the entire
  augmented raw cloud using the existing E.1 shared-grid regeneration.

## Training

The config differs from E.1b only by `copy_paste.mode: basic` and
`max_objects_per_scan: 3`. The segmentation model starts from scratch.
AdamW LR 0.002, weight decay 0.005, batch3, seed0, EMA0.99, maximum70 epochs,
warmup5 + cosine65, BF16, and all losses/augmentation/validation are unchanged.
Initial warmup LR is 0.0002. Early stopping remains patience12, delta0.001,
minimum epoch30 (existing zero-based convention).

```bash
.venv/bin/python -B -u verify_basic_raw_copy_paste_e1c.py
.venv/bin/python -B -u train_pointnet_basic_raw_copy_paste_e1c.py
```

Preflight is required and the formal command refuses an existing directory
unless resuming. A completed preflight report is immutable; use a new
`--output` and pass its `verification.json` via training `--gate` to reverify.

```bash
.venv/bin/python -B -u train_pointnet_basic_raw_copy_paste_e1c.py \
  --resume logs/SJNet3_PointNet_stepE1c_raw_cp_basic3/latest.path
```

Training and geometry implementations are inherited/reused from E.1; only
the independent dataset mode and object-count reporting are new. OFF and
advanced dispatch directly delegate to the unchanged original dataset.

## Verification And Outputs

Preflight: `logs/SJNet3_PointNet_stepE1c_raw_cp_basic3_preflight/`.
47 CPU tests passed. In 64 real scans (32 configured-probability scans and
32 forced-ON diagnostics), 13 had zero objects and 51 had three. All five
donor classes were exercised. Controlled tests also cover one/two successes
and bounded all-failure retries. Three independent whole-cloud projections
matched PointNet16, labels, and masks exactly; handcrafted7 matched the
existing FP16 tolerance.

Batch3 forward/backward and frozen encoder checks passed. Model, EMA,
optimizer, scheduler, scaler, and CPU/CUDA RNG restore exactly. The next
CUDA update is not guaranteed bitwise equal under the retained E.1 settings;
`resume_diagnostics.json` records this separately (observed maximum parameter
difference 0.0003445, identical next-update loss). No training settings were
changed to force determinism. The unavailable historical C.2 checkpoint is
not used by scratch training; all present historical hashes, E.1a/E.1b
artifacts, initialization, and the validation cache manifest were verified.

Training outputs include `train_log.txt`, `console.log`, `best_val.path`,
`latest.path`, training/validation histories, all19 class IoUs, and CP stats.
`cp_scan_counts.jsonl` records each processed scan's 0..3 successes; after an
unclean restart, deduplicate journal entries by `(epoch, scan_id)`.
`basic_cp_progress.json` and `basic_cp_epoch_history.json` are checkpoint-aligned
counts excluding replayed updates. Existing CP CSVs retain class totals.

## Comparison

`comparison.json` and `comparison.md` record all19 class IoUs, focus classes,
and Basic-minus-OFF/advanced deltas after training completes or stops cleanly.
References: E.1a OFF best0.496889916 at epoch8; E.1b advanced/max1 best0.499935152
at epoch15. E.1c values remain pending until validation.

E.1b versus E.1c changes both method and maximum object count. Differences
cannot be attributed solely to the method. These are BEV-cell validation
metrics, not the official SemanticKITTI point-wise test metric.
