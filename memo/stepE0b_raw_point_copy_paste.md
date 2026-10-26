# Step E.0b: Raw Point Copy-Paste Technical Memo

This work is isolated verification only. Do not start E.1 from these outputs.
No production Dataset, Parser, Trainer, Model, config, raw scan, label, donor,
feature cache or historical checkpoint is edited. E.0 remains unchanged.

## Sources And Outputs

- `audit_copy_paste_database_e0b.py`: full read-only donor audit and quality manifest.
- `lib/utils/raw_point_copy_paste_e0b.py`: pure patch checks, train raw loader,
  rigid placement, ground/collision/visibility, shared-grid regeneration, isolated
  dropout policy and unchanged OFF delegation.
- `verify_raw_copy_paste_e0b.py`: independent references, actual-scan trials,
  original-cache/OFF comparisons, per-case raw archives,3D PNGs, cost reports.
- `tests/test_raw_point_copy_paste_e0b.py`: synthetic safety/boundary tests.
- Main output: `logs/SJNet3_PointNet_stepE0b_raw_copy_paste_integrity/`.

## Database Audit

Measured 28,615 files /245,523,860 bytes. Quality-usable for **audit**:12,216
(42.6909%). Excluded16,399. Mixed raw-semantic labels:472 (all excluded).
Complete original-byte hashes, shape/dtype, point counts, mean/min/maxXYZ,
PCA extent, remission range, semantic sets and exclusion reasons are retained
in CSV/JSON. No file was deleted, fixed, recentered, relabeled or regenerated.

Class-specific minimum points: bicycle30, motorcycle40, truck150,
other-vehicle100, person30; also conservative PCA width/length/height gates.
These were selected after a full distribution probe, not after tuning paste
success rates or looking at validation accuracy. Low counts and partial surfaces
are common; DBSCAN min_samples10 does not guarantee every retained cluster has
10 points. Geometry gates are not single-instance or complete-object proofs.

| Learning class | Filename-class files | Audit usable |
| --- | ---: | ---: |
| bicycle2 | 3,460 | 2,013 |
| motorcycle3 | 2,857 | 1,932 |
| truck4 | 4,631 | 1,212 |
| other-vehicle5 | 9,481 | 3,747 |
| person6 | 6,379 | 3,312 |
| bicyclist7 | 1,276 | 0 |
| motorcyclist8 | 531 | 0 |
| pole18 / traffic-sign19 | 0 | 0 |

Raw11 maps to learning2, raw18 to4 (not pole18), raw257 to5; YAML is authoritative.
Raw31/32 may occur only as points inside rejected mixed patches. Riders7/8 are
not planted as ground-standing people: original bike/motorcycle alignment is lost,
so they require a separate source-traceable paired-object design. No missing class
is synthesized or silently borrowed from validation.

## Provenance Gate

`build_database.py` and its inspected Git history use training sequences only.
The current DB stores only relative XYZ, remission and raw semantic ID, not source
sequence/frame/instance or original center. Current files cannot be proven to
come only from that code/config. Status: **出典未確認**. There is no claim that
validation leakage has been excluded. File number and timestamp are not provenance.

`load_patch` rejects unknown provenance by default. The verifier explicitly opts
into **audit-only** unverified donors, after quality and hash checks. None of the
12,216 candidates is authorized for formal training yet. Recover a trustworthy
generation manifest or build a NEW instance-aware train-only DB with source
sequence/frame, packed labels, original center/pose, extraction code/config hashes,
content hashes and geometry/vehicle metadata; preserve this old DB unchanged.

## Rigid Geometry And Visibility

The official Polar3 `SemLaserScan.open_scan/open_label` reads training raw data
without sorting points. Target packed instance labels are retained. Donor instances
are unknown; donor instance0 is explicitly not an original instance ID.

Apply `relative_xyz @ Rz.T + translation` with no scaling, shear or polar cell
translation. Sample observed class-appropriate ground, fit a robust local plane,
reject insufficient support/steep or rough plane. Set minimum observed patch z
3cm above ground, reject buried/floating bottom or sparse support. This cannot
prove the observed patch bottom equals the complete object's true contact surface.

Reject any non-ground point in an expanded oriented donor bounding box. Do not
overwrite an entire BEV cell. An explicit2048x64 azimuth/elevation depth buffer
uses sensor limits-23..3 degrees and15cm tolerance: foreground hides donor returns;
retained donor returns remove farther target returns in the same angular bin.
Self-hidden donor returns are filtered; require80% visibility and class-minimum
surviving points. Keep raw-point/label/remission correspondence and all masks.
This is a conservative angular-bin approximation, **not** calibrated64-beam firing,
mesh-based occlusion, unseen-surface synthesis or complete-scene LiDAR simulation.

## Shared Grid And Independent Reference

All24 output channels use one mapping from the **existing** float32 PointNet
point-feature API. Handcrafted7 and deterministic labels/mask are regenerated
from those same changed points/groups. PointNet uses the frozen eval/no_grad
default encoder. No cached donor or original target feature is pasted.

NumPy legacy versus Torch quantization differences at floating-point cell edges
are checked explicitly. Such cases are rejected as boundary-limited candidates,
not snapped, silently corrected or accepted under the legacy1% occupancy allowance.
The successful cases must agree exactly on cell IDs/mask/labels. Fractional feature
comparisons retain exact error counts plus documented FP16 atol1e-5/rtol0.002.

The independent path recomputes NumPy7/labels/mask, applies precision-matched GPU
point-MLP and **separate NumPy max pooling**, then GPU cell-MLP with the same frozen weights.
Selected after-clouds are also passed to the actual original Polar4 projector.
It is not `regenerate_hybrid` called twice. Authoritative point-input calculation
is deliberately shared: an initial all-NumPy input reference exhibited GPU atan2
roundoff magnified by small output cancellation. Those discrepancies are retained
in a separate probe, and per-channel NumPy/input errors are recorded in final JSON.
The tolerance was not enlarged to make that mathematical-precision mismatch pass.
An additional full CPU-MLP/pooling reference is retained as a cross-device
diagnostic: an intermediate run found two candidates with one value each outside
the same tolerance, while cell IDs/labels/masks/raw geometry were exact. The final
primary reference matches the generator's device/precision, with NumPy aggregation
and dense reconstruction still independent. CPU diagnostic violations remain in
the final JSON; they are not presented as bitwise CPU/GPU equivalence.
An additional same-GPU probe found that default scatter-add ordering could change
cell z-means across calls. New CUDA regeneration therefore requires deterministic
algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8. A repeated full scan's point inputs
and encoder outputs were exactly equal under that setting. Original cached OFF
generation/reproduction is left unchanged; this execution policy is explicitly
part of the new raw-regeneration arm and any future matched raw OFF control.

## Tie And Dropout Protocol

Raw semantic priority is unchanged. The new ON-only rule picks maximum raw ID
among equally highest-priority labels, independent of point ordering or point
count elsewhere. Changes relative to baseline caches are counted and must lie
within ambiguous-priority cells. The legacy OFF path delegates the original
Dataset without another random draw, normalization or label rule.

Baseline dropout still means intentional denoising if retained: zero features/
observation mask with nonzero semantic supervision. Main/aux Focal and Lovasz
ignore label0, not observation mask; boundary BCE is mask-weighted. The isolated
E.0b helper offers `ignore_dropped` (also clear labels0) and an explicit
`legacy_denoising` option. Main/aux/Lovasz/boundary ignore behavior is tested.
No baseline loss is edited, and no proposed policy is silently enabled in C.2.
Boundary nuance: clearing labels to0 before the existing Sobel boundary generator
can create artificial boundaries at neighboring kept cells. The gradient test
only proves dropped-cell BCE masking, not neighbor-target correctness. A future
E.1 Trainer should derive boundary targets from the post-geometry **pre-dropout**
semantic map and apply dropout validity (or an approved neighborhood-ignore rule).
This remains an integration gate, not a completed production loss fix.

An E.1 CP-only comparison must use a matched raw-regeneration OFF control with
the same tie rule, dropout policy, raw augmentation and cache rounding. Comparing
new tie/dropout ON only against historical cached OFF confounds the CP effect.
Feature-only roll/flip is not automatically raw geometric equivalence; future raw
rotation/reflection must precede feature generation and be controlled in both arms.

## E.1 Runtime Integration

Prefer online raw A after provenance/protocol gates. Keep DataLoader workers
CPU-only; run the frozen encoder in the main CUDA process and feed detached24ch
to the unchanged C.2 model. Benchmark joint segmentation/encoder memory and
throughput, retry limits/fallback rate and class selection before formal launch.
Current successful-path preprocessing and illustrative epoch overhead are **not**
measured training epoch times. Reference/plot/archive time is excluded and reported.

One offline B example is stored under E.0b using the actual original gzip torch
7ch/sparse16 formats and loaded by the unchanged Dataset. Use an immutable bank
with full provenance/label-policy metadata if full-workflow A cost is too high;
finite diversity, storage and variant sampling are explicit tradeoffs.

C.2 model, Dataset, Parser, saved config, EMA best checkpoint and current losses
exist, but the historical Local3 entrypoint and StableLR Trainer are absent.
Current base Trainer scheduler/aux decay must not be assumed identical to C.2.
Reconstruct/archive an approved exact recipe from source history/records before
claiming a strict reproduction. E.0b does not restore or launch training.

## Status

Authoritative measured success counts, class rates, independent errors, performance
and final verdict are in the main output's `raw_copy_paste_report.md` and
`verification.json`. Missing donor provenance alone prevents unconditional PASS;
physical visibility/partial-instance limitations and matched-control protocol
remain explicit conditions even when feature/label/mask regeneration succeeds.
