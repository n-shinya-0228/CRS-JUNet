# Step C.4: Spatial-aware Learnable Alpha for PointNet Local3

Implementation and preflight completed on 2026-10-12.
FULL TRAINING HAS NOT STARTED. Smoke-test weights are discarded, not reused.

## Files And Preservation

New files:

- lib/models/SJNet3_PointNet_Gated_Local3_SpatialAlpha.py
- train_pointnet_hybrid_gated_local3_spatial_alpha.py
- config/arch/sjnet_pointnet_gated_local3_spatial_alpha_stablelr.yaml
- lib/utils/pointnet_spatial_alpha_diagnostics.py
- tests/test_pointnet_spatial_alpha.py
- tests/smoke_pointnet_spatial_alpha.py
- memo/stepC4_spatial_alpha.md

One existing file changed: lib/trainer/trainer_PointNet_Gated_StableLR.py.
Additions select the C.4 model/optimizer, collect validation diagnostics and
persist the matching best summary. The original trainer is backed up as
logs/StepC4_preflight_20261012/trainer_StableLR_before_C4.py.

C.2/C.3 model files, configs, datasets, logs and checkpoints are preserved.
The protected_files.sha256 manifest in the preflight directory verifies C.2
and C.3 artifacts. Real latest checkpoints from both experiments still pass
the shared resume validator (epoch 30). Old optimizer group counts remain
one for C.2 and two for C.3. C.4 refuses cross-experiment resume and refuses
an existing log directory for a fresh run.

AST comparisons confirm unchanged train(), train_epoch(), loss functions,
EMA, LR scheduler and EarlyStopping. Dataset/Parser and C.3 diagnostics were
not edited. No PointNet feature regeneration or encoder training occurs.

## Structure And Difference From C.2/C.3

The complete C.2 forward is inherited without modification. Only the
point_context residual coefficient changes:

```python
point_base = point_stem(point_input)
local_residual = local3(point_base)
alpha_delta_map = alpha_predictor(point_base)
alpha_map = alpha_global + alpha_delta_map
point_local = point_base + alpha_map * local_residual
gate = point_gate(torch.cat([hand_feat, point_local], dim=1))
point_gated = point_local * gate
fusion_input = torch.cat([hand_feat, point_gated], dim=1)
```

alpha_global is a scalar float32 nn.Parameter initialized to 1.0.
alpha_predictor is exactly Conv2d(16,1,1,bias=False), with all weights zero.
No ReLU, sigmoid, clamp, extra attention or channel-dependent scale is added.
The map is one scalar per cell, broadcast across the 16 residual channels.

The Local3 block is the SAME C.2 instance: RingConv2d(16,16,3,bias=False),
InstanceNorm2d(16,affine=True), ReLU(inplace=True). Handcrafted processing,
fusion, encoder/decoder, ASPP, CSWin and all five output heads are unchanged.

Registered keys:

- point_context.alpha.alpha_global
- point_context.alpha.alpha_predictor.weight

Both are accessible through model.alpha_global / model.alpha_predictor.
Creating the zero-initialized Conv uses a CPU RNG fork, so the common model
weights AND the post-construction random state match C.2 at the same seed.

## Parameter Counts And Optimizer

| Model | Total | Trainable | Difference From C.2 |
| --- | --- | --- | --- |
| C.2 | 28,876,586 | 28,876,586 | 0 |
| C.3 | 28,876,587 | 28,876,587 | +1 |
| C.4 | 28,876,603 | 28,876,603 | +17 |

AdamW group 0 contains the original 28,876,586 parameters in their existing
order, weight_decay=.005. Group 1 contains alpha_global and predictor.weight,
17 scalar values across two parameter tensors, weight_decay=0.0. Both groups
use the same LR/scheduler. No parameters are duplicated or omitted.

## Tests A-D And Preservation Checks

40 unit/regression tests and syntax compilation passed.

- A: global=1 and zero predictor produce an all-one map. All five C.4 eval
  outputs exactly match C.2 with the same common weights on CPU, for mixed,
  all-observed and all-unobserved masks. CUDA smoke also asserts the local
  residual equation at initial alpha=1 with atol=1e-6.
- B: global=.641107 and zero predictor match C.3's scalar equation and all
  five complete eval outputs with matching common weights on CPU.
- C: nonzero predictor weights produce different alpha values at cells with
  different PointNet features. Negative/extreme scales remain unclamped.
- D: global and predictor gradients exist, are finite/nonzero, and both
  parameters change after optimizer steps.
- Gate/fusion use point_local, not point_base. Model input is unchanged.
- Diagnostic hooks do not change outputs/RNG and are removed after use,
  including exceptional exits. Labels are never arguments to model forward.
- Population mean/std, online merging, quantiles, negative/>3 fractions,
  invalid labels, unobserved cells, empty and low-count classes are tested.
- Raw/EMA weights, both optimizer groups, scheduler, best statistics and
  alpha parameters round-trip through best/latest checkpoints.

## Observation Mask And Class Statistics

Input ordering is unchanged: 0:7 handcrafted, 7:23 PointNet, 23:24 mask.
The existing soft modulation remains `s0 *= 1 + tanh(stem_mask(mask))`.
No new hard masking is added. Nonzero latent features at unobserved cells
still enter gate/fusion, as in C.2/C.3. Alpha does not zero point_base there.

After each validation forward, a detached CPU alpha map for that batch is
combined with the GT labels and observation mask ONLY for statistics.
All 19 non-ignored classes are aggregated over the entire validation loader;
the condition is a valid class label AND observation mask > 0. Prediction and
loss do not consume class-alpha summaries or labels through this pathway.

Class records contain mean, population std and cell count. Fewer than five
cells yields N/A (JSON null), with the count retained. This diagnostic-only
threshold is config diagnostics.alpha_min_class_cells=5, not a claim of
statistical significance. Non-finite groups also have unavailable moments.

Global map, observed/unobserved groups and each class use online merged
moments. No validation-wide array of alpha maps is retained. Exact p05/p50/p95
are computed only for validation batch 0, not approximated as epoch-wide
percentiles. The collector consumes each batch's map after forward.

Batch-0 diagnostics additionally report alpha_delta mean/std/min/max,
point_base/local_residual/scaled_local_residual/point_local mean/std and
mean(abs(scaled_local_residual)) / (mean(abs(point_base)) + 1e-8).
Global alpha and predictor weight norm are recorded from the evaluated model,
which is EMA under the unchanged configuration. Map fractions <0 and >3,
finite checks and observations split are logged. Values <=-5 or >=10, or
non-finite values, warn without clamping.

## Real-Data Smoke Test

RTX 5060 Ti (16 GB), separate C.2 control and C.4 processes, each with three
training batches and two validation batches. Batch size remained 3.
No NaN/Inf, OOM or AMP-skipped optimizer step occurred. Every C.4 update
checked 334 gradient tensors. All paste masks were zero.

| Alpha Gradient (After Existing Unscale/Clip) | Update 1 | Update 2 | Update 3 |
| --- | --- | --- | --- |
| global gradient | .017690485 | .027877467 | .030405814 |
| predictor gradient norm | .127188340 | .190291718 | .099521026 |

After three updates, training global=.999403715 and predictor norm=.001886362.
Evaluated EMA global=.999988139 and predictor norm=.000040460.
These are startup smoke values, NOT a trained best model or accuracy evidence.
The formal run will initialize global=1/predictor=0 again.

All 19 class counts were independently checked against the real validation
labels/masks. The two smoke batches contain 244,546 observed cells, of which
238,856 have valid non-ignored labels; 1,328,318 cells are unobserved.
Examples: car=2,616 cells, bicycle=81, person=303, terrain=75,611,
pole=356, traffic-sign=556. Other-ground=2 is N/A; absent classes have count 0.
These counts describe the smoke subset, not the complete validation split.

Smoke EMA map mean=.999954903, std=.000069683, min=.999268651,
max=.999982178. Negative and >3 fractions are zero; all values are finite.
Batch-0 p05=.999842286, p50=.999976039, p95=.999976099.
Observed mean/std=.999839717/.000124580; unobserved=.999976109/.000000087.
Batch-0 local contribution ratio=1.222788246.

## Verified Shapes

| Tensor | Shape |
| --- | --- |
| Dataset features | [3,23,512,512] |
| Model input | [3,24,512,512] |
| hand / point / mask | [3,7,512,512] / [3,16,512,512] / [3,1,512,512] |
| hand_feat | [3,16,512,512] |
| point_base | [3,16,512,512] |
| local_residual | [3,16,512,512] |
| alpha_global | [] |
| alpha_delta / alpha_map | [3,1,512,512] each |
| scaled_local_residual / point_local / gate | [3,16,512,512] each |
| Fusion input / output | [3,32,512,512] / [3,48,512,512] |
| logits | [3,20,512,512] |

## GPU Memory

Measured using reset_peak_memory_stats() and max_memory_allocated():

| Peak | C.2 Control | C.4 |
| --- | --- | --- |
| Training allocated GiB | 13.914830 | 13.920117 |
| Training reserved GiB | 14.449219 | 14.451172 |
| Validation allocated GiB | 10.807534 | 10.805727 |

Training allocated increased by 5.414063 MiB. Batch size was never reduced.
These are observed smoke peaks; later batches/backend workspaces can differ.

## Best Checkpoints And Comparison

Best/latest remain separate. Checkpoints include raw and EMA alpha parameters,
optimizer moments, scheduler/scaler/early-stopping state. Real GPU smoke saves
and CPU reloads verified exact global/predictor equality for both checkpoints.

On best mIoU improvement, the trainer saves/displays epoch, mIoU, accuracy,
validation loss, evaluated global alpha, full-validation map mean/std,
predictor norm, all class IoUs and class-alpha moments/counts. The same values
are in checkpoint best_spatial_summary and best_summary.json. Latest retains
the best summary separately from the current validation summary, including
after resume. EMA summaries correspond to the EMA inference weights in best.

The comparison table covers all 19 classes for Baseline, C.1b, C.2, C.3, C.4.
Sources are the actual best-epoch logs. References retain original precision
(three decimals, or six where detailed comparison log lines are available).

| Reference | Best mIoU | Epoch |
| --- | --- | --- |
| Baseline 8ch, logs/SJNet3_48 | .507000 | 90 |
| C.1b | .504596 | 9 |
| C.2 | .508253 | 18 |
| C.3 | .505478 | 9 |

C.3 best global alpha=.641107. No causal interpretation of alpha distributions
or automatic claim of performance improvement is made by the diagnostics.

## Unchanged Training Conditions

- Batch=3, AdamW, base weight decay=.005, alpha group weight decay=0.
- Max LR=.002, min LR=.00001, epoch 0-4 warmup, epoch 5-69 cosine.
- Maximum 70 epochs, early stopping enabled/patience=12/delta=.001/min_epochs=30.
- Torch and numpy seeds remain 0; other backend/random settings unchanged.
- Copy-Paste OFF, copy_paste_prob=0, w_paste=0, fixed pointnet_16ch features.
- Resume=None, pretrained=None. No initialization from C.2/C.3 checkpoints.
- train/dataset/post config sections equal C.2/C.3; model differs only by name.

## Artifacts And Formal Command (Not Executed)

Preflight logs/reports: logs/StepC4_preflight_20261012/c2/ and c4/.
C.4 also has smoke_best_summary.json, a test artifact, NOT a training result.
Full smoke checkpoints were temporary and are not used for the formal run.

The new formal log directory has NOT been created:

```bash
.venv/bin/python -B -u train_pointnet_hybrid_gated_local3_spatial_alpha.py \
  --dataset SemanticKitti \
  --arch_cfg config/arch/sjnet_pointnet_gated_local3_spatial_alpha_stablelr.yaml \
  --data_cfg config/labels/semantic-kitti.yaml \
  --log logs/SJNet3_PointNet_stepC4_local3_spatial_alpha_gated_lr2e3_bs3
```
