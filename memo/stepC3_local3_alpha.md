# Step C.3: Gated PointNet Hybrid + Local3 + Learnable Scalar Alpha

Implementation and preflight report, 2026-10-09. Full training has NOT started.
Smoke-test models are discarded; the formal run starts from scratch.

## 1. New Files

- lib/models/SJNet3_PointNet_Gated_Local3_Alpha.py
- train_pointnet_hybrid_gated_local3_alpha.py
- config/arch/sjnet_pointnet_gated_local3_alpha_stablelr.yaml
- lib/utils/pointnet_alpha_diagnostics.py
- tests/test_pointnet_local3_alpha.py
- tests/smoke_pointnet_local3_alpha.py
- memo/stepC3_local3_alpha.md

Preflight artifacts are separate in logs/StepC3_preflight_20261009/.

## 2. Existing File Changes And Preservation

Only lib/trainer/trainer_PointNet_Gated_StableLR.py was changed: additive C.3
model/experiment selection, alpha optimizer groups, alpha diagnostics, alpha
checkpoint metadata and best-comparison logs. The trainer is shared rather
than duplicated. C.1b/C.2 still use their original model and single AdamW group.

The C.2 model, train entry, config, Dataset, Parser and diagnostics are unchanged.
C.2 log.txt, best_val.path and latest.path hashes are unchanged as well.
The original shared trainer is backed up as
logs/StepC3_preflight_20261009/trainer_StableLR_before_C3.py.
Protected file hashes are in the same directory's protected_files.sha256.

The actual C.2 latest.path still passes the shared resume validator (epoch 30,
best mIoU 0.508252934833837, best epoch 18). C.3 rejects C.2 checkpoints and
refuses existing log directories for a fresh run.

## 3-6. Scalar Placement, Initialization, Shape And Residual

The existing C.2 model's entire forward is inherited without modification.
Only point_context is wrapped to replace its residual addition:

```python
point_base = point_stem(point)
local_residual = point_context.local(point_base)
point_local = point_base + local_alpha * local_residual
```

The same initialized C.2 local block is reused:
RingConv2d(16,16,3,bias=False) -> InstanceNorm2d(16,affine=True) -> ReLU(inplace=True).
No extra convolution, dropout, attention or masking is introduced.

The parameter is registered once as point_context.local_alpha and is also
accessible as model.local_alpha. It is nn.Parameter(float32), shape [],
initial value 1.0, unconstrained (no clamp/sigmoid/softmax).

Gate input is concat(hand_feat, point_local). Its output multiplies point_local,
then concat(hand_feat, point_local * gate) enters the unchanged 32->48 fusion.
Handcrafted stem, mask branch, backbone, decoder and all outputs are unchanged.

All shared initial weights and the RNG state match C.2 with the same seed.
CPU regression tests compare all five complete model outputs exactly at
alpha=1 for mixed, all-zero and all-one masks. Real CUDA smoke asserts that
point_base + alpha * local_residual equals point_base + local_residual at
initial alpha=1 (atol=1e-6).

Mask behavior remains the existing soft modulation:
`s0 *= 1 + tanh(stem_mask(mask))`. Alpha does not zero unobserved point_base
features, and no hard mask is added. Nonzero latent features at unobserved
cells still enter gate/fusion, just as in C.2.

## 7-8. Optimizer Membership And Weight Decay

AdamW group 0 contains all 28,876,586 existing C.2 parameters, in their original
order, with weight_decay=0.005. Group 1 contains only the scalar alpha, with
weight_decay=0.0. No parameter is duplicated or omitted.

Both groups use the same LR and the same scheduler. Unit tests verify the LR
of both groups across the full schedule and verify that zero alpha gradient
does not cause weight-decay shrinkage. Alpha's optimizer moments are included
in resume checkpoints.

## 9-10. Parameter Counts

| Count | C.2 | C.3 | Difference |
| --- | --- | --- | --- |
| Total | 28,876,586 | 28,876,587 | +1 |
| Trainable | 28,876,586 | 28,876,587 | +1 |

## 11. Shapes Verified On Real Data

| Tensor | Shape |
| --- | --- |
| Dataset features | [3,23,512,512] |
| Model input | [3,24,512,512] |
| hand / point / mask | [3,7,512,512] / [3,16,512,512] / [3,1,512,512] |
| hand_feat | [3,16,512,512] |
| point_base | [3,16,512,512] |
| local_residual | [3,16,512,512] |
| alpha | [] |
| scaled_local_residual | [3,16,512,512] |
| point_local | [3,16,512,512] |
| gate | [3,16,512,512] |
| Fusion input / output | [3,32,512,512] / [3,48,512,512] |
| logits | [3,20,512,512] |

## 12-13. Gradients And Forward/Backward

Each model completed three real train batches and two validation batches on
an RTX 5060 Ti (16 GB), batch size 3. C.3 checked 333 gradient tensors on each
update, including alpha. All gradients, losses, features and gate values were
finite. There were no OOMs or skipped optimizer updates.

First C.3 smoke run, alpha gradients after existing unscale/gradient clipping:

| Update | Alpha Before Update | Alpha Gradient |
| --- | --- | --- |
| 1 | 1.000000000 | +0.023702964 |
| 2 | 0.999800026 | +0.010402910 |
| 3 | 0.999617457 | -0.000698122 |

Training alpha after three updates: 0.999479830.
EMA alpha used for validation: 0.999989092.
These are smoke-test values, NOT a trained best model or evidence of accuracy
improvement. Formal training will initialize alpha to 1.0 again.

31 unit/regression tests and py_compile passed. Tests include scalar-only
parameter changes, inherited forward, identical initial weights/RNG, negative
and large alpha without clamping, gate/fusion wiring, read-only diagnostics,
all optimizer groups, best/raw/EMA checkpoint round trips and old-experiment
isolation. AST comparisons also confirm unchanged train_epoch, loss functions,
EMA functions, scheduler and EarlyStopping implementations.

## 14. GPU Peak Memory

Peaks use reset_peak_memory_stats() and max_memory_allocated(), measured in
separate CUDA processes under unchanged C.2 backend settings.

| First Measurement | C.2 | C.3 |
| --- | --- | --- |
| Train peak allocated GiB | 13.914830 | 12.082395 |
| Train peak reserved GiB | 14.449219 | 13.056641 |
| Validation peak allocated GiB | 10.807534 | 10.807261 |

A second independent C.3 smoke run (c3_repeat/) also passed and measured exactly
12.082395 GiB allocated / 13.056641 GiB reserved during training. The observed
C.3 peak is lower, but this is NOT evidence of an intrinsic memory saving from
alpha. The cause of the difference has not been isolated. Startup peaks include
backend/workspace allocations; C.2's existing cuDNN benchmark settings are
unchanged. Future batches can have different peaks. Batch size 3 was maintained
throughout.

## 15-17. Unchanged Training Conditions

- Batch size 3, 19,130 scans, 6,376 training batches per epoch.
- AdamW, max LR=0.002, min LR=0.00001, existing weight decay=0.005.
- Epoch 0-4: linear warmup; epoch 5-69: cosine decay; maximum 70 epochs.
- LR preview epoch 0-5: .0002, .00056, .00092, .00128, .00164, .002.
- Early stopping enabled, patience=12, min_delta=.001, min_epochs=30 (zero-based).
- Existing deferred-stop logging before min_epochs remains unchanged.
- Copy-Paste OFF, copy_paste_prob=0.0, w_paste=0.0, paste_mask sum=0.0.
- Existing PointNet encoder/features remain fixed; pointnet_16ch files reused.
- train/dataset/post config sections are exactly equal to C.2; model section
  differs only in name. All loss weights, augmentation and EMA are unchanged.
- Same explicit seeds: torch.manual_seed(0), numpy.random.seed(0). No new
  Python random seed or backend determinism setting was added.
- Resume=None, Pretrained=None. No C.2 weights are used for initialization.

## Diagnostics And Best Model Reporting

Each validation reports Local alpha for the evaluated model (EMA by default)
and the separate training alpha. Extreme values <=-5 or >=10 trigger a warning,
not a clamp. Best saves and end-of-training logs report Best local alpha from
the corresponding inference model, not from the last/raw model. Checkpoints
store local_alpha_train, local_alpha_eval and best_local_alpha metadata, as
well as the actual parameter in raw/EMA state dictionaries.

Validation batch 0 logs the existing gate/local/unobserved-cell statistics and
alpha, point_base, local_residual, scaled_local_residual, point_local statistics.
The contribution ratio is mean(abs(scaled_local_residual)) /
(mean(abs(point_base)) + 1e-8). Hooks are detached, removed after use, and never
modify the forward result or loss.

First C.3 smoke validation-batch-0 statistics (EMA):

| Feature | Mean | Std |
| --- | --- | --- |
| point_base | 0.214113 | 0.659863 |
| local_residual | 0.261843 | 0.644160 |
| scaled_local_residual | 0.261840 | 0.644153 |
| point_local | 0.475953 | 1.017645 |

Alpha=0.999989092, local contribution ratio=1.222906019.
Gate mean=.504216, std=.095869, min=.000089, max=.998087.

## Comparison References

Baseline is NOT N/A: per the user's clarification, it uses epoch 90 of
logs/SJNet3_48/log.txt (8-channel config, mIoU .507 in the log).
The user confirmed this reference explicitly: log_dir=logs/SJNet3_48,
best_epoch=90, mIoU=.507, acc=.891, and all 19 evaluated class IoUs.
C.1b uses epoch 9 and C.2 uses epoch 18. Best saves log all four columns,
Baseline / C.1b / C.2 / C.3, for these 12 classes. Baseline and C.1b values
are rounded to 3 decimals in their source logs; C.2 comparisons use 6 decimals.

| Class | Baseline Epoch 90 | C.1b Epoch 9 | C.2 Epoch 18 |
| --- | --- | --- | --- |
| car | .947 | .907 | .912672 |
| bicycle | .247 | .238 | .203021 |
| motorcycle | .313 | .365 | .369755 |
| truck | .645 | .610 | .674159 |
| other-vehicle | .306 | .357 | .293603 |
| person | .411 | .412 | .415712 |
| bicyclist | .492 | .454 | .471364 |
| motorcyclist | .000 | .000 | .000000 |
| parking | .495 | .495 | .480234 |
| terrain | .723 | .790 | .782028 |
| pole | .487 | .451 | .479191 |
| traffic-sign | .474 | .422 | .483528 |

Startup references: Baseline=.507000, C.1b=.504596, C.2=.508253 (epoch 18).
First target: exceed .508253; clear-improvement reference: .510.
No interpretation of learned alpha or final accuracy is automated.

## 18-19. Formal Log Directory And Command (Not Executed)

```bash
.venv/bin/python -B -u train_pointnet_hybrid_gated_local3_alpha.py \
  --dataset SemanticKitti \
  --arch_cfg config/arch/sjnet_pointnet_gated_local3_alpha_stablelr.yaml \
  --data_cfg config/labels/semantic-kitti.yaml \
  --log logs/SJNet3_PointNet_stepC3_local3_alpha_gated_lr2e3_bs3
```

The formal log directory has not been created. Only preflight results are
written to logs/StepC3_preflight_20261009/c2/, c3/ and c3_repeat/, with smoke.log
and report.json in each. There is no full C.3 training result or learned best
alpha yet.
