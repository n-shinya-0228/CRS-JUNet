# Step C.2: Gated PointNet Hybrid + Local 3x3 Context

Implementation and preflight are complete. Full training has NOT been started.
The previous C.1b training remains stopped and its logs/checkpoints are intact.

## Files

New source/config/test files:

- lib/models/SJNet3_PointNet_Gated_Local3.py
- train_pointnet_hybrid_gated_local3.py
- config/arch/sjnet_pointnet_gated_local3_stablelr.yaml
- lib/utils/pointnet_local3_diagnostics.py
- tests/test_pointnet_local3.py
- tests/smoke_pointnet_local3.py
- memo/stepC2_local3.md

One existing file changed: lib/trainer/trainer_PointNet_Gated_StableLR.py.
It supports both model/experiment identities, retains C.1b checkpoint
compatibility, adds C.2 diagnostics/references, and logs when early stopping
is deferred by min_epochs. It is reused, not duplicated.

The original C.1b model, config, entry point, dataset, parser, RingConv and
experiment log have unchanged SHA-256 hashes. A copy of the original trainer
and before/after hashes are in logs/StepC2_preflight_20261007/.

## Exact Architecture Difference

Only the PointNet branch receives a new residual block, directly after
point_stem and before the existing gate:

```python
point_base = point_stem(point)
local = ReLU(InstanceNorm2d(RingConv2d(point_base)))
point_local = point_base + local
gate = point_gate(torch.cat([hand_feat, point_local], dim=1))
point_gated = point_local * gate
fusion_input = torch.cat([hand_feat, point_gated], dim=1)
```

The block is exactly RingConv2d(16,16,3,bias=False), followed by
InstanceNorm2d(16,affine=True), then ReLU(inplace=True). There is one block,
no alpha, no new dropout and no new attention. Existing RingConv provides
circular angular padding and zero radial padding. No change was made to
the handcrafted stem, sigmoid gate, fusion stem, mask branch, encoder,
ASPP, decoder, CSWin, auxiliary heads or boundary head. Initial values of
all shared parameters match C.1b under the same seed.

## Shapes Verified On Real Data

| Tensor | Shape |
| --- | --- |
| Dataset features | [3,23,512,512] |
| Model input | [3,24,512,512] |
| hand | [3,7,512,512] |
| point | [3,16,512,512] |
| mask | [3,1,512,512] |
| hand_feat | [3,16,512,512] |
| point_base | [3,16,512,512] |
| local | [3,16,512,512] |
| point_local | [3,16,512,512] |
| gate | [3,16,512,512] |
| Fusion input | [3,32,512,512] |
| Fusion output | [3,48,512,512] |
| logits | [3,20,512,512] |

## Mask Audit: Important Caveat

C.1b does NOT hard-zero learned features at unobserved cells. Its existing
rule, preserved verbatim in C.2, is:

```python
s0 = s0 * (1.0 + torch.tanh(stem_mask(mask)))
```

This is learned modulation, not guaranteed suppression when mask=0.
gate_scale is still registered but unused. No extra mask has been added
to the residual branch, gate or skip paths.

The raw PointNet input is zero at unobserved cells, confirmed in the smoke
test. InstanceNorm/ReLU already produces nonzero point_base features there,
and Local3 spreads further context there. In the first C.2 validation batch:

| At Unobserved Cells | Mean | Std | Nonzero Feature-Element Fraction |
| --- | --- | --- | --- |
| Raw PointNet input | 0 | 0 | 0 |
| point_base | 0.088349 | 0.123582 | 0.562500 |
| local residual | 0.193947 | 0.471513 | 0.545257 |
| point_local | 0.282296 | 0.484696 | 0.759284 |

These latent features DO enter gate/fusion; the observation mask remains
unchanged and does not label them as observed. It would be incorrect to claim
that existing masking removes all such features. C.2 preserves the requested
single architectural difference. Hard-zeroing the residual or fused features
would introduce an additional masking intervention and is not implemented.
This caveat should be considered before starting the full experiment.

## Unchanged Training Conditions

The train, dataset and post sections are identical to C.1b. Model config
differs only in its name; extra experiment fields are diagnostic references.

- Batch size 3, 19,130 train scans, 6,376 batches/epoch.
- AdamW; max LR 0.002, min LR 0.00001, weight decay 0.005.
- Epoch 0-4 linear warmup; epoch 5-69 cosine, maximum 70 epochs.
- Actual LR preview 0-5: 0.0002, 0.00056, 0.00092, 0.00128, 0.00164, 0.002.
- Same two-stage scheduler, no constant phase; step after each training epoch.
- Early stopping: patience 12, min_delta 0.001, min_epochs 30 (zero-based).
- Bad-epoch count may exceed 12 before epoch 30. A deferred-stop log now
  explicitly explains continued training; stopping logic is unchanged.
- Copy-Paste probability=0.0; w_paste=0.0; paste_mask sum=0.0.
- Same losses, class weights, Focal gamma, Lovasz, aux decay, boundary loss,
  EMA, augmentation, resolution, and dropout.
- Existing pointnet_16ch/*.pt features are fixed. No PointNet encoder
  training or feature regeneration is performed.
- Fresh experiment: resume=None, pretrained=None.

## Preflight Results (2026-10-07)

24 unit/regression tests passed. Syntax compilation passed. Tests cover
residual/gate wiring, matching shared initialization, all five outputs with
Local3 bypassed, mask preservation, angular padding, finite gradients,
read-only diagnostic hooks, C.1b/C.2 checkpoint isolation and deferred early
stopping. Losses, EMA, train_epoch, scheduler and EarlyStopping code were also
AST-compared against the pre-edit trainer and are unchanged. An actual C.1b
epoch-28 checkpoint still passes the updated resume validator.

Separate CUDA processes tested C.1b and C.2 on the same RTX 5060 Ti (16 GB),
each with three real train batches and two validation batches, batch size 3.
All forward/backward passes, losses, feature checks and gradients were finite;
there was no OOM or skipped optimizer step. C.2 checked 332 gradient tensors
on each update, including every Local3 parameter. Train losses were 5.170988,
5.041318, 4.914346. These are startup smoke losses, not accuracy results.

| Measure | C.1b | C.2 |
| --- | --- | --- |
| Total parameters | 28,874,250 | 28,876,586 |
| Trainable parameters | 28,874,250 | 28,876,586 |
| Train peak allocated GiB | 13.826466 | 13.914830 |
| Train peak reserved GiB | 14.349609 | 14.449219 |
| Validation peak allocated GiB | 10.760815 | 10.807534 |

Increase: 2,336 parameters (2,304 convolution weights + 32 normalization
parameters) and 90.485 MiB peak allocated training memory. Peaks were measured
with reset_peak_memory_stats() and max_memory_allocated(). Batch size was
never reduced. Full-run peak usage may depend on subsequent inputs.

Validation-batch-0 diagnostics after the three C.2 smoke updates:

| Feature | Mean | Std |
| --- | --- | --- |
| point_base | 0.214111 | 0.659863 |
| local residual | 0.261843 | 0.644139 |
| point_local | 0.475954 | 1.017606 |

Gate mean=0.504219, std=0.095868, min=0.000089, max=0.998084.
Fraction <=0.01: 0.037003%; fraction >=0.99: 0.002964%.
It is not broadly saturated in this startup test. These statistics are
logged only for validation batch 0, detached from autograd and unused by loss.

Artifacts: logs/StepC2_preflight_20261007/c1b/ and local3/ each contain
smoke.log and report.json. No preflight weights will be used for training.

## Comparison References

Startup/validation logs retain Baseline 8ch ~0.507, Naive Hybrid ~0.497,
C.1=0.497326, C.1b=0.504596 at epoch 9, and a 0.510 clear-improvement target.
The C.1b class references below come from its epoch-9 log (rounded to 3
decimals). Each requested class is compared on every validation.

| Class | C.1b IoU |
| --- | --- |
| car | 0.907 |
| bicycle | 0.238 |
| motorcycle | 0.365 |
| truck | 0.610 |
| other-vehicle | 0.357 |
| person | 0.412 |
| bicyclist | 0.454 |
| motorcyclist | 0.000 |
| parking | 0.495 |
| terrain | 0.790 |
| pole | 0.451 |
| traffic-sign | 0.422 |

## Full Training Command (Not Executed)

```bash
.venv/bin/python -B -u train_pointnet_hybrid_gated_local3.py \
  --dataset SemanticKitti \
  --arch_cfg config/arch/sjnet_pointnet_gated_local3_stablelr.yaml \
  --data_cfg config/labels/semantic-kitti.yaml \
  --log logs/SJNet3_PointNet_stepC2_local3_gated_lr2e3_bs3
```

The entry point refuses an existing log directory unless resuming a matching
C.2 checkpoint. It never resumes C.1b as C.2. Latest/best checkpoints retain
the existing optimizer, scheduler, scaler, EMA and early-stopping state schema
with a distinct C.2 experiment/model identity.
