# Step D.5: Shared-weight Two-pass Soft Alpha Scratch

New experiment only. Existing C.2/C.3/C.4/D.1/D.2/D.4 source, checkpoints and
logs are read-only. This experiment is exploratory sequence-08 development
evaluation, not independent test performance: that sequence also selected the
fixed alpha table. No alpha retuning or learned alpha predictor is allowed.

## Model and Initialization

`SJNet3_PointNet_Gated_Local3_SharedSoftAlpha` inherits the C.2-compatible D.2
external-alpha model. It adds no parameters, heads or additional network.
There is one trainable parameter set (28,876,586 parameters) and one ordinary
EMA copy. Stage 1 and Stage 2 call the same model object, not separate models.

Stage 1 temporarily switches all modules to eval, uses `no_grad` (not
`inference_mode`), and runs global alpha=1.2 in FP32. Raw main logits yield the
original D.2 temperature-1 softmax-weighted fixed table. Ground truth and height
correction are not inputs to this calculation. Unobserved cells use alpha=1.2.
The logits/map are detached; prior module modes are restored even on exception.
Stage 2 uses the same 24ch tensor and scales the existing local residual by the
map. Original masks, gates, RingConv, skips and all five outputs are unchanged.

All formal segmentation weights are random, using C.2's constructor path and
seed 0. The initial hash must equal a freshly generated RANDOM C.2 hash and
differ from C.2's trained EMA hash. Formal Trainer construction never loads
trained C.2/D.2/D.4 weights. The only trained C.2 loading is in a separate,
disposable regression test; its models are deleted before scratch smoke.
The fixed, precomputed PointNet 16ch input remains unchanged: this is scratch
training of the segmentation network, not end-to-end PointNet pretraining.

## C.2 Learning Conditions

Values come from `config/arch/sjnet_pointnet_gated_local3_stablelr.yaml` and are
checked at startup. The historical StableLR Trainer was inspected at commit
`396385a5051d3358b9d4c471268ca8be4e09ff3d` because that file is absent from the
current checkout. Its unchanged loss/boundary/EMA methods are reused via the
current base Trainer, and its exact scheduler/early-stop helpers via the
read-only D.4 support module. The D.4 Teacher initialization is never called.

- AdamW base LR .002, weight decay .005, warmup 5 + cosine 65, min LR **1e-5**.
- 70 epochs, C.2 zero-based epoch numbering 0..69. Minimum stopping epoch 30,
  patience 12, significant min_delta .001. Strict Stage 2 EMA best is separate
  from the significant-improvement threshold.
- Warmup LR starts .0002, then .00056/.00092/.00128/.00164/.002 at epoch 5.
- Original C.2 BF16 autocast + GradScaler for Stage 2 training, clip norm 5;
  Stage 1/alpha and both validation passes use FP32. D.4's FP32-only fine-tuning
  settings, LR 1e-4 and 20-epoch budget are not reused.
- EMA .99 on the single shared model only. No fixed Teacher exists.
- Copy-Paste 0, all paste masks must be zero, no sampling/mixing/paste loss.
  Original aligned augmentation and the same Dataset/Parser are reused.
- Original content-normalized weighted Focal gamma 1, ignore 0; main 1,
  aux2 .30, aux4 .15, aux8 .10, Lovasz .50, masked boundary BCE .20. Aux2/aux4
  linearly decay in epochs 50..60 then become zero, exactly as C.2.
- Only Stage 2 has loss; no Stage 1 supervised/KL/consistency/pseudo-label loss.
- First try native batch 3. Only actual OOM permits 2+1 or 1+1+1 accumulation,
  weighted by sample count. Weighted Focal/Lovasz reductions and dropout are
  not generally exactly equal to an unsplit batch. No BatchNorm exists in the
  production model. This caveat is recorded, not hidden by changing the loss.
- CUDA allocator limit .85 leaves display/driver memory on the WSL GPU;
  deterministic=True, benchmark=True and matmul precision=high match C.2.
- Training workers are seeded/recreated each epoch for epoch-boundary resume;
  validation worker behavior remains unchanged.

## Unified Logging

Logging is observational: it changes neither loss/gradients nor optimizer,
scheduler, EMA, checkpoint selection or metric definitions. Native batch and
accumulation both update progress at completed effective groups of 3 samples.
The denominator is 6,376 effective batches for 19,130 train scans, never the
microbatch count 12,752 when using 2+1.

```text
Epoch 00 | Learning rate: 2.000000e-04
Lr: 2.000e-04 | Epoch: [0][0/6376] | Loss current (average) | acc current (average) | IoU current (average)
Lr: ... | Epoch: [0][3000/6376] | ...
Lr: ... | Epoch: [0][6000/6376] | ...
```

Indices are zero-based as in C.2. Training metrics use Stage 2 outputs only:
current accuracy/IoU are from the cumulative confusion matrix after the
effective batch; the averages are sample-weighted means of these cumulative
metrics, exactly C.2's convention. Loss is sample-weighted. No per-microbatch
progress or alpha statistics are printed.

After full validation, logs show both Stage 1/2 mIoU and accuracy, Stage 2 loss,
Stage2-minus-Stage1 mIoU, LR, best Stage 2 score/epoch and early stopping state.
Both stages' 19 class IoUs are separately logged every epoch, including each
best epoch, and saved at full precision. Observed alpha mean/std/min/max are
one concise line. Detailed observed/unobserved/all statistics, histogram CSVs,
raw Stage 1 observed-class histogram and confusion matrices are saved separately.
Raw logits generate alpha before final-prediction height correction (7->2 and
8->3 at max_z < -.8). Both stages' reported metrics use that original correction.

All progress/error messages have JST timestamps and are mirrored to console
and `train_log.txt`. Resume appends; it never clears previous logs. Validation
progress is available as JSON without printing per-batch messages.

## Gates and Checkpoints

Before formal training: syntax + CPU tests; actual C.2 global-alpha and D.2b
two-pass regression on all five outputs; at least three scratch optimizer/EMA
updates; actual native checkpoint/RNG/scaler/optimizer/EMA resume plus another
disposable update. Every preflight model is then discarded. Formal construction
restarts seed 0, a new optimizer/scheduler/scaler and a random EMA.

`epoch0_random_init.path` stores the untrained state, internally epoch=-1,
next epoch=0. It contains no C.2 trained weights. A small real validation batch
checks Stage 1/2 shapes/probabilities/alpha/detachment; random initialization is
not expected to reproduce D.2b mIoU. No partial random-score best is invented.
`best_val.path` is first produced after the first complete trained validation.

Native checkpoints contain raw model, EMA, AdamW, scheduler, scaler, epoch,
strict best, early stopping, RNG, fixed table/configs/seed/CP=0/source identity.
Only matching native D.5 checkpoints can resume. Interrupted epochs restart
from the last full epoch boundary. Preflight identities cannot resume a formal
experiment. Initial/latest `model` is raw; best `model` is EMA, with raw model
still retained for resume, as in C.2.

## Commands and Outputs

New run (refuses existing directories):

```bash
.venv/bin/python -B -u train_pointnet_shared_soft_alpha_d5.py \
  --dataset SemanticKitti \
  --arch_cfg config/arch/sjnet_pointnet_shared_soft_alpha_d5_cp0.yaml
```

Prepare only with all GPU gates and zero formal updates: add `--prepare-only`.
Resume an initialized or interrupted run without deleting logs:

```bash
.venv/bin/python -B -u train_pointnet_shared_soft_alpha_d5.py \
  --dataset SemanticKitti \
  --resume logs/SJNet3_PointNet_stepD5_shared_soft_alpha_scratch_cp0/latest.path
```

Re-evaluate both Stage 1/Stage 2 of the completed best EMA:

```bash
.venv/bin/python -B -u train_pointnet_shared_soft_alpha_d5.py \
  --dataset SemanticKitti --evaluate
```

Output root: `logs/SJNet3_PointNet_stepD5_shared_soft_alpha_scratch_cp0/`.
Includes config/data/table/metadata/manifest; regression/smoke/initialization
reports; random-init/latest/best checkpoints; append-only train log; epoch CSV
and full JSON history; alpha CSVs/histograms and Stage 1 class histograms.
After training, the best EMA is re-evaluated on all 4,071 scans; automatic
`comparison.md`, `results.json`, all-19-class comparison CSV and an inference
package compare D.5 against the saved D.2b and other reference results.
A matched-budget scratch global-alpha=1.2 control remains future work.

## Verified Run: 2026-10-21 JST

- Syntax checks and **32 CPU tests** passed, including all previous tests and
  11 D.5 tests. The latter cover logging boundaries/format, effective batch
  denominators, read-only logging, unchanged optimizer/EMA/metrics with logging
  enabled/disabled, Stage 2 best selection, both stages' 19 class logs, timestamped
  append behavior, mode/BatchNorm restoration, sharing/gradients and final export.
- Actual C.2 global-alpha/D.2b Soft compatibility passed on real input: all five
  outputs had maximum absolute difference **0**. Regression models discarded.
- Scratch GPU smoke succeeded with **native batch 3**, not accumulation. Three
  optimizer updates plus one native checkpoint-resume update passed. Peak CUDA
  allocated/reserved **12.0418 / 13.2695 GiB**. No OOM occurred in this run.
- Initial Stage 1/2 output shapes were correct; alpha range 1.019858..1.325305,
  finite/detached with unobserved cells 1.2.
- Formal model/EMA and random C.2 state SHA256:
  `4ca99449969b24576dbd8d9166ab429a760400bd46decaeca8bf252fab1e4b49`.
  This differs from trained C.2 EMA:
  `aa0aa93de1b0b85987843ef074ef36c1a11bec850a786d8352da13e7c6d13cc4`.
- `epoch0_random_init.path` was checked: epoch -1, no AdamW state entries,
  raw/EMA hashes equal the random hash, pretrained_initialization=False.
  All 193 protected source/artifact hashes were unchanged.
- Formal scratch epoch 0 started at **13:28:30 JST**, from that untrained D.5
  initialization checkpoint, not from C.2/D.4 trained weights. PID 1347748.
  Resuming the prepared initial checkpoint preserves scratch initialization.
- First requested-format line at 13:29:00 JST:

```text
Lr: 2.000e-04 | Epoch: [0][0/6376] | Loss 5.1963 (5.1963) | acc 0.151 (0.151) | IoU 0.015 (0.015)
```

At 13:29:59 GPU use was 87%, memory 13,649 MiB including display/driver,
power about 120 W. Process-average CPU use was 111% of one logical core,
not 111% of the host. Cold smoke took 26.46 s for three updates, including
initial cuDNN autotuning; this is not a reliable steady epoch-time estimate.
Training is still in progress; full Stage 1/2 validation and best performance
are pending. These will be generated automatically after each full epoch.
