# Step D.4: Teacher-conditioned Student Fine-tuning

This is an exploratory/development experiment on sequence 08, not an independent
test. That sequence was also used to select earlier checkpoints and the fixed
alpha table. No improvement is claimed before training and final evaluation.

## Read-only References

- Teacher and Student start from C.2 epoch-18 **best EMA**, not latest:
  `logs/SJNet3_PointNet_stepC2_local3_gated_lr2e3_bs3/best_val.path`.
- Checkpoint SHA256: `23b581814e920afe8f463666f1cbf6d7a3e1b3d131577bef82239bb7fdb5fc9b`.
- Initial Teacher/Student/Student EMA state SHA256:
  `aa0aa93de1b0b85987843ef074ef36c1a11bec850a786d8352da13e7c6d13cc4`.
- Each network has 28,876,586 parameters, loaded with `strict=True`.
- D.2b reference mIoU: **0.5097046791309** on all 4,071 sequence-08 scans.
- D.2 source, D.1b table, C.2 checkpoint, original lib/config files are hashed
  and never overwritten. Native D.4 resume checks these identities again.

The historical `trainer_PointNet_Gated_StableLR.py` is absent in the current
checkout. Its implementation was inspected in commit
`396385a5051d3358b9d4c471268ca8be4e09ff3d`. Its loss/EMA/boundary helpers are
AST-identical to current `trainer_PointNet.py`; D.4 inherits those helpers without
calling its training initializer. The historical scheduler and EarlyStopping are
copied without behavioral changes into `lib/utils/pointnet_d4_training.py`.

## Model and Loss

The same augmented float32 `[B,24,512,512]` tensor goes first to the frozen,
eval-only Teacher, then to the trainable Student. Channels are 7 handcrafted,
16 precomputed PointNet, 1 observation mask. Teacher forward uses `no_grad`,
not `inference_mode`. Raw main logits at global alpha 1.2 generate the original
D.2 soft map with temperature 1; unobserved cells use alpha 1.2. Ground truth is
never an input to this calculation. The detached map scales the Student's local
residual; its existing network is otherwise unchanged.

Only Student enters AdamW and EMA. Teacher parameters have no gradients and its
state hash is checked after smoke, every train epoch and every validation.
The 20-value D.2 table is a fixed tuple, not a parameter/buffer/predictor.

Original C.2 losses: content-normalized weighted Focal gamma 1, ignore 0,
main 1, aux2 0.30, aux4 0.15, aux8 0.10, Lovasz 0.50, masked boundary BCE 0.20.
Paste weight is 0, paste masks must be empty, Parser/Dataset CP probability is
0. No paste sampling, region mixing, KL, pseudo-label or consistency loss.
Existing aligned roll/flip/dropout augmentation remains enabled for training.

## Settings and Gates

- AdamW LR 1e-4, weight decay 0.005, clip norm 5, Student EMA 0.99, seed 0.
- 20 fine-tuning epochs, 1 warmup + 19 cosine, minimum LR 1e-6.
- Original scheduler semantics: first epoch LR 1e-5, second epoch LR 1e-4;
  scheduler advances after training, and validation/checkpoints follow it.
- EMA early stopping: patience 6, minimum 8 epochs, min_delta 0.0003.
- Best checkpoint uses the strict maximum, independently of min_delta, and
  includes epoch0 as a fallback if no trained checkpoint improves it.
- FP32, no AMP/scaler. **C.2 training used BF16; D.2 evaluation used FP32.**
- First try actual batch 3. Only a real CUDA OOM permits microbatch 2 then 1,
  with groups `[2,1]` or `[1,1,1]` per optimizer update. No BatchNorm modules.
  Microbatching is not mathematically identical for batch-level Lovasz/Focal
  normalization and dropout RNG; actual choice is recorded in config/metadata.
- CUDA allocator is capped at 85% of VRAM to leave display/driver headroom on
  this display-attached Windows/WSL GPU. The initial uncapped disposable trial
  used about 15.6 GiB including display and did not finish its first update in
  over three minutes; it was terminated without formal updates. Its log is
  retained in the `_preflight_uncapped_20261019` directory. This cap changes
  neither precision nor model/loss and allows explicit OOM fallback rather than
  relying on slow driver host-memory fallback.
- Original train loss/accuracy/mIoU metric definitions are retained; final
  global train confusion metrics are also saved. Validation accumulates the
  full confusion matrix and excludes class 0.
- Validation uses batch 3 and original height correction (7->2, 8->3 only for
  final predictions if max_z < -0.8). Teacher maps precede postprocessing.
- Training workers are recreated and seeded per epoch for reproducible
  epoch-boundary resume; validation worker behavior remains unchanged.
- D.4 uses cuDNN deterministic=True and benchmark=False. D.2 used benchmark=True;
  a disposable preflight with that setting matched all five outputs inside one
  process but its Teacher alpha map differed by up to 0.000954 from the saved
  D.2 map. Timing-based algorithm selection is disabled to stabilize frozen
  Teacher maps across process restarts. FP32/TF32 policy is otherwise unchanged,
  and the full D.2b reproduction gate still must pass. The diagnostic trial is
  retained in `_preflight_benchmark_20261019`, without formal Student updates.

Before formal updates, disposable GPU smoke checks all five outputs exactly
against D.2 and performs at least three complete Student optimizer/EMA updates.
Those weights are discarded. A freshly initialized formal Student must then
pass full epoch0 evaluation within 1e-4 of D.2b. Failure stops the run. A separate
epoch0 checkpoint is saved only after the gate passes. Each soft validation
also checks Teacher map examples and observed/unobserved alpha distributions
against D.2, without using GT to generate them.

## Commands

New run (refuses an existing output directory):

```bash
.venv/bin/python -B -u train_pointnet_teacher_student_d4.py \
  --dataset SemanticKitti \
  --arch_cfg config/arch/sjnet_pointnet_teacher_student_d4_cp0.yaml \
  --log logs/SJNet3_PointNet_stepD4_soft_teacher_student_cp0_lr1e4
```

`--prepare-only` performs all preflight/epoch0 gates and exits with zero formal
Student updates. Resume that prepared run, or resume a later interruption:

```bash
.venv/bin/python -B -u train_pointnet_teacher_student_d4.py \
  --dataset SemanticKitti \
  --resume logs/SJNet3_PointNet_stepD4_soft_teacher_student_cp0_lr1e4/latest.path
```

Re-evaluate the D.4 best EMA Soft, best raw Student Soft, and best EMA global 1.2:

```bash
.venv/bin/python -B -u train_pointnet_teacher_student_d4.py \
  --dataset SemanticKitti --evaluate
```

Resume restores raw Student, Student EMA, AdamW, scheduler, early stopping,
epoch, strict best, history, RNG, and frozen Teacher identity. It resumes at an
epoch boundary; an interrupted epoch is rerun, never partially checkpointed.

## Outputs

All new artifacts are under
`logs/SJNet3_PointNet_stepD4_soft_teacher_student_cp0_lr1e4/`:

- `train_log.txt`, `status.json`, errors with traceback in `error.json`.
- Actual `arch_cfg.yaml`, `data_cfg.yaml`, table/metadata/validation manifest.
- `smoke_report.json`: attempted microbatches, 3-step checks, peak CUDA memory.
- `epoch0_validation.json`, `epoch0_class_comparison.csv`.
- `epoch0_baseline.path`, strict `best_val.path`, resumable `latest.path`.
- `validation_history.csv`, full history JSON, per-validation JSON diagnostics.
- After training: `best_ema_soft.json`, `best_raw_soft.json`,
  `best_ema_global12.json`, all-19-class `best_class_comparison.csv`,
  `comparison.md`, `results.json`, and bundled `inference_package.path`
  (frozen Teacher + best EMA Student + fixed table/configs).

Final diagnostics and report generation run automatically after max epochs or
early stopping. A running/prepared status is not a completed experiment.

The final inference package is self-contained; no original C.2 file is needed
to load its weights. With `x` in the recorded channel order and dtype:

```python
import torch
from lib.models.SJNet3_PointNet_Gated_Local3_TeacherStudent import (
    SJNet3_PointNet_Gated_Local3_TeacherStudent,
)

package = torch.load(
    "logs/SJNet3_PointNet_stepD4_soft_teacher_student_cp0_lr1e4/inference_package.path",
    map_location="cpu", weights_only=False,
)
system = SJNet3_PointNet_Gated_Local3_TeacherStudent(
    in_channels=24, nclasses=20, drop=package["arch_cfg"]["model"]["dropout"],
)
system.initialize(package["teacher"])
system.student.load_state_dict(package["student"], strict=True)
system.to("cuda").eval()
with torch.no_grad():
    outputs = system(x.to("cuda", dtype=torch.float32))
```

Use the recorded D.4 cuDNN/FP32 settings for numeric reproducibility and the
original final-prediction height correction/IoU evaluator for metric parity.

## Verification

The 7 new CPU unit tests pass, covering five-output equivalence, masks, detached
maps, Teacher eval/gradients, 3 optimizer/EMA updates, original loss formula,
CP=0, effective batch splitting, scheduler/early stopping, RNG/identity restore,
atomic checkpoints, read-only output protection and the reproduction gate.
The automated final diagnostics/package/report path is also exercised on CPU.
The 14 existing fixed/class-alpha tests also pass. Existing Matplotlib tests
emit library Pyparsing deprecation warnings; they do not affect training.
The benchmark-enabled diagnostic GPU smoke passed with microbatch 2,
accumulated as 2+1 for effective batch 3.
All five D.2b outputs matched exactly (max absolute difference 0); three real
optimizer updates on nine augmented scans changed Student/EMA but not Teacher.
Peak allocated/reserved: **12.7766 / 13.4434 GiB**. FP32 microbatch 3 had CUDA OOM
under the recorded 85% allocator limit. Disposable weights were discarded, and
the formal Student/EMA were freshly reinitialized to the C.2 EMA hash above.
One preflight-only device-index API error was corrected (`cuda` -> `cuda:0`);
its traceback is retained in `_preflight_devicefix_20261019` with no updates.
Final-setting GPU smoke and epoch0 results are recorded by the running
entrypoint before formal training.

Final-setting smoke also passed: 3 optimizer updates / 9 scans, all five output
max differences 0, loss 0.3741059734, Teacher unchanged, disposable Student/EMA
changed and discarded. Peak allocated/reserved **12.3137 / 13.5039 GiB**;
update portion took 5.99 s (startup/short-run timing, not a full-epoch estimate).
The WSL driver's OOM message reports an obviously invalid enormous
"non-PyTorch memory" figure; use actual allocated/reserved CUDA counters instead.
The run is an independent background process, with stdout/stderr in `console.log`.

## Epoch0 Gate and Training Start

On 2026-10-19 the complete 4,071-scan / 1,357-batch gate passed:

- Epoch0 Student EMA Soft mIoU: **0.5096905511292307** (50.969055%).
- D.2b Soft: 0.5097046791309 (50.970468%).
- Signed difference: **-0.0000141280016693 mIoU**, or **-0.0014128 pp**;
  absolute difference is below the required 0.0001 tolerance.
- Epoch0 evaluation wall time: approximately 918.2 seconds.
- All observed/unobserved/all Teacher alpha distribution checks passed against
  D.2's original definitions and saved statistics.
- `epoch0_baseline.path`, initial `best_val.path`, and `latest.path` were saved
  before any formal Student optimizer update. Class-level numeric differences
  are retained in `epoch0_class_comparison.csv`, not rounded or forced to zero.
- Formal fine-tuning started at **2026-10-19 17:03:56 JST**, epoch 1.

Training is still in progress. The initial best is epoch0; this is not a
fine-tuned result or evidence of improvement. `results.json` and `comparison.md`
are produced only after training and the final three complete evaluations.

At 17:05:15 JST, formal epoch 1 reached 50 optimizer updates / 101 microbatches:
loss 0.336709, original cumulative-metric accuracy 0.940793 and mIoU 0.718423,
79.2 seconds elapsed. These train metrics are not validation results. This short
sample projects roughly 2.8 h training + 0.26 h validation per epoch, around
3 h/epoch, subject to data/GPU load. Minimum 8 to maximum 20 epochs therefore
suggest about 24-61 h, plus final diagnostics; this is an estimate, not a deadline.
GPU usage during formal training was 92%, about 14,061 MiB including display and
driver. Process CPU time averaged 132% of one logical core (not 132% of the host).
