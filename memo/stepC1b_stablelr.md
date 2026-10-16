# Step C.1b: Gated PointNet Hybrid + Stable LR

## Scope

New files only:

- train_pointnet_hybrid_gated_stablelr.py
- lib/trainer/trainer_PointNet_Gated_StableLR.py
- config/arch/sjnet_pointnet_gated70_stablelr.yaml
- tests/test_pointnet_stablelr.py
- tests/smoke_pointnet_stablelr.py
- memo/stepC1b_stablelr.md

The model file SJNet3_PointNet_Gated.py is reused unchanged, including its
initialization, separate stems, sigmoid gate, RingConv fusion and backbone.
Parser_PointNet.py and SemanticKitti_PointNet.py are unchanged. PointNet
features are the existing fixed pointnet_16ch files; no encoder is trained.
The new trainer is a copy of Step C.1 with changes limited to LR scheduling,
early stopping, diagnostic logs, and experiment identification in checkpoints.
The existing Step C.1 trainer, entry point, config and logs are preserved.

AdamW, weight decay 0.005, batch size 3, dropout, loss/class weighting,
Focal gamma, Lovasz, auxiliary-loss weights/decay, boundary loss, EMA,
mask handling, resolution, and data augmentation are unchanged.
Copy-Paste probability and paste loss weight both remain zero.

## Learning Rate

The new config sets train.lr=0.002, min_lr=0.00001, wup_epochs=5,
cos_epochs=65 and max_epochs=70. build_scheduler() in the new trainer uses
SequentialLR with exactly two schedulers: LambdaLR and CosineAnnealingLR.
There is no constant phase. scheduler.step() is called once after training
each epoch, as in Step C.1.

The LR at the start of epoch e < 5 is max_lr * (0.1 + 0.9 * e / 5).
Cosine starts at epoch 5 with max_lr. T_max=65 makes the lower LR bound
1e-5 occur at the boundary after epoch 69; the last training epoch uses
about 1.11619e-5. The final scheduler boundary is not an extra training epoch.

Actual scheduler sanity-test output (not completed full training epochs):

| Epoch | LR |
| --- | --- |
| 0 | 2.000000e-4 |
| 1 | 5.600000e-4 |
| 2 | 9.200000e-4 |
| 3 | 1.280000e-3 |
| 4 | 1.640000e-3 |
| 5 | 2.000000e-3 |
| 6 | 1.998838e-3 |
| 7 | 1.995355e-3 |
| 8 | 1.989559e-3 |
| 9 | 1.981463e-3 |
| 10 | 1.971087e-3 |

The startup log labels these values as a preview. Separately, each actual
training epoch logs its current optimizer LR before processing any batches.

## Early Stopping

Patience 12, min_delta 0.001, min_epochs 30. To honor the explicit request
not to stop before epoch 30, Step C.1b disallows stopping while the zero-based
epoch number is below 30. A gain of at least min_delta over the significant
best resets patience. Strict best-checkpoint selection is independent, so
smaller gains can still update best_val.path. Current mIoU, strict best,
best epoch, patience count and significant-best reference are logged.

Both latest and best checkpoints retain the early-stopping state and the
new two-stage scheduler state. Earlier Step C / C.1 checkpoints cannot be
resumed with this entry point. This experiment starts from scratch with
resume=None and pretrained=None.

## Diagnostics And Comparisons

Validation reads mean/std/min/max of the existing sigmoid gate via a
temporary forward hook for only the first batch. The hook detaches its
statistics, returns no replacement output, consumes no random values and
is removed immediately. Model forward, weights and predictions are unchanged.
The standard deviation is population std over all batch/channel/spatial
elements, including unobserved cells.

Comparison targets, recorded in config and startup logs:

- Baseline 8ch: approximately 0.507 (user-provided reference).
- Naive PointNet Hybrid: approximately 0.497 (user-provided reference).
- Step C.1 Gated Hybrid: 0.497326, epoch 23 (confirmed in the previous log).

The previous run ended at epoch 33 via early stopping. Class references below
come from its best-epoch log and are rounded to three decimals. Every new
validation reports current values and deltas; these comparisons do not enter
the loss or training decisions.

| Class | Step C.1 Epoch 23 IoU |
| --- | --- |
| bicycle | 0.255 |
| motorcycle | 0.401 |
| person | 0.423 |
| bicyclist | 0.471 |
| terrain | 0.802 |
| vegetation | 0.835 |
| truck | 0.541 |
| parking | 0.367 |
| car | 0.906 |
| other-vehicle | 0.250 |

## Verification

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_pointnet*.py' -v
.venv/bin/python -B tests/smoke_pointnet_stablelr.py \
  --output logs/StepC1b_preflight_20261005
```

Use a new output directory when repeating the smoke test. It uses temporary
trainer state, three real training batches and two validation batches. It
checks finite losses and gradients on every optimizer update, input shapes,
empty paste masks, gate diagnostics and GPU allocation. No smoke checkpoint
is passed to formal training. Reports are stored separately from formal logs.

Preflight completed on 2026-10-05: all 17 unit tests and syntax checks passed.
The CUDA smoke test passed on an RTX 5060 Ti (16 GB): 329 gradient tensors
were finite on each of three optimizer updates, with no OOM or skipped step.
Training losses were 5.152254, 5.039968 and 4.848295. The first validation
batch's gate had mean=0.501157, std=0.074797, min=0.005383, max=0.998000.
Peak PyTorch allocation was 13.83 GiB in training and 10.76 GiB in validation;
peak reserved training memory was 14.35 GiB. These smoke metrics are not an
accuracy evaluation of the experiment.

Detailed output: logs/StepC1b_preflight_20261005/report.json and smoke.log.
protected_files.sha256 records unchanged previous code and Step C.1 log.

## Formal Training Command

```bash
.venv/bin/python -B -u train_pointnet_hybrid_gated_stablelr.py \
  --dataset SemanticKitti \
  --arch_cfg config/arch/sjnet_pointnet_gated70_stablelr.yaml \
  --data_cfg config/labels/semantic-kitti.yaml \
  --log logs/SJNet3_PointNet_stepC1b_gated_lr2e3_bs3
```

After an interruption, resume only this experiment with its latest.path.
Training uses 19,130 scans, batch size 3, drop_last=True and 6,376 batches
per epoch. Do not rerun the scratch command into an existing log directory.

The formal command was launched in detached tmux session
stepc1b_gated_lr2e3_bs3_20261005. The training process is independent of the
smoke process and initializes new model, optimizer and scheduler state.
