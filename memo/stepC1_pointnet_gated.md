# Step C.1: Gated PointNet Hybrid

This is a separate experiment, not a modification of Step C. The Step C
dataset and parser are reused without edits. No existing experiment logs
or checkpoints are overwritten. Start from scratch by default; a Step C
checkpoint cannot be resumed as Step C.1.

## Architecture

The dataset normalizes the seven handcrafted channels, appends the sixteen
PointNet channels, and returns [23, H, W]. The trainer appends the observation
mask to form [B, 24, H, W]. The channel contract is:

- 0:7: max_z, mean_z, max_r, density, z_diff, x_diff, y_diff
- 7:23: precomputed PointNet features
- 23:24: observation mask

Both feature branches use Conv2d(1x1, bias=False), InstanceNorm2d(affine=True),
and ReLU to produce 16 channels each. A Conv2d(32, 16, 1) plus sigmoid gate
scales PointNet channels independently at each pixel. Gate weights use the
normal PyTorch Conv2d initialization; the bias is zero. Concatenating the
handcrafted branch and gated PointNet branch gives 32 channels, fused by
RingConv2d(3x3), InstanceNorm, and ReLU into the existing 48-channel backbone.

The base model's encoder, decoder, attention gates, auxiliary heads (aux2,
aux4, aux8), and boundary head are reused. Angular padding remains circular;
radial padding remains zero-filled.

Mask behavior is unchanged: s0 *= 1 + tanh(stem_mask(mask)). In the actual
Step C forward, gate_scale is registered but unused, encoder blocks are
called without a hard mask, and skips are not explicitly hard-masked. Step
C.1 preserves these facts rather than introducing additional masking.

## Training

- Maximum 70 epochs (zero-based epoch 0 through 69).
- Validation mIoU early stopping: patience 10, min_delta 0.001, min_epochs 30.
- Warmup 3 epochs, cosine decay 67 epochs; base LR 0.003, minimum LR 2e-5.
- Batch size 3; Copy-Paste OFF; paste-region loss weight 0.
- Existing dropout 0.3, AdamW weight decay 0.005, losses, EMA, data
  augmentation, metric postprocessing, and auxiliary-loss decay are retained.

The early-stopping counter starts with validation and counts consecutive
epochs without an improvement greater than min_delta over its reference
best. Improvements accumulate against that reference, not the preceding
epoch. Stopping is prohibited before 30 completed epochs. Checkpoint
selection is independent: even improvements smaller than min_delta update
best_val.path if they are a new strict maximum.

Regularization has not been broadly retuned so the fusion change can be
evaluated separately. Early stopping limits late training; reduced
overfitting or a gain over the reported Step C mIoU of about 0.497 is not
guaranteed and must be measured on validation.

## Commands

Start a new experiment from the repository root:

```bash
.venv/bin/python -B -u train_pointnet_hybrid_gated.py \
  --dataset SemanticKitti \
  --arch_cfg config/arch/sjnet_pointnet_gated70.yaml \
  --data_cfg config/labels/semantic-kitti.yaml \
  --log logs/SJNet3_PointNet_stepC1_gated70_bs3
```

After an interruption, use the same command with:

```bash
  --resume logs/SJNet3_PointNet_stepC1_gated70_bs3/latest.path
```

latest.path is saved atomically after each completed epoch. It retains raw
training weights, EMA weights, optimizer, scheduler, scaler, best metrics,
best epoch, early-stopping state, and configuration. best_val.path exposes
EMA weights as model for inference, plus model_train for consistent resume.
Resume starts at the following epoch; an unfinished epoch is repeated.
RNG and data-loader position are not restored for bitwise-exact replay.
Resuming a checkpoint that already reached early stopping does not restart
training. Resume requires matching configs and a Step C.1 checkpoint; use
a new log directory for a changed experiment.

## Checks

```bash
.venv/bin/python -B -m unittest discover -s tests -p test_pointnet_gated.py -v
.venv/bin/python -B tests/smoke_pointnet_gated.py --dataset SemanticKitti
```

The CUDA smoke test performs one real train batch and one validation batch
at [3, 24, 512, 512], using temporary logs and no permanent checkpoints. It
reports gate statistics, finite loss/gradients, and peak GPU allocation.
Unit tests cover channel ordering, masks and unchanged backbone outputs,
gate gradients, angular padding, early stopping, checkpoint resume, and
existing-log protection. None of these commands starts the full experiment.

Verified on 2026-10-01 with RTX 5060 Ti (16 GB): real-data batch size 3
forward/backward and EMA validation passed. Initial gate mean was 0.5014;
peak allocated GPU memory was 13.83 GiB for training and 10.76 GiB for
validation. These are smoke-test results, not accuracy measurements or
steady-state throughput benchmarks.
