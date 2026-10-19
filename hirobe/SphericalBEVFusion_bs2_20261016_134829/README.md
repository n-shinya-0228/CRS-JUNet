# SphericalBEVFusion Training

Started on 2026-10-16 at 13:48:29 JST, from random initialization.
No resume or pretrained checkpoint is used. The five original source files
and the original config/labels are unchanged.

Settings: batch size 2, 100 epochs, AdamW lr=.001, cosine schedule,
float32, workers=4. Original training augmentations, including Copy-Paste,
remain enabled. BEV is 7x512x512 and spherical input is 6x64x512.

All training outputs are stored in this directory:

- log.txt: per-epoch training/validation loss, mIoU and class IoUs.
- console.log: standard output, warnings and error tracebacks.
- metrics.jsonl: metrics after each completed validation epoch.
- best_val.path and latest.path: checkpoints after validation.
- results.json: final summary when all configured epochs complete.
- tfrecord/: TensorBoard events.
- arch_cfg.yaml and data_cfg.yaml: exact configuration snapshots.
- launch.json and used_files.txt: launch details and source hashes.

The original Trainer logs at epoch completion, not after every batch.
log.txt and console.log can be empty while the first epoch is running.
The first epoch is index 0 internally and logged as Epoch 1.

Metrics are polar-BEV-pixel metrics, not official SemanticKITTI 3D scores.
The validation sequence is 08 (4071 scans); training uses 19130 scans.
Smoke-test weights were discarded before this independent training run.

To resume this same run after stopping it, from the repository root:

```bash
.venv/bin/python -B -u -m spherical_bev_fusion.train \
  --dataset SemanticKitti \
  --arch_cfg hirobe/SphericalBEVFusion_bs2_20261016_134829/arch_cfg.yaml \
  --data_cfg hirobe/SphericalBEVFusion_bs2_20261016_134829/data_cfg.yaml \
  --log hirobe/SphericalBEVFusion_bs2_20261016_134829 \
  --resume hirobe/SphericalBEVFusion_bs2_20261016_134829/latest.path \
  >> hirobe/SphericalBEVFusion_bs2_20261016_134829/console.log 2>&1
```

Resume only once latest.path exists, and do not run two processes in this
directory at the same time. Use the explicit latest.path for continuation.
