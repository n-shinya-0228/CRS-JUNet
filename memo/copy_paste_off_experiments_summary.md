# Copy-Paste OFF Experiments: Evidence Review

Audit date: 2026-10-23 (JST), Step E.0. No new training or full-validation model
evaluation was performed for this review. Scores are fractions, not percentages,
unless marked `%`. Unknown facts are explicitly marked **未確認**.

## Scope And Comparability

The segmentation experiments use handcrafted 7ch + frozen/precomputed PointNet16
+ observation mask = 24ch, except the original 8ch baseline (7+mask). Dataset
features are 23ch; Trainer appends observation mask at channel 23. The PointNet
encoder is pretrained even when segmentation weights are initialized from scratch.
"Scratch" below does not mean the cached point encoder is randomly initialized.

Full validation is SemanticKITTI sequence 08 (4,071 scans for the recorded D runs).
Point-encoder-only subset experiments are reported separately and cannot be ranked
against the full segmentation scores. No held-out test-set result or multi-seed
confidence interval was found for these comparisons (**未確認**).

## Full Segmentation Results

| Experiment | Main change / purpose | CP | Weight initialization / evaluation | Best epoch | Validation mIoU |
| --- | --- | --- | --- | ---: | ---: |
| Original 8ch baseline | 7 handcrafted + mask, SJNet3 base48 | OFF in recorded Polar2 Dataset | Archived run includes `--resume .../best_val.path`; initial launch **未確認** | 90 | 0.507 (rounded log) |
| C.1 | Separate 7->16/16->16 stems, sigmoid PointNet gate, RingConv32->48 | OFF | Segmentation scratch, pretrained cached encoder | 23 | 0.49732601127212633 |
| C.1b | C.1 architecture retained; stable LR/warmup settings | OFF | Segmentation scratch | 9 | 0.504595929222469 |
| C.2 | Add PointNet residual Local3 RingConv context, fixed alpha=1 | OFF | Segmentation scratch | 18 | 0.508252934833837 |
| C.3 | C.2 local residual with learnable global alpha | OFF | Segmentation scratch | 9 | 0.5054781060415194 |
| C.4 | Spatial-aware learnable local alpha | OFF | Segmentation scratch | 12 | 0.5016311373118368 |
| D.1 | Fixed global alpha sweep 0.4/0.6/0.8/1.0/1.2 | OFF | C.2 best EMA, **inference only, no retraining** | Source 18 | 0.5086269996818946 (alpha=1.2) |
| D.1b | Add high-alpha 1.1/1.3/1.4/1.6 | OFF | Same C.2 best EMA, **inference only** | Source 18 | 0.5086269996818946 (alpha=1.2 retained) |
| D.2a Hard | First-pass class argmax chooses class-fixed local alpha | OFF | Same C.2 best EMA, shared-weight two-pass **inference only** | Source 18 | 0.509595264387397 |
| D.2b Soft | First-pass probability-weighted class alpha | OFF | Same C.2 best EMA, shared-weight two-pass **inference only** | Source 18 | **0.5097046791309** |
| D.4 | Fixed C.2 EMA teacher supplies soft alpha, train student | OFF | C.2 initialization / fine-tuning; EMA student evaluation | **0** | 0.5096905511292307 |
| D.5 | Shared-weight two-pass soft-alpha scratch segmentation training | OFF | Verified random segmentation initialization; fixed pretrained encoder cache | **12, provisional** | **0.49635787093375333** |

C.1-C.4 epochs/scores were read directly from their existing `best_val.path`
checkpoint scalars. Baseline checkpoint contains epoch 90 but not a high-precision
best-mIoU scalar, so its rounded 0.507 must not be promoted to a fabricated exact
value. Baseline epoch90 accuracy is 0.891 and validation loss 0.6770.

### What Improved And What Did Not

- **Baseline:** remains a competitive 8ch reference. `used_files.txt` records
  `lib/dataset/SemanticKitti_Polar2.py`, which has no Copy-Paste operation; do not
  confuse it with the later step4_5 CP Dataset. Historical source snapshot identity
  and complete initialization history are **未確認**. It is not an exact matched
  one-variable C.2 ablation because training settings/history differ.
- **C.1:** explicit feature-space alignment and gates did not surpass the baseline.
  Initial LR 0.003, warmup3, minLR2e-5, dropout0.3, weight decay0.005, batch3,
  max70, ES patience10/min_delta0.001/min_epochs30. Log records ES at epoch33.
- **C.1b:** LR0.002, warmup5, minLR1e-5, cosine65, ES patience12, other recorded
  regularization unchanged. Best improves by about 0.727 percentage points over
  C.1, but remains below baseline. Log ends during training, not a documented
  normal completion; exact reason for that termination is **未確認**.
- **C.2:** one 3x3 angular-circular residual block on the PointNet branch improves
  by 0.3657 percentage points over C.1b. Best training-run mIoU is about 0.1253
  points above the *rounded* baseline, not evidence of a robust statistically
  significant win. Same stable-LR/batch3/max70 settings as C.1b; log says Finished
  Training. Current source model preserves observation gating, skip paths,
  auxiliary outputs and boundary output.
- **C.3:** learned alpha at the best evaluation is 0.641106903553009 (training
  alpha 0.6464779376983643), but best score is 0.2775 points below C.2. Optimizing
  alpha during scratch training is not equivalent to changing alpha on C.2 weights.
- **C.4:** spatial flexibility does not improve the aggregate score; 0.6622 points
  below C.2. The log ends just after the epoch25 checkpoint and the start of the
  next epoch, without a completed-training message; interruption reason is
  **未確認**, not assumed to be early stopping or convergence.
- **D.1/D.1b:** alpha1.2 is the best of the recorded grid, not a universal optimum.
  mIoU rises by 0.0374 points relative to C.2's stored scalar. D.1 alpha1
  reevaluation is 0.5082557747585773, slightly different from stored C.2 mIoU;
  preserve this numerical evaluation difference, do not overwrite the source.
- **D.2a/D.2b:** D.2's reevaluated global1.2 baseline is 0.5086245904876063.
  Hard improves by 0.0971 points and Soft by 0.1080 points against that same-run
  reference. Soft exceeds Hard by only 0.01094 points. Overall gains are small;
  higher total mIoU does not imply all classes improve. Soft is the best recorded
  CP-OFF full-validation result, with no weight training in D.2.
  Against the same-run global1.2 reference, Soft improves truck/person/parking/
  fence/vegetation/trunk/terrain/traffic-sign/road/sidewalk, but worsens car/bicycle/
  motorcycle/bicyclist/other-ground/building/pole; motorcyclist remains zero.
  See the archived `D.2/comparison.md` and per-class CSVs for numerical deltas.
- **D.4:** `results.json` says complete, best_epoch=0. The baseline student/EMA
  evaluation at epoch0 is best; subsequent fine-tuning through epoch8 does not
  improve it and early stopping ends the run. The 0.00141-point difference from
  D.2b is a small negative reproduction difference, not a fine-tuning benefit.
- **D.5:** an independent segmentation scratch experiment, not C.2 fine-tuning.
  Stage1 and Stage2 use shared weights; Stage1 alpha selection has no gradient,
  Stage2 supplies training loss, no separate teacher. Last completed epoch18
  has Stage1 mIoU0.4724694282935731 and Stage2 mIoU0.48260239345924044; train IoU
  about0.7321. This gap is consistent with overfitting but alone does not establish
  a unique cause. The run is incomplete, so final convergence is **未確認**.

## D.5 Interruption Verification

Read `status.json`, `train_log.txt`, `training_history.json`,
`best_validation.json`, and `latest.path` under
`logs/SJNet3_PointNet_stepD5_shared_soft_alpha_scratch_cp0/`.

- Status is `interrupted`, not complete; error trace is `KeyboardInterrupt` after
  the user's stop request, not a claimed numerical training crash.
- `latest.path`: epoch18, best_val_epoch12,
  best_val_iou0.49635787093375333, copy_paste_prob0.0,
  pretrained_initialization=false, initialization_verified=true, seed0.
- Log last progress is epoch19 `[3000/6376]`, then stop at 2026-10-23 11:14 JST.
- Completed validation is available for epochs0..18. Partial epoch19 is not an
  evaluated result and is not included in best selection.
- Best epoch12 Stage1 mIoU0.4912172902271449; Stage2 mIoU0.49635787093375333,
  difference +0.00514058070660843. Stage2 accuracy0.8814681672296529.
- Config: LR0.002, max70, batch3/microbatch3, BF16; latest initialization digest
  `4ca99449969b24576dbd8d9166ab429a760400bd46decaeca8bf252fab1e4b49`.
- This review neither resumes D.5 nor labels its provisional best a completed run.

## Point Encoder Only: Subset Studies

These experiments use **500 train / 1,000 validation scans, three epochs**, not the
full sequence08 segmentation protocol. The raw-point encoder has pointMLP7->32->64
with BatchNorm, per-cell max pooling, cellMLP64->32->16 and a pretraining classifier.
The seven input values are normalized x,y,z,remission, radial offset, angle offset
and within-cell z offset. The cached feature generator discards the classifier and
uses a frozen eval-mode encoder. Encoder weighting and loss-cell sampling do not
change the segmentation architecture.

### Stage 1: Class Weighting

Evidence: `point_encoder_tuning/stage1_20260902_seed31415/stage1_report.md`,
`stage1_results.csv` and each `run_config.json` / `summary.json`.

Train manifest seed31415, validation31416, lr1e-3, weighted CE. Six runs share
byte-identical manifests. No Copy-Paste operation is present in the point-encoder
Dataset. Best overall and small result is `exp_006_eps1p05`, epoch3:

| Run | Overall mIoU | Small mIoU | Accuracy | Core-small positive |
| --- | ---: | ---: | ---: | ---: |
| Control eps1.2 | 14.1646% | 0.0038% | 63.0455% | 0/5 |
| Best eps1.05, core boost2, pole/sign boost1.5 | 14.5044% | 1.7672% | 60.9298% | 0/5 |

All bicycle/motorcycle/person/bicyclist/motorcyclist IoUs remain exactly zero in
all six runs. Improvement comes from pole1.1703% and traffic-sign11.2004%, not
the five core-small classes. All five classes exist in the subsets, so absence
of those classes is not the explanation. Longer/full-data outcomes are **未確認**.

### Stage 2A: Class-Balanced Loss Sampling

Evidence: `point_encoder_balanced_sampling/stage2a_report.md`,
`stage2a_results.json`, per-run configs/summaries and cell-sampling statistics.

Same manifests and LR, global/sampling seed42, eps1.05/coreboost2/pole-sign1.5.
All rare-class cells2,3,6,7,8,18,19 retained. Other nonzero classes capped per scan
for the **training loss only**. All-cell feature generation/pooling is unchanged;
validation is all occupied cells with class0 ignored, not sampled validation.

| Run | Selected epoch | Overall mIoU | Small mIoU | Core-small positive |
| --- | ---: | ---: | ---: | ---: |
| Control | 3 | 14.4788% | 1.7288% | 0/5 |
| Cap4000 | 3 | 13.7303% | 2.4493% | 0/5 |
| Cap2000 | 3 | 13.0507% | 2.4608% | 0/5 |
| Cap1000 | 3 | 12.1361% | 2.5162% | 0/5 |
| Cap500 | 2 | 10.6986% | 2.3672% | 0/5 |

Core-small IoUs are zero in all15 run/epoch pairs. Smaller caps sacrifice overall
accuracy without resolving core-small recognition. Cap1000 has the largest small
mean, but not the largest overall score. Cap500's selected epoch2 follows the
recorded core/small-prioritized rule, not an invented overall-mIoU epoch choice.
Stage1 and Stage2A control use different global seeds, so their small difference
is not a pure sampling effect. Adoption of any tuned checkpoint for existing
`pointnet_16ch` caches is **未確認**: cache files do not embed encoder checkpoint
provenance. E.0 numerically tests the default `point_encoder_runs/best_point_encoder.pth`
and records its hash; it does not relabel that checkpoint as a Stage1/2A winner.

## Baseline Class Reference

These epoch90 numbers were confirmed in `logs/SJNet3_48/log.txt` at
2026-05-17 18:46:58 and agree with the user's supplied comparison. They are
rounded to three decimals, not newly evaluated high-precision IoUs.

| Class | IoU | Class | IoU |
| --- | ---: | --- | ---: |
| car | 0.947 | bicycle | 0.247 |
| motorcycle | 0.313 | truck | 0.645 |
| other-vehicle | 0.306 | person | 0.411 |
| bicyclist | 0.492 | motorcyclist | 0.000 |
| road | 0.937 | parking | 0.495 |
| sidewalk | 0.802 | other-ground | 0.009 |
| building | 0.839 | fence | 0.272 |
| vegetation | 0.807 | trunk | 0.432 |
| terrain | 0.723 | pole | 0.487 |
| traffic-sign | 0.474 | | |

## Sources And Reproduction Limits

For C.1/C.1b/C.2/C.3/C.4, inspect each named run directory's `best_val.path`,
`arch_cfg.yaml`, `log.txt` and `used_files.txt`:

- `logs/SJNet3_48/`
- `logs/SJNet3_PointNet_stepC1_gated70_bs3/`
- `logs/SJNet3_PointNet_stepC1b_gated_lr2e3_bs3/`
- `logs/SJNet3_PointNet_stepC2_local3_gated_lr2e3_bs3/`
- `logs/SJNet3_PointNet_stepC3_local3_alpha_gated_lr2e3_bs3/`
- `logs/SJNet3_PointNet_stepC4_local3_spatial_alpha_gated_lr2e3_bs3/`
- `logs/SJNet3_PointNet_stepD1_fixed_alpha_sensitivity/results.json`
- `logs/SJNet3_PointNet_stepD1b_high_alpha_sensitivity/results.json`
- `logs/SJNet3_PointNet_stepD2_class_fixed_alpha/results.json`
- `logs/SJNet3_PointNet_stepD4_soft_teacher_student_cp0_lr1e4/results.json`
- `logs/SJNet3_PointNet_stepD5_shared_soft_alpha_scratch_cp0/`

C.2's recorded command uses `train_pointnet_hybrid_gated_local3.py` and
`trainer_PointNet_Gated_StableLR.py`. Neither is present in the current checkout.
Dataset, Parser, C.2 Model and saved config/checkpoints do exist. The existing
`lib/trainer/trainer_PointNet.py` is inspectable but must not be asserted byte-
identical to the historical StableLR Trainer. Complete executable C.2 source
reproduction from this checkout alone is **未確認**. E.0 does not recreate a
missing Trainer or change any baseline source/config.

## Interpretation And Next Step

D.2b is currently the best recorded CP-OFF inference result, not a newly trained
model. Local3 C.2 is the best recorded scratch-training backbone among these C
variants; learned global/spatial alpha and teacher-student fine-tuning have not
improved its best full-validation result. D.5 is only provisional and should not
be used to claim that all two-pass scratch training is ineffective.

Repeated architecture selection, checkpoint selection and global/class alpha
selection on **the same sequence08** create validation-selection bias. The small
D.1/D.2 improvements are exploratory; they are not independent test evidence.
Report a fixed protocol, multiple seeds and an untouched held-out evaluation before
making generalization claims. Dataset split isolation for CP prevents leakage of
objects but does not eliminate this model-selection bias.

E.0 separately audits CP alignment and finds reasons to withhold E.1 training;
see `logs/SJNet3_PointNet_stepE0_copy_paste_integrity/integrity_report.md`.
No inference or training score in this review was recalculated, overwritten,
or treated as evidence that feature-level CP equals point-level augmentation.
