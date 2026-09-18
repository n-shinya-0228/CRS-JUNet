# Point Encoder Stage 2A: Class-Balanced Cell Sampling

## Scope and conditions

- Train: 500 scans from the Stage 1 manifest
- Validation: 1000 scans from the Stage 1 manifest
- Manifest seeds: train 31415, validation 31416
- Global seed: 42
- Cell-sampling seed: 42
- Epochs: 3
- Learning rate: 1e-3
- Class-weight epsilon: 1.05
- Core-small boost: 2.0
- Pole / traffic-sign boost: 1.5
- PointNet input: 7 dimensions
- PointNet feature dimension: 16
- Polar grid: 512 x 512
- Maximum radius: 51.2

All five runs reused byte-identical copies of the Stage 1 train and validation
manifests. No test split was used. Point feature generation, cell pooling, and
cell logits were computed for every occupied cell. Sampling was applied only to
the training loss inputs. Validation used all occupied cells, with class 0
ignored by the unchanged criterion and confusion-matrix logic.

## Implementation and verification

The training-only selector keeps every cell from classes 2, 3, 6, 7, 8, 18,
and 19. Other nonzero classes are sampled independently within each scan up to
the requested cap. Class 0 is never selected. A SHA-256 digest of the global
sampling seed, epoch, sequence, frame, and class ID seeds each class-specific
generator. This gives identical selections for identical inputs while changing
the majority-cell selection between epochs.

Verification completed before Stage 2A:

- `uv run python -m py_compile train_point_encoder.py`: passed
- Synthetic selector test: cap, class-0 exclusion, small-class retention,
  same-epoch reproducibility, and next-epoch variation all passed
- Smoke control: 20 train / 20 validation scans, 1 epoch, passed
- Smoke cap 2000: 20 train / 20 validation scans, 1 epoch, passed
- Both smoke runs produced finite losses, metrics, five required checkpoints,
  epoch JSONL, summary JSON, and sampling CSV / JSON
- Smoke control retained all nonzero classes at 100%
- Smoke cap 2000 retained all seven small classes at 100%
- Validation contains no sampling call and completed on all validation cells

The first Stage 2A cap-2000 scan (`00/000039`) gave this direct sanity check:

| Class | Before | After |
| --- | ---: | ---: |
| unlabeled | 160 | 0 |
| road | 7,364 | 2,000 |
| parking | 3,840 | 2,000 |
| sidewalk | 2,929 | 2,000 |
| building | 2,863 | 2,000 |
| vegetation | 3,614 | 2,000 |
| person | 1 | 1 |
| bicyclist | 4 | 4 |
| motorcyclist | 12 | 12 |
| pole | 39 | 39 |
| traffic-sign | 45 | 45 |

## Stage 2A results

Percentages are reported below. The selected epoch follows the specified
priority: core nonzero count, core-small mIoU, core-class IoUs, small mIoU,
overall mIoU, then accuracy. All core-class metrics tied at zero, so cap 500
selects epoch 2 because its small mIoU was higher than at epoch 3.

| Experiment | Cap | Epoch | Val mIoU | Val Acc | Small mIoU | Core mIoU | Core >0 | Core >1% | Bicycle | Motorcycle | Person | Bicyclist | Motorcyclist | Pole | Traffic-sign |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | none | 3 | 14.4788 | 60.8645 | 1.7288 | 0.0000 | 0/5 | 0/5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.2627 | 10.8387 |
| cap4000 | 4,000 | 3 | 13.7303 | 54.5282 | 2.4493 | 0.0000 | 0/5 | 0/5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.0231 | 14.1220 |
| cap2000 | 2,000 | 3 | 13.0507 | 51.6840 | 2.4608 | 0.0000 | 0/5 | 0/5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.2704 | 13.9550 |
| cap1000 | 1,000 | 3 | 12.1361 | 48.3617 | 2.5162 | 0.0000 | 0/5 | 0/5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.3134 | 14.2998 |
| cap500 | 500 | 2 | 10.6986 | 42.7729 | 2.3672 | 0.0000 | 0/5 | 0/5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.9662 | 13.6043 |

The five core-small classes were all exactly zero in every one of the 15
experiment epochs, not only in the selected epochs. Therefore the increase in
small mIoU came entirely from pole and traffic-sign.

## Control reproducibility

| Result | Val mIoU | Val Acc | Small mIoU | Core mIoU | Core >0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 1 `exp_006_eps1p05` | 14.5044 | 60.9298 | 1.7672 | 0.0000 | 0/5 |
| Stage 2A control | 14.4788 | 60.8645 | 1.7288 | 0.0000 | 0/5 |
| Difference | -0.0256 | -0.0653 | -0.0384 | 0.0000 | 0 |

The Stage 2A control closely reproduced the Stage 1 best result, so the
balanced-run differences are meaningful relative to the same-run control.

## Effect relative to the Stage 2A control

| Experiment | Val mIoU delta | Val Acc delta | Small mIoU delta |
| --- | ---: | ---: | ---: |
| cap4000 | -0.7485 | -6.3363 | +0.7205 |
| cap2000 | -1.4281 | -9.1805 | +0.7320 |
| cap1000 | -2.3427 | -12.5028 | +0.7874 |
| cap500 | -3.7802 | -18.0916 | +0.6384 |

Reducing the cap caused a monotonic degradation in overall mIoU and accuracy.
The apparent small-mIoU gain did not include any core-small improvement.

## Sampling statistics

Counts are accumulated over all three training epochs. Because every run used
the same 500 scans, the before count is common to all experiments. Each run
entry is shown as `after cells (retained percentage)`.

| Class | Before | Control | Cap 4000 | Cap 2000 | Cap 1000 | Cap 500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unlabeled | 593,004 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| car | 1,249,944 | 1,249,944 (100.0%) | 1,249,944 (100.0%) | 1,167,777 (93.4%) | 871,035 (69.7%) | 546,747 (43.7%) |
| bicycle | 11,157 | 11,157 (100.0%) | 11,157 (100.0%) | 11,157 (100.0%) | 11,157 (100.0%) | 11,157 (100.0%) |
| motorcycle | 9,303 | 9,303 (100.0%) | 9,303 (100.0%) | 9,303 (100.0%) | 9,303 (100.0%) | 9,303 (100.0%) |
| truck | 51,270 | 51,270 (100.0%) | 51,270 (100.0%) | 51,270 (100.0%) | 49,707 (97.0%) | 40,674 (79.3%) |
| other-vehicle | 53,355 | 53,355 (100.0%) | 53,355 (100.0%) | 53,355 (100.0%) | 50,076 (93.9%) | 44,697 (83.8%) |
| person | 9,144 | 9,144 (100.0%) | 9,144 (100.0%) | 9,144 (100.0%) | 9,144 (100.0%) | 9,144 (100.0%) |
| bicyclist | 4,476 | 4,476 (100.0%) | 4,476 (100.0%) | 4,476 (100.0%) | 4,476 (100.0%) | 4,476 (100.0%) |
| motorcyclist | 1,827 | 1,827 (100.0%) | 1,827 (100.0%) | 1,827 (100.0%) | 1,827 (100.0%) | 1,827 (100.0%) |
| road | 10,556,211 | 10,556,211 (100.0%) | 5,982,024 (56.7%) | 3,000,000 (28.4%) | 1,500,000 (14.2%) | 750,000 (7.1%) |
| parking | 864,516 | 864,516 (100.0%) | 855,288 (98.9%) | 666,231 (77.1%) | 427,902 (49.5%) | 247,119 (28.6%) |
| sidewalk | 7,933,488 | 7,933,488 (100.0%) | 5,305,647 (66.9%) | 2,820,858 (35.6%) | 1,429,572 (18.0%) | 717,096 (9.0%) |
| other-ground | 238,497 | 238,497 (100.0%) | 224,934 (94.3%) | 178,593 (74.9%) | 114,948 (48.2%) | 72,291 (30.3%) |
| building | 2,384,061 | 2,384,061 (100.0%) | 2,382,741 (99.9%) | 2,022,219 (84.8%) | 1,216,542 (51.0%) | 649,560 (27.2%) |
| fence | 1,591,260 | 1,591,260 (100.0%) | 1,572,141 (98.8%) | 1,419,549 (89.2%) | 1,022,199 (64.2%) | 609,444 (38.3%) |
| vegetation | 10,609,857 | 10,609,857 (100.0%) | 5,676,027 (53.5%) | 2,976,321 (28.1%) | 1,499,355 (14.1%) | 750,000 (7.1%) |
| trunk | 141,816 | 141,816 (100.0%) | 141,816 (100.0%) | 141,816 (100.0%) | 141,816 (100.0%) | 140,814 (99.3%) |
| terrain | 5,764,293 | 5,764,293 (100.0%) | 3,003,015 (52.1%) | 1,933,662 (33.5%) | 1,115,313 (19.3%) | 610,980 (10.6%) |
| pole | 61,455 | 61,455 (100.0%) | 61,455 (100.0%) | 61,455 (100.0%) | 61,455 (100.0%) | 61,455 (100.0%) |
| traffic-sign | 21,075 | 21,075 (100.0%) | 21,075 (100.0%) | 21,075 (100.0%) | 21,075 (100.0%) | 21,075 (100.0%) |

The statistics confirm that the implementation behaved as intended. The seven
small classes were retained at 100%, class 0 was excluded, and majority classes
were progressively reduced as the cap decreased.

## Interpretation

1. Balanced sampling did not move any core-small class above zero. The success
   criterion was not met for any cap.
2. No cap qualifies as a successful setting. Under the declared metric priority,
   cap 1000 ranks highest among the balanced runs because it has the highest
   small mIoU after all core metrics tie at zero. That gain is only pole/sign.
   Cap 4000 is the least damaging balanced setting for overall performance.
3. Overall performance worsened at every cap. The loss relative to control grew
   from -0.7485 mIoU points at cap 4000 to -3.7802 points at cap 500, with an
   even larger accuracy loss.
4. Class weighting plus balanced sampling did not improve the five core-small
   classes over Stage 1 class weighting alone. It improved pole/sign while
   sacrificing majority-class performance.
5. A larger training subset and full validation could be used once to confirm
   this negative result, but the present result does not justify broad follow-up
   training. If confirmation is later approved, cap 4000 is the conservative
   choice; cap 1000 is the diagnostic choice. No such run was started here.
6. Class-balanced cell sampling should not be adopted as the Point Encoder
   pretraining setting based on Stage 2A.
7. Since substantial core-small cell counts were retained at 100% and all four
   caps still produced zero IoU, simple class imbalance is unlikely to be the
   only limitation. The next hypothesis should be insufficient information in
   a single Polar cell, motivating future consideration of neighboring-cell
   context, object-aware grouping, local multi-cell PointNet, or object-level
   pretraining. None of those methods was implemented or run in this stage.

This experiment stops at Stage 2A as requested.
