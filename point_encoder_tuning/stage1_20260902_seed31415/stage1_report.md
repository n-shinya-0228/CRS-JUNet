# Point Encoder Stage 1

## Conditions

- Train subset: 500 scans, fixed seed 31415
- Validation subset: 1000 scans, fixed seed 31416
- Epochs: 3
- Learning rate: 1e-3
- Loss: weighted CrossEntropyLoss
- Polar grid: 512 x 512
- Maximum radius: 51.2

All six experiments used byte-identical train and validation manifests.

## Result

The best overall and best small result was exp_006_eps1p05. Relative to the
control, overall mIoU increased from 14.1646% to 14.5044%, while small mIoU
increased from 0.0038% to 1.7672%. Accuracy decreased from 63.0455% to 60.9298%.

The small-mIoU increase came entirely from pole (1.1703%) and traffic-sign
(11.2004%). Bicycle, motorcycle, person, bicyclist, and motorcyclist remained
at 0% in every experiment, so no experiment improved core-small mIoU.

## Subset coverage

Occupied-cell ground-truth counts show that the zero core-small result was not
caused by missing classes in the subsets.

| Class | Train cells | Validation cells |
| --- | ---: | ---: |
| bicycle | 3720 | 18972 |
| motorcycle | 3101 | 14055 |
| person | 3050 | 14929 |
| bicyclist | 1492 | 11057 |
| motorcyclist | 609 | 1142 |
| pole | 20480 | 46816 |
| traffic-sign | 7030 | 18293 |

## Recommendation

For a Stage 2 diagnostic, compare the control with exp_005_eps1p10 and
exp_006_eps1p05 on a larger training subset and full validation. However,
Stage 1 does not show that class weighting alone is sufficient for the five
core-small classes. Class-balanced cell sampling should be the next method to
consider after approval, while retaining rare-class cells and limiting
majority-class cells within each scan.
