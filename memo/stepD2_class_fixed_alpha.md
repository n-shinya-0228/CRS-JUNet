# Step D.2: Class-fixed Alpha Two-stage Inference

## Current Status (2026-10-19)

Implementation, syntax checks and all 14 CPU tests pass. The actual C.2
checkpoint also loads strictly on CPU with the expected EMA hash and
28,876,586 frozen parameters. The fixed learning-ID table matches D.1b.

GPU execution has NOT started. Three attempts to launch the evaluation
through the required execution approval service failed with:

```
Automatic approval review failed: Selected model is at capacity.
The action was not executed because automatic approval review could not be completed.
```

This is an approval-service availability failure, not a GPU/implementation
failure or a finding that the action is unsafe. No bypass is attempted.
The real-data GPU smoke, full baseline reproduction and Hard/Soft validation
are still required. No D.2 accuracy results are claimed.

Setup verification and launch-error records are saved separately under
logs/SJNet3_PointNet_stepD2_setup_checks_20261019/ so the intended D.2 output
directory remains fresh for the first successful execution. Start with the
command below WITHOUT --resume; no evaluation run exists yet.

## Experiment

Evaluation only: C.2 epoch-18 best EMA checkpoint, all 28,876,586 parameters
frozen. Stage1 uses global alpha=1.2. Hard/Soft use the same raw main logits
and the fixed 20-class learning-ID table from the instruction and D.1b CSV.
GT is used only in metrics and post-forward diagnostics. Unobserved cells
use 1.2. Height correction is applied to final predictions only.

New model, evaluator, streaming statistics/reporting helpers and focused
tests are separate files. Existing tracked files are not modified.

The current checkout was missing C.1/C.2/D.1 reference source files although
their result directories/checkpoints were present. Seven missing references
were added byte-for-byte from historical commit
396385a5051d3358b9d4c471268ca8be4e09ff3d:

- lib/models/SJNet3_PointNet_Gated.py
- lib/models/SJNet3_PointNet_Gated_Local3.py
- lib/models/SJNet3_PointNet_Gated_Local3_FixedAlpha.py
- evaluate_pointnet_fixed_alpha_sensitivity.py
- evaluate_pointnet_high_alpha_sensitivity.py
- config/arch/sjnet_pointnet_gated_local3_stablelr.yaml
- tests/test_pointnet_fixed_alpha.py

Existing Parser/Dataset/base model/IoU source matches that commit. No branch
switch, tracked-file overwrite, checkpoint rewrite or D.1/D.1b result update
is performed. D.1's EMA loader, real-data preflight, baseline evaluator,
scan manifest, postprocessing, metrics, atomic saves and CSV helpers are
reused. C.4 source was inspected in history but its predictor is not used.

The required full alpha=1.2 reproduction gate runs before the three-condition
evaluation. Baseline/Hard/Soft then use at most three forwards per batch.
Completed-batch progress commits all three confusion matrices and alpha/error
statistics together. Partial batches are not committed. Resume verifies
checkpoint/config/features/order/code/environment identity.

Alpha moments use float64 sums/squared sums. Hard quantiles use fixed-value
histograms. Soft quantiles use bounded-memory 0.0001-wide histograms; this
approximation and its maximum bin error are reported. Class-conditioned
statistics include both observed GT and raw predicted classes. Stage1 error
diagnostics use observed labeled cells, not GT to select inference alpha.

Output: logs/SJNet3_PointNet_stepD2_class_fixed_alpha/.

```bash
MPLCONFIGDIR=/tmp/stepd2_mpl .venv/bin/python -B -u evaluate_pointnet_class_fixed_alpha.py \
  --dataset SemanticKitti
```

Add --resume after interruption. --preflight-only performs actual-data smoke
checks without full baseline/Hard/Soft evaluation.

This is exploratory/development evaluation, not an independent test score:
sequence 08 already selected the C.2 checkpoint and fixed alpha table.
Retrospectively splitting sequence 08 does not create independent evaluation.
The lookup table is never adjusted using D.2 results.
