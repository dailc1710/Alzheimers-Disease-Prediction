# V3 source patch notes

## Changes applied

- Removed the unused second holdout split.
- Training and validation now partition the complete non-locked development cohort.
- The final calibrated model is refitted on all development rows after model and threshold decisions are fixed.
- XGBoost tuning now uses the imbalance strategy selected by cross-validation.
- The imbalance tie rule now prioritizes F2, recall, Brier score, then simplicity within the PR-AUC tie band.
- Metadata now records `development`, `final_training`, and checksum methods.
- Artifact verification now checks source-dataset and locked-test checksums.
- Tests cover screening-oriented imbalance selection and complete split accounting.

## Commands required locally

```powershell
python -m pip install -r requirements-dev.txt
python train_pipeline.py
python scripts/verify_artifact.py
pytest --cov=ml_pipeline --cov-report=term-missing --cov-fail-under=80
ruff check .
mypy ml_pipeline.py scripts --ignore-missing-imports --follow-imports=skip
```

Alternatively, from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/finalize_project.ps1
```

## Acceptance conditions

- `final_training + locked_test == rows_clean`.
- `train + validation == development`.
- Active model and metadata versions match.
- Dataset and locked-test checksums pass verification.
- The report uses metrics from the newly generated active metadata.
- External validation is not claimed unless an independent labelled cohort is evaluated.

## Important

Artifact regeneration is part of applying this patch. The active artifact must
be generated after the patched source, verified with the checksum command, and
only then used for final metrics or report updates.
