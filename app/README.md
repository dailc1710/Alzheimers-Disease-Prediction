# Alzheimer's Disease Screening

This project implements the pipeline described in report V3: deterministic
dataset cleaning, exploratory diagnostics, permutation-based feature ranking,
five-feature selection, imbalance comparison, repeated stratified validation,
XGBoost grid search, sigmoid (Platt) calibration, F2 threshold selection, test
evaluation with bootstrap confidence intervals, and a versioned Streamlit app.

The original multi-stage notebook is archived under `archive/legacy/` because
it derives the target from MMSE while retaining MMSE as an input feature. Its
metrics must not be used for final conclusions. The V3 source of truth is
`ml_pipeline.py`, `train_pipeline.py`, `app.py`, and the active artifact pair
(`active_metadata.json` schema 2 + `active_model.joblib`).

The data-cleaning decision record and evidence audit are documented in
`docs/DATA_CLEANING_RESEARCH.md`.

The combined tab audit and prioritized improvement plan are documented in
`docs/TAB_AUDIT_RESEARCH.md`. The maintained module map is in
`docs/ARCHITECTURE.md`.

The model is an educational screening demonstration. It is not a medical
diagnostic device and must not be used as the sole basis for medical decisions.

## Release consistency

The pipeline uses every non-locked row for final training. A single development
split is used for training and validation decisions; after feature, imbalance,
tuning, and threshold decisions are fixed, the final calibrated model is
refitted on the complete development cohort. The locked test remains untouched.

Imbalance strategies are compared only within cross-validation. Methods inside
the PR-AUC tie band are ranked by F2, recall, Brier score, and then simplicity.
XGBoost tuning uses the selected imbalance strategy rather than a fixed
weighting strategy.

Whenever `ml_pipeline.py`, `train_pipeline.py`, or the source CSV changes,
regenerate and verify the active artifact before reporting metrics:

```powershell
python train_pipeline.py
python scripts/verify_artifact.py
pytest --cov=ml_pipeline --cov-report=term-missing --cov-fail-under=80
```

Update the report only after these commands create and verify new active
metadata.

The raw source CSV is intentionally kept outside the application directory.
When the project is stored inside a parent workspace, place
`alzheimers_disease_data.csv` beside the project folder, or pass its full path
with `--data` to the training and verification scripts.

## Run the pipeline

From this directory:

```powershell
pip install -r requirements.txt
python train_pipeline.py
```

The default run creates `artifacts/active_model.joblib` and
`artifacts/active_metadata.json`, plus immutable files under
`artifacts/versions/`. It also creates `data/locked_test.csv` once. That cohort
is excluded from training and is used unchanged for final evaluation and
champion/challenger comparisons. Use `python train_pipeline.py --fast` to skip
the three diagnostic comparisons while keeping tuning and final evaluation.
Use `python train_pipeline.py --selection-mode nested_cv` when a nested-CV
feature-stability diagnostic is required; the default `report_v3` mode keeps
the fixed five-feature set for reproducibility.

Verify the artifact contract and run a ten-row smoke prediction with:

```powershell
python scripts/verify_artifact.py
python scripts/run_audit.py
```

The audit command writes `artifacts/audit/bp_removed_profile.csv` and
`artifacts/audit/cleaning_sensitivity.csv`. The latter compares remove,
swap-if-plausible, and keep-with-flag on a common internal evaluation cohort;
it is not external validation.

## Run the Streamlit app

```powershell
pip install -r requirements.txt
streamlit run app.py
```

`streamlit_app/app.py` remains a thin deployment wrapper for hosts that expect
the entry point inside a `streamlit_app` directory.

The app has four main tabs. The **Prediction** tab has two visible sub-tabs:

1. **Single case** accepts the final five features and shows the model class,
   confidence percentage, calibrated Alzheimer score, and a non-identifying
   `CASE-...` follow-up code.
2. **Batch CSV** accepts the five model features (plus optional `PatientID`),
   screens every eligible row, exports results, and can save uploaded cases for
   later clinical feedback. Full V3 CSVs belong in **Project dataset**;
   labelled CSVs belong in **Model update**. `PatientID` is not copied into the
   feedback database.

The separate **Project dataset** tab is the CSV input/output workflow. It
accepts an optional CSV, automatically detects a Full V3 or five-feature
structure from its headers, validates and cleans rows, and exports the cleaned
CSV, complete issue log, and removed/rejected rows. With no upload, it uses the
bundled dataset. This tab does not predict or retrain; the bundled data are not
an independent external validation set.
An error-only review table highlights problematic input values in red, actual
corrected values in green, and fields requiring manual review in amber.

The **Model information** tab contains the model card, EDA evidence, and the
clinical feedback workflow. The separate **Model update (CSV)** tab accepts
labelled cases for retraining. The batch-prediction CSV and retraining CSV
uploaders are intentionally separate; only the latter requires `Diagnosis`.

Authentication is enabled by default. On first launch, the app asks for the
first administrator account and stores only a salted PBKDF2-SHA256 password
hash in `data/app_state.sqlite3`. Administrators can create `viewer`, `doctor`,
and `admin` accounts from the sidebar. Every authenticated session has a
logout action. `ALZHEIMER_AUTH_DISABLED=1` exists only for automated test runs;
do not use it in a deployed application. `ALZHEIMER_STATE_DB` can point the
runtime state database to a persistent mounted volume.

Batch processing also reports privacy-minimized quality and feature-drift
alerts when the active artifact contains a training monitoring baseline.

## Clinical feedback loop

Single-case and uploaded-batch predictions can be saved under non-identifying
case codes. A viewer can report the later examination outcome, which remains
pending. A doctor or administrator can add or correct the label; each change is
an append-only revision, so the audit history is not overwritten. Both
five-feature and Full V3 cases with a verified doctor/admin label are eligible
for retraining. Updates use only the five fixed model inputs and the observed
`Diagnosis`; no unused clinical fields are fabricated. When a verified result
differs from the saved screening prediction, the app warns that the model was
wrong for that case and shows the mismatch in the case table. A later pending
viewer report does not replace the most recent verified label or enter training.
The feedback screen separates the read-only AI result from the doctor's
examination conclusion. No clinical conclusion is preselected; the user must
choose one and confirm that it came from the examination. The latest verified
result displays the verifying account, time, and revision. Viewer-submitted
results are visibly pending, not presented as verified medical conclusions.

## Retraining

The Streamlit **Model update (CSV)** tab accepts verified feedback or a
labelled CSV with `MMSE`, `FunctionalAssessment`, `ADL`, `MemoryComplaints`,
`BehavioralProblems`, and `Diagnosis`. `PatientID` is optional for duplicate
tracking and is not a model input. It shows validation/rejected-row logs,
skips unchanged accepted cases, replaces a previously accepted case when its
identified and verified label is corrected, preserves the original 35-column dataset,
stores promoted additions in `artifacts/accepted_retraining_rows.csv`,
writes a timestamped challenger version, and compares champion/challenger
metrics on the same locked test rows. Only an authenticated administrator can
start the loop. A file containing only old or locked-test rows (including
`data/locked_test.csv`) is rejected so the locked evaluation cohort cannot leak
into training.

After the administrator starts the loop, promotion is automatic but still
strictly gated: PR-AUC, recall, Brier, F2, paired bootstrap confidence, and
common estimable subgroup checks must all pass. If they pass, the challenger
becomes active and the new rows are persisted into the training dataset. If any
condition fails, the champion and dataset remain unchanged. Every attempt,
including failures and retained-champion decisions, is written to the local
retraining session log. The same workflow is available by calling
`retrain_with_new_data()` from `ml_pipeline.py` with the original 32 feature
columns plus `Diagnosis`.

All screening results are educational only. A positive or negative screening
signal is not a diagnosis, and a negative signal does not rule out disease.

## Development and deployment

```powershell
python -m pip install -r requirements-dev.txt
pytest
ruff check .
docker build -t alzheimer-screening .
docker run --read-only -p 8501:8501 alzheimer-screening
```

CI runs linting, tests with coverage, static type checking, dependency
vulnerability scanning, and the artifact contract check. The
container runs as a non-root user and does not include legacy experiments or
version backups. Uploads are size/row limited and exported text is protected
against spreadsheet formula injection. Before public deployment, put the
retrain/promote workflow behind authentication and authorization.
