# Operations and release checklist

## Local setup

Use the locked Python 3.12 dependency file:

```powershell
python -m pip install -r requirements.txt
python scripts/verify_artifact.py
python -m unittest discover -s tests -p "test_*.py"
```

For development checks, install `requirements-dev.txt` and run `pytest` plus
`ruff check .`. Full retraining is intentionally not part of every CI job.

## Artifact safety

Only the checked-in active artifact directory is loaded by the app. The loader
verifies model SHA-256, metadata SHA-256, schema version, feature order,
threshold, and imputation statistics before deserializing the joblib payload.
Never upload an arbitrary joblib file. Production artifact directories should
be mounted read-only; retraining and promotion should run in a restricted
administrative process.

## Retrain, promote, and rollback

1. Validate verified feedback or a labelled CSV with the five fixed model
   features plus `Diagnosis`; optional `PatientID` identifies repeated cases.
   At least one genuinely new row must remain. Promoted additions are stored
   separately from the original 35-column V3 dataset.
2. Keep `data/locked_test.csv` unchanged and excluded from fitting.
3. Review champion/challenger metrics, paired differences, confidence intervals,
   and every promotion-policy check.
4. Require an authenticated administrator to start the loop. Treat that action
   as approval to promote only if every fixed policy condition passes.
5. Preserve the timestamped version and dataset backup for rollback.

The Streamlit retraining view is role-gated: only an `admin` account can start
the persistent loop. The first administrator is created through the one-time
first-run setup, and administrators can create doctor/viewer accounts from the
sidebar. Promotion automatically checks every common subgroup with an estimable
sample size: recall and PR-AUC may not fall by more than the configured subgroup
tolerance. Groups with fewer than 20 rows or only one class remain visible for
audit but are not automatic promotion gates. Every promoted, retained-champion,
or failed attempt is written to the SQLite retraining-session log.

## Privacy-minimized monitoring

Operational summaries contain UTC time, model/schema version, batch counts,
rejection/imputation counts, and score distribution summaries. They must not
contain PatientID, raw feature values, or uploaded file contents. Monitor
rejection rate, missingness, positive screening rate, score drift, and—when
delayed labels become available—PR-AUC, recall, calibration, and Brier score.
Each newly trained artifact also stores a privacy-minimized baseline for the
five selected features. The batch tab compares feature means against that
baseline and raises quality/drift alerts. This remains operational monitoring,
not clinical validation or external validation.

The local state database contains password hashes, stored prediction features,
feedback revisions, and retraining logs. It deliberately omits source
`PatientID` values and uses non-identifying `CASE-...` codes, but the feature
payload can still be sensitive health data. Put `data/app_state.sqlite3` on a
restricted persistent volume, include it in the retention/backup policy, and do
not expose or commit it. Five-feature cases can receive follow-up feedback but
are never used for retraining because the other 27 values are unavailable.
