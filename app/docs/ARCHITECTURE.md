# Project architecture

## Runtime flow

1. `app.py` orchestrates authentication and four main tabs: Prediction,
   Project dataset, Model information, and Model update. Prediction has two
   sub-tabs (single case and uploaded batch CSV); Project dataset can inspect
   and clean another uploaded CSV or fall back to the bundled Full V3 data,
   then export processed, rejected, and audit CSVs without predicting; Model
   information holds EDA and clinical feedback; Model update has a labelled-CSV
   uploader for retraining.
2. `ui/` owns reusable presentation helpers, formatting, and the shared stylesheet.
3. `auth.py` owns password hashing, users, and role checks.
4. `feedback_store.py` owns case codes, feedback revisions, verified retraining
   rows, and retraining-session logs in local SQLite state.
5. `ml_pipeline.py` owns validation, cleaning, training, evaluation, artifact versioning, monitoring, and prediction contracts.
6. `train_pipeline.py` is the command-line entry point for training.
7. `streamlit_app/app.py` is a deployment-only wrapper around the root application.

## Directory responsibilities

```text
app/
├── app.py                 Streamlit orchestration
├── auth.py                Authentication and role storage
├── feedback_store.py      Feedback/retraining audit persistence
├── ml_pipeline.py         ML and data-domain logic
├── train_pipeline.py      Training CLI
├── ui/                    UI helpers and styles
├── scripts/               Audit, verification, and report utilities
├── tests/                 Automated regression tests
├── artifacts/             Active model plus local generated history
├── data/                  Locked test cohort and external-data placeholder
├── docs/                  Research notes and maintenance documentation
└── archive/legacy/        Superseded notebook and report
```

Generated report files do not belong in the application tree. They are written
to the workspace-level `deliverables/` directory. Only
`artifacts/active_model.joblib` and `artifacts/active_metadata.json` are needed
for normal application startup. Existing model history is archived under
`deliverables/model-history/`; future training runs recreate local history
folders under `artifacts/` as needed.
