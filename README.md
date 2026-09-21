# Alzheimer's Disease Prediction workspace

The workspace is intentionally split into three clear areas:

- `app/` — maintained Python source, tests, active model artifacts, and runtime configuration.
- `alzheimers_disease_data.csv` — canonical source dataset kept beside the app so the default training path remains stable.
- `deliverables/` — reports, rendered pages, report assets, and packaged submissions.

Run development commands from `app/`:

```powershell
cd app
python -m pip install -r requirements-dev.txt
pytest
streamlit run app.py
```

For compatibility, training can also be launched directly from this workspace
root:

```powershell
python train_pipeline.py
```

The Streamlit interface has the same workspace-level launcher:

```powershell
streamlit run app.py
```

Historical report renders are retained under `deliverables/renders/archive/`.
The current aligned render is under `deliverables/renders/current/`.
Archived model versions and dataset backups are retained under
`deliverables/model-history/`; new training runs recreate their local runtime
directories under `app/artifacts/` as needed.
