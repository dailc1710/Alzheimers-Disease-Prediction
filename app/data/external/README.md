# External validation input

Place a separately collected, labelled Full V3 CSV here only when an approved
external cohort is available. Do not treat another random split of the project
CSV as external validation.

Evaluate a frozen artifact without fitting it:

```powershell
python scripts/evaluate_external.py data/external/your_cohort.csv --output artifacts/external_validation.json
```

Until a real external cohort is supplied, artifact metadata correctly reports
`external_validation_status: not performed`. The script reuses the stored model,
feature order, imputer, and threshold and reports overall plus subgroup metrics.
