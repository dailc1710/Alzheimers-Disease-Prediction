# Audit of the Batch CSV + Data Processing Tab

## Executive verdict

The combined tab was initially functional for a clean feature-only CSV, but the audit found gaps in the data-quality and prediction workflow. The most important issue was that the same validator was used for two different purposes: screening prediction and labelled-data quality/retraining. As a result, a row with a missing `Diagnosis` was rejected even when `Diagnosis` was not required to make a prediction. The tab also relied on reference-data medians/modes rather than preprocessing parameters saved with the active model.

The recommended next version keeps one main processing tab and separates model updates:

1. **Batch CSV + Data processing** — screening only; `Diagnosis` is optional and never used as an input. Only feature quality affects whether a row can be scored.
2. **Retrain / Model update** — the full V3 schema and binary `Diagnosis` are required; missing or invalid targets are rejected before a versioned model update.

This separation is now implemented in the UI. The Batch tab no longer mixes labelled-data audit with screening, while the Retrain tab owns the model-update workflow. The artifact caveat below records the pre-patch state; the finalized release must regenerate the active artifact after any source or dataset change.

## Scope and evidence

The audit checked the combined Streamlit tab in `app.py`, the validation and cleaning functions in `ml_pipeline.py`, the active artifact metadata, the canonical CSV, and the V3 report `../../deliverables/reports/I1_2510_E0_-_Nhom_1-_Alzheimer_project_report_V3_revised.docx`.

The canonical audit remains consistent with V3: 2,149 rows, 35 columns, no exact duplicates or missing cells, and 186 impossible blood-pressure rows where `SystolicBP <= DiastolicBP`. The source dictionary documents the model domains, including Age 60–90, MMSE 0–30, FunctionalAssessment/ADL 0–10, and binary 0/1 fields. [Source data dictionary](https://www.varprotools.org/reference/alzheimers.html)

The current code has a regression suite covering negative values, one missing feature, all-five-features-missing rows, invalid categories, invalid targets, screening-mode labels, conflicting patient IDs, saved imputation statistics, and source-row preservation in validator tests.

## What is already correct

- The tab is now one end-to-end path: load, profile, deduplicate, apply the V3 BP rule, validate/impute, preview, export, and predict.
- Negative and out-of-range values are rejected rather than converted to zero or absolute values.
- One recoverable missing feature is imputed with a reference median/mode and logged.
- A row with all five selected features missing is rejected instead of becoming five invented imputed values.
- Exact duplicate rows and impossible BP ordering are handled for full labelled V3 files.
- The tab displays input rows, columns, missing cells, duplicate rows, negative cells, BP removals, valid rows, rejected rows, the issue log, rejected rows, cleaned CSV, and predictions.
- The five official model features remain `MMSE`, `FunctionalAssessment`, `ADL`, `MemoryComplaints`, and `BehavioralProblems`.

## Findings that should be fixed

### Historical findings — Separate scoring mode from training/retraining mode

The earlier implementation treated a present `Diagnosis` column as a target that had to be valid in training mode. The patched Batch tab uses screening mode, so an optional `Diagnosis` column is ignored for scoring; the Retrain tab uses training mode and rejects missing or invalid labels.

Required behavior:

- In **screening prediction**, `Diagnosis` is optional, is excluded from prediction, and any supplied label is displayed as “provided label; not used for scoring.”
- In **training/retraining audit**, require `Diagnosis` and reject missing or invalid target values. Never impute the target.
- The UI should make the selected CSV schema and the screening-only purpose visible before showing the result.

### Historical findings — Persist and reuse the model’s preprocessing parameters

The earlier active artifact metadata contained model, threshold, split, and cleaning information, but no imputation parameters. The patched pipeline now persists reusable imputation statistics in the artifact and the app uses them for scoring; regenerate the artifact after source changes so this contract is actually present.

This can make a prediction depend on a changed reference CSV and can use rows that were removed by the BP-cleaning rule. A 2026 study using the same 2,149-record dataset reports training-split-only median/mode imputation and applies those fitted parameters unchanged to validation and test data. Scikit-learn gives the same rule: split first, fit preprocessing only on training data, and use a pipeline to prevent leakage. [FUSION-AD preprocessing](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2026.1799307/full) [Scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)

Required behavior:

- Fit the imputer on the training data during model training.
- Save the fitted imputer or its learned statistics inside the model artifact.
- Apply that exact transformer during batch prediction.
- Show the artifact version and preprocessing version in the tab.

### Historical findings — Detect conflicting duplicate patient identifiers

The earlier cleaning removed exact duplicate rows, but two different rows with the same `PatientID` could pass validation. The patched validator rejects conflicting patient records and preserves their source rows for review.

Required behavior:

- If `PatientID` is present, report missing IDs and duplicate IDs.
- If duplicate IDs have identical content, keep one and log the duplicate.
- If duplicate IDs conflict, reject all conflicting records for manual review.
- Keep the original `PatientID` and source row number in all audit exports.

### Historical findings — Make schema mode explicit instead of inferring it only from columns

The earlier tab classified a file as “full V3” only when all 32 features and `Diagnosis` were present. The patched UI distinguishes five-feature scoring, full-V3 scoring, and full labelled retraining schemas.

The UI should offer schema choices or clearly identify them:

- **Five-feature scoring CSV** — require the five selected model columns.
- **Full V3 scoring CSV** — require all 32 feature columns; `Diagnosis` optional and ignored for scoring.
- **Full V3 labelled audit/retraining CSV** — require all 32 features plus `Diagnosis`.

The application should list missing required columns and unexpected columns before processing rather than silently accepting a partial schema.

### Historical P1 — Improve CSV parsing and profile accuracy

The app uses `pd.read_csv()` with defaults. Pandas recognizes common missing markers such as empty strings, `NaN`, `N/A`, and `NULL`, but delimiter, decimal convention, encoding, and malformed-line behavior still matter. Pandas documents explicit controls for `na_values`, `encoding`, `decimal`, and `on_bad_lines`; the tab should expose a clear parse error rather than only a generic exception. [pandas `read_csv` documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

The current profile counts only cells that are already `NaN`. A string such as `not_available` is counted as non-missing initially, even though the validator later converts it to missing and imputes it. The profile should distinguish:

- parser missing markers;
- format errors converted to missing;
- domain-invalid values;
- values imputed;
- rows rejected.

### Historical P1 — Add row-level quality status to the prediction export

The prediction file currently contains prediction columns, but the user must open a separate processing log to know whether a row was imputed. Add these audit fields to each prediction row:

- source row number;
- `quality_status`: `clean`, `imputed`, or `rejected`;
- imputed fields;
- validation issue count;
- model artifact version;
- threshold used for the prediction.

Rejected rows should remain in a separate download with the exact rule and original value. This makes the result traceable without joining multiple files manually.

### Historical P1 — Show model context and medical-use guardrails in the combined tab

The single-case tab displays a screening disclaimer, but the batch results section does not display the same warning, model version, threshold, or the fact that this is not a diagnosis. Put the warning next to the batch prediction result, not only in the model-information tab. TRIPOD+AI emphasizes transparent reporting of preprocessing, missing-data handling, evaluation, and implementation details for AI prediction models. [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378)

### Historical P2 — Add a template and column dictionary

The tab should offer a downloadable template for the selected schema and show the allowed domain beside each field. This would prevent common errors before upload and make the five-feature versus full-V3 distinction obvious. The source data dictionary defines the feature ranges and category codes. [Source data dictionary](https://www.varprotools.org/reference/alzheimers.html)

### Historical P2 — Make upload limits and file identity visible

Streamlit's uploader accepts CSV by extension as a best-effort check and has a default per-file limit of 200 MB; applications can set a per-widget limit. The tab should show filename, byte size, detected encoding/delimiter, and the configured limit. [Streamlit `st.file_uploader`](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)

### Historical P2 — Do not silently score the project reference dataset

The current source selector defaults to “Project dataset,” so opening the combined tab immediately processes and predicts the canonical data. Keep the project dataset as a preview option, but require an explicit “Run batch prediction” action or default to upload mode when the user intends to score a new CSV.

## Recommended final workflow

```text
Choose scoring schema
        |
Read CSV + record filename/size/parser diagnostics
        |
Check required/unexpected columns and patient-record grain
        |
Exact duplicates + PatientID conflict check
        |
Domain rules + BP consistency for screening; target rules only in retraining
        |
Apply saved training imputer to recoverable feature missingness
        |
Show clean / imputed / rejected counts and downloadable audit files
        |
Screen only eligible rows, with model version + threshold + quality flags
```

## Priority order for implementation

1. Keep **Screening prediction** in the Batch tab and isolate **Training/retraining audit** in the Retrain / Model update tab.
2. Save and reuse fitted imputation parameters with the model artifact.
3. Add `PatientID` uniqueness/conflict checks and explicit schema diagnostics.
4. Add row-level quality flags and model metadata to predictions.
5. Add parser diagnostics, schema templates, visible domain dictionary, and upload metadata.

## Sources

1. [V3 project report](<../../deliverables/reports/I1_2510_E0_-_Nhom_1-_Alzheimer_project_report_V3_revised.docx>) — supplied project requirement and cleaning table.
2. [Alzheimer’s Disease Dataset data dictionary](https://www.varprotools.org/reference/alzheimers.html) — domains, codes, and dataset structure.
3. [FUSION-AD methods](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2026.1799307/full) — training-only preprocessing and imputation protocol for the same dataset.
4. [Scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) — preprocessing leakage and pipeline guidance.
5. [Pandas `read_csv`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) — missing markers, encoding, delimiter, and malformed-line controls.
6. [Streamlit `st.file_uploader`](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader) — upload limits and file-type handling.
7. [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378) — transparent reporting and implementation expectations for AI prediction models.

## Implementation update

The code now implements the priority items from this audit:

- The Batch CSV + Data processing tab is screening-only, while the Retrain / Model update tab owns labelled-data validation and model promotion.
- A separate EDA tab profiles the raw CSV before processing, including schema coverage, missingness, duplicates, negative cells, invalid formats/codes, V3 ranges, BP consistency, distributions, diagnosis balance, and final-feature correlation.
- It distinguishes five-feature scoring CSVs from full V3 CSVs and reports missing/unexpected columns.
- It detects conflicting duplicate `PatientID` values and rejects the conflicting rows.
- Newly trained artifacts save reusable imputation statistics; the app uses them during scoring.
- Prediction exports include source row, quality status, imputed fields, issue count, model version, and threshold.
- Uploaded CSVs expose filename, size, encoding, delimiter, parser behavior, templates, and allowed-value rules.
- Batch output includes the screening disclaimer and explicitly ignores a supplied `Diagnosis` in screening mode.
- A dedicated `Retrain / Model update` tab exposes the V3 retrain loop with labelled-CSV validation, backup, PR-AUC comparison, and conditional promotion.

The finalized active artifact contains the `imputation_statistics` field and
the app uses those saved values during scoring. A clearly labelled fallback is
retained only for legacy artifacts that predate this field.
