# Data Cleaning Research for Alzheimer V3

## Executive conclusion

The correct cleaning policy for this project is not to delete or impute rows indiscriminately. The supplied V3 dataset is already complete and unique at the patient-record level. The only confirmed source-data defect is an impossible blood-pressure ordering rule: `SystolicBP <= DiastolicBP`. Those 186 rows should be quarantined before exploratory analysis, splitting, feature selection, and model training.

For new CSVs, the application should use two different actions. Recoverable feature problems such as missing values or parseable formatting errors may be converted to missing and imputed with the training/reference median or mode. Invalid domain values, unknown category codes, missing/invalid targets, duplicate patient records, and impossible blood-pressure ordering should be rejected and logged row by row. The target must never be imputed.

## Evidence from the project files

The primary project requirement is `../../deliverables/reports/I1_2510_E0_-_Nhom_1-_Alzheimer_project_report_V3_revised.docx`. Its cleaning table specifies:

- 2,149 rows and 35 columns.
- 0 exact duplicate rows.
- 0 missing cells.
- 186 rows where `SystolicBP <= DiastolicBP`; those rows are removed.
- One unique `DoctorInCharge` value, so it is not a useful predictor.
- 2,149 unique `PatientID` values, so it is an identifier rather than a predictor.

The local audit of `alzheimers_disease_data.csv` reproduces those figures. It also confirms that all documented binary codes and numeric source ranges are valid before the blood-pressure rule is applied. After removing the 186 BP-invalid rows, the dataset has 1,963 rows. The positive-label rate changes from 760/2,149 = 35.36% to 694/1,963 = 35.35%, so the cleaning rule does not materially change class prevalence.

The V3 validator test table further specifies the intended behavior: missing feature values are imputed; formatting errors are converted to missing and then imputed; negative or out-of-range values are rejected; unknown categorical codes are rejected; and the BP ordering rule is rejected.

### Negative values and a missing value in a new CSV

For this project, a negative number in a feature is an invalid domain value, not a missing value. The validator therefore rejects the complete row and records the field in the processing log. It must not apply `abs()`, replace the value with zero, or pass the negative value into imputation. This follows the V3 examples for impossible negative Age/Cholesterol values and is consistent with the source dictionary, where the model fields are binary, bounded scores, or positive measurements.

If exactly one selected feature is blank or has a recoverable formatting error, the row remains usable: the value is converted to missing, imputed with the reference median for numeric scores or mode for categorical/binary fields, and logged. The reference values must come from the training/reference data, not be re-estimated from the uploaded scoring CSV. Scikit-learn documents median/mode imputers and recommends fitting preprocessing only on the training data and applying the fitted transform unchanged to later data. [Scikit-learn imputation guide](https://scikit-learn.org/stable/modules/impute.html) [Scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)

If all five selected model features are missing in the same row, the row is rejected instead of being filled five times. Five imputed values would conceal a broken record and create an invented average patient with no observed model evidence. During retraining, a missing or invalid `Diagnosis` is also rejected because the target must never be imputed. The Streamlit processing tab displays these rules, counts negative cells, and exports both the log and rejected rows.

## External evidence and data dictionary

The dataset data dictionary documents the source domains: Age 60–90, BMI 15–40, SleepQuality 4–10, systolic BP 90–180 mmHg, diastolic BP 60–120 mmHg, total cholesterol 150–300 mg/dL, LDL 50–200 mg/dL, HDL 20–100 mg/dL, triglycerides 50–400 mg/dL, MMSE 0–30, and FunctionalAssessment/ADL 0–10. It also defines the binary and four-level categorical codes. These ranges are used as domain checks for full V3 files. [Alzheimer’s Disease Dataset data dictionary](https://www.varprotools.org/reference/alzheimers.html)

An independent 2026 study using the same 2,149-record synthetic dataset describes the same identifier range and recommends training-split-only estimates for imputation, categorical modes, standardization, and winsorization. It explicitly applies preprocessing parameters learned on training data to validation and test data unchanged. [FUSION-AD methods](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2026.1799307/full)

Scikit-learn’s guidance reaches the same methodological conclusion: split before fitting preprocessing, never fit transforms on the test set, and use a `Pipeline` so imputation, feature selection, and the estimator are fitted on the correct fold. [Scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) [Scikit-learn imputation guide](https://scikit-learn.org/stable/modules/impute.html)

For a health prediction study, TRIPOD+AI recommends transparent reporting of the development/evaluation data, model selection, missing-data handling, performance evaluation, subgroup performance, and open-science details. [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378)

## Correct processing contract

### 1. Ingest and identify the data grain

Require the V3 schema for training/retraining. Treat one row as one patient record. Preserve `PatientID` only for traceability and audit; exclude both `PatientID` and `DoctorInCharge` from model features. Reject missing required columns before any transformation.

### 2. Normalize types without inventing clinical values

Convert numeric columns with `errors='coerce'`. Record every parse error. For feature columns, a parse error follows the V3 recoverable-error rule: convert to missing, then impute using a value learned from the training/reference data. Do not silently coerce an invalid target into a label.

### 3. Remove duplicates and protect identifiers

Remove exact duplicate rows before splitting. Check `PatientID` uniqueness separately. Exact duplicates are safe to drop; conflicting rows sharing a `PatientID` should be rejected for manual review rather than silently merged.

### 4. Apply domain and cross-field rules

Reject rows that violate a documented numeric domain, contain an unknown categorical/binary code, have a missing/invalid `Diagnosis` during training, or satisfy `SystolicBP <= DiastolicBP`. Domain checks are validity rules, not imputation rules. They should produce a row-level issue log and a rejected-row export.

### 5. Split before learned preprocessing

After deterministic row cleaning, lock the test set. Fit imputation, scaling, winsorization, feature selection, class balancing, calibration, and threshold selection only on the appropriate training/validation data. Apply the fitted transformations unchanged to validation, test, and new scoring data. This prevents leakage and keeps the reported metrics reproducible.

### 6. Keep scoring and retraining behavior distinct

For prediction-only CSVs, `Diagnosis` is absent and the validator returns valid feature rows plus screening results. For retraining CSVs, `Diagnosis` is required and invalid/missing target rows are rejected. The retraining flow should clean the new batch with the same duplicate, domain, BP, and target rules before concatenating it with the existing clean dataset.

## Audit of the current implementation

The pipeline already implemented the most important V3 rule correctly: exact duplicate removal and removal of the 186 impossible BP rows before the model split. The model also keeps the official five-feature set and preserves the V3 calibrated-threshold workflow.

The main correctness gap was in the validator. `Ethnicity` was not included in the numeric validation loop, so its unknown codes could bypass the categorical check. `EducationLevel` was not treated as a categorical field for imputation semantics. Several domain ranges were broader than the source data dictionary, and a missing/invalid target could previously be imputed instead of rejected. These gaps are now corrected in `ml_pipeline.py`.

## Implementation status

The current code now:

- Uses the V3 data-domain ranges for full-dataset validation.
- Validates `Ethnicity` codes and treats `EducationLevel` as categorical for imputation.
- Rejects missing or invalid `Diagnosis` values instead of fabricating labels.
- Preserves V3 behavior for recoverable feature formatting/missingness.
- Rejects negative/out-of-range values with the observed value in the issue message.
- Keeps a row with one missing feature through reference imputation, but rejects a row when all five selected model features are missing.
- Keeps BP-ordering violations as rejected rows.
- Applies the same deterministic duplicate/BP cleaning to new labelled retraining batches before they are merged.
- Exposes the full step-by-step processing flow and downloadable logs in the Streamlit data-processing tab.
- Separates screening validation from training/retraining validation so an optional `Diagnosis` is ignored during scoring but required during training audit.
- Saves reusable imputation statistics in newly trained model artifacts and includes row-level quality metadata in batch predictions.
- Checks conflicting duplicate `PatientID` values and exposes CSV schema/parser diagnostics in the Batch CSV + Data processing tab; labelled-data checks remain in the Retrain / Model update tab.

The current source audit passes with 9 unit tests. The canonical dataset still produces 2,149 raw rows, 1,963 cleaned rows, 186 BP rows removed, 0 validation issues, and 0 rejected rows after deterministic cleaning.

## Limitations and open questions

The dataset is synthetic and should be treated as a reproducible benchmark, not as evidence of clinical performance in a real patient population. The National Institute on Aging describes Alzheimer’s diagnosis as a broader clinical assessment that may include history, cognitive testing, laboratory evaluation, imaging, and biomarkers; a five-feature tabular screen cannot replace that process. [National Institute on Aging diagnosis guidance](https://www.nia.nih.gov/health/alzheimers-symptoms-and-diagnosis/how-alzheimers-disease-diagnosed)

The app should remain explicitly labelled as an educational/research screening demonstration. WHO guidance emphasizes human control of medical decisions, safety, transparency, accountability, and equity for AI used in health. [WHO ethics and governance guidance](https://www.who.int/publications/i/item/9789240037403)

Before any real-world claim, the project still needs external validation, subgroup performance checks, a documented source/license for the dataset, and a locked version of the code, dataset, and artifacts that reproduces every reported metric.
