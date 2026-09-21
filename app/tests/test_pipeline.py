import unittest

import numpy as np
import pandas as pd

from ml_pipeline import (
    DEFAULT_DATA_PATH,
    SELECTED_FEATURES,
    _exclude_existing_patient_ids,
    audit_bp_rows,
    clean_dataset,
    read_dataset,
    run_cleaning_sensitivity,
    threshold_table,
    validate_dataframe,
)


class PipelineSchemaTests(unittest.TestCase):
    def test_retraining_skips_patient_ids_already_in_existing_dataset(self):
        existing = pd.DataFrame({"PatientID": [4751, 4752]})
        uploaded = pd.DataFrame({"PatientID": [4752, 9001, 9002]})

        remaining, skipped = _exclude_existing_patient_ids(uploaded, existing)

        self.assertEqual(skipped, 1)
        self.assertEqual(remaining["PatientID"].tolist(), [9001, 9002])

    def test_cleaning_reproduces_v3_row_count(self):
        data = read_dataset(DEFAULT_DATA_PATH)
        cleaned, report = clean_dataset(data)
        self.assertEqual(data.shape, (2149, 35))
        self.assertEqual(len(cleaned), 1963)
        self.assertEqual(report["invalid_blood_pressure_rows_removed"], 186)
        self.assertEqual(int(cleaned["Diagnosis"].isna().sum()), 0)

    def test_validator_imputes_recoverable_values_and_rejects_bad_rows(self):
        reference = read_dataset(DEFAULT_DATA_PATH)
        frame = pd.DataFrame(
            {
                "MMSE": [np.nan, "N/A", -1, 31],
                "FunctionalAssessment": [5, 5, 5, 5],
                "ADL": [5, 5, 5, 5],
                "MemoryComplaints": [0, 0, 0, 0],
                "BehavioralProblems": [0, 0, 0, 0],
            }
        )
        result = validate_dataframe(frame, reference_data=reference)
        self.assertEqual(result.summary["valid_rows"], 2)
        self.assertEqual(result.summary["rejected_rows"], 2)
        self.assertGreaterEqual(result.summary["imputed_issue_count"], 2)
        self.assertTrue(result.valid["MMSE"].between(0, 30).all())

    def test_selected_schema_is_exactly_five_features(self):
        self.assertEqual(
            SELECTED_FEATURES,
            ["MMSE", "FunctionalAssessment", "ADL", "MemoryComplaints", "BehavioralProblems"],
        )

    def test_validator_rejects_dataset_domain_and_target_errors(self):
        reference = read_dataset(DEFAULT_DATA_PATH)
        frame = pd.DataFrame(
            {
                "Age": [75, 50],
                "Ethnicity": [0, 9],
                "EducationLevel": [2, 4],
                "MMSE": [20, 20],
                "FunctionalAssessment": [5, 5],
                "ADL": [5, 5],
                "MemoryComplaints": [0, 0],
                "BehavioralProblems": [0, 0],
                "Diagnosis": [0, np.nan],
            }
        )
        result = validate_dataframe(frame, reference_data=reference, mode="training")
        self.assertEqual(result.summary["valid_rows"], 1)
        self.assertEqual(result.summary["rejected_rows"], 1)
        rejected_fields = set(result.issues.loc[result.issues["action"] == "rejected row", "field"])
        self.assertTrue({"Age", "Ethnicity", "EducationLevel", "Diagnosis"}.issubset(rejected_fields))

    def test_validator_rejects_negative_and_all_missing_feature_rows(self):
        reference = read_dataset(DEFAULT_DATA_PATH)
        frame = pd.DataFrame(
            {
                "MMSE": [-5, np.nan, np.nan],
                "FunctionalAssessment": [5, 5, np.nan],
                "ADL": [5, 5, np.nan],
                "MemoryComplaints": [0, 0, np.nan],
                "BehavioralProblems": [0, 0, np.nan],
            }
        )
        result = validate_dataframe(frame, reference_data=reference)
        self.assertEqual(result.summary["valid_rows"], 1)
        self.assertEqual(result.summary["rejected_rows"], 2)
        self.assertEqual(result.summary["negative_issue_count"], 1)
        self.assertTrue(
            result.issues["issue"].str.contains("all selected model features missing").any()
        )
        self.assertTrue(result.valid["MMSE"].between(0, 30).all())

    def test_screening_does_not_require_diagnosis(self):
        reference = read_dataset(DEFAULT_DATA_PATH)
        frame = reference.iloc[[0]][
            ["MMSE", "FunctionalAssessment", "ADL", "MemoryComplaints", "BehavioralProblems", "Diagnosis"]
        ].copy()
        frame["Diagnosis"] = np.nan
        screening = validate_dataframe(frame, reference_data=reference, mode="screening")
        training = validate_dataframe(frame, reference_data=reference, mode="training")
        self.assertEqual(screening.summary["valid_rows"], 1)
        self.assertEqual(screening.summary["rejected_rows"], 0)
        self.assertEqual(training.summary["valid_rows"], 0)
        self.assertEqual(training.summary["rejected_rows"], 1)

    def test_validator_rejects_conflicting_patient_ids_and_exposes_quality_fields(self):
        reference = read_dataset(DEFAULT_DATA_PATH)
        frame = reference.iloc[[0, 1]][
            [
                "PatientID",
                "MMSE",
                "FunctionalAssessment",
                "ADL",
                "MemoryComplaints",
                "BehavioralProblems",
            ]
        ].copy()
        frame.iloc[1, frame.columns.get_loc("PatientID")] = frame.iloc[0]["PatientID"]
        frame.iloc[0, frame.columns.get_loc("MMSE")] = np.nan
        result = validate_dataframe(frame, reference_data=reference)
        self.assertEqual(result.summary["valid_rows"], 0)
        self.assertEqual(result.summary["rejected_rows"], 2)
        self.assertTrue(result.issues["issue"].eq("conflicting duplicate PatientID").any())
        self.assertIn("_quality_status", result.rejected.columns)
        self.assertIn("_imputed_fields", result.rejected.columns)

    def test_validator_uses_saved_imputation_statistics(self):
        reference = read_dataset(DEFAULT_DATA_PATH)
        frame = pd.DataFrame(
            {
                "MMSE": [np.nan],
                "FunctionalAssessment": [5],
                "ADL": [5],
                "MemoryComplaints": [0],
                "BehavioralProblems": [0],
            }
        )
        result = validate_dataframe(
            frame,
            reference_data=reference,
            imputation_statistics={"MMSE": 22.5},
        )
        self.assertEqual(result.valid.iloc[0]["MMSE"], 22.5)
        self.assertEqual(result.valid.iloc[0]["_quality_status"], "imputed")
        self.assertEqual(result.valid.iloc[0]["_imputed_fields"], "MMSE")

    def test_validator_preserves_source_row_after_cleaning(self):
        reference = read_dataset(DEFAULT_DATA_PATH)
        frame = reference.iloc[[0, 1]][
            [
                "PatientID",
                "MMSE",
                "FunctionalAssessment",
                "ADL",
                "MemoryComplaints",
                "BehavioralProblems",
            ]
        ].copy()
        frame["_source_row"] = [8, 12]
        result = validate_dataframe(frame, reference_data=reference)
        self.assertEqual(result.valid["_row_number"].tolist(), [8, 12])

    def test_bp_audit_and_sensitivity_are_reproducible(self):
        data = read_dataset(DEFAULT_DATA_PATH)
        audit = audit_bp_rows(data)
        self.assertEqual(len(audit), len(data))
        self.assertEqual(int(audit["bp_order_invalid"].sum()), 186)
        result = run_cleaning_sensitivity(data)
        self.assertEqual(result.set_index("strategy").loc["remove", "rows"], 1963)
        self.assertEqual(result.set_index("strategy").loc["keep-with-flag", "bp_invalid_rows_retained"], 186)

    def test_threshold_table_has_explicit_operating_points(self):
        table = threshold_table([0, 1, 0, 1], [0.1, 0.8, 0.4, 0.6], thresholds=[0.5])
        row = table.iloc[0]
        self.assertEqual(row["false_negatives"], 0)
        self.assertEqual(row["true_positives"], 2)
        self.assertEqual(row["specificity"], 1.0)


if __name__ == "__main__":
    unittest.main()
