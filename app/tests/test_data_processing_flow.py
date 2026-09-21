import unittest

import pandas as pd

from ml_pipeline import ALL_FEATURES, SELECTED_FEATURES
from ui.data_processing import (
    build_error_review_table,
    cleaning_removed_rows,
    detect_v3_schema,
    validate_five_feature_scoring_columns,
)


class DataProcessingFlowTests(unittest.TestCase):
    def test_schema_is_detected_from_columns_without_a_manual_choice(self):
        self.assertEqual(
            "Full V3 scoring CSV",
            detect_v3_schema(["PatientID", *ALL_FEATURES, "Diagnosis"]),
        )
        self.assertEqual(
            "Five-feature scoring CSV",
            detect_v3_schema([*SELECTED_FEATURES, "Diagnosis"]),
        )
        with self.assertRaisesRegex(ValueError, "BehavioralProblems"):
            detect_v3_schema(SELECTED_FEATURES[:-1])

    def test_batch_prediction_accepts_only_five_model_inputs_and_optional_id(self):
        validate_five_feature_scoring_columns(SELECTED_FEATURES)
        validate_five_feature_scoring_columns(["PatientID", *SELECTED_FEATURES])
        with self.assertRaisesRegex(ValueError, "Dataset dự án"):
            validate_five_feature_scoring_columns(["PatientID", *ALL_FEATURES])
        with self.assertRaisesRegex(ValueError, "Cập nhật mô hình"):
            validate_five_feature_scoring_columns([*SELECTED_FEATURES, "Diagnosis"])
        with self.assertRaisesRegex(ValueError, "BehavioralProblems"):
            validate_five_feature_scoring_columns(SELECTED_FEATURES[:-1])
        with self.assertRaisesRegex(ValueError, "Age"):
            validate_five_feature_scoring_columns([*SELECTED_FEATURES, "Age"])

    def test_cleaning_audit_accounts_for_duplicates_and_invalid_bp(self):
        raw = pd.DataFrame(
            {
                "_source_row": [1, 2, 3, 4],
                "PatientID": [10, 10, 11, 12],
                "SystolicBP": [120, 120, 80, 130],
                "DiastolicBP": [80, 80, 90, 90],
            }
        )
        cleaned = raw.loc[[0, 3]].copy()
        removed = cleaning_removed_rows(raw, cleaned, full_schema=True)
        self.assertEqual([2, 3], removed["_source_row"].tolist())
        self.assertEqual(
            ["duplicate row", "SystolicBP <= DiastolicBP"],
            removed["_removal_reason"].tolist(),
        )
        self.assertEqual(len(raw), len(cleaned) + len(removed))

    def test_error_review_marks_corrected_rejected_and_unhandled_values(self):
        raw = pd.DataFrame(
            {
                "_source_row": [1, 2, 3, 4],
                "PatientID": [10, 11, -4751, 12],
                "MMSE": [None, -1, 20, 20],
                "FunctionalAssessment": [5, 5, 5, 5],
                "ADL": [5, 5, 5, 5],
                "MemoryComplaints": [0, 0, 0, 0],
                "BehavioralProblems": [0, 0, 0, 0],
                "SystolicBP": [120, 120, 120, 80],
                "DiastolicBP": [80, 80, 80, 90],
            }
        )
        valid = pd.DataFrame(
            {"_row_number": [1, 3], "MMSE": [20, 20], "PatientID": [10, -4751]}
        )
        issues = pd.DataFrame(
            [
                {"row": 1, "field": "MMSE", "issue": "missing value", "action": "imputed with 20"},
                {"row": 2, "field": "MMSE", "issue": "negative value -1", "action": "rejected row"},
                {"row": 4, "field": "SystolicBP/DiastolicBP", "issue": "SystolicBP <= DiastolicBP", "action": "removed before validation"},
            ]
        )
        review = build_error_review_table(raw, valid, issues)
        self.assertEqual(4, len(review))
        self.assertEqual(
            ["corrected", "rejected", "review", "rejected"],
            review["status"].tolist(),
        )
        self.assertEqual("Thiếu", review.loc[0, "original"])
        self.assertEqual("20", review.loc[0, "processed"])
        self.assertEqual("PatientID", review.loc[2, "field"])
        self.assertIn("chưa được", review.loc[2, "issue"])


if __name__ == "__main__":
    unittest.main()
