import tempfile
import unittest
from pathlib import Path

import pandas as pd

from auth import authenticate, create_user, hash_password, list_users, user_count, verify_password
from feedback_store import (
    case_counts,
    feedback_training_frame,
    get_retraining_run,
    list_cases,
    list_retraining_runs,
    record_prediction_cases,
    record_retraining_run,
    submit_feedback,
)
from ml_pipeline import ALL_FEATURES, SELECTED_FEATURES


class AuthenticationTests(unittest.TestCase):
    def test_password_hash_and_role_round_trip(self):
        encoded = hash_password("StrongPass123", salt=b"0" * 16, iterations=100_000)
        self.assertTrue(verify_password("StrongPass123", encoded))
        self.assertFalse(verify_password("WrongPass123", encoded))

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "state.sqlite3"
            created = create_user(
                "Doctor.One",
                "ClinicalPass123",
                "doctor",
                created_by="admin",
                database_path=database,
            )
            self.assertEqual("doctor.one", created.username)
            self.assertEqual(1, user_count(database))
            self.assertEqual("doctor", authenticate("DOCTOR.ONE", "ClinicalPass123", database_path=database).role)
            self.assertIsNone(authenticate("doctor.one", "bad-password", database_path=database))
            self.assertEqual(["doctor.one"], [row["username"] for row in list_users(database)])


class FeedbackStoreTests(unittest.TestCase):
    def test_verified_five_feature_case_is_retrain_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "state.sqlite3"
            values = {
                "MMSE": 20.0,
                "FunctionalAssessment": 5.0,
                "ADL": 5.0,
                "MemoryComplaints": 0,
                "BehavioralProblems": 0,
            }
            frame = pd.DataFrame([{**values, "calibrated_score": 0.3, "predicted_label": 0}])
            case_id = record_prediction_cases(
                frame, actor="doctor.one", source="single_case",
                model_version="v1", threshold=0.5, database_path=database,
            )[0]
            feedback = submit_feedback(
                case_id, 1, submitted_by="doctor.one", verified=True,
                database_path=database,
            )
            self.assertTrue(feedback["verified_model_mismatch"])
            case = list_cases(database_path=database).iloc[0]
            self.assertEqual(1, int(case["verified_diagnosis"]))
            self.assertEqual(1, int(case["verified_revision"]))
            self.assertEqual("doctor.one", case["verified_by"])
            self.assertTrue(str(case["verified_at_utc"]).startswith("20"))
            training = feedback_training_frame(database)
            self.assertEqual(["PatientID", *SELECTED_FEATURES, "Diagnosis"], training.columns.tolist())
            self.assertEqual(1, int(training.iloc[0]["Diagnosis"]))
            self.assertEqual(1, case_counts(database)["retrain_ready_cases"])

    def test_verified_full_v3_feedback_becomes_retraining_data(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "state.sqlite3"
            row = {feature: 1.0 for feature in ALL_FEATURES}
            frame = pd.DataFrame(
                [
                    {
                        **row,
                        "source_row": 1,
                        "calibrated_score": 0.8,
                        "predicted_label": 1,
                    }
                ]
            )
            first_ids = record_prediction_cases(
                frame,
                actor="doctor.one",
                source="batch_upload",
                model_version="v1",
                threshold=0.5,
                batch_key="same-upload",
                database_path=database,
            )
            second_ids = record_prediction_cases(
                frame,
                actor="doctor.one",
                source="batch_upload",
                model_version="v1",
                threshold=0.5,
                batch_key="same-upload",
                database_path=database,
            )
            self.assertEqual(first_ids, second_ids)
            self.assertEqual(1, len(list_cases(database_path=database)))

            pending = submit_feedback(
                first_ids[0],
                0,
                submitted_by="viewer.one",
                verified=False,
                database_path=database,
            )
            self.assertFalse(pending["verified_model_mismatch"])
            self.assertTrue(feedback_training_frame(database).empty)
            correction = submit_feedback(
                first_ids[0],
                1,
                submitted_by="doctor.one",
                verified=True,
                database_path=database,
            )
            self.assertEqual(2, correction["revision"])
            self.assertFalse(correction["verified_model_mismatch"])
            training = feedback_training_frame(database)
            self.assertEqual(1, len(training))
            self.assertEqual(1, int(training.iloc[0]["Diagnosis"]))
            self.assertEqual(1, case_counts(database)["retrain_ready_cases"])

            submit_feedback(
                first_ids[0], 0, submitted_by="viewer.one", verified=False,
                database_path=database,
            )
            case = list_cases(database_path=database).iloc[0]
            self.assertEqual(0, int(case["latest_diagnosis"]))
            self.assertEqual(0, int(case["latest_verified"]))
            self.assertEqual(1, int(case["verified_diagnosis"]))
            self.assertEqual(2, int(case["verified_revision"]))
            self.assertEqual("doctor.one", case["verified_by"])
            self.assertEqual(1, int(feedback_training_frame(database).iloc[0]["Diagnosis"]))

    def test_retraining_session_is_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "state.sqlite3"
            run_id = record_retraining_run(
                started_by="admin",
                source="Verified clinical feedback",
                source_rows=3,
                status="retained_champion",
                policy_passed=False,
                promoted=False,
                champion_version="v1",
                challenger_version="v2",
                old_pr_auc=0.91,
                new_pr_auc=0.90,
                details={"promotion_decision": {"checks": [{"metric": "pr_auc", "passed": False}]}},
                database_path=database,
            )
            runs = list_retraining_runs(database_path=database)
            self.assertEqual(run_id, runs.iloc[0]["run_id"])
            self.assertEqual("retained_champion", runs.iloc[0]["status"])
            saved = get_retraining_run(run_id, database_path=database)
            self.assertEqual("pr_auc", saved["details"]["promotion_decision"]["checks"][0]["metric"])
            self.assertIsNone(get_retraining_run("RUN-UNKNOWN", database_path=database))


if __name__ == "__main__":
    unittest.main()
