import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import ml_pipeline
from ml_pipeline import (
    ARTIFACT_SCHEMA_VERSION,
    SELECTED_FEATURES,
    _exclude_locked_rows,
    _promote_artifacts,
    calibration_summary,
    choose_imbalance_method,
    evaluate_payload,
    evaluate_promotion_policy,
    feature_importance_percentages,
    paired_bootstrap_difference,
    predict_dataframe,
    prepare_retraining_rows,
    validate_artifact_contract,
)


class FakeCalibratedModel:
    def predict_proba(self, features):
        score = np.clip(pd.to_numeric(features["MMSE"], errors="coerce").to_numpy() / 30.0, 0.01, 0.99)
        return np.column_stack([1.0 - score, score])


class FakeImportanceEstimator:
    def __init__(self, values):
        self.feature_importances_ = np.asarray(values, dtype=float)


class FakeCalibratedFold:
    def __init__(self, values):
        self.estimator = type(
            "FakePipeline",
            (),
            {"named_steps": {"model": FakeImportanceEstimator(values)}},
        )()


class FakeImportanceModel:
    def __init__(self):
        self.calibrated_classifiers_ = [
            FakeCalibratedFold([1, 2, 3, 4, 5]),
            FakeCalibratedFold([2, 3, 4, 5, 6]),
        ]


def _payload(threshold=0.5):
    return {
        "model": FakeCalibratedModel(),
        "features": SELECTED_FEATURES,
        "threshold": threshold,
        "version": "test",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "imputation_statistics": {},
    }


class ArtifactAndPromotionTests(unittest.TestCase):
    def test_feature_importance_percentages_average_folds_and_sum_to_100(self):
        rows = feature_importance_percentages(
            {"model": FakeImportanceModel(), "features": SELECTED_FEATURES}
        )
        self.assertEqual(SELECTED_FEATURES, [row["feature"] for row in rows])
        self.assertAlmostEqual(100.0, sum(row["percent"] for row in rows))
        self.assertGreater(rows[-1]["percent"], rows[0]["percent"])

    def test_five_feature_retraining_validates_label_and_holds_out_locked_match(self):
        def case(mmse, diagnosis, patient_id=None):
            return {
                "PatientID": patient_id,
                "MMSE": mmse,
                "FunctionalAssessment": 5.0,
                "ADL": 5.0,
                "MemoryComplaints": 0,
                "BehavioralProblems": 0,
                "Diagnosis": diagnosis,
            }

        base = pd.DataFrame([case(25.0, 0, "BASE-1")])
        locked = pd.DataFrame([case(10.0, 1, "LOCK-1")])
        accepted = pd.DataFrame([case(24.0, 0)])
        upload = pd.DataFrame([
            case(18.0, 1),
            case(10.0, 0),  # Same model inputs as locked test, even with a different label.
            case(24.0, 0),  # Repeated anonymous case.
            case(22.0, None),
        ])
        validation, fresh, skipped = prepare_retraining_rows(upload, base, accepted, locked)
        self.assertEqual(1, len(fresh))
        self.assertEqual(18.0, fresh.iloc[0]["MMSE"])
        self.assertEqual(1, len(validation.rejected))
        self.assertEqual(1, skipped["locked_feature_rows_skipped"])
        self.assertEqual(1, skipped["previous_rows_skipped"])

    def test_verified_label_correction_is_not_discarded_as_duplicate(self):
        def case(diagnosis, patient_id):
            return {
                "PatientID": patient_id,
                "MMSE": 18.0,
                "FunctionalAssessment": 5.0,
                "ADL": 5.0,
                "MemoryComplaints": 0,
                "BehavioralProblems": 0,
                "Diagnosis": diagnosis,
            }

        base = pd.DataFrame([case(0, "BASE-1")])
        base["MMSE"] = 25.0
        locked = pd.DataFrame([case(1, "LOCK-1")])
        locked["MMSE"] = 10.0
        accepted = pd.DataFrame([case(0, "FB-1")])
        upload = pd.DataFrame([case(1, "FB-1")])
        _, fresh, skipped = prepare_retraining_rows(upload, base, accepted, locked)
        self.assertEqual(1, len(fresh))
        self.assertEqual(1, skipped["corrected_rows"])
        self.assertEqual(0, skipped["previous_rows_skipped"])

    def test_evaluate_payload_returns_metrics_confusion_and_row_predictions(self):
        frame = pd.DataFrame(
            {
                "PatientID": [10, 11, 12, 13],
                "_source_row": [2, 4, 6, 8],
                "MMSE": [3, 27, 6, 24],
                "FunctionalAssessment": [5] * 4,
                "ADL": [5] * 4,
                "MemoryComplaints": [0] * 4,
                "BehavioralProblems": [0] * 4,
                "Diagnosis": [0, 1, 0, 1],
            }
        )
        def confusion(actual, predicted, labels):
            return np.array([[2, 0], [0, 2]])
        with patch.object(ml_pipeline, "_metric_values", return_value={"pr_auc": 1.0}), patch.object(
            ml_pipeline, "_import_sklearn", return_value={"confusion_matrix": confusion}
        ):
            result = evaluate_payload(_payload(), frame)
        self.assertEqual(result["n_rows"], 4)
        self.assertEqual(result["row_predictions"]["_source_row"].tolist(), [2, 4, 6, 8])
        self.assertEqual(result["confusion_matrix"]["counts"], [[2, 0], [0, 2]])

    def test_champion_and_challenger_use_same_locked_rows(self):
        frame = pd.DataFrame(
            {
                "_source_row": [11, 12],
                "PatientID": [101, 102],
                "MMSE": [8, 22],
                "FunctionalAssessment": [5, 5],
                "ADL": [5, 5],
                "MemoryComplaints": [0, 1],
                "BehavioralProblems": [0, 0],
                "Diagnosis": [0, 1],
            }
        )
        with patch.object(ml_pipeline, "_metric_values", return_value={"pr_auc": 1.0}), patch.object(
            ml_pipeline, "_import_sklearn", return_value={"confusion_matrix": lambda y, p, labels: np.array([[1, 0], [0, 1]])}
        ):
            champion = evaluate_payload(_payload(), frame)
            challenger = evaluate_payload(_payload(threshold=0.4), frame)
        identity = ["_source_row", "PatientID"]
        self.assertTrue(champion["row_predictions"][identity].equals(challenger["row_predictions"][identity]))

    def test_paired_bootstrap_is_deterministic(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        old = np.array([0.2, 0.6, 0.3, 0.7, 0.4, 0.5])
        new = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.6])
        first = paired_bootstrap_difference(y, old, new, lambda actual, scores: float(np.mean(actual == (scores >= 0.5))), 200)
        second = paired_bootstrap_difference(y, old, new, lambda actual, scores: float(np.mean(actual == (scores >= 0.5))), 200)
        self.assertEqual(first, second)

    def test_better_pr_auc_but_bad_recall_is_not_promoted(self):
        champion = {"version": "old", "metrics": {"pr_auc": 0.80, "recall": 0.90, "brier": 0.10, "f2": 0.80}}
        challenger = {"version": "new", "metrics": {"pr_auc": 0.85, "recall": 0.70, "brier": 0.09, "f2": 0.90}}
        result = evaluate_promotion_policy(champion, challenger, paired_differences={"f2": {"ci_low": 0.05}})
        self.assertFalse(result["passed"])
        self.assertEqual([check["status"] for check in result["checks"] if check["metric"] == "recall"], ["Fail"])

    def test_worse_challenger_is_not_promoted(self):
        champion = {"version": "old", "metrics": {"pr_auc": 0.80, "recall": 0.90, "brier": 0.10, "f2": 0.80}}
        challenger = {"version": "new", "metrics": {"pr_auc": 0.70, "recall": 0.80, "brier": 0.20, "f2": 0.70}}
        result = evaluate_promotion_policy(champion, challenger, paired_differences={"f2": {"ci_low": -0.20}})
        self.assertFalse(result["passed"])

    def test_no_locked_patient_enters_training(self):
        data = pd.DataFrame({"PatientID": [1, 2, 3], "_source_row": [1, 2, 3]})
        locked = pd.DataFrame({"PatientID": [2], "_source_row": [2]})
        remaining, count = _exclude_locked_rows(data, locked)
        self.assertEqual(count, 1)
        self.assertEqual(remaining["PatientID"].tolist(), [1, 3])

    def test_promotion_writes_model_and_metadata_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "active_model.joblib").write_bytes(b"old-model")
            (root / "active_metadata.json").write_bytes(b"old-metadata")
            (root / "versions").mkdir()
            version_model = root / "versions" / "model.joblib"
            version_metadata = root / "versions" / "metadata.json"
            version_model.write_bytes(b"new-model")
            version_metadata.write_bytes(b"new-metadata")

            original_copy = ml_pipeline.shutil.copy2

            def fail_metadata(source, destination, *args, **kwargs):
                if Path(source) == version_metadata:
                    raise OSError("simulated staging failure")
                return original_copy(source, destination, *args, **kwargs)

            with patch.object(ml_pipeline.shutil, "copy2", side_effect=fail_metadata):
                with self.assertRaises(OSError):
                    _promote_artifacts(version_model, version_metadata, root)
            self.assertEqual((root / "active_model.joblib").read_bytes(), b"old-model")
            self.assertEqual((root / "active_metadata.json").read_bytes(), b"old-metadata")

    def test_artifact_contract_rejects_wrong_feature_order(self):
        metadata = {
            "schema_version": 2,
            "version": "test",
            "selected_features": list(reversed(SELECTED_FEATURES)),
            "threshold": 0.5,
            "imputation_statistics": {},
            "dataset_sha256": "dataset",
            "model_sha256": "model",
            "locked_test_sha256": "locked",
            "python_version": "3.12",
            "package_versions": {},
            "trained_at_utc": "now",
            "training_command": "python train_pipeline.py",
            "validation_scope": "internal",
        }
        with self.assertRaises(RuntimeError):
            validate_artifact_contract({**_payload(), "features": list(reversed(SELECTED_FEATURES))}, metadata)

    def test_prediction_changes_at_exact_threshold(self):
        frame = pd.DataFrame({
            "MMSE": [15.0], "FunctionalAssessment": [5], "ADL": [5],
            "MemoryComplaints": [0], "BehavioralProblems": [0],
        })
        prediction = predict_dataframe(_payload(threshold=0.5), frame)
        self.assertEqual(prediction.iloc[0]["screen_prediction"], "Positive")

    def test_calibrated_score_summary_is_bounded_and_has_null_brier(self):
        summary = calibration_summary([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.9])
        self.assertLessEqual(summary["brier"], summary["null_brier"])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in summary["mean_predicted"]))

    def test_imbalance_selection_uses_simple_tie_rule_when_screening_metrics_are_absent(self):
        comparison = pd.DataFrame([
            {"method": "Baseline", "pr_auc": 0.900, "fold_count": 10, "status": "available"},
            {"method": "scale_pos_weight", "pr_auc": 0.904, "fold_count": 10, "status": "available"},
        ])
        self.assertEqual(choose_imbalance_method(comparison)["method"], "Baseline")

    def test_screening_tie_prefers_higher_f2_and_recall(self):
        comparison = pd.DataFrame([
            {"method": "Baseline", "pr_auc": 0.939, "f2": 0.932, "recall": 0.934, "brier": 0.049, "fold_count": 10, "status": "available"},
            {"method": "scale_pos_weight", "pr_auc": 0.940, "f2": 0.936, "recall": 0.943, "brier": 0.053, "fold_count": 10, "status": "available"},
            {"method": "SMOTENC", "pr_auc": 0.938, "f2": 0.933, "recall": 0.940, "brier": 0.054, "fold_count": 10, "status": "available"},
        ])
        result = choose_imbalance_method(comparison)
        self.assertEqual(result["method"], "scale_pos_weight")
        self.assertAlmostEqual(result["selected_mean_f2"], 0.936)


if __name__ == "__main__":
    unittest.main()
