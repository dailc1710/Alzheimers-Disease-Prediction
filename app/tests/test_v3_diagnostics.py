import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import ml_pipeline
from ml_pipeline import (
    DEFAULT_DATA_PATH,
    REQUIRED_METADATA_KEYS,
    SELECTED_FEATURES,
    _atomic_write_text,
    _dataframe_sha256,
    _exclude_locked_rows,
    _locked_test_frame,
    _monitoring_baseline,
    _package_versions,
    _save_artifacts,
    _source_release_sha256,
    audit_bp_rows,
    bootstrap_confidence_intervals,
    choose_f2_threshold,
    exploratory_summary,
    load_artifacts,
    make_candidate_models,
    monitor_prediction_batch,
    rank_features,
    run_training,
    save_eda_plots,
    subgroup_metrics,
    summarize_bp_audit,
)


class DiagnosticModel:
    def predict_proba(self, features):
        score = np.clip(pd.to_numeric(features["MMSE"], errors="coerce").fillna(15).to_numpy() / 30.0, 0.01, 0.99)
        return np.column_stack([1.0 - score, score])


class V3DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ml_pipeline.read_dataset(DEFAULT_DATA_PATH)
        cls.cleaned, _ = ml_pipeline.clean_dataset(cls.data)

    def test_hashes_metadata_and_locking_helpers(self):
        with self.assertRaises(ValueError):
            ml_pipeline._require_columns(pd.DataFrame(), ["missing"])
        self.assertEqual(len(_dataframe_sha256(self.data)), 64)
        self.assertEqual(len(_source_release_sha256()), 64)
        self.assertIn("pandas", _package_versions())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "value.txt"
            _atomic_write_text(path, "ok")
            self.assertEqual(path.read_text(encoding="utf-8"), "ok")
            first, created = _locked_test_frame(self.data, Path(directory) / "locked.csv")
            second, created_again = _locked_test_frame(self.data, Path(directory) / "locked.csv")
            self.assertTrue(created)
            self.assertFalse(created_again)
            self.assertEqual(first["PatientID"].tolist(), second["PatientID"].tolist())

    def test_locked_row_exclusion_prefers_stable_patient_ids(self):
        data = pd.DataFrame(
            {
                "PatientID": [100, 200, 300],
                "_source_row": [1, 2, 3],
            }
        )
        locked = pd.DataFrame(
            {
                "PatientID": [300],
                "_source_row": [1],
            }
        )

        remaining, excluded = _exclude_locked_rows(data, locked)

        self.assertEqual(1, excluded)
        self.assertEqual([100, 200], remaining["PatientID"].tolist())

    def test_locked_row_exclusion_falls_back_to_source_rows(self):
        data = pd.DataFrame({"_source_row": [10, 20, 30]})
        locked = pd.DataFrame({"_source_row": [20]})

        remaining, excluded = _exclude_locked_rows(data, locked)

        self.assertEqual(1, excluded)
        self.assertEqual([10, 30], remaining["_source_row"].tolist())

    def test_cleaning_duplicate_and_audit_summary(self):
        duplicated = pd.concat([self.data.iloc[[0]], self.data.iloc[[0]]], ignore_index=True)
        cleaned, report = ml_pipeline.clean_dataset(duplicated)
        self.assertEqual(report["duplicates_removed"], 1)
        audit = audit_bp_rows(self.data)
        summary = summarize_bp_audit(audit)
        self.assertIn("status", summary.columns)
        with self.assertRaises(ValueError):
            summarize_bp_audit(pd.DataFrame({"status": ["kept"]}))
        self.assertEqual(ml_pipeline._reference_median(None, "MMSE"), 0.0)
        self.assertEqual(ml_pipeline._reference_mode(None, "Gender"), 0.0)
        statistics = ml_pipeline.fit_imputation_statistics(self.cleaned)
        self.assertIn("MMSE", statistics)

    def test_eda_and_model_factories(self):
        summary = exploratory_summary(self.cleaned)
        self.assertIn("correlation", summary)
        with tempfile.TemporaryDirectory() as directory:
            plots = save_eda_plots(self.cleaned, summary, directory)
            self.assertGreaterEqual(len(plots), 0)
        features = self.cleaned[SELECTED_FEATURES].iloc[:80]
        target = self.cleaned["Diagnosis"].iloc[:80]
        candidates = make_candidate_models(target)
        self.assertEqual(len(candidates), 7)
        metrics = ml_pipeline.repeated_cv_metrics(candidates["DummyClassifier"], features, target, n_splits=2, n_repeats=1)
        self.assertIn("pr_auc", metrics)
        ranking = rank_features(self.cleaned[ml_pipeline.ALL_FEATURES].iloc[:80], target, self.cleaned[ml_pipeline.ALL_FEATURES].iloc[80:120], self.cleaned["Diagnosis"].iloc[80:120])
        self.assertEqual(set(ranking.columns), {"feature", "importance_mean", "importance_std"})

    def test_benchmark_and_nested_selection_paths(self):
        features = self.cleaned[SELECTED_FEATURES].iloc[:80]
        target = self.cleaned["Diagnosis"].iloc[:80]
        fake_metrics = {
            "accuracy": 0.8,
            "roc_auc": 0.8,
            "pr_auc": 0.8,
            "f1": 0.8,
            "f2": 0.8,
            "recall": 0.8,
            "brier": 0.1,
        }
        def fake_repeated_cv(model, features, target, return_folds=False, **kwargs):
            return {"mean": fake_metrics, "fold_metrics": [fake_metrics]} if return_folds else fake_metrics

        with patch.object(ml_pipeline, "repeated_cv_metrics", side_effect=fake_repeated_cv):
            benchmark = ml_pipeline.benchmark_models(features, target)
            imbalance = ml_pipeline.compare_imbalance_methods(features, target)
        self.assertEqual(len(benchmark), 7)
        self.assertIn("method", imbalance.columns)

        ranking = pd.DataFrame({"feature": SELECTED_FEATURES, "importance_mean": [1, 0, 0, 0, 0], "importance_std": [0] * 5})
        from sklearn.dummy import DummyClassifier

        with patch.object(ml_pipeline, "rank_features", return_value=ranking), patch.object(
            ml_pipeline, "_xgb_classifier", return_value=DummyClassifier(strategy="prior")
        ):
            stability = ml_pipeline.nested_cv_feature_stability(features, target, feature_count=2, n_splits=2, n_repeats=1)
        self.assertEqual(stability["outer_folds"], 2)
        self.assertEqual(len(stability["feature_stability"]), len(SELECTED_FEATURES))

    def test_threshold_subgroups_and_monitoring(self):
        threshold, score = choose_f2_threshold(pd.Series([0, 1, 0, 1]), [0.1, 0.8, 0.4, 0.7])
        self.assertGreaterEqual(threshold, 0.05)
        self.assertGreaterEqual(score, 0.0)
        frame = self.cleaned.iloc[:120].copy()
        payload = {"model": DiagnosticModel(), "features": SELECTED_FEATURES, "threshold": 0.5}
        subgroup = subgroup_metrics(payload, frame, n_bootstrap=5)
        self.assertIn("status", subgroup.columns)
        ci = bootstrap_confidence_intervals(frame["Diagnosis"], DiagnosticModel().predict_proba(frame[SELECTED_FEATURES])[:, 1], 0.5, n_bootstrap=5)
        self.assertIn("recall", ci)
        baseline = _monitoring_baseline(frame, SELECTED_FEATURES)
        self.assertEqual(set(baseline), set(SELECTED_FEATURES))
        monitoring = monitor_prediction_batch(frame[SELECTED_FEATURES], {"input_rows": len(frame)}, {"version": "test", "monitoring_baseline": baseline})
        self.assertIn(monitoring["status"], {"ok", "alert"})

    def test_artifact_writer_creates_version_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {
                "version": "test-version",
                "selected_features": SELECTED_FEATURES,
                "threshold": 0.5,
                "imputation_statistics": {},
            }
            paths = _save_artifacts(DiagnosticModel(), metadata, root, promote=False)
            self.assertTrue(Path(paths["version_model"]).exists())
            self.assertTrue(Path(paths["version_metadata"]).exists())
            self.assertEqual(metadata["model_sha256"].__len__(), 64)

    def test_artifact_loader_validates_complete_promoted_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {
                "schema_version": 2,
                "version": "loadable-version",
                "selected_features": SELECTED_FEATURES,
                "threshold": 0.5,
                "imputation_statistics": {},
                "dataset_sha256": "dataset",
                "locked_test_sha256": "locked",
                "python_version": "3.12",
                "package_versions": {},
                "trained_at_utc": "now",
                "training_command": "test",
                "validation_scope": "internal",
            }
            self.assertTrue(REQUIRED_METADATA_KEYS.issuperset(metadata))
            _save_artifacts(DiagnosticModel(), metadata, root, promote=True)
            payload, loaded = load_artifacts(root)
            self.assertEqual(payload["features"], SELECTED_FEATURES)
            self.assertEqual(loaded["version"], "loadable-version")

    def test_prediction_rejects_wrong_payload_schema(self):
        with self.assertRaises(ValueError):
            ml_pipeline.predict_dataframe({"model": DiagnosticModel(), "features": ["MMSE"]}, self.cleaned.iloc[:1])

    def test_fast_training_creates_pipeline_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_training(
                self.data,
                artifacts_dir=root / "artifacts",
                locked_test_path=root / "locked_test.csv",
                fast=True,
                promote=False,
            )
            self.assertEqual(result["metadata"]["preprocessing"]["type"], "sklearn Pipeline")
            self.assertIn("imputer", result["model"].estimator.named_steps)
            self.assertIn("intercept", result["metadata"]["calibration"]["validation_summary"])
            split_sizes = result["metadata"]["split_sizes"]
            self.assertEqual(
                split_sizes["final_training"] + split_sizes["locked_test"],
                result["metadata"]["rows_clean"],
            )
            self.assertEqual(
                split_sizes["train"] + split_sizes["validation"],
                split_sizes["development"],
            )
            loaded_payload, loaded_metadata = load_artifacts(root / "artifacts") if (root / "artifacts" / "active_model.joblib").exists() else ({}, {})
            self.assertEqual(loaded_payload, {})
            self.assertEqual(loaded_metadata, {})


if __name__ == "__main__":
    unittest.main()
