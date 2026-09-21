import unittest

import numpy as np
import pandas as pd

from ml_pipeline import (
    calibration_intercept_slope,
    calibration_summary,
    compare_subgroup_metrics,
    monitor_prediction_batch,
)


class QualityAndMonitoringTests(unittest.TestCase):
    def test_calibration_intercept_and_slope_are_reported(self):
        intercept, slope = calibration_intercept_slope(
            [0, 1, 0, 1, 0, 1, 0, 1],
            [0.05, 0.80, 0.20, 0.70, 0.10, 0.90, 0.30, 0.60],
        )
        summary = calibration_summary(
            [0, 1, 0, 1, 0, 1, 0, 1],
            [0.05, 0.80, 0.20, 0.70, 0.10, 0.90, 0.30, 0.60],
        )
        self.assertTrue(np.isfinite(intercept))
        self.assertTrue(np.isfinite(slope))
        self.assertEqual(summary["intercept_slope_status"], "estimable")
        self.assertEqual(summary["intercept"], intercept)
        self.assertEqual(summary["slope"], slope)

    def test_subgroup_recall_degradation_fails_promotion_check(self):
        champion = pd.DataFrame(
            [
                {"group": "Gender", "level": "0", "n": 40, "status": "estimable", "recall": 0.90, "pr_auc": 0.80},
            ]
        )
        challenger = pd.DataFrame(
            [
                {"group": "Gender", "level": "0", "n": 40, "status": "estimable", "recall": 0.80, "pr_auc": 0.81},
            ]
        )
        checks = compare_subgroup_metrics(champion, challenger, {"subgroup_recall_tolerance": 0.05})
        recall_checks = [check for check in checks if check["metric"].startswith("subgroup_recall:")]
        self.assertEqual(len(recall_checks), 1)
        self.assertFalse(recall_checks[0]["passed"])

    def test_monitoring_alerts_on_quality_and_feature_shift(self):
        data = pd.DataFrame({"MMSE": [30.0, 30.0], "ADL": [10.0, 10.0]})
        result = monitor_prediction_batch(
            data,
            {"input_rows": 4, "rejected_rows": 2, "imputed_rows": 2},
            {
                "version": "test",
                "monitoring_baseline": {
                    "MMSE": {"mean": 15.0, "std": 2.0},
                    "ADL": {"mean": 5.0, "std": 1.0},
                },
            },
        )
        self.assertEqual(result["status"], "alert")
        self.assertGreaterEqual(len(result["alerts"]), 3)
        self.assertTrue(any(alert["type"] == "drift" for alert in result["alerts"]))


if __name__ == "__main__":
    unittest.main()
