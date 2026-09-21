"""Command-line entry point for the V3 Alzheimer's training pipeline."""

from __future__ import annotations

import argparse
import json

from ml_pipeline import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_PATH, read_dataset, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and version the V3 Alzheimer's screening model")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="CSV containing the training data")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR), help="Directory for model versions")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip repeated benchmark/imbalance/feature-count diagnostics; keep tuning and final evaluation",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Save a version without making it the active Streamlit model",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["report_v3", "nested_cv"],
        default="report_v3",
        help="Keep the reproducible V3 feature set or record nested-CV selection stability",
    )
    args = parser.parse_args()

    data = read_dataset(args.data)
    result = run_training(
        data,
        artifacts_dir=args.artifacts_dir,
        dataset_label=args.data,
        fast=args.fast,
        promote=not args.no_promote,
        selection_mode=args.selection_mode,
    )
    metadata = result["metadata"]
    print(json.dumps(
        {
            "version": metadata["version"],
            "rows_raw": metadata["rows_raw"],
            "rows_clean": metadata["rows_clean"],
            "selected_features": metadata["selected_features"],
            "threshold": metadata["threshold"],
            "test_metrics": metadata["test_metrics"],
            "artifact_paths": result["paths"],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
