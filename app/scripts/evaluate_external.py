"""Evaluate a frozen artifact on a separately supplied, labelled CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_pipeline import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATA_PATH,
    TARGET_COLUMN,
    clean_dataset,
    evaluate_payload,
    load_artifacts,
    subgroup_metrics,
    validate_dataframe,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen artifact without fitting it")
    parser.add_argument("csv", type=Path, help="External labelled CSV")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    payload, metadata = load_artifacts(args.artifacts_dir)
    external = pd.read_csv(args.csv)
    if TARGET_COLUMN not in external.columns:
        raise ValueError("External validation requires a binary Diagnosis column; no fitting was performed.")
    cleaned, cleaning = clean_dataset(external)
    validation = validate_dataframe(
        cleaned,
        reference_data=pd.read_csv(DEFAULT_DATA_PATH),
        mode="training",
        imputation_statistics=metadata.get("imputation_statistics"),
    )
    if validation.valid.empty:
        raise ValueError("No valid labelled rows remain after the fixed validation rules.")
    valid = validation.valid.drop(columns=["_quality_status", "_imputed_fields", "_issue_count"], errors="ignore")
    evaluation = evaluate_payload(payload, valid)
    summary = {
        "validation_scope": "external",
        "source": args.csv.name,
        "rows_input": int(len(external)),
        "rows_valid": int(len(valid)),
        "prevalence": float(valid[TARGET_COLUMN].mean()),
        "cleaning": cleaning,
        "validation_summary": validation.summary,
        "metrics": evaluation["metrics"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "subgroup_metrics": subgroup_metrics(payload, valid, n_bootstrap=200).to_dict(orient="records"),
        "model_version": metadata.get("version"),
        "note": "The artifact, imputer, feature order, and threshold were reused unchanged; no fitting was performed.",
    }
    serialized = json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
