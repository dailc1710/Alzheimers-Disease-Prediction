"""Verify the active artifact contract and run a small screening smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline import (  # noqa: E402
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_LOCKED_TEST_PATH,
    _dataframe_sha256,
    _sha256_file,
    _with_source_rows,
    clean_dataset,
    load_artifacts,
    predict_dataframe,
    read_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the active Alzheimer screening artifact")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--locked-test", default=str(DEFAULT_LOCKED_TEST_PATH))
    args = parser.parse_args()

    payload, metadata = load_artifacts(args.artifacts_dir)

    data_path = Path(args.data)
    dataset_method = metadata.get("checksum_methods", {}).get("dataset_sha256", "file_bytes")
    if dataset_method == "file_bytes" and data_path.exists():
        if _sha256_file(data_path) != metadata["dataset_sha256"]:
            raise RuntimeError("Dataset SHA-256 mismatch; the training CSV changed after artifact creation.")

    locked_path = Path(args.locked_test)
    if not locked_path.exists():
        raise FileNotFoundError(f"Locked test file is missing: {locked_path}")
    locked = _with_source_rows(read_dataset(locked_path))
    locked, _ = clean_dataset(locked)
    if _dataframe_sha256(locked) != metadata["locked_test_sha256"]:
        raise RuntimeError("Locked-test SHA-256 mismatch; refusing to verify a modified evaluation cohort.")
    source = read_dataset(args.data)
    cleaned, _ = clean_dataset(source)
    smoke_input = cleaned.head(10).drop(columns=["Diagnosis"], errors="ignore")
    predictions = predict_dataframe(payload, smoke_input)
    result = {
        "schema_version": metadata["schema_version"],
        "version": metadata["version"],
        "rows_smoke_tested": len(predictions),
        "locked_test_rows_verified": len(locked),
        "dataset_checksum_verified": dataset_method == "file_bytes" and data_path.exists(),
        "selected_features": metadata["selected_features"],
        "threshold": metadata["threshold"],
        "model_sha256": metadata["model_sha256"],
        "metadata_sha256": metadata["metadata_sha256"],
        "locked_test_sha256": metadata["locked_test_sha256"],
        "status": "ok",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
