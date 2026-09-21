"""Reproducible Alzheimer's screening pipeline described in report V3.

The project is an educational screening demo, not a clinical diagnostic system.
The module keeps data cleaning, validation, training, calibration, evaluation,
artifact versioning, and retraining in one place so the Streamlit app uses the
same schema as the training code.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT.parent / "alzheimers_disease_data.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_LOCKED_TEST_PATH = PROJECT_ROOT / "data" / "locked_test.csv"
TARGET_COLUMN = "Diagnosis"
RANDOM_STATE = 42
ARTIFACT_SCHEMA_VERSION = 2
PROMOTION_POLICY = {
    "pr_auc_min_delta": 0.0,
    "recall_tolerance": 0.01,
    "brier_max_delta": 0.005,
    "f2_min_delta": 0.005,
    "f2_ci_low_min": 0.0,
    "subgroup_recall_tolerance": 0.05,
    "subgroup_pr_auc_tolerance": 0.05,
    "subgroup_min_group_size": 20,
}
REQUIRED_METADATA_KEYS = {
    "schema_version",
    "version",
    "selected_features",
    "threshold",
    "imputation_statistics",
    "dataset_sha256",
    "model_sha256",
    "locked_test_sha256",
    "metadata_sha256",
    "python_version",
    "package_versions",
    "trained_at_utc",
    "training_command",
    "validation_scope",
}

ALL_FEATURES = [
    "Age",
    "Gender",
    "Ethnicity",
    "EducationLevel",
    "BMI",
    "Smoking",
    "AlcoholConsumption",
    "PhysicalActivity",
    "DietQuality",
    "SleepQuality",
    "FamilyHistoryAlzheimers",
    "CardiovascularDisease",
    "Diabetes",
    "Depression",
    "HeadInjury",
    "Hypertension",
    "SystolicBP",
    "DiastolicBP",
    "CholesterolTotal",
    "CholesterolLDL",
    "CholesterolHDL",
    "CholesterolTriglycerides",
    "MMSE",
    "FunctionalAssessment",
    "MemoryComplaints",
    "BehavioralProblems",
    "ADL",
    "Confusion",
    "Disorientation",
    "PersonalityChanges",
    "DifficultyCompletingTasks",
    "Forgetfulness",
]

# The official compact feature set from report V3.
SELECTED_FEATURES = [
    "MMSE",
    "FunctionalAssessment",
    "ADL",
    "MemoryComplaints",
    "BehavioralProblems",
]

IDENTIFIER_COLUMNS = ["PatientID", "DoctorInCharge"]
DATASET_COLUMNS = IDENTIFIER_COLUMNS + ALL_FEATURES + [TARGET_COLUMN]
COMPACT_TRAINING_COLUMNS = ["PatientID", *SELECTED_FEATURES, TARGET_COLUMN]
VALIDATION_META_COLUMNS = [
    "_row_number",
    "_quality_status",
    "_imputed_fields",
    "_issue_count",
]

# Bounds are deliberately conservative sanity checks, not medical guidelines.
RANGE_RULES: dict[str, tuple[float, float]] = {
    # Dataset-domain ranges documented for the V3 source dataset.
    "Age": (60, 90),
    "EducationLevel": (0, 3),
    "BMI": (15, 40),
    "AlcoholConsumption": (0, 20),
    "PhysicalActivity": (0, 10),
    "DietQuality": (0, 10),
    "SleepQuality": (4, 10),
    "SystolicBP": (90, 180),
    "DiastolicBP": (60, 120),
    "CholesterolTotal": (150, 300),
    "CholesterolLDL": (50, 200),
    "CholesterolHDL": (20, 100),
    "CholesterolTriglycerides": (50, 400),
    "MMSE": (0, 30),
    "FunctionalAssessment": (0, 10),
    "ADL": (0, 10),
    "BehavioralProblems": (0, 1),
}

CATEGORICAL_ALLOWED_VALUES: dict[str, tuple[int, ...]] = {
    "Ethnicity": (0, 1, 2, 3),
}

BINARY_COLUMNS = [
    "Gender",
    "Smoking",
    "FamilyHistoryAlzheimers",
    "CardiovascularDisease",
    "Diabetes",
    "Depression",
    "HeadInjury",
    "Hypertension",
    "MemoryComplaints",
    "BehavioralProblems",
    "Confusion",
    "Disorientation",
    "PersonalityChanges",
    "DifficultyCompletingTasks",
    "Forgetfulness",
]
CATEGORICAL_COLUMNS = [
    "Gender",
    "Ethnicity",
    "EducationLevel",
    *[c for c in BINARY_COLUMNS if c != "Gender"],
]


@dataclass
class ValidationResult:
    """Output of the batch validator used by the Streamlit app."""

    valid: pd.DataFrame
    rejected: pd.DataFrame
    issues: pd.DataFrame
    summary: dict[str, Any]


def _require_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def _with_source_rows(data: pd.DataFrame, start: int = 1) -> pd.DataFrame:
    """Return a copy with stable source-row identifiers for audit and locking."""

    frame = data.copy()
    if "_source_row" not in frame.columns:
        frame.insert(0, "_source_row", np.arange(start, start + len(frame)))
    return frame


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataframe_sha256(data: pd.DataFrame) -> str:
    """Hash a deterministic CSV representation, including column order."""

    csv_bytes = data.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return _sha256_bytes(csv_bytes)


def _source_release_sha256() -> str:
    files = [Path(__file__), PROJECT_ROOT / "train_pipeline.py"]
    digest = hashlib.sha256()
    for path in files:
        if not path.exists():
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    packages = {
        "joblib": "joblib",
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "xgboost": "xgboost",
    }
    versions: dict[str, str] = {}
    for label, distribution in packages.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = "not installed"
    return versions


def _atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _artifact_lock(artifacts_dir: str | Path):
    """Serialize active-artifact writes on Windows and POSIX."""

    lock_path = Path(artifacts_dir) / ".active_artifacts.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)
    handle = lock_path.open("r+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            handle.truncate(1)
            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def read_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the supplied CSV and check the modeling schema."""

    frame = pd.read_csv(path)
    _require_columns(frame, [*ALL_FEATURES, TARGET_COLUMN])
    return frame


def clean_dataset(
    data: pd.DataFrame,
    require_target: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply deterministic training-time cleaning from report V3.

    Rows with impossible blood-pressure ordering are removed before any split.
    The function does not learn medians, thresholds, or model parameters.
    """

    required = [*ALL_FEATURES] + ([TARGET_COLUMN] if require_target else [])
    _require_columns(data, required)
    frame = data.copy()
    before = len(frame)
    duplicate_basis = frame.drop(columns=["_source_row"], errors="ignore")
    duplicate_mask = duplicate_basis.duplicated()
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        frame = frame.loc[~duplicate_mask].reset_index(drop=True)

    frame["SystolicBP"] = pd.to_numeric(frame["SystolicBP"], errors="coerce")
    frame["DiastolicBP"] = pd.to_numeric(frame["DiastolicBP"], errors="coerce")
    invalid_bp = frame["SystolicBP"].notna() & frame["DiastolicBP"].notna() & (
        frame["SystolicBP"] <= frame["DiastolicBP"]
    )
    invalid_bp_count = int(invalid_bp.sum())
    frame = frame.loc[~invalid_bp].reset_index(drop=True)

    report = {
        "rows_before": before,
        "rows_after": len(frame),
        "duplicates_removed": duplicate_count,
        "invalid_blood_pressure_rows_removed": invalid_bp_count,
        "missing_cells_after_cleaning": int(frame.isna().sum().sum()),
        "positive_rate_after_cleaning": (
            float(frame[TARGET_COLUMN].mean())
            if TARGET_COLUMN in frame.columns
            else None
        ),
    }
    return frame, report


def audit_bp_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Return a row-level audit of the blood-pressure cleaning decision."""

    _require_columns(data, ["SystolicBP", "DiastolicBP"])
    frame = _with_source_rows(data)
    systolic = pd.to_numeric(frame["SystolicBP"], errors="coerce")
    diastolic = pd.to_numeric(frame["DiastolicBP"], errors="coerce")
    invalid = systolic.notna() & diastolic.notna() & (systolic <= diastolic)
    output = pd.DataFrame(
        {
            "_source_row": frame["_source_row"].to_numpy(),
            "PatientID": frame["PatientID"].to_numpy() if "PatientID" in frame.columns else "",
            "Diagnosis": frame[TARGET_COLUMN].to_numpy() if TARGET_COLUMN in frame.columns else np.nan,
            "SystolicBP": systolic.to_numpy(),
            "DiastolicBP": diastolic.to_numpy(),
            "bp_order_invalid": invalid.to_numpy(),
        }
    )
    for column in [
        "Age", "Gender", "Ethnicity", "EducationLevel", "MMSE",
        "FunctionalAssessment", "ADL", "MemoryComplaints", "BehavioralProblems",
    ]:
        if column in frame.columns:
            output[column] = frame[column].to_numpy()
    output["status"] = np.where(output["bp_order_invalid"], "removed", "kept")
    return output


def run_cleaning_sensitivity(
    data: pd.DataFrame,
    strategies: Sequence[str] = ("remove", "swap-if-plausible", "keep-with-flag"),
) -> pd.DataFrame:
    """Compare the row-count and label impact of BP cleaning strategies."""

    _require_columns(data, [*ALL_FEATURES, TARGET_COLUMN])
    source = _with_source_rows(data)
    records: list[dict[str, Any]] = []
    audit = audit_bp_rows(source)
    invalid_count = int(audit["bp_order_invalid"].sum())
    for strategy in strategies:
        if strategy == "remove":
            candidate, _ = clean_dataset(source)
            swapped = 0
            removed = invalid_count
        elif strategy == "swap-if-plausible":
            candidate = source.copy()
            systolic = pd.to_numeric(candidate["SystolicBP"], errors="coerce")
            diastolic = pd.to_numeric(candidate["DiastolicBP"], errors="coerce")
            invalid = systolic.notna() & diastolic.notna() & (systolic <= diastolic)
            plausible = invalid & diastolic.between(*RANGE_RULES["SystolicBP"]) & systolic.between(*RANGE_RULES["DiastolicBP"])
            candidate.loc[plausible, "SystolicBP"], candidate.loc[plausible, "DiastolicBP"] = (
                diastolic.loc[plausible],
                systolic.loc[plausible],
            )
            swapped = int(plausible.sum())
            candidate, clean_report = clean_dataset(candidate)
            removed = int(clean_report["invalid_blood_pressure_rows_removed"])
        elif strategy == "keep-with-flag":
            candidate = source.copy()
            systolic = pd.to_numeric(candidate["SystolicBP"], errors="coerce")
            diastolic = pd.to_numeric(candidate["DiastolicBP"], errors="coerce")
            candidate["bp_order_invalid"] = (
                systolic.notna() & diastolic.notna() & (systolic <= diastolic)
            ).astype(bool)
            swapped = 0
            removed = 0
        else:
            raise ValueError(f"Unknown cleaning sensitivity strategy: {strategy}")
        records.append(
            {
                "strategy": strategy,
                "rows": int(len(candidate)),
                "bp_invalid_rows_removed": removed,
                "bp_rows_swapped": swapped,
                "bp_invalid_rows_retained": int(candidate.get("bp_order_invalid", pd.Series(False, index=candidate.index)).sum()),
                "positive_rate": float(candidate[TARGET_COLUMN].mean()),
            }
        )
    return pd.DataFrame(records)


def summarize_bp_audit(audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize kept/removed BP rows without exposing patient identifiers."""

    required = {"status", "Diagnosis"}
    missing = sorted(required.difference(audit.columns))
    if missing:
        raise ValueError("BP audit is missing columns: " + ", ".join(missing))
    frame = audit.copy()
    frame["Diagnosis"] = pd.to_numeric(frame["Diagnosis"], errors="coerce")
    records: list[dict[str, Any]] = []
    numeric_columns = [
        "Age", "MMSE", "FunctionalAssessment", "ADL", "Gender", "Ethnicity",
        "EducationLevel", "MemoryComplaints", "BehavioralProblems",
    ]
    for status, group in frame.groupby("status", sort=True):
        row: dict[str, Any] = {
            "status": status,
            "rows": int(len(group)),
            "row_pct": round(float(len(group) / len(frame) * 100), 4) if len(frame) else 0.0,
            "diagnosis_rate": round(float(group["Diagnosis"].mean()), 6) if group["Diagnosis"].notna().any() else np.nan,
        }
        for column in numeric_columns:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}_median"] = float(values.median()) if not values.empty else np.nan
            row[f"{column}_q1"] = float(values.quantile(0.25)) if not values.empty else np.nan
            row[f"{column}_q3"] = float(values.quantile(0.75)) if not values.empty else np.nan
        records.append(row)

    summary = pd.DataFrame(records)
    if len(summary) == 2 and set(summary["status"]) == {"kept", "removed"}:
        kept = summary.loc[summary["status"] == "kept"].iloc[0]
        removed = summary.loc[summary["status"] == "removed"].iloc[0]
        for column in numeric_columns:
            kept_value = kept.get(f"{column}_median", np.nan)
            removed_value = removed.get(f"{column}_median", np.nan)
            kept_q1, kept_q3 = kept.get(f"{column}_q1", np.nan), kept.get(f"{column}_q3", np.nan)
            removed_q1, removed_q3 = removed.get(f"{column}_q1", np.nan), removed.get(f"{column}_q3", np.nan)
            pooled_sd = np.sqrt(max(((kept_q3 - kept_q1) / 1.349) ** 2 + ((removed_q3 - removed_q1) / 1.349) ** 2, 0) / 2)
            summary.loc[summary["status"] == "removed", f"{column}_smd_vs_kept"] = (
                (removed_value - kept_value) / pooled_sd if pooled_sd and np.isfinite(pooled_sd) else np.nan
            )
    return summary


def _locked_test_frame(
    source_data: pd.DataFrame,
    locked_test_path: str | Path,
) -> tuple[pd.DataFrame, bool]:
    """Load the fixed evaluation cohort, creating it once when absent."""

    path = Path(locked_test_path)
    if path.exists():
        locked = pd.read_csv(path)
        _require_columns(locked, [*ALL_FEATURES, TARGET_COLUMN])
        locked = _with_source_rows(locked)
        locked, _ = clean_dataset(locked)
        if locked.empty:
            raise ValueError("Locked test file contains no usable rows.")
        return locked, False

    source = _with_source_rows(source_data)
    cleaned, _ = clean_dataset(source)
    s = _import_sklearn()
    _, locked = s["train_test_split"](
        cleaned,
        test_size=0.20,
        stratify=cleaned[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    sort_columns = [column for column in ["_source_row", "PatientID"] if column in locked.columns]
    if sort_columns:
        locked = locked.sort_values(sort_columns, kind="stable")
    locked = locked.reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, locked.to_csv(index=False, lineterminator="\n"))
    return locked, True


def _exclude_locked_rows(
    data: pd.DataFrame,
    locked_test: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Remove locked-test identities from a training frame."""

    if "PatientID" in data.columns and "PatientID" in locked_test.columns:
        locked_patient_ids = _patient_id_keys(locked_test["PatientID"])
        usable_locked_ids = locked_patient_ids.dropna()
        usable_locked_ids = usable_locked_ids[usable_locked_ids.ne("")]
        if len(usable_locked_ids) == len(locked_test):
            locked_ids = set(usable_locked_ids.tolist())
            data_patient_ids = _patient_id_keys(data["PatientID"])
            excluded = data_patient_ids.isin(locked_ids)
            return data.loc[~excluded].copy(), int(excluded.sum())

    excluded = pd.Series(False, index=data.index)
    if "_source_row" in data.columns and "_source_row" in locked_test.columns:
        locked_rows = set(locked_test["_source_row"].astype("string").dropna().tolist())
        if locked_rows:
            excluded |= data["_source_row"].astype("string").isin(locked_rows)
    return data.loc[~excluded].copy(), int(excluded.sum())


def _exclude_existing_patient_ids(
    data: pd.DataFrame,
    existing_data: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Drop uploaded rows whose PatientID is already in the training data."""

    if "PatientID" not in data.columns or "PatientID" not in existing_data.columns:
        return data.copy(), 0
    existing_ids = set(_patient_id_keys(existing_data["PatientID"]).dropna().tolist())
    if not existing_ids:
        return data.copy(), 0
    duplicate_mask = _patient_id_keys(data["PatientID"]).isin(existing_ids)
    return data.loc[~duplicate_mask].copy(), int(duplicate_mask.sum())


def _patient_id_keys(values: pd.Series) -> pd.Series:
    """Keep text IDs while matching CSV-inferred integer IDs such as 12.0."""

    return (
        values.astype("string").str.strip()
        .str.replace(r"^([+-]?\d+)\.0+$", r"\1", regex=True)
        .replace("", pd.NA)
    )


def _reference_median(reference: pd.DataFrame | None, column: str) -> float:
    if reference is not None and column in reference:
        values = pd.to_numeric(reference[column], errors="coerce").dropna()
        if not values.empty:
            return float(values.median())
    return 0.0


def _reference_mode(reference: pd.DataFrame | None, column: str) -> float:
    if reference is not None and column in reference:
        values = pd.to_numeric(reference[column], errors="coerce").dropna()
        if not values.empty:
            return float(values.mode().iloc[0])
    return 0.0


def fit_imputation_statistics(data: pd.DataFrame) -> dict[str, float]:
    """Fit reusable median/mode values for the V3 feature schema."""

    statistics: dict[str, float] = {}
    for column in ALL_FEATURES:
        if column not in data.columns:
            continue
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            continue
        if column in CATEGORICAL_COLUMNS:
            statistics[column] = float(values.mode().iloc[0])
        else:
            statistics[column] = float(values.median())
    return statistics


def _imputation_value(
    reference: pd.DataFrame | None,
    statistics: Mapping[str, float] | None,
    column: str,
) -> float:
    if statistics is not None and column in statistics:
        return float(statistics[column])
    if column in CATEGORICAL_COLUMNS:
        return _reference_mode(reference, column)
    return _reference_median(reference, column)


def validate_dataframe(
    data: pd.DataFrame,
    reference_data: pd.DataFrame | None = None,
    mode: str = "screening",
    imputation_statistics: Mapping[str, float] | None = None,
) -> ValidationResult:
    """Validate uploaded rows and retain recoverable formatting errors.

    Missing or non-numeric values are converted to NaN and imputed with a
    reference median, as documented in V3. Negative/out-of-range values,
    unknown category codes, and impossible blood-pressure ordering are rejected.
    """

    if mode not in {"screening", "training"}:
        raise ValueError("mode must be 'screening' or 'training'")
    required = [*SELECTED_FEATURES] + ([TARGET_COLUMN] if mode == "training" else [])
    _require_columns(data, required)
    frame = data.copy()
    if "_source_row" in frame.columns:
        source_rows = frame["_source_row"].to_numpy()
        frame = frame.drop(columns=["_source_row"])
    else:
        source_rows = np.arange(1, len(frame) + 1)
    if "_row_number" in frame.columns:
        frame = frame.drop(columns=["_row_number"])
    frame.insert(0, "_row_number", source_rows)
    issues: list[dict[str, Any]] = []
    reject = pd.Series(False, index=frame.index)
    quality_status = pd.Series("clean", index=frame.index, dtype="object")
    imputed_fields: dict[Any, list[str]] = {index: [] for index in frame.index}

    # A row with one missing feature is recoverable. A row with no usable
    # selected feature evidence is not: imputing all five values would create
    # a synthetic "average" patient and hide a broken CSV row.
    selected_features_missing = pd.Series(True, index=frame.index)
    for selected_column in SELECTED_FEATURES:
        selected_features_missing &= pd.to_numeric(
            frame[selected_column], errors="coerce"
        ).isna()
    if selected_features_missing.any():
        reject.loc[selected_features_missing] = True
        for index in frame.index[selected_features_missing]:
            issues.append(
                {
                    "row": int(frame.at[index, "_row_number"]),
                    "field": "model_features",
                    "issue": "all selected model features missing",
                    "action": "rejected row",
                }
            )

    numeric_columns = list(
        dict.fromkeys(
            [
                *[column for column in RANGE_RULES if column in frame.columns],
                *[column for column in CATEGORICAL_ALLOWED_VALUES if column in frame.columns],
                *[
                    column
                    for column in BINARY_COLUMNS + ([TARGET_COLUMN] if mode == "training" else [])
                    if column in frame.columns
                ],
            ]
        )
    )
    for column in numeric_columns:
        original = frame[column].copy()
        converted = pd.to_numeric(original, errors="coerce")
        parse_errors = original.notna() & converted.isna()
        for index in frame.index[parse_errors]:
            issues.append(
                {
                    "row": int(frame.at[index, "_row_number"]),
                    "field": column,
                    "issue": "invalid format",
                    "action": "rejected row" if column == TARGET_COLUMN else "converted to missing then imputed",
                }
            )
        frame[column] = converted

        missing = frame[column].isna()
        if missing.any():
            if column == TARGET_COLUMN:
                reject.loc[missing] = True
                for index in frame.index[missing]:
                    issues.append(
                        {
                            "row": int(frame.at[index, "_row_number"]),
                            "field": column,
                            "issue": "missing or invalid target",
                            "action": "rejected row",
                        }
                    )
            else:
                fill_value = _imputation_value(
                    reference_data,
                    imputation_statistics,
                    column,
                )
                recoverable_missing = missing & ~selected_features_missing
                for index in frame.index[recoverable_missing]:
                    quality_status.loc[index] = "imputed"
                    imputed_fields[index].append(column)
                    issues.append(
                        {
                            "row": int(frame.at[index, "_row_number"]),
                            "field": column,
                            "issue": "missing value",
                            "action": f"imputed with {fill_value:g}",
                        }
                    )
                frame.loc[recoverable_missing, column] = fill_value

        if column in RANGE_RULES:
            lower, upper = RANGE_RULES[column]
            bad = frame[column].notna() & ((frame[column] < lower) | (frame[column] > upper))
            negative = bad & (frame[column] < 0)
            for index in frame.index[negative]:
                reject.loc[index] = True
                issues.append(
                    {
                        "row": int(frame.at[index, "_row_number"]),
                        "field": column,
                        "issue": (
                            f"negative value {frame.at[index, column]:g} outside allowed "
                            f"range [{lower:g}, {upper:g}]"
                        ),
                        "action": "rejected row",
                    }
                )
            for index in frame.index[bad & ~negative]:
                reject.loc[index] = True
                issues.append(
                    {
                        "row": int(frame.at[index, "_row_number"]),
                        "field": column,
                        "issue": f"outside allowed range [{lower:g}, {upper:g}]",
                        "action": "rejected row",
                    }
                )

        if column in BINARY_COLUMNS:
            bad = frame[column].notna() & ~frame[column].isin([0, 1])
            for index in frame.index[bad]:
                reject.loc[index] = True
                issue = (
                    f"negative value {frame.at[index, column]:g}; expected 0 or 1"
                    if frame.at[index, column] < 0
                    else "unknown binary code"
                )
                issues.append(
                    {
                        "row": int(frame.at[index, "_row_number"]),
                        "field": column,
                        "issue": issue,
                        "action": "rejected row",
                    }
                )

        if column in CATEGORICAL_ALLOWED_VALUES:
            allowed = CATEGORICAL_ALLOWED_VALUES[column]
            bad = frame[column].notna() & ~frame[column].isin(allowed)
            for index in frame.index[bad]:
                reject.loc[index] = True
                issue = (
                    f"negative value {frame.at[index, column]:g}; expected code in {list(allowed)}"
                    if frame.at[index, column] < 0
                    else "unknown categorical code"
                )
                issues.append(
                    {
                        "row": int(frame.at[index, "_row_number"]),
                        "field": column,
                        "issue": issue,
                        "action": "rejected row",
                    }
                )

        if column == TARGET_COLUMN:
            bad = frame[column].notna() & ~frame[column].isin([0, 1])
            for index in frame.index[bad]:
                reject.loc[index] = True
                issue = (
                    f"negative target {frame.at[index, column]:g}; expected 0 or 1"
                    if frame.at[index, column] < 0
                    else "target must be 0 or 1"
                )
                issues.append(
                    {
                        "row": int(frame.at[index, "_row_number"]),
                        "field": column,
                        "issue": issue,
                        "action": "rejected row",
                    }
                )

    if "SystolicBP" in frame.columns and "DiastolicBP" in frame.columns:
        bad_bp = frame["SystolicBP"] <= frame["DiastolicBP"]
        for index in frame.index[bad_bp]:
            reject.loc[index] = True
            issues.append(
                {
                    "row": int(frame.at[index, "_row_number"]),
                    "field": "SystolicBP/DiastolicBP",
                    "issue": "SystolicBP must be greater than DiastolicBP",
                    "action": "rejected row",
                }
            )

    if "PatientID" in frame.columns:
        patient_ids = _patient_id_keys(frame["PatientID"])
        duplicate_ids = patient_ids.notna() & patient_ids.duplicated(keep=False)
        for index in frame.index[duplicate_ids]:
            reject.loc[index] = True
            issues.append(
                {
                    "row": int(frame.at[index, "_row_number"]),
                    "field": "PatientID",
                    "issue": "conflicting duplicate PatientID",
                    "action": "rejected row",
                }
            )

    issue_frame = pd.DataFrame(issues, columns=["row", "field", "issue", "action"])
    issue_counts = issue_frame["row"].value_counts().to_dict() if not issue_frame.empty else {}
    frame["_quality_status"] = np.where(reject, "rejected", quality_status)
    frame["_imputed_fields"] = [", ".join(imputed_fields[index]) for index in frame.index]
    frame["_issue_count"] = (
        frame["_row_number"].map(issue_counts).fillna(0).astype(int)
    )
    valid = frame.loc[~reject].copy()
    rejected = frame.loc[reject].copy()
    summary = {
        "input_rows": len(frame),
        "valid_rows": len(valid),
        "rejected_rows": len(rejected),
        "issue_count": len(issue_frame),
        "imputed_issue_count": int(issue_frame["action"].str.contains("imputed", na=False).sum()) if not issue_frame.empty else 0,
        "negative_issue_count": int(issue_frame["issue"].str.contains("negative", na=False).sum()) if not issue_frame.empty else 0,
        "clean_rows": int(((frame["_quality_status"] == "clean") & ~reject).sum()),
        "imputed_rows": int(((frame["_quality_status"] == "imputed") & ~reject).sum()),
    }
    return ValidationResult(valid=valid, rejected=rejected, issues=issue_frame, summary=summary)


def _import_sklearn() -> dict[str, Any]:
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        fbeta_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import (
        GridSearchCV,
        RepeatedStratifiedKFold,
        StratifiedKFold,
        train_test_split,
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return {
        "clone": clone,
        "CalibratedClassifierCV": CalibratedClassifierCV,
        "calibration_curve": calibration_curve,
        "DummyClassifier": DummyClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "permutation_importance": permutation_importance,
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "average_precision_score": average_precision_score,
        "brier_score_loss": brier_score_loss,
        "f1_score": f1_score,
        "fbeta_score": fbeta_score,
        "confusion_matrix": confusion_matrix,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "GridSearchCV": GridSearchCV,
        "RepeatedStratifiedKFold": RepeatedStratifiedKFold,
        "StratifiedKFold": StratifiedKFold,
        "train_test_split": train_test_split,
        "KNeighborsClassifier": KNeighborsClassifier,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "SVC": SVC,
    }


def _xgb_classifier(scale_pos_weight: float = 1.0, **overrides: Any) -> Any:
    from xgboost import XGBClassifier

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "scale_pos_weight": float(scale_pos_weight),
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "tree_method": "hist",
    }
    params.update(overrides)
    return XGBClassifier(**params)


def _with_fold_imputer(estimator: Any) -> Any:
    """Put learned imputation inside the estimator cloned for every CV fold."""

    s = _import_sklearn()
    return s["Pipeline"](
        steps=[
            ("imputer", s["SimpleImputer"](strategy="median")),
            ("model", estimator),
        ]
    )


def _positive_weight(y: pd.Series | np.ndarray) -> float:
    values = pd.Series(y)
    positive = int((values == 1).sum())
    negative = int((values == 0).sum())
    return float(negative / positive) if positive else 1.0


def _cohens_d(group_zero: pd.Series, group_one: pd.Series) -> float:
    first = pd.to_numeric(group_zero, errors="coerce").dropna().astype(float)
    second = pd.to_numeric(group_one, errors="coerce").dropna().astype(float)
    if len(first) < 2 or len(second) < 2:
        return 0.0
    pooled_variance = (
        (len(first) - 1) * first.var(ddof=1) + (len(second) - 1) * second.var(ddof=1)
    ) / (len(first) + len(second) - 2)
    pooled_std = float(np.sqrt(pooled_variance))
    return float((second.mean() - first.mean()) / pooled_std) if pooled_std else 0.0


def exploratory_summary(data: pd.DataFrame) -> dict[str, Any]:
    """Build the reproducible EDA summaries referenced in report V3."""

    _require_columns(data, [*ALL_FEATURES, TARGET_COLUMN])
    numeric_features = [
        column for column in ALL_FEATURES if pd.api.types.is_numeric_dtype(data[column])
    ]
    group_zero = data.loc[data[TARGET_COLUMN] == 0]
    group_one = data.loc[data[TARGET_COLUMN] == 1]
    cohen = pd.DataFrame(
        [
            {"feature": column, "cohens_d": _cohens_d(group_zero[column], group_one[column])}
            for column in numeric_features
        ]
    )
    cohen["absolute_cohens_d"] = cohen["cohens_d"].abs()
    cohen = cohen.sort_values("absolute_cohens_d", ascending=False).head(15)
    correlation_columns = [
        column
        for column in [
            "MMSE",
            "FunctionalAssessment",
            "ADL",
            "SleepQuality",
            "PhysicalActivity",
            "Diabetes",
            "Hypertension",
            "CholesterolTotal",
            TARGET_COLUMN,
        ]
        if column in data.columns
    ]
    grouped = data.groupby(TARGET_COLUMN)[SELECTED_FEATURES].mean().round(6)
    return {
        "diagnosis_counts": {str(key): int(value) for key, value in data[TARGET_COLUMN].value_counts().sort_index().items()},
        "positive_rate": float(data[TARGET_COLUMN].mean()),
        "cohens_d_top15": cohen.to_dict(orient="records"),
        "correlation": data[correlation_columns].corr().round(6).to_dict(),
        "selected_feature_means_by_diagnosis": grouped.to_dict(),
    }


def save_eda_plots(
    data: pd.DataFrame,
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> list[str]:
    """Save the core V3 EDA figures when plotting dependencies are available."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return []

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    counts = data[TARGET_COLUMN].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(6, 4))
    counts.plot.bar(ax=axis, color=["#4C72B0", "#DD8452"])
    axis.set_title("Diagnosis distribution after BP cleaning")
    axis.set_xlabel("Diagnosis")
    axis.set_ylabel("Patients")
    figure.tight_layout()
    path = output / "diagnosis_distribution.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(str(path))

    cohen = pd.DataFrame(summary["cohens_d_top15"])
    figure, axis = plt.subplots(figsize=(8, 6))
    cohen = cohen.sort_values("cohens_d")
    axis.barh(cohen["feature"], cohen["cohens_d"], color="#4C72B0")
    axis.set_title("Top 15 features by Cohen's d")
    axis.set_xlabel("Cohen's d")
    figure.tight_layout()
    path = output / "cohens_d_top15.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(str(path))

    correlation = pd.DataFrame(summary["correlation"])
    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axis)
    axis.set_title("Selected feature correlation matrix")
    figure.tight_layout()
    path = output / "correlation_heatmap.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(str(path))

    long_frame = data[[TARGET_COLUMN, *SELECTED_FEATURES]].melt(
        id_vars=TARGET_COLUMN, var_name="feature", value_name="value"
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=long_frame, x="feature", y="value", hue=TARGET_COLUMN, ax=axis)
    axis.set_title("Selected feature distributions by diagnosis")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    path = output / "selected_feature_boxplots.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(str(path))
    return paths


def make_candidate_models(y: pd.Series, xgb_params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Create the two baselines and five candidate models from V3."""

    s = _import_sklearn()
    weight = _positive_weight(y)
    xgb_options = dict(xgb_params or {})
    xgb_options.setdefault("scale_pos_weight", weight)
    return {
        "DummyClassifier": _with_fold_imputer(s["DummyClassifier"](strategy="most_frequent")),
        "Logistic Regression": s["Pipeline"](
            [
                ("imputer", s["SimpleImputer"](strategy="median")),
                ("scale", s["StandardScaler"]()),
                ("model", s["LogisticRegression"](max_iter=2000, class_weight="balanced")),
            ]
        ),
        "KNN": s["Pipeline"](
            [
                ("imputer", s["SimpleImputer"](strategy="median")),
                ("scale", s["StandardScaler"]()),
                ("model", s["KNeighborsClassifier"](n_neighbors=5)),
            ]
        ),
        "SVM (RBF)": s["Pipeline"](
            [
                ("imputer", s["SimpleImputer"](strategy="median")),
                ("scale", s["StandardScaler"]()),
                ("model", s["SVC"](kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced")),
            ]
        ),
        "Random Forest": _with_fold_imputer(
            s["RandomForestClassifier"](
                n_estimators=300,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=1,
            )
        ),
        "Gradient Boosting": _with_fold_imputer(s["GradientBoostingClassifier"](random_state=RANDOM_STATE)),
        "XGBoost": _with_fold_imputer(_xgb_classifier(**xgb_options)),
    }


def _probabilities(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        return np.asarray(probabilities)[:, 1]
    scores = np.asarray(model.decision_function(features), dtype=float)
    return 1.0 / (1.0 + np.exp(-scores))


def _metric_values(y_true: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5) -> dict[str, float]:
    s = _import_sklearn()
    actual = np.asarray(y_true)
    scores = np.asarray(probabilities)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = s["confusion_matrix"](actual, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(s["accuracy_score"](actual, predictions)),
        "roc_auc": float(s["roc_auc_score"](actual, scores)),
        "pr_auc": float(s["average_precision_score"](actual, scores)),
        "f1": float(s["f1_score"](actual, predictions, zero_division=0)),
        "f2": float(s["fbeta_score"](actual, predictions, beta=2, zero_division=0)),
        "precision": float(s["precision_score"](actual, predictions, zero_division=0)),
        "recall": float(s["recall_score"](actual, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "brier": float(s["brier_score_loss"](actual, scores)),
        "true_negatives": float(tn),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_positives": float(tp),
    }


def evaluate_payload(
    payload: Mapping[str, Any],
    frame: pd.DataFrame,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate a calibrated payload on one explicit, shared cohort.

    The returned row-level table is intentionally kept alongside the aggregate
    metrics so champion/challenger comparisons can prove that they used the
    same rows in the same order.
    """

    selected = list(payload.get("features", SELECTED_FEATURES))
    _require_columns(frame, [*selected, TARGET_COLUMN])
    working = frame.reset_index(drop=True).copy()
    for column in selected:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    if working[selected].isna().any().any():
        missing = working[selected].columns[working[selected].isna().any()].tolist()
        raise ValueError("Evaluation frame has missing or non-numeric model features: " + ", ".join(missing))
    actual = pd.to_numeric(working[TARGET_COLUMN], errors="coerce")
    if actual.isna().any() or not actual.isin([0, 1]).all():
        raise ValueError("Evaluation frame must contain a binary 0/1 Diagnosis column.")

    model = payload.get("model")
    if model is None:
        raise ValueError("Artifact payload is missing its model.")
    chosen_threshold = float(
        payload.get("threshold", 0.5) if threshold is None else threshold
    )
    scores = _probabilities(model, working[selected])
    predictions = (scores >= chosen_threshold).astype(int)
    metrics = _metric_values(actual.astype(int), scores, chosen_threshold)
    s = _import_sklearn()
    confusion = s["confusion_matrix"](actual.astype(int), predictions, labels=[0, 1])
    identity_columns = [
        column
        for column in ["_source_row", "PatientID"]
        if column in working.columns
    ]
    rows = working[identity_columns].reset_index(drop=True) if identity_columns else pd.DataFrame(index=range(len(working)))
    rows["y_true"] = actual.astype(int).to_numpy()
    rows["calibrated_score"] = scores
    rows["screen_prediction"] = np.where(predictions == 1, "Positive", "Negative")
    rows["threshold"] = chosen_threshold
    return {
        "metrics": metrics,
        "confusion_matrix": {
            "labels": ["Negative", "Positive"],
            "counts": confusion.tolist(),
            "threshold": chosen_threshold,
        },
        "row_predictions": rows,
        "predictions": rows,
        "threshold": chosen_threshold,
        "n_rows": int(len(rows)),
    }


def repeated_cv_metrics(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 2,
    return_folds: bool = False,
) -> dict[str, Any]:
    """Run leakage-safe repeated stratified CV and average fold metrics."""

    s = _import_sklearn()
    cv = s["RepeatedStratifiedKFold"](n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    fold_metrics: list[dict[str, float]] = []
    for train_indices, validation_indices in cv.split(features, target):
        fitted = s["clone"](model)
        fitted.fit(features.iloc[train_indices], target.iloc[train_indices])
        probabilities = _probabilities(fitted, features.iloc[validation_indices])
        fold_metrics.append(_metric_values(target.iloc[validation_indices], probabilities))
    result = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
    mean_metrics = {key: float(value) for key, value in result.items()}
    if return_folds:
        return {"mean": mean_metrics, "fold_metrics": fold_metrics}
    return mean_metrics


def benchmark_models(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    rows = []
    for name, model in make_candidate_models(target).items():
        metrics = repeated_cv_metrics(model, features, target)
        rows.append(
            {
                "model": name,
                "pr_auc_cv": metrics["pr_auc"],
                "roc_auc_cv": metrics["roc_auc"],
                "f1": metrics["f1"],
                "recall": metrics["recall"],
            }
        )
    return pd.DataFrame(rows).sort_values("pr_auc_cv", ascending=False).reset_index(drop=True)


def rank_features(
    features: pd.DataFrame,
    target: pd.Series,
    validation_features: pd.DataFrame | None = None,
    validation_target: pd.Series | None = None,
) -> pd.DataFrame:
    """Rank all features with validation-only permutation importance."""

    s = _import_sklearn()
    if validation_features is None or validation_target is None:
        validation_features, validation_target = features, target
    model = _with_fold_imputer(_xgb_classifier(scale_pos_weight=_positive_weight(target)))
    model.fit(features, target)
    importance = s["permutation_importance"](
        model,
        validation_features,
        validation_target,
        scoring="accuracy",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    ranking = pd.DataFrame(
        {
            "feature": validation_features.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    )
    return ranking.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def nested_cv_feature_stability(
    features: pd.DataFrame,
    target: pd.Series,
    feature_count: int = 5,
    n_splits: int = 5,
    n_repeats: int = 3,
) -> dict[str, Any]:
    """Estimate feature-selection stability with an inner ranking per outer fold.

    The outer validation rows are never passed to ``rank_features``. This is a
    diagnostic for selection optimism; the final V3 artifact still retains its
    fixed five-feature contract.
    """

    s = _import_sklearn()
    cv = s["RepeatedStratifiedKFold"](
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )
    counts = {feature: 0 for feature in features.columns}
    outer_metrics: list[dict[str, float]] = []
    for outer_train_idx, outer_test_idx in cv.split(features, target):
        outer_x = features.iloc[outer_train_idx]
        outer_y = target.iloc[outer_train_idx]
        inner_x, _selection_x, inner_y, _selection_y = s["train_test_split"](
            outer_x, outer_y, test_size=0.25, stratify=outer_y, random_state=RANDOM_STATE
        )
        ranking = rank_features(inner_x, inner_y, _selection_x, _selection_y)
        selected = ranking["feature"].head(feature_count).tolist()
        for feature in selected:
            counts[feature] += 1
        model = _with_fold_imputer(_xgb_classifier(scale_pos_weight=_positive_weight(outer_y)))
        model.fit(outer_x[selected], outer_y)
        outer_probabilities = _probabilities(model, features.iloc[outer_test_idx][selected])
        outer_metrics.append(_metric_values(target.iloc[outer_test_idx], outer_probabilities))
    total_folds = n_splits * n_repeats
    stability = [
        {
            "feature": feature,
            "selected_count": int(count),
            "selection_rate": float(count / total_folds),
        }
        for feature, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {
        "outer_folds": total_folds,
        "feature_count": feature_count,
        "feature_stability": stability,
        "outer_metrics_mean": pd.DataFrame(outer_metrics).mean(numeric_only=True).to_dict(),
        "outer_metrics_std": pd.DataFrame(outer_metrics).std(ddof=1, numeric_only=True).to_dict(),
    }


def _build_imbalance_estimator(
    method: str,
    features: pd.DataFrame,
    target: pd.Series,
    params: Mapping[str, Any] | None = None,
) -> Any:
    """Build the selected imbalance strategy without sampling outside a fold."""

    if method != "SMOTENC":
        weight = _positive_weight(target) if method == "scale_pos_weight" else 1.0
        return _with_fold_imputer(_xgb_classifier(scale_pos_weight=weight, **dict(params or {})))
    try:
        from imblearn.over_sampling import SMOTENC
        from imblearn.pipeline import Pipeline as ImbPipeline
    except ImportError as exc:
        raise RuntimeError("SMOTENC was selected but imbalanced-learn is unavailable.") from exc
    categorical_indices = [
        index for index, name in enumerate(features.columns) if name in CATEGORICAL_COLUMNS
    ]
    return ImbPipeline(
        steps=[
            ("imputer", _import_sklearn()["SimpleImputer"](strategy="median")),
            ("sampler", SMOTENC(categorical_features=categorical_indices, random_state=RANDOM_STATE)),
            ("model", _xgb_classifier(scale_pos_weight=1.0, **dict(params or {}))),
        ]
    )


def compare_feature_counts(
    features: pd.DataFrame,
    target: pd.Series,
    ranked_features: Sequence[str],
    counts: Sequence[int] = (3, 5, 8, 12, 20, 32),
) -> pd.DataFrame:
    rows = []
    for count in counts:
        selected = list(ranked_features[: min(count, len(ranked_features))])
        metrics = repeated_cv_metrics(
            _with_fold_imputer(_xgb_classifier(scale_pos_weight=_positive_weight(target))),
            features[selected],
            target,
        )
        rows.append(
            {
                "feature_count": len(selected),
                "features": selected,
                "accuracy_cv": metrics["accuracy"],
                "pr_auc_cv": metrics["pr_auc"],
            }
        )
    return pd.DataFrame(rows)


def compare_imbalance_methods(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Compare baseline, class weighting, and SMOTENC where imblearn is installed."""

    rows: list[dict[str, Any]] = []
    weight = _positive_weight(target)
    candidates: list[tuple[str, Any]] = [
        ("Baseline", _with_fold_imputer(_xgb_classifier(scale_pos_weight=1.0))),
        ("scale_pos_weight", _with_fold_imputer(_xgb_classifier(scale_pos_weight=weight))),
    ]
    try:
        from imblearn.over_sampling import SMOTENC
        from imblearn.pipeline import Pipeline as ImbPipeline

        categorical_indices = [index for index, name in enumerate(features.columns) if name in CATEGORICAL_COLUMNS]
        candidates.append(
            (
                "SMOTENC",
                ImbPipeline(
                    steps=[
                        ("imputer", _import_sklearn()["SimpleImputer"](strategy="median")),
                        (
                            "sampler",
                            SMOTENC(categorical_features=categorical_indices, random_state=RANDOM_STATE),
                        ),
                        ("model", _xgb_classifier(scale_pos_weight=1.0)),
                    ]
                ),
            )
        )
        smotenc_status = "available"
    except ImportError:
        smotenc_status = "imbalanced-learn not installed"

    for name, model in candidates:
        metrics = repeated_cv_metrics(model, features, target, return_folds=True)
        fold_frame = pd.DataFrame(metrics["fold_metrics"])
        rows.append(
            {
                "method": name,
                "accuracy": metrics["mean"]["accuracy"],
                "accuracy_std": float(fold_frame["accuracy"].std(ddof=1)),
                "pr_auc": metrics["mean"]["pr_auc"],
                "pr_auc_std": float(fold_frame["pr_auc"].std(ddof=1)),
                "f1": metrics["mean"]["f1"],
                "f1_std": float(fold_frame["f1"].std(ddof=1)),
                "recall": metrics["mean"]["recall"],
                "recall_std": float(fold_frame["recall"].std(ddof=1)),
                "f2": metrics["mean"]["f2"],
                "f2_std": float(fold_frame["f2"].std(ddof=1)),
                "brier": metrics["mean"]["brier"],
                "brier_std": float(fold_frame["brier"].std(ddof=1)),
                "fold_count": int(len(fold_frame)),
                "status": "available",
            }
        )
    if smotenc_status != "available":
        rows.append(
            {
                "method": "SMOTENC",
                "accuracy": np.nan,
                "accuracy_std": np.nan,
                "pr_auc": np.nan,
                "pr_auc_std": np.nan,
                "f1": np.nan,
                "f1_std": np.nan,
                "recall": np.nan,
                "recall_std": np.nan,
                "f2": np.nan,
                "f2_std": np.nan,
                "brier": np.nan,
                "brier_std": np.nan,
                "fold_count": 0,
                "status": smotenc_status,
            }
        )
    return pd.DataFrame(rows)


def choose_imbalance_method(comparison: pd.DataFrame, tie_delta: float = 0.005) -> dict[str, Any]:
    """Choose a screening-oriented imbalance strategy without using the locked test.

    Mean PR-AUC defines the eligible band. Within that band, F2 and recall are
    preferred because false negatives are costly in screening; Brier score and
    implementation simplicity resolve the remaining ties.
    """

    if "pr_auc" not in comparison:
        return {"method": "scale_pos_weight", "rule": "fallback: PR-AUC comparison unavailable"}
    status = comparison["status"] if "status" in comparison else pd.Series("available", index=comparison.index)
    available = comparison.loc[(status == "available") & comparison["pr_auc"].notna()].copy()
    if available.empty:
        return {"method": "scale_pos_weight", "rule": "fallback: no available comparison"}

    best_pr_auc = float(available["pr_auc"].max())
    eligible = available[available["pr_auc"] >= best_pr_auc - float(tie_delta)].copy()
    preference = {"Baseline": 0, "scale_pos_weight": 1, "SMOTENC": 2}
    f2_values = eligible["f2"] if "f2" in eligible else pd.Series(np.nan, index=eligible.index)
    recall_values = eligible["recall"] if "recall" in eligible else pd.Series(np.nan, index=eligible.index)
    brier_values = eligible["brier"] if "brier" in eligible else pd.Series(np.nan, index=eligible.index)
    eligible["_f2"] = pd.to_numeric(f2_values, errors="coerce").fillna(-np.inf)
    eligible["_recall"] = pd.to_numeric(recall_values, errors="coerce").fillna(-np.inf)
    eligible["_brier"] = pd.to_numeric(brier_values, errors="coerce").fillna(np.inf)
    eligible["_preference"] = eligible["method"].map(preference).fillna(99)
    selected = eligible.sort_values(
        ["_f2", "_recall", "_brier", "_preference", "pr_auc"],
        ascending=[False, False, True, True, False],
        kind="stable",
    ).iloc[0]
    return {
        "method": str(selected["method"]),
        "rule": (
            f"PR-AUC within {tie_delta:.3f} of best; then highest F2, highest recall, "
            "lowest Brier, and simplest method"
        ),
        "best_mean_pr_auc": best_pr_auc,
        "selected_mean_pr_auc": float(selected["pr_auc"]),
        "selected_mean_f2": None if not np.isfinite(selected["_f2"]) else float(selected["_f2"]),
        "selected_mean_recall": None if not np.isfinite(selected["_recall"]) else float(selected["_recall"]),
        "selected_mean_brier": None if not np.isfinite(selected["_brier"]) else float(selected["_brier"]),
        "selected_fold_count": int(selected.get("fold_count", 0)),
    }


def tune_xgboost(
    features: pd.DataFrame,
    target: pd.Series,
    imbalance_method: str = "scale_pos_weight",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune the same imbalance estimator family that will be fitted finally."""

    s = _import_sklearn()
    model = _build_imbalance_estimator(imbalance_method, features, target)
    grid = {
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [3, 4, 5],
        "model__n_estimators": [200, 300, 400],
    }
    search = s["GridSearchCV"](
        model,
        param_grid=grid,
        scoring="average_precision",
        cv=s["StratifiedKFold"](n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
        refit=True,
    )
    search.fit(features, target)
    results = pd.DataFrame(search.cv_results_)[
        [
            "param_model__learning_rate",
            "param_model__max_depth",
            "param_model__n_estimators",
            "mean_test_score",
            "rank_test_score",
        ]
    ].sort_values("rank_test_score")
    best_params = {
        key.removeprefix("model__"): value
        for key, value in search.best_params_.items()
    }
    results = results.rename(
        columns={
            "param_model__learning_rate": "param_learning_rate",
            "param_model__max_depth": "param_max_depth",
            "param_model__n_estimators": "param_n_estimators",
        }
    )
    return best_params, results.reset_index(drop=True)


def choose_f2_threshold(target: pd.Series, probabilities: Sequence[float]) -> tuple[float, float]:
    s = _import_sklearn()
    thresholds = np.round(np.arange(0.05, 0.951, 0.001), 3)
    scores = []
    for threshold in thresholds:
        predictions = (np.asarray(probabilities) >= threshold).astype(int)
        scores.append(s["fbeta_score"](target, predictions, beta=2, zero_division=0))
    best_score = max(scores)
    candidates = [float(thresholds[index]) for index, score in enumerate(scores) if score == best_score]
    threshold = min(candidates, key=lambda value: abs(value - 0.490))
    return threshold, float(best_score)


def threshold_table(
    target: Sequence[int],
    probabilities: Sequence[float],
    thresholds: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Return threshold operating points using labels from validation only."""

    s = _import_sklearn()
    actual = np.asarray(target).astype(int)
    scores = np.asarray(probabilities, dtype=float)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if thresholds is None else np.asarray(thresholds, dtype=float)
    records: list[dict[str, Any]] = []
    for threshold in grid:
        predictions = (scores >= threshold).astype(int)
        tn, fp, fn, tp = s["confusion_matrix"](actual, predictions, labels=[0, 1]).ravel()
        records.append(
            {
                "threshold": float(threshold),
                "precision": float(s["precision_score"](actual, predictions, zero_division=0)),
                "recall": float(s["recall_score"](actual, predictions, zero_division=0)),
                "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
                "f1": float(s["f1_score"](actual, predictions, zero_division=0)),
                "f2": float(s["fbeta_score"](actual, predictions, beta=2, zero_division=0)),
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "true_positives": int(tp),
            }
        )
    return pd.DataFrame(records)


def _expected_calibration_error(
    target: Sequence[int],
    probabilities: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Calculate a quantile-binned expected calibration error."""

    actual = np.asarray(target).astype(int)
    scores = np.asarray(probabilities, dtype=float)
    if len(actual) == 0 or len(actual) != len(scores):
        return float("nan")
    if not np.isfinite(scores).all():
        return float("nan")
    order = np.argsort(scores, kind="stable")
    boundaries = np.linspace(0, len(scores), int(n_bins) + 1, dtype=int)
    error = 0.0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        indices = order[start:end]
        error += (len(indices) / len(scores)) * abs(
            float(actual[indices].mean()) - float(scores[indices].mean())
        )
    return float(error)


def calibration_summary(
    target: Sequence[int],
    probabilities: Sequence[float],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Summarize validation calibration against prevalence and a null model."""

    s = _import_sklearn()
    actual = np.asarray(target).astype(int)
    scores = np.asarray(probabilities, dtype=float)
    prevalence = float(actual.mean()) if len(actual) else np.nan
    try:
        observed, predicted = s["calibration_curve"](actual, scores, n_bins=n_bins, strategy="quantile")
    except ValueError:
        observed, predicted = np.array([]), np.array([])
    intercept, slope = calibration_intercept_slope(actual, scores)
    return {
        "brier": float(s["brier_score_loss"](actual, scores)),
        "ece": _expected_calibration_error(actual, scores, n_bins=n_bins),
        "prevalence": prevalence,
        "null_brier": float(prevalence * (1 - prevalence)) if np.isfinite(prevalence) else np.nan,
        "intercept": intercept,
        "slope": slope,
        "intercept_slope_status": "estimable" if np.isfinite(intercept) and np.isfinite(slope) else "not estimable",
        "n_bins": int(len(observed)),
        "mean_predicted": [float(value) for value in predicted],
        "fraction_positive": [float(value) for value in observed],
    }


def calibration_intercept_slope(
    target: Sequence[int],
    probabilities: Sequence[float],
) -> tuple[float, float]:
    """Estimate calibration intercept and slope on the logit score scale.

    A calibrated model has an intercept near zero and a slope near one. The
    regression is fitted only on the explicitly supplied evaluation cohort;
    it is a diagnostic, not another calibration step.
    """

    actual = np.asarray(target).astype(int)
    scores = np.asarray(probabilities, dtype=float)
    if len(actual) == 0 or len(actual) != len(scores) or len(np.unique(actual)) < 2:
        return float("nan"), float("nan")
    s = _import_sklearn()
    logits = np.log(np.clip(scores, 1e-6, 1 - 1e-6) / np.clip(1 - scores, 1e-6, 1 - 1e-6))
    try:
        calibration_model = s["LogisticRegression"](solver="lbfgs", max_iter=2000)
        calibration_model.fit(logits.reshape(-1, 1), actual)
        return float(calibration_model.intercept_[0]), float(calibration_model.coef_[0][0])
    except (ValueError, FloatingPointError):
        return float("nan"), float("nan")


def subgroup_metrics(
    payload: Mapping[str, Any],
    frame: pd.DataFrame,
    group_columns: Sequence[str] | None = None,
    min_group_size: int = 20,
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    """Calculate guarded subgroup metrics on one explicitly supplied cohort."""

    selected = list(payload.get("features", SELECTED_FEATURES))
    _require_columns(frame, [*selected, TARGET_COLUMN])
    working = frame.reset_index(drop=True).copy()
    for column in selected:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    age = pd.to_numeric(working.get("Age", pd.Series(index=working.index)), errors="coerce")
    working["AgeGroup"] = pd.cut(
        age, bins=[59, 69, 79, 90], labels=["60-69", "70-79", "80-90"], include_lowest=True
    )
    requested = list(group_columns or [
        "Gender", "AgeGroup", "Ethnicity", "EducationLevel", "Diabetes",
        "Hypertension", "FamilyHistoryAlzheimers",
    ])
    scores = _probabilities(payload["model"], working[selected])
    actual = pd.to_numeric(working[TARGET_COLUMN], errors="coerce").astype(int).to_numpy()
    threshold = float(payload.get("threshold", 0.5))
    records: list[dict[str, Any]] = []
    for column in requested:
        if column not in working.columns:
            continue
        for value, indices in working.groupby(column, dropna=False, observed=False).groups.items():
            indices_array = np.asarray(list(indices), dtype=int)
            n = len(indices_array)
            row: dict[str, Any] = {
                "group": column,
                "level": "Missing" if pd.isna(value) else str(value),
                "n": n,
                "prevalence": float(actual[indices_array].mean()) if n else np.nan,
                "status": "estimable" if n >= min_group_size and len(np.unique(actual[indices_array])) == 2 else "not estimable",
            }
            if row["status"] == "estimable":
                metrics = _metric_values(actual[indices_array], scores[indices_array], threshold)
                row.update({name: float(value) for name, value in metrics.items()})
                ci = bootstrap_confidence_intervals(
                    actual[indices_array], scores[indices_array], threshold, n_bootstrap=n_bootstrap
                )
                for name, bounds in ci.items():
                    row[f"{name}_ci_low"] = float(bounds[0])
                    row[f"{name}_ci_high"] = float(bounds[1])
            else:
                for name in ["accuracy", "roc_auc", "pr_auc", "f1", "f2", "precision", "recall", "brier"]:
                    row[name] = np.nan
            records.append(row)
    return pd.DataFrame(records)


def summarize_prediction_batch(
    predictions: pd.DataFrame,
    validation_summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Return privacy-minimized operational counters without row-level data."""

    score = pd.to_numeric(predictions.get("calibrated_score", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": metadata.get("version", "unknown"),
        "schema_version": metadata.get("schema_version", "unknown"),
        "batch_size": int(len(predictions)),
        "valid_count": int(validation_summary.get("valid_rows", len(predictions))),
        "rejected_count": int(validation_summary.get("rejected_rows", 0)),
        "imputed_count": int(validation_summary.get("imputed_rows", 0)),
        "positive_screening_rate": float((predictions.get("screen_prediction", pd.Series(dtype=str)) == "Positive").mean()) if len(predictions) else 0.0,
        "score_min": float(score.min()) if not score.empty else None,
        "score_median": float(score.median()) if not score.empty else None,
        "score_max": float(score.max()) if not score.empty else None,
    }


def bootstrap_confidence_intervals(
    target: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
    n_bootstrap: int = 1000,
) -> dict[str, list[float]]:
    actual = np.asarray(target)
    scores = np.asarray(probabilities)
    rng = np.random.default_rng(RANDOM_STATE)
    names = ["roc_auc", "pr_auc", "precision", "recall"]
    samples = {name: [] for name in names}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(actual), len(actual))
        boot_y = actual[indices]
        if len(np.unique(boot_y)) < 2:
            continue
        metrics = _metric_values(boot_y, scores[indices], threshold)
        for name in names:
            samples[name].append(metrics[name])
    return {
        name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for name, values in samples.items()
        if values
    }


def paired_bootstrap_difference(
    y_true: Sequence[int],
    old_scores: Sequence[float],
    new_scores: Sequence[float],
    metric_fn: Any,
    n_bootstrap: int = 2000,
) -> dict[str, float | int]:
    """Estimate new-minus-old performance using paired row resampling."""

    actual = np.asarray(y_true)
    old = np.asarray(old_scores)
    new = np.asarray(new_scores)
    if not (len(actual) == len(old) == len(new)):
        raise ValueError("Paired bootstrap inputs must have the same number of rows.")
    if len(actual) == 0:
        raise ValueError("Paired bootstrap requires at least one row.")

    def difference(indices: np.ndarray) -> float:
        return float(metric_fn(actual[indices], new[indices]) - metric_fn(actual[indices], old[indices]))

    try:
        estimate = difference(np.arange(len(actual)))
    except Exception as exc:
        raise ValueError(f"Could not calculate paired metric: {exc}") from exc

    rng = np.random.default_rng(RANDOM_STATE)
    samples: list[float] = []
    for _ in range(int(n_bootstrap)):
        indices = rng.integers(0, len(actual), len(actual))
        try:
            value = difference(indices)
        except Exception:
            continue
        if np.isfinite(value):
            samples.append(value)
    if not samples:
        return {
            "estimate": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_bootstrap": 0,
        }
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "n_bootstrap": len(samples),
    }


def compare_subgroup_metrics(
    champion_metrics: pd.DataFrame | Sequence[Mapping[str, Any]],
    challenger_metrics: pd.DataFrame | Sequence[Mapping[str, Any]],
    policy: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Create promotion checks for every common, estimable subgroup.

    Small or one-class groups remain visible in the model card but are not
    used as automatic promotion gates because their metric is not estimable.
    Every common estimable group is checked for recall and PR-AUC degradation.
    """

    chosen_policy = {**PROMOTION_POLICY, **dict(policy or {})}
    old = pd.DataFrame(champion_metrics)
    new = pd.DataFrame(challenger_metrics)
    required_columns = {"group", "level", "n", "status", "recall", "pr_auc"}
    if old.empty or new.empty or not required_columns.issubset(old.columns) or not required_columns.issubset(new.columns):
        return [
            {
                "condition": "Estimable subgroup metrics are required",
                "metric": "subgroup_metrics_available",
                "old": 0.0,
                "new": 0.0,
                "delta": 0.0,
                "required": 1.0,
                "passed": False,
                "status": "Fail",
            }
        ]
    key_columns = ["group", "level"]
    merged = old.merge(new, on=key_columns, suffixes=("_old", "_new"))
    min_size = int(chosen_policy["subgroup_min_group_size"])
    checks: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        estimable = (
            row.get("status_old") == "estimable"
            and row.get("status_new") == "estimable"
            and int(row.get("n_old", 0)) >= min_size
            and int(row.get("n_new", 0)) >= min_size
        )
        if not estimable:
            continue
        label = f"{row['group']}={row['level']}"
        for metric, tolerance_key in (
            ("recall", "subgroup_recall_tolerance"),
            ("pr_auc", "subgroup_pr_auc_tolerance"),
        ):
            old_value = float(row[f"{metric}_old"])
            new_value = float(row[f"{metric}_new"])
            change = new_value - old_value
            tolerance = float(chosen_policy[tolerance_key])
            checks.append(
                {
                    "condition": f"{label} {metric} may not fall by more than tolerance",
                    "metric": f"subgroup_{metric}:{label}",
                    "old": old_value,
                    "new": new_value,
                    "delta": change,
                    "required": -tolerance,
                    "passed": change >= -tolerance,
                    "group": row["group"],
                    "level": row["level"],
                    "n_old": int(row["n_old"]),
                    "n_new": int(row["n_new"]),
                }
            )
    if not checks:
        return [
            {
                "condition": "At least one common estimable subgroup is required",
                "metric": "estimable_subgroup_count",
                "old": 0.0,
                "new": 0.0,
                "delta": 0.0,
                "required": 1.0,
                "passed": False,
                "status": "Fail",
            }
        ]
    for check in checks:
        check["status"] = "Pass" if check["passed"] else "Fail"
    return checks


def _monitoring_baseline(data: pd.DataFrame, features: Sequence[str]) -> dict[str, dict[str, float | int]]:
    baseline: dict[str, dict[str, float | int]] = {}
    for column in features:
        if column not in data.columns:
            continue
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            continue
        standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        baseline[column] = {
            "n": int(len(values)),
            "mean": float(values.mean()),
            "std": standard_deviation,
            "q01": float(values.quantile(0.01)),
            "q99": float(values.quantile(0.99)),
        }
    return baseline


def monitor_prediction_batch(
    data: pd.DataFrame,
    validation_summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
    drift_z_threshold: float = 0.5,
    rejection_rate_threshold: float = 0.20,
    imputation_rate_threshold: float = 0.20,
) -> dict[str, Any]:
    """Return privacy-minimized batch drift counters and actionable alerts."""

    input_rows = int(validation_summary.get("input_rows", len(data)))
    rejected_rows = int(validation_summary.get("rejected_rows", 0))
    imputed_rows = int(validation_summary.get("imputed_rows", 0))
    rejection_rate = rejected_rows / input_rows if input_rows else 0.0
    imputation_rate = imputed_rows / input_rows if input_rows else 0.0
    alerts: list[dict[str, Any]] = []
    if rejection_rate > rejection_rate_threshold:
        alerts.append({
            "type": "quality",
            "metric": "rejection_rate",
            "value": rejection_rate,
            "threshold": rejection_rate_threshold,
            "message": "Rejected-row rate is above the configured data-quality threshold.",
        })
    if imputation_rate > imputation_rate_threshold:
        alerts.append({
            "type": "quality",
            "metric": "imputation_rate",
            "value": imputation_rate,
            "threshold": imputation_rate_threshold,
            "message": "Imputation rate is above the configured data-quality threshold.",
        })

    drift_rows: list[dict[str, Any]] = []
    baseline = metadata.get("monitoring_baseline", {})
    for column, reference in baseline.items():
        if column not in data.columns:
            continue
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if values.empty:
            continue
        reference_mean = float(reference.get("mean", np.nan))
        reference_std = float(reference.get("std", 0.0))
        current_mean = float(values.mean())
        denominator = max(reference_std, 1e-6)
        standardized_shift = abs(current_mean - reference_mean) / denominator
        drift_row = {
            "feature": column,
            "reference_mean": reference_mean,
            "current_mean": current_mean,
            "standardized_mean_shift": standardized_shift,
            "status": "alert" if standardized_shift > drift_z_threshold else "within_threshold",
        }
        drift_rows.append(drift_row)
        if standardized_shift > drift_z_threshold:
            alerts.append({
                "type": "drift",
                "metric": f"feature_mean:{column}",
                "value": standardized_shift,
                "threshold": drift_z_threshold,
                "message": f"Feature {column} mean shifted beyond the configured monitoring threshold.",
            })
    if not baseline:
        alerts.append({
            "type": "configuration",
            "metric": "monitoring_baseline",
            "value": None,
            "threshold": None,
            "message": "Training monitoring baseline is not stored in this artifact; retrain before relying on drift alerts.",
        })
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": metadata.get("version", "unknown"),
        "batch_rows": int(len(data)),
        "rejection_rate": float(rejection_rate),
        "imputation_rate": float(imputation_rate),
        "drift_features": drift_rows,
        "alerts": alerts,
        "status": "alert" if alerts else "ok",
        "thresholds": {
            "drift_z": drift_z_threshold,
            "rejection_rate": rejection_rate_threshold,
            "imputation_rate": imputation_rate_threshold,
        },
    }


def evaluate_promotion_policy(
    champion_evaluation: Mapping[str, Any],
    challenger_evaluation: Mapping[str, Any],
    policy: Mapping[str, float] | None = None,
    paired_differences: Mapping[str, Mapping[str, float | int]] | None = None,
    subgroup_checks: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply the fixed promotion policy to same-cohort evaluations."""

    chosen_policy = {**PROMOTION_POLICY, **dict(policy or {})}
    old_metrics = dict(champion_evaluation["metrics"])
    new_metrics = dict(challenger_evaluation["metrics"])
    differences = dict(paired_differences or {})

    def delta(name: str) -> float:
        return float(new_metrics[name] - old_metrics[name])

    checks = [
        {
            "condition": "PR-AUC must not decrease",
            "metric": "pr_auc",
            "old": float(old_metrics["pr_auc"]),
            "new": float(new_metrics["pr_auc"]),
            "delta": delta("pr_auc"),
            "required": float(chosen_policy["pr_auc_min_delta"]),
            "passed": delta("pr_auc") >= float(chosen_policy["pr_auc_min_delta"]),
        },
        {
            "condition": "Recall may not fall by more than tolerance",
            "metric": "recall",
            "old": float(old_metrics["recall"]),
            "new": float(new_metrics["recall"]),
            "delta": delta("recall"),
            "required": -float(chosen_policy["recall_tolerance"]),
            "passed": delta("recall") >= -float(chosen_policy["recall_tolerance"]),
        },
        {
            "condition": "Brier score may not worsen beyond tolerance",
            "metric": "brier",
            "old": float(old_metrics["brier"]),
            "new": float(new_metrics["brier"]),
            "delta": delta("brier"),
            "required": float(chosen_policy["brier_max_delta"]),
            "passed": delta("brier") <= float(chosen_policy["brier_max_delta"]),
        },
        {
            "condition": "F2 must improve or have a positive paired CI",
            "metric": "f2",
            "old": float(old_metrics["f2"]),
            "new": float(new_metrics["f2"]),
            "delta": delta("f2"),
            "required": float(chosen_policy["f2_min_delta"]),
            "passed": (
                delta("f2") >= float(chosen_policy["f2_min_delta"])
                or float(differences.get("f2", {}).get("ci_low", float("-inf")))
                > float(chosen_policy["f2_ci_low_min"])
            ),
        },
    ]
    if subgroup_checks is not None:
        checks.extend(dict(check) for check in subgroup_checks)
    for check in checks:
        check["passed"] = bool(check["passed"])
        check["status"] = "Pass" if check["passed"] else "Fail"
    passed = all(bool(check["passed"]) for check in checks)
    return {
        "policy": chosen_policy,
        "checks": checks,
        "passed": passed,
        "reason": "All promotion conditions passed." if passed else "One or more promotion conditions failed.",
        "champion_version": champion_evaluation.get("version"),
        "challenger_version": challenger_evaluation.get("version"),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _metadata_integrity_hash(metadata: Mapping[str, Any]) -> str:
    """Hash metadata without the self-referential checksum field."""

    unsigned = {key: value for key, value in metadata.items() if key != "metadata_sha256"}
    canonical = json.dumps(
        _json_safe(unsigned),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def _save_artifacts(
    model: Any,
    metadata: dict[str, Any],
    artifacts_dir: str | Path,
    promote: bool = True,
) -> dict[str, str]:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to save model artifacts; install requirements.txt") from exc

    root = Path(artifacts_dir)
    versions = root / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    version = str(metadata["version"])
    version_model = versions / f"model_{version}.joblib"
    version_metadata = versions / f"metadata_{version}.json"
    active_model = root / "active_model.joblib"
    active_metadata = root / "active_metadata.json"
    paths = {
        "version_model": str(version_model),
        "version_metadata": str(version_metadata),
        "active_model": str(active_model),
        "active_metadata": str(active_metadata),
    }
    metadata["artifact_paths"] = {
        name: str(Path(value).relative_to(root))
        for name, value in paths.items()
    }
    joblib.dump(
        {
            "model": model,
            "features": metadata["selected_features"],
            "threshold": metadata["threshold"],
            "version": version,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "imputation_statistics": metadata.get("imputation_statistics", {}),
        },
        version_model,
    )
    metadata["model_sha256"] = _sha256_file(version_model)
    metadata["metadata_sha256"] = _metadata_integrity_hash(metadata)
    _atomic_write_text(
        version_metadata,
        json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False) + "\n",
    )

    if promote:
        _promote_artifacts(version_model, version_metadata, root)
    return paths


def _promote_artifacts(
    version_model: str | Path,
    version_metadata: str | Path,
    artifacts_dir: str | Path,
    acquire_lock: bool = True,
) -> None:
    """Promote a complete version with staged copies and rollback."""

    root = Path(artifacts_dir)
    active_model = root / "active_model.joblib"
    active_metadata = root / "active_metadata.json"

    def promote_locked() -> None:
        model_temp = active_model.with_name(f".{active_model.name}.{uuid.uuid4().hex}.tmp")
        metadata_temp = active_metadata.with_name(f".{active_metadata.name}.{uuid.uuid4().hex}.tmp")
        old_model = active_model.read_bytes() if active_model.exists() else None
        old_metadata = active_metadata.read_bytes() if active_metadata.exists() else None
        model_replaced = False
        metadata_replaced = False
        try:
            shutil.copy2(version_model, model_temp)
            shutil.copy2(version_metadata, metadata_temp)
            os.replace(model_temp, active_model)
            model_replaced = True
            os.replace(metadata_temp, active_metadata)
            metadata_replaced = True
        except Exception:
            # Restore both old files if either half of the pair was replaced.
            if model_replaced:
                if old_model is None:
                    active_model.unlink(missing_ok=True)
                else:
                    rollback_model = active_model.with_name(f".{active_model.name}.{uuid.uuid4().hex}.rollback")
                    rollback_model.write_bytes(old_model)
                    os.replace(rollback_model, active_model)
            if metadata_replaced:
                if old_metadata is None:
                    active_metadata.unlink(missing_ok=True)
                else:
                    rollback_metadata = active_metadata.with_name(f".{active_metadata.name}.{uuid.uuid4().hex}.rollback")
                    rollback_metadata.write_bytes(old_metadata)
                    os.replace(rollback_metadata, active_metadata)
            raise
        finally:
            model_temp.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)

    if acquire_lock:
        with _artifact_lock(root):
            promote_locked()
    else:
        promote_locked()


def run_training(
    data: pd.DataFrame,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    dataset_label: str = "alzheimers_disease_data.csv",
    fast: bool = False,
    promote: bool = True,
    locked_test_path: str | Path = DEFAULT_LOCKED_TEST_PATH,
    selection_mode: str = "report_v3",
    selection_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train V3 without allowing the locked evaluation cohort into fitting."""

    s = _import_sklearn()
    if selection_mode not in {"report_v3", "nested_cv", "fixed_five"}:
        raise ValueError("selection_mode must be 'report_v3', 'nested_cv', or 'fixed_five'.")
    source = _with_source_rows(data)
    fixed_five = selection_mode == "fixed_five"
    if fixed_five:
        _require_columns(source, [*SELECTED_FEATURES, TARGET_COLUMN])
        clean = source.copy()
        cleaning = {
            "rows_before": len(source),
            "rows_after": len(clean),
            "duplicates_removed": 0,
            "invalid_blood_pressure_rows_removed": 0,
            "missing_cells_after_cleaning": int(clean[SELECTED_FEATURES].isna().sum().sum()),
            "positive_rate_after_cleaning": float(clean[TARGET_COLUMN].mean()),
            "note": "Fixed five-feature retraining; full-schema BP cleaning applies to the baseline only.",
        }
    else:
        clean, cleaning = clean_dataset(source)
    locked_test, locked_was_created = _locked_test_frame(source, locked_test_path)
    training_frame, locked_rows_excluded = _exclude_locked_rows(clean, locked_test)
    if locked_rows_excluded != len(locked_test):
        raise ValueError(
            "Locked test rows could not be matched to the supplied training data; "
            "refusing to train with an unprotected evaluation cohort."
        )
    if training_frame.empty:
        raise ValueError("No training rows remain after excluding the locked test cohort.")

    features = training_frame[SELECTED_FEATURES if fixed_five else ALL_FEATURES].copy()
    target = training_frame[TARGET_COLUMN].astype(int)
    eda_summary = dict((selection_reference or {}).get("eda", {})) if fixed_five else exploratory_summary(training_frame)
    eda_files = [] if fast or fixed_five else [
        Path(path).name
        for path in save_eda_plots(training_frame, eda_summary, Path(artifacts_dir) / "eda")
    ]

    train_x, validation_x, train_y, validation_y = s["train_test_split"](
        features,
        target,
        test_size=0.20,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    # Production preprocessing statistics are fitted on the full development
    # cohort after the locked test has been excluded.
    imputation_statistics = fit_imputation_statistics(features)

    ranking = (
        pd.DataFrame((selection_reference or {}).get("feature_ranking", []))
        if fixed_five else rank_features(train_x, train_y, validation_x, validation_y)
    )
    ranked = ranking["feature"].tolist() if "feature" in ranking else []
    selected = [feature for feature in SELECTED_FEATURES if feature in features.columns]
    selection_warning = None
    if not fixed_five and ranked[: len(selected)] != selected:
        selection_warning = {
            "official_features": selected,
            "validation_permutation_top_features": ranked[: len(selected)],
            "message": "The report V3 feature set is retained; the ranking is stored for auditability.",
        }

    nested_selection = None
    if selection_mode == "nested_cv":
        nested_selection = nested_cv_feature_stability(train_x, train_y)

    if fixed_five:
        feature_comparison = pd.DataFrame((selection_reference or {}).get("feature_selection", []))
        imbalance_comparison = pd.DataFrame((selection_reference or {}).get("imbalance_comparison", []))
        benchmark = pd.DataFrame((selection_reference or {}).get("benchmark", []))
        grid_results = pd.DataFrame()
        imbalance_selection = {
            "method": "scale_pos_weight",
            "rule": "Fixed-five updates use class weighting; no rows are resampled.",
        }
    elif fast:
        feature_comparison = pd.DataFrame()
        imbalance_comparison = pd.DataFrame()
        benchmark = pd.DataFrame()
        grid_results = pd.DataFrame()
        imbalance_selection = {
            "method": "scale_pos_weight",
            "rule": "fast mode skips imbalance comparison; report_v3 default retained",
        }
    else:
        feature_comparison = compare_feature_counts(train_x, train_y, ranked)
        imbalance_comparison = compare_imbalance_methods(train_x[selected], train_y)
        benchmark = benchmark_models(train_x[selected], train_y)
        imbalance_selection = choose_imbalance_method(imbalance_comparison)

    selected_imbalance_method = str(imbalance_selection["method"])
    best_params, grid_results = tune_xgboost(
        train_x[selected],
        train_y,
        imbalance_method=selected_imbalance_method,
    )
    uncalibrated_for_validation = _build_imbalance_estimator(
        selected_imbalance_method,
        train_x[selected],
        train_y,
        best_params,
    )
    uncalibrated_for_validation.fit(train_x[selected], train_y)
    raw_validation_probabilities = _probabilities(uncalibrated_for_validation, validation_x[selected])
    calibration_before = calibration_summary(validation_y, raw_validation_probabilities)
    calibrated_for_validation = s["CalibratedClassifierCV"](
        estimator=_build_imbalance_estimator(selected_imbalance_method, train_x[selected], train_y, best_params),
        method="sigmoid",
        cv=5,
        n_jobs=1,
    )
    calibrated_for_validation.fit(train_x[selected], train_y)
    validation_probabilities = _probabilities(calibrated_for_validation, validation_x[selected])
    threshold, validation_f2 = choose_f2_threshold(validation_y, validation_probabilities)

    final_weight = _positive_weight(target)
    effective_scale_pos_weight = final_weight if selected_imbalance_method == "scale_pos_weight" else 1.0
    final_model = s["CalibratedClassifierCV"](
        estimator=_build_imbalance_estimator(selected_imbalance_method, features[selected], target, best_params),
        method="sigmoid",
        cv=5,
        n_jobs=1,
    )
    # Refit on every non-locked row after model/threshold decisions are fixed.
    final_model.fit(features[selected], target)
    locked_test_x = locked_test[selected].copy()
    locked_test_y = locked_test[TARGET_COLUMN].astype(int)
    test_probabilities = _probabilities(final_model, locked_test_x)
    test_metrics = _metric_values(locked_test_y, test_probabilities, threshold)
    bootstrap = bootstrap_confidence_intervals(locked_test_y, test_probabilities, threshold)
    test_predictions = (test_probabilities >= threshold).astype(int)
    test_confusion = s["confusion_matrix"](locked_test_y, test_predictions, labels=[0, 1])
    validation_threshold_table = threshold_table(validation_y, validation_probabilities)
    validation_calibration = calibration_summary(validation_y, validation_probabilities)
    validation_calibration["threshold_selection"] = {
        "dataset_split": "validation",
        "metric": "F2",
        "tie_break": "closest to 0.490",
    }
    tuning_best_pr_auc = (
        float(grid_results.iloc[0]["mean_test_score"])
        if not grid_results.empty and pd.notna(grid_results.iloc[0].get("mean_test_score"))
        else None
    )
    official_payload = {
        "model": final_model,
        "features": selected,
        "threshold": threshold,
    }
    subgroup = subgroup_metrics(official_payload, locked_test, n_bootstrap=200)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    label_path = Path(str(dataset_label))
    dataset_display = label_path.name if label_path.exists() or "/" in str(dataset_label) or "\\" in str(dataset_label) else str(dataset_label)
    dataset_sha256 = _sha256_file(label_path) if label_path.exists() else _dataframe_sha256(clean)
    source_release_sha256 = _source_release_sha256()
    git_commit = os.environ.get("ALZHEIMER_GIT_COMMIT")
    if not git_commit:
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit = None
    metadata: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "version": timestamp,
        "dataset": dataset_display,
        "dataset_sha256": dataset_sha256,
        "checksum_methods": {
            "model_sha256": "file_bytes",
            "metadata_sha256": "canonical_json_without_metadata_sha256",
            "dataset_sha256": "file_bytes" if label_path.exists() else "canonical_dataframe",
            "locked_test_sha256": "canonical_dataframe",
        },
        "source_release_sha256": source_release_sha256,
        "git_commit": git_commit,
        "source_revision": git_commit or f"source-sha256:{source_release_sha256[:16]}",
        "git_commit_status": "available" if git_commit else "unversioned checkout; source hash recorded",
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_command": "Streamlit fixed-five retraining" if fixed_five else "python train_pipeline.py",
        "locked_test_path": Path(locked_test_path).name,
        "locked_test_sha256": _dataframe_sha256(locked_test),
        "validation_scope": "internal",
        "external_validation_status": "not performed",
        "locked_test_created": locked_was_created,
        "random_state": RANDOM_STATE,
        "rows_raw": int(len(data)),
        "rows_clean": int(len(clean)),
        "split_sizes": {
            "train": int(len(train_x)),
            "validation": int(len(validation_x)),
            "development": int(len(features)),
            "final_training": int(len(features)),
            "train_validation": int(len(features)),
            "test": int(len(locked_test)),
            "locked_test": int(len(locked_test)),
        },
        "all_features": ALL_FEATURES,
        "selected_features": selected,
        "feature_selection_provenance": (
            f"Retained from model {(selection_reference or {}).get('version', 'unknown')}; "
            "new clinical rows provide only the fixed five features."
            if fixed_five else "Recomputed from the full V3 dataset."
        ),
        "imputation_statistics": imputation_statistics,
        "preprocessing": {
            "type": "sklearn Pipeline",
            "imputer": "SimpleImputer(strategy='median') fitted inside each training fold",
            "sampler_scope": "SMOTENC, when selected, runs after imputation inside each fold",
        },
        "monitoring_baseline": _monitoring_baseline(features, selected),
        "feature_ranking": ranking.to_dict(orient="records"),
        "eda": eda_summary,
        "eda_files": eda_files,
        "feature_selection": feature_comparison.to_dict(orient="records"),
        "imbalance_comparison": imbalance_comparison.to_dict(orient="records"),
        "benchmark": benchmark.to_dict(orient="records"),
        "imbalance_selection": imbalance_selection,
        "selected_imbalance_method": selected_imbalance_method,
        "selection_mode": selection_mode,
        "nested_cv_selection": nested_selection,
        "hyperparameters": {
            **best_params,
            "scale_pos_weight": effective_scale_pos_weight,
            "selected_imbalance_method": selected_imbalance_method,
        },
        "grid_search_candidates": int(len(grid_results)) if not grid_results.empty else 27,
        "tuning_best_pr_auc": tuning_best_pr_auc,
        "calibration": {
            "method": "Platt scaling via sigmoid calibration",
            "validation_f2": validation_f2,
            "threshold": threshold,
            "before": {
                **calibration_before,
                "dataset_split": "validation",
                "method": "uncalibrated estimator",
            },
            "after": {
                **validation_calibration,
                "dataset_split": "validation",
                "method": "sigmoid/Platt calibration",
            },
            "validation_summary": validation_calibration,
        },
        "threshold": threshold,
        "threshold_selection": {
            "dataset_split": "validation",
            "metric": "F2",
            "tie_break": "closest to 0.490",
            "selected_threshold": threshold,
            "selected_f2": validation_f2,
            "table": validation_threshold_table.to_dict(orient="records"),
        },
        "validation_metrics": _metric_values(validation_y, validation_probabilities, threshold),
        "test_metrics": test_metrics,
        "test_confusion_matrix": {
            "labels": ["Negative", "Positive"],
            "counts": test_confusion.tolist(),
            "threshold": threshold,
        },
        "bootstrap_95_ci": bootstrap,
        "subgroup_metrics": subgroup.to_dict(orient="records"),
        "cleaning": cleaning,
        "evaluation_cohort": {
            "name": "locked_test",
            "rows": int(len(locked_test)),
            "patient_ids_present": "PatientID" in locked_test.columns,
        },
        "promotion_decision": None,
        "warnings": [
            "Educational screening demo only; not a medical diagnostic device.",
            "Metrics are dataset-specific and require external validation before clinical interpretation.",
        ],
    }
    if selection_warning:
        metadata["warnings"].append(selection_warning)

    paths = _save_artifacts(final_model, metadata, artifacts_dir, promote=promote)
    return {"model": final_model, "metadata": metadata, "paths": paths}


def validate_artifact_contract(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    model_path: str | Path | None = None,
) -> None:
    """Reject stale, incomplete, or mismatched model/metadata pairs."""

    missing = sorted(REQUIRED_METADATA_KEYS.difference(metadata.keys()))
    if missing:
        raise RuntimeError("Artifact metadata is missing required keys: " + ", ".join(missing))
    if metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported artifact schema_version={metadata.get('schema_version')}; "
            f"expected {ARTIFACT_SCHEMA_VERSION}. Run python train_pipeline.py."
        )
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("Model payload schema_version does not match artifact schema v2.")
    if payload.get("version") != metadata.get("version"):
        raise RuntimeError("Model payload version does not match metadata version.")
    metadata_features = list(metadata.get("selected_features", []))
    payload_features = list(payload.get("features", []))
    if metadata_features != payload_features or metadata_features != SELECTED_FEATURES:
        raise RuntimeError("Artifact feature order does not match the official five-feature schema.")
    if float(payload.get("threshold", -1)) != float(metadata.get("threshold", -2)):
        raise RuntimeError("Artifact threshold does not match between payload and metadata.")
    if dict(payload.get("imputation_statistics", {})) != dict(metadata.get("imputation_statistics", {})):
        raise RuntimeError("Artifact imputation statistics do not match between payload and metadata.")
    if model_path is not None:
        actual_hash = _sha256_file(model_path)
        if actual_hash != metadata.get("model_sha256"):
            raise RuntimeError("Artifact model SHA-256 does not match metadata; refusing to load it.")
    if _metadata_integrity_hash(metadata) != metadata.get("metadata_sha256"):
        raise RuntimeError("Artifact metadata SHA-256 does not match its contents; refusing to load it.")


def load_artifacts(artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the active model and metadata, with a clear missing-artifact error."""

    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to load model artifacts; install requirements.txt") from exc

    root = Path(artifacts_dir)
    model_path = root / "active_model.joblib"
    metadata_path = root / "active_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Active artifacts are missing in {root}. Run: python train_pipeline.py"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_METADATA_KEYS.difference(metadata.keys()))
    if missing:
        raise RuntimeError("Artifact metadata is missing required keys: " + ", ".join(missing))
    if metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported artifact schema_version={metadata.get('schema_version')}; "
            f"expected {ARTIFACT_SCHEMA_VERSION}. Run python train_pipeline.py."
        )
    if _metadata_integrity_hash(metadata) != metadata.get("metadata_sha256"):
        raise RuntimeError("Artifact metadata SHA-256 does not match its contents; refusing to load it.")
    if _sha256_file(model_path) != metadata.get("model_sha256"):
        raise RuntimeError("Artifact model SHA-256 does not match metadata; refusing to deserialize it.")
    payload = joblib.load(model_path)
    validate_artifact_contract(payload, metadata, model_path=model_path)
    return payload, metadata


def predict_dataframe(
    model_payload: Mapping[str, Any],
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Return calibrated scores and screen labels for validated rows."""

    selected = list(model_payload.get("features", SELECTED_FEATURES))
    if selected != SELECTED_FEATURES:
        raise ValueError("Model payload feature order does not match the official five-feature schema.")
    _require_columns(data, selected)
    model = model_payload["model"]
    threshold = float(model_payload.get("threshold", 0.5))
    scores = _probabilities(model, data[selected])
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Model returned calibrated scores outside the [0, 1] interval.")
    output = data.copy()
    output["calibrated_score"] = scores
    positive = scores >= threshold
    output["predicted_label"] = positive.astype(int)
    output["prediction_confidence"] = np.where(positive, scores, 1 - scores)
    output["screen_prediction"] = np.where(positive, "Positive", "Negative")
    return output


def _accepted_retraining_rows(artifacts_dir: str | Path) -> pd.DataFrame:
    """Read previously promoted five-feature rows without changing source V3 data."""

    path = Path(artifacts_dir) / "accepted_retraining_rows.csv"
    if not path.exists():
        return pd.DataFrame(columns=COMPACT_TRAINING_COLUMNS)
    rows = pd.read_csv(path, dtype={"PatientID": "string"})
    _require_columns(rows, [*SELECTED_FEATURES, TARGET_COLUMN])
    return rows.reindex(columns=COMPACT_TRAINING_COLUMNS)


def _feature_keys(data: pd.DataFrame, *, include_target: bool = False) -> list[tuple[float | None, ...]]:
    """Compare compact rows conservatively, including missing-value positions."""

    columns = [*SELECTED_FEATURES, *([TARGET_COLUMN] if include_target else [])]
    numeric = data[columns].apply(pd.to_numeric, errors="coerce")
    return [
        tuple(None if pd.isna(value) else float(value) for value in row)
        for row in numeric.itertuples(index=False, name=None)
    ]


def prepare_retraining_rows(
    new_data: pd.DataFrame,
    base_clean: pd.DataFrame,
    accepted: pd.DataFrame,
    locked_test: pd.DataFrame,
    imputation_statistics: Mapping[str, float] | None = None,
) -> tuple[ValidationResult, pd.DataFrame, dict[str, int]]:
    """Validate only the final five predictors and the verified clinical label."""

    _require_columns(new_data, [*SELECTED_FEATURES, TARGET_COLUMN])
    columns = [*(["PatientID"] if "PatientID" in new_data else []), *SELECTED_FEATURES, TARGET_COLUMN]
    validation = validate_dataframe(
        new_data[columns],
        reference_data=base_clean[SELECTED_FEATURES],
        mode="training",
        imputation_statistics=imputation_statistics,
    )
    valid = validation.valid.drop(columns=VALIDATION_META_COLUMNS, errors="ignore")
    if "PatientID" not in valid:
        valid.insert(0, "PatientID", pd.NA)
    valid = valid.reindex(columns=COMPACT_TRAINING_COLUMNS)
    if valid.empty:
        return validation, valid, {"existing_patient_id_rows_skipped": 0, "locked_feature_rows_skipped": 0, "previous_rows_skipped": 0, "corrected_rows": 0}

    # A new upload with the same five inputs as a locked-test patient is
    # conservatively held out, even if its optional ID is absent or changed.
    locked_keys = set(_feature_keys(locked_test))
    locked_mask = pd.Series(
        [key in locked_keys for key in _feature_keys(valid)], index=valid.index
    )
    locked_count = int(locked_mask.sum())
    valid = valid.loc[~locked_mask].copy()

    base_ids = set(_patient_id_keys(base_clean.get("PatientID", pd.Series(dtype="string"))).dropna())
    valid_ids = _patient_id_keys(valid["PatientID"])
    id_mask = valid_ids.isin(base_ids)
    id_count = int(id_mask.sum())
    valid = valid.loc[~id_mask].copy()

    # Existing accepted IDs are upserted when a clinician corrects the label.
    # Exact repeats are ignored; the earlier accepted version is backed up.
    accepted_by_id = {}
    if not accepted.empty:
        accepted_by_id = {
            patient_id: key
            for patient_id, key in zip(
                _patient_id_keys(accepted["PatientID"]),
                _feature_keys(accepted, include_target=True),
                strict=True,
            )
            if pd.notna(patient_id)
        }
    valid_ids = _patient_id_keys(valid["PatientID"])
    valid_keys = _feature_keys(valid, include_target=True)
    unchanged_id_mask = pd.Series(
        [bool(patient_id in accepted_by_id and accepted_by_id[patient_id] == key)
         for patient_id, key in zip(valid_ids, valid_keys, strict=True)],
        index=valid.index,
    )
    corrected_count = int(sum(
        bool(patient_id in accepted_by_id and accepted_by_id[patient_id] != key)
        for patient_id, key in zip(valid_ids, valid_keys, strict=True)
    ))

    # Without an ID, an exact repeat cannot be distinguished from a second
    # patient with the same values. Accept it only once across promotions.
    prior_anonymous = accepted.loc[
        _patient_id_keys(accepted["PatientID"]).isna()
    ] if not accepted.empty else accepted
    prior_keys = set(_feature_keys(prior_anonymous, include_target=True))
    anonymous = _patient_id_keys(valid["PatientID"]).isna()
    anonymous_repeat_mask = pd.Series(
        [bool(is_anon and key in prior_keys) for is_anon, key in zip(anonymous, valid_keys, strict=True)],
        index=valid.index,
    )
    prior_mask = unchanged_id_mask | anonymous_repeat_mask
    prior_count = int(prior_mask.sum())
    valid = valid.loc[~prior_mask].copy()
    return validation, valid, {
        "existing_patient_id_rows_skipped": id_count,
        "locked_feature_rows_skipped": locked_count,
        "previous_rows_skipped": prior_count,
        "corrected_rows": corrected_count,
    }


def retrain_with_new_data(
    new_data: pd.DataFrame,
    existing_data_path: str | Path = DEFAULT_DATA_PATH,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    locked_test_path: str | Path = DEFAULT_LOCKED_TEST_PATH,
    promotion_policy: Mapping[str, float] | None = None,
    approve_promotion: bool = True,
) -> dict[str, Any]:
    """Validate, evaluate, and safely promote a challenger model.

    The champion and challenger are always evaluated on the same locked rows.
    ``approve_promotion=False`` lets a UI show the policy result before a user
    confirms the final promotion.
    """

    required = [*SELECTED_FEATURES, TARGET_COLUMN]
    _require_columns(new_data, required)
    root = Path(artifacts_dir)
    with _artifact_lock(root):
        active_model_path = root / "active_model.joblib"
        active_metadata_path = root / "active_metadata.json"
        champion_payload: dict[str, Any] | None = None
        champion_metadata: dict[str, Any] | None = None
        if active_model_path.exists() or active_metadata_path.exists():
            if not (active_model_path.exists() and active_metadata_path.exists()):
                raise RuntimeError("Active model and metadata must exist as a pair before retraining.")
            champion_payload, champion_metadata = load_artifacts(root)

        base = _with_source_rows(read_dataset(existing_data_path))
        base_clean, _ = clean_dataset(base)
        locked_test, _ = _locked_test_frame(base, locked_test_path)
        accepted = _accepted_retraining_rows(root)
        base_imputation_statistics = fit_imputation_statistics(base_clean[SELECTED_FEATURES])
        validation, validated_new, skip_counts = prepare_retraining_rows(
            new_data, base_clean, accepted, locked_test, base_imputation_statistics,
        )
        if validated_new.empty:
            raise ValueError(
                "No new labelled five-feature rows remain after validation and duplicate/locked-test checks. "
                "Add a new case with Diagnosis 0 or 1; do not upload the original or locked-test dataset."
            )
        if accepted.empty:
            accepted_plus_new = validated_new.reset_index(drop=True)
        else:
            updated_ids = set(_patient_id_keys(validated_new["PatientID"]).dropna())
            retained = accepted.loc[~_patient_id_keys(accepted["PatientID"]).isin(updated_ids)]
            accepted_plus_new = pd.concat(
                [retained.reindex(columns=COMPACT_TRAINING_COLUMNS), validated_new],
                ignore_index=True,
            )
        compact_base = base_clean.reindex(columns=["_source_row", *COMPACT_TRAINING_COLUMNS])
        appended = accepted_plus_new.copy()
        appended.insert(0, "_source_row", range(len(base) + 1, len(base) + 1 + len(appended)))
        combined = pd.concat([compact_base, appended], ignore_index=True)

        backup_dir = root / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = backup_dir / f"dataset_{timestamp}.csv"
        _atomic_write_text(backup_path, base.drop(columns=["_source_row"], errors="ignore").to_csv(index=False, lineterminator="\n"))
        accepted_path = root / "accepted_retraining_rows.csv"
        accepted_backup_path = None
        if accepted_path.exists():
            accepted_backup_path = backup_dir / f"accepted_retraining_rows_{timestamp}.csv"
            shutil.copy2(accepted_path, accepted_backup_path)

        training_frame = combined
        result = run_training(
            training_frame,
            artifacts_dir=root,
            dataset_label=f"retrain_{timestamp}",
            promote=False,
            locked_test_path=locked_test_path,
            selection_mode="fixed_five",
            selection_reference=champion_metadata,
        )
        challenger_payload = {
            "model": result["model"],
            "features": result["metadata"]["selected_features"],
            "threshold": result["metadata"]["threshold"],
            "version": result["metadata"]["version"],
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "imputation_statistics": result["metadata"].get("imputation_statistics", {}),
        }
        challenger_evaluation = evaluate_payload(challenger_payload, locked_test)
        challenger_evaluation["version"] = result["metadata"]["version"]
        challenger_subgroups = subgroup_metrics(challenger_payload, locked_test, n_bootstrap=200)

        if champion_payload is None or champion_metadata is None:
            promotion_decision: dict[str, Any] = {
                "policy": {**PROMOTION_POLICY, **dict(promotion_policy or {})},
                "checks": [],
                "passed": True,
                "reason": "No existing champion; initial model may be promoted.",
                "champion_version": None,
                "challenger_version": result["metadata"]["version"],
            }
            champion_evaluation = None
            metric_differences: dict[str, Any] = {}
            subgroup_checks: list[dict[str, Any]] = []
            champion_subgroups = pd.DataFrame()
        else:
            champion_evaluation = evaluate_payload(champion_payload, locked_test)
            champion_evaluation["version"] = champion_metadata.get("version")
            champion_subgroups = subgroup_metrics(champion_payload, locked_test, n_bootstrap=200)
            subgroup_checks = compare_subgroup_metrics(
                champion_subgroups,
                challenger_subgroups,
                policy=promotion_policy,
            )
            old_rows = champion_evaluation["row_predictions"]
            new_rows = challenger_evaluation["row_predictions"]
            if not old_rows[[column for column in ["_source_row", "PatientID"] if column in old_rows]].equals(
                new_rows[[column for column in ["_source_row", "PatientID"] if column in new_rows]]
            ):
                raise RuntimeError("Champion and challenger evaluation rows are not identical.")
            actual = new_rows["y_true"].to_numpy()
            old_scores = old_rows["calibrated_score"].to_numpy()
            new_scores = new_rows["calibrated_score"].to_numpy()
            old_predictions = (old_scores >= champion_evaluation["threshold"]).astype(int)
            new_predictions = (new_scores >= challenger_evaluation["threshold"]).astype(int)
            s = _import_sklearn()
            metric_differences = {
                "pr_auc": paired_bootstrap_difference(actual, old_scores, new_scores, s["average_precision_score"]),
                "recall": paired_bootstrap_difference(actual, old_predictions, new_predictions, lambda y, scores: s["recall_score"](y, scores, zero_division=0)),
                "f2": paired_bootstrap_difference(actual, old_predictions, new_predictions, lambda y, scores: s["fbeta_score"](y, scores, beta=2, zero_division=0)),
                "brier": paired_bootstrap_difference(actual, old_scores, new_scores, s["brier_score_loss"]),
            }
            promotion_decision = evaluate_promotion_policy(
                champion_evaluation,
                challenger_evaluation,
                policy=promotion_policy,
                paired_differences=metric_differences,
                subgroup_checks=subgroup_checks,
            )

        policy_passed = bool(promotion_decision["passed"])
        promoted = policy_passed and bool(approve_promotion)
        promotion_decision["approved"] = bool(approve_promotion)
        promotion_decision["promoted"] = promoted
        if policy_passed and not approve_promotion:
            promotion_decision["reason"] = "Policy passed; manual approval is required before promotion."
        result["metadata"]["promotion_decision"] = promotion_decision
        result["metadata"]["metric_differences"] = metric_differences
        result["metadata"]["champion_subgroup_metrics"] = champion_subgroups.to_dict(orient="records")
        result["metadata"]["challenger_subgroup_metrics"] = challenger_subgroups.to_dict(orient="records")
        result["metadata"]["subgroup_promotion_checks"] = subgroup_checks
        result["metadata"]["champion_version"] = champion_metadata.get("version") if champion_metadata else None
        result["metadata"]["promotion_approved"] = bool(approve_promotion)
        result["metadata"]["metadata_sha256"] = _metadata_integrity_hash(result["metadata"])
        _atomic_write_text(
            result["paths"]["version_metadata"],
            json.dumps(_json_safe(result["metadata"]), indent=2, ensure_ascii=False) + "\n",
        )
        dataset_persisted = False
        if promoted:
            previous_accepted = accepted_path.read_bytes() if accepted_path.exists() else None
            try:
                _atomic_write_text(
                    accepted_path,
                    accepted_plus_new.to_csv(index=False, lineterminator="\n"),
                )
                _promote_artifacts(
                    result["paths"]["version_model"],
                    result["paths"]["version_metadata"],
                    root,
                    acquire_lock=False,
                )
            except Exception:
                if previous_accepted is None:
                    accepted_path.unlink(missing_ok=True)
                else:
                    restore = accepted_path.with_name(f".{accepted_path.name}.{uuid.uuid4().hex}.rollback")
                    restore.write_bytes(previous_accepted)
                    os.replace(restore, accepted_path)
                raise
            dataset_persisted = True
        result["retrain"] = {
            "validated_new_rows": int(len(validated_new)),
            "rejected_new_rows": int(len(validation.rejected)),
            **skip_counts,
            "new_data_cleaning": {"input_rows": len(new_data), "validated_rows": len(validated_new)},
            "backup_path": str(backup_path),
            "accepted_backup_path": str(accepted_backup_path) if accepted_backup_path else None,
            "old_pr_auc": champion_evaluation["metrics"]["pr_auc"] if champion_evaluation else None,
            "new_pr_auc": challenger_evaluation["metrics"]["pr_auc"],
            "promoted": promoted,
            "dataset_persisted": dataset_persisted,
            "dataset_path": str(accepted_path) if dataset_persisted else None,
            "policy_passed": policy_passed,
            "approval_required": policy_passed and not approve_promotion,
            "validation_summary": validation.summary,
            "champion_metrics": champion_evaluation["metrics"] if champion_evaluation else None,
            "challenger_metrics": challenger_evaluation["metrics"],
            "promotion_decision": promotion_decision,
            "metric_differences": metric_differences,
            "champion_subgroup_metrics": champion_subgroups.to_dict(orient="records"),
            "challenger_subgroup_metrics": challenger_subgroups.to_dict(orient="records"),
            "subgroup_promotion_checks": subgroup_checks,
            "locked_test_sha256": _dataframe_sha256(locked_test),
        }
        return result


__all__ = [
    "ALL_FEATURES",
    "ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_ARTIFACTS_DIR",
    "DEFAULT_DATA_PATH",
    "DEFAULT_LOCKED_TEST_PATH",
    "PROMOTION_POLICY",
    "SELECTED_FEATURES",
    "TARGET_COLUMN",
    "VALIDATION_META_COLUMNS",
    "ValidationResult",
    "clean_dataset",
    "audit_bp_rows",
    "calibration_intercept_slope",
    "calibration_summary",
    "choose_imbalance_method",
    "compare_subgroup_metrics",
    "evaluate_payload",
    "evaluate_promotion_policy",
    "exploratory_summary",
    "fit_imputation_statistics",
    "load_artifacts",
    "predict_dataframe",
    "paired_bootstrap_difference",
    "read_dataset",
    "retrain_with_new_data",
    "run_training",
    "run_cleaning_sensitivity",
    "save_eda_plots",
    "monitor_prediction_batch",
    "subgroup_metrics",
    "summarize_bp_audit",
    "summarize_prediction_batch",
    "threshold_table",
    "validate_artifact_contract",
    "validate_dataframe",
]
