"""Persistent, auditable prediction feedback and retraining-run storage."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from auth import state_database_path
from ml_pipeline import ALL_FEATURES, SELECTED_FEATURES


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _open_connection(database_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(database_path) if database_path is not None else state_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_cases (
            case_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            created_by TEXT NOT NULL,
            source TEXT NOT NULL,
            model_version TEXT NOT NULL,
            threshold REAL NOT NULL,
            calibrated_score REAL NOT NULL,
            predicted_label INTEGER NOT NULL CHECK (predicted_label IN (0, 1)),
            feature_payload_json TEXT NOT NULL,
            full_v3_schema INTEGER NOT NULL CHECK (full_v3_schema IN (0, 1))
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS clinical_feedback (
            feedback_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            submitted_at_utc TEXT NOT NULL,
            submitted_by TEXT NOT NULL,
            diagnosis INTEGER NOT NULL CHECK (diagnosis IN (0, 1)),
            verified INTEGER NOT NULL CHECK (verified IN (0, 1)),
            revision INTEGER NOT NULL,
            FOREIGN KEY (case_id) REFERENCES prediction_cases(case_id),
            UNIQUE (case_id, revision)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS retraining_runs (
            run_id TEXT PRIMARY KEY,
            started_at_utc TEXT NOT NULL,
            started_by TEXT NOT NULL,
            source TEXT NOT NULL,
            source_rows INTEGER NOT NULL,
            status TEXT NOT NULL,
            policy_passed INTEGER,
            promoted INTEGER NOT NULL CHECK (promoted IN (0, 1)),
            champion_version TEXT,
            challenger_version TEXT,
            old_pr_auc REAL,
            new_pr_auc REAL,
            details_json TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


@contextmanager
def _connect(database_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield one initialized connection and always release its Windows file lock."""

    connection = _open_connection(database_path)
    try:
        yield connection
    finally:
        connection.close()


def _number_or_none(value: Any) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    number = float(value)
    return int(number) if number.is_integer() else number


def _case_id(
    *,
    actor: str,
    model_version: str,
    batch_key: str | None,
    row_key: str,
) -> str:
    if batch_key:
        seed = f"alzheimer-feedback:{actor}:{model_version}:{batch_key}:{row_key}"
        token = uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:12]
    else:
        token = uuid.uuid4().hex[:12]
    return "CASE-" + token.upper()


def record_prediction_cases(
    frame: pd.DataFrame,
    *,
    actor: str,
    source: str,
    model_version: str,
    threshold: float,
    batch_key: str | None = None,
    database_path: str | Path | None = None,
) -> list[str]:
    """Store prediction inputs without PatientID and return stable case IDs."""

    if "calibrated_score" not in frame.columns:
        raise ValueError("Prediction frame is missing calibrated_score.")
    timestamp = _utc_now()
    case_ids: list[str] = []
    with _connect(database_path) as connection:
        for position, (_, row) in enumerate(frame.iterrows(), start=1):
            row_key = str(row.get("source_row", row.get("_row_number", position)))
            case_id = _case_id(
                actor=actor,
                model_version=model_version,
                batch_key=batch_key,
                row_key=row_key,
            )
            features = {
                feature: _number_or_none(row.get(feature))
                for feature in ALL_FEATURES
                if feature in frame.columns
            }
            full_v3 = int(
                len(features) == len(ALL_FEATURES)
                and all(value is not None for value in features.values())
            )
            score = float(row["calibrated_score"])
            predicted_label = int(row.get("predicted_label", score >= threshold))
            connection.execute(
                """
                INSERT OR IGNORE INTO prediction_cases (
                    case_id, created_at_utc, created_by, source, model_version,
                    threshold, calibrated_score, predicted_label,
                    feature_payload_json, full_v3_schema
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    timestamp,
                    actor,
                    source,
                    model_version,
                    float(threshold),
                    score,
                    predicted_label,
                    json.dumps(features, ensure_ascii=False, sort_keys=True),
                    full_v3,
                ),
            )
            case_ids.append(case_id)
        connection.commit()
    return case_ids


def submit_feedback(
    case_id: str,
    diagnosis: int,
    *,
    submitted_by: str,
    verified: bool,
    allowed_owner: str | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Append a feedback revision; previous labels remain available for audit."""

    normalized_case_id = case_id.strip().upper()
    if diagnosis not in (0, 1):
        raise ValueError("Diagnosis must be 0 or 1.")
    with _connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        case = connection.execute(
            "SELECT case_id, created_by, predicted_label FROM prediction_cases WHERE case_id = ?",
            (normalized_case_id,),
        ).fetchone()
        if case is None:
            raise ValueError("Case ID was not found.")
        if allowed_owner is not None and str(case["created_by"]) != allowed_owner:
            raise PermissionError("This case belongs to a different user.")
        row = connection.execute(
            "SELECT COALESCE(MAX(revision), 0) + 1 AS revision FROM clinical_feedback WHERE case_id = ?",
            (normalized_case_id,),
        ).fetchone()
        revision = int(row["revision"])
        feedback_id = "FDBK-" + uuid.uuid4().hex[:12].upper()
        timestamp = _utc_now()
        connection.execute(
            """
            INSERT INTO clinical_feedback (
                feedback_id, case_id, submitted_at_utc, submitted_by,
                diagnosis, verified, revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                normalized_case_id,
                timestamp,
                submitted_by,
                int(diagnosis),
                int(verified),
                revision,
            ),
        )
        connection.commit()
    return {
        "feedback_id": feedback_id,
        "case_id": normalized_case_id,
        "diagnosis": int(diagnosis),
        "verified": bool(verified),
        "verified_model_mismatch": bool(verified and int(case["predicted_label"]) != diagnosis),
        "revision": revision,
        "submitted_at_utc": timestamp,
    }


def list_cases(
    *,
    actor: str | None = None,
    database_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return case summaries; viewers can be restricted to their own cases."""

    where_clause = "WHERE p.created_by = ?" if actor else ""
    parameters: tuple[str, ...] = (actor,) if actor else ()
    query = f"""
        SELECT
            p.case_id,
            p.created_at_utc,
            p.created_by,
            p.source,
            p.model_version,
            p.calibrated_score,
            p.predicted_label,
            p.full_v3_schema,
            f.diagnosis AS latest_diagnosis,
            f.verified AS latest_verified,
            f.revision AS feedback_revision,
            f.submitted_by AS feedback_by,
            f.submitted_at_utc AS feedback_at_utc,
            vf.diagnosis AS verified_diagnosis,
            vf.revision AS verified_revision,
            vf.submitted_by AS verified_by,
            vf.submitted_at_utc AS verified_at_utc
        FROM prediction_cases AS p
        LEFT JOIN clinical_feedback AS f
          ON f.case_id = p.case_id
         AND f.revision = (
             SELECT MAX(f2.revision)
             FROM clinical_feedback AS f2
             WHERE f2.case_id = p.case_id
         )
        LEFT JOIN clinical_feedback AS vf
          ON vf.case_id = p.case_id
         AND vf.verified = 1
         AND vf.revision = (
             SELECT MAX(vf2.revision)
             FROM clinical_feedback AS vf2
             WHERE vf2.case_id = p.case_id AND vf2.verified = 1
         )
        {where_clause}
        ORDER BY p.created_at_utc DESC
    """
    with _connect(database_path) as connection:
        rows = connection.execute(query, parameters).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def feedback_training_frame(
    database_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build five-feature rows from the latest verified label for each case."""

    query = """
        SELECT p.case_id, p.feature_payload_json, f.diagnosis
        FROM prediction_cases AS p
        JOIN clinical_feedback AS f
          ON f.case_id = p.case_id
         AND f.verified = 1
         AND f.revision = (
             SELECT MAX(f2.revision)
             FROM clinical_feedback AS f2
             WHERE f2.case_id = p.case_id AND f2.verified = 1
         )
        ORDER BY p.created_at_utc
    """
    with _connect(database_path) as connection:
        rows = connection.execute(query).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        features = json.loads(str(row["feature_payload_json"]))
        records.append(
            {
                "PatientID": "FB-" + str(row["case_id"]),
                **{feature: features.get(feature) for feature in SELECTED_FEATURES},
                "Diagnosis": int(row["diagnosis"]),
            }
        )
    return pd.DataFrame(records)


def record_retraining_run(
    *,
    started_by: str,
    source: str,
    source_rows: int,
    status: str,
    promoted: bool,
    policy_passed: bool | None = None,
    champion_version: str | None = None,
    challenger_version: str | None = None,
    old_pr_auc: float | None = None,
    new_pr_auc: float | None = None,
    details: dict[str, Any] | None = None,
    database_path: str | Path | None = None,
) -> str:
    """Append one retraining session to the operational audit log."""

    run_id = "RUN-" + uuid.uuid4().hex[:12].upper()
    with _connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO retraining_runs (
                run_id, started_at_utc, started_by, source, source_rows, status,
                policy_passed, promoted, champion_version, challenger_version,
                old_pr_auc, new_pr_auc, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                _utc_now(),
                started_by,
                source,
                int(source_rows),
                status,
                None if policy_passed is None else int(policy_passed),
                int(promoted),
                champion_version,
                challenger_version,
                old_pr_auc,
                new_pr_auc,
                json.dumps(details or {}, ensure_ascii=False, default=str, sort_keys=True),
            ),
        )
        connection.commit()
    return run_id


def list_retraining_runs(
    *,
    limit: int = 50,
    database_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return recent retraining sessions without the verbose JSON payload."""

    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT run_id, started_at_utc, started_by, source, source_rows, status,
                   policy_passed, promoted, champion_version, challenger_version,
                   old_pr_auc, new_pr_auc
            FROM retraining_runs
            ORDER BY started_at_utc DESC
            LIMIT ?
            """,
            (max(1, min(int(limit), 500)),),
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def get_retraining_run(
    run_id: str,
    *,
    database_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Load one saved run, including its policy checks for later review."""

    with _connect(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM retraining_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["details"] = json.loads(result.pop("details_json"))
    return result


def case_counts(database_path: str | Path | None = None) -> dict[str, int]:
    """Return compact feedback readiness counts."""

    with _connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_cases,
                SUM(CASE WHEN full_v3_schema = 1 THEN 1 ELSE 0 END) AS full_v3_cases,
                SUM(CASE WHEN EXISTS (
                    SELECT 1 FROM clinical_feedback AS f
                    WHERE f.case_id = p.case_id AND f.verified = 1
                ) THEN 1 ELSE 0 END) AS verified_cases,
                SUM(CASE WHEN EXISTS (
                    SELECT 1 FROM clinical_feedback AS f
                    WHERE f.case_id = p.case_id AND f.verified = 1
                ) THEN 1 ELSE 0 END) AS retrain_ready_cases
            FROM prediction_cases AS p
            """
        ).fetchone()
    return {key: int(row[key] or 0) for key in row.keys()} if row else {}


def attach_case_ids(frame: pd.DataFrame, case_ids: Iterable[str]) -> pd.DataFrame:
    """Return a copy with case IDs inserted as the first column."""

    output = frame.copy()
    output.insert(0, "case_id", list(case_ids))
    return output


__all__ = [
    "attach_case_ids",
    "case_counts",
    "feedback_training_frame",
    "get_retraining_run",
    "list_cases",
    "list_retraining_runs",
    "record_prediction_cases",
    "record_retraining_run",
    "submit_feedback",
]
