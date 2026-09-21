"""Create reproducible cleaning and evaluation sensitivity tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ml_pipeline as pipeline


def _swap_if_plausible(source: pd.DataFrame) -> pd.DataFrame:
    candidate = pipeline._with_source_rows(source)
    systolic = pd.to_numeric(candidate["SystolicBP"], errors="coerce")
    diastolic = pd.to_numeric(candidate["DiastolicBP"], errors="coerce")
    invalid = systolic.notna() & diastolic.notna() & (systolic <= diastolic)
    plausible = invalid & diastolic.between(*pipeline.RANGE_RULES["SystolicBP"]) & systolic.between(*pipeline.RANGE_RULES["DiastolicBP"])
    candidate.loc[plausible, "SystolicBP"], candidate.loc[plausible, "DiastolicBP"] = (
        diastolic.loc[plausible], systolic.loc[plausible]
    )
    return candidate


def _fit_sensitivity_model(candidate: pd.DataFrame, locked: pd.DataFrame) -> dict[str, float]:
    """Fit one lightweight, identically split model for strategy comparison."""

    s = pipeline._import_sklearn()
    locked_ids = set(locked["_source_row"].tolist())
    train = candidate.loc[~candidate["_source_row"].isin(locked_ids)].copy()
    train_x, validation_x, train_y, validation_y = s["train_test_split"](
        train[pipeline.SELECTED_FEATURES], train[pipeline.TARGET_COLUMN].astype(int),
        test_size=0.25, stratify=train[pipeline.TARGET_COLUMN].astype(int), random_state=pipeline.RANDOM_STATE,
    )
    model = s["CalibratedClassifierCV"](
        estimator=pipeline._build_imbalance_estimator("scale_pos_weight", train_x, train_y),
        method="sigmoid", cv=3, n_jobs=1,
    )
    model.fit(train_x, train_y)
    validation_scores = pipeline._probabilities(model, validation_x)
    threshold, _ = pipeline.choose_f2_threshold(validation_y, validation_scores)
    locked_scores = pipeline._probabilities(model, locked[pipeline.SELECTED_FEATURES])
    return pipeline._metric_values(locked[pipeline.TARGET_COLUMN].astype(int), locked_scores, threshold)


def main() -> None:
    source = pipeline._with_source_rows(pipeline.read_dataset(pipeline.DEFAULT_DATA_PATH))
    audit = pipeline.audit_bp_rows(source)
    output_dir = pipeline.DEFAULT_ARTIFACTS_DIR / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline._atomic_write_text(
        output_dir / "bp_removed_profile.csv",
        pipeline.summarize_bp_audit(audit).to_csv(index=False, lineterminator="\n"),
    )

    sensitivity = pipeline.run_cleaning_sensitivity(source)
    cleaned, _ = pipeline.clean_dataset(source)
    s = pipeline._import_sklearn()
    _, locked = s["train_test_split"](
        cleaned, test_size=0.20, stratify=cleaned[pipeline.TARGET_COLUMN], random_state=pipeline.RANDOM_STATE
    )
    strategies = {
        "remove": cleaned,
        "swap-if-plausible": pipeline.clean_dataset(_swap_if_plausible(source))[0],
        "keep-with-flag": source.assign(
            bp_order_invalid=audit["bp_order_invalid"].astype(bool).to_numpy()
        ),
    }
    metric_rows: list[dict[str, object]] = []
    for strategy, candidate in strategies.items():
        # The fixed cohort is drawn from rows that survive the canonical remove rule,
        # so all three strategies are evaluated on identical row identities.
        evaluation = _fit_sensitivity_model(candidate, locked)
        base = sensitivity.loc[sensitivity["strategy"] == strategy].iloc[0].to_dict()
        metric_rows.append({**base, **{f"{name}_locked": value for name, value in evaluation.items()}})
    pipeline._atomic_write_text(
        output_dir / "cleaning_sensitivity.csv",
        pd.DataFrame(metric_rows).to_csv(index=False, lineterminator="\n"),
    )
    manifest = {
        "source": pipeline.DEFAULT_DATA_PATH.name,
        "raw_rows": len(source),
        "canonical_clean_rows": len(cleaned),
        "sensitivity_locked_rows": len(locked),
        "sensitivity_locked_source_rows_sha256": pipeline._dataframe_sha256(locked[["_source_row"]]),
        "outputs": ["bp_removed_profile.csv", "cleaning_sensitivity.csv"],
        "disclaimer": "Sensitivity metrics are internal, same-cohort research diagnostics; no external validation was performed.",
    }
    pipeline._atomic_write_text(output_dir / "audit_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
