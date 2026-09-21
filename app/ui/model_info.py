"""Pure model-card data helpers kept outside the Streamlit entry point."""

from __future__ import annotations

import pandas as pd


def retraining_metric_comparison(checks: list[dict]) -> pd.DataFrame:
    """Prepare the four overall same-cohort checks for a signed comparison."""

    metrics = (
        ("pr_auc", "PR-AUC", True),
        ("recall", "Recall", True),
        ("f2", "F2", True),
        ("brier", "Brier", False),
    )
    by_metric = {check.get("metric"): check for check in checks}
    rows = []
    for key, label, higher_is_better in metrics:
        check = by_metric.get(key)
        if not check:
            continue
        old = pd.to_numeric(check.get("old"), errors="coerce")
        new = pd.to_numeric(check.get("new"), errors="coerce")
        if pd.isna(old) or pd.isna(new):
            continue
        delta = float(new - old)
        direction = "same" if abs(delta) < 1e-12 else ("up" if delta > 0 else "down")
        outcome = (
            "same" if direction == "same"
            else "better" if (delta > 0) == higher_is_better
            else "worse"
        )
        rows.append(
            {
                "metric": key,
                "label": label,
                "old": float(old),
                "new": float(new),
                "delta": delta,
                "direction": direction,
                "outcome": outcome,
                "policy_passed": bool(check.get("passed")),
            }
        )
    return pd.DataFrame(rows)


def model_influence_table(metadata: dict) -> pd.DataFrame:
    """Combine permutation dependence with the report's univariate effect size."""

    ranking = pd.DataFrame(metadata.get("feature_ranking", []))
    if ranking.empty or "feature" not in ranking.columns:
        return pd.DataFrame()
    columns = ["feature", "importance_mean", "importance_std"]
    ranking = ranking[[column for column in columns if column in ranking.columns]].copy()
    eda_effects = pd.DataFrame(metadata.get("eda", {}).get("cohens_d_top15", []))
    if not eda_effects.empty and {"feature", "cohens_d"}.issubset(eda_effects.columns):
        ranking = ranking.merge(eda_effects[["feature", "cohens_d"]], on="feature", how="left")
    ranking = ranking.rename(
        columns={
            "feature": "Feature",
            "importance_mean": "Permutation importance mean",
            "importance_std": "Permutation importance std",
            "cohens_d": "Cohen's d",
        }
    )
    for column in ["Permutation importance mean", "Permutation importance std", "Cohen's d"]:
        if column in ranking.columns:
            ranking[column] = pd.to_numeric(ranking[column], errors="coerce").round(4)
    return ranking


def format_model_value(value, kind: str = "score") -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if kind == "percent":
        return f"{numeric * 100:.1f}%"
    return f"{numeric:.3f}"


def model_metric_table(metadata: dict) -> pd.DataFrame:
    metric_labels = {
        "accuracy": "Accuracy",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "f1": "F1",
        "f2": "F2",
        "precision": "Precision",
        "recall": "Recall",
        "brier": "Brier score",
    }
    validation = metadata.get("validation_metrics", {})
    test = metadata.get("test_metrics", {})
    metric_order = [key for key in metric_labels if key in validation or key in test]
    rows = []
    for key in metric_order:
        row = {"Metric": metric_labels[key]}
        for column, values in [("Validation", validation), ("Test", test)]:
            value = values.get(key)
            row[column] = "—" if value is None else f"{float(value):.3f}"
        rows.append(row)
    return pd.DataFrame(rows, columns=["Metric", "Validation", "Test"])


def feature_selection_summary_table(metadata: dict, selected_count: int) -> pd.DataFrame:
    """Summarize the stored feature-count experiment."""

    selection = pd.DataFrame(metadata.get("feature_selection", []))
    required = {"feature_count", "accuracy_cv", "pr_auc_cv"}
    if selection.empty or not required.issubset(selection.columns):
        return pd.DataFrame()
    view = selection[["feature_count", "accuracy_cv", "pr_auc_cv"]].copy()
    full_row = view.loc[view["feature_count"].eq(view["feature_count"].max())]
    if not full_row.empty:
        full_accuracy = pd.to_numeric(full_row.iloc[0]["accuracy_cv"], errors="coerce")
        full_pr_auc = pd.to_numeric(full_row.iloc[0]["pr_auc_cv"], errors="coerce")
        view["Δ Accuracy vs full"] = pd.to_numeric(view["accuracy_cv"], errors="coerce") - full_accuracy
        view["Δ PR-AUC vs full"] = pd.to_numeric(view["pr_auc_cv"], errors="coerce") - full_pr_auc
    else:
        view["Δ Accuracy vs full"] = pd.NA
        view["Δ PR-AUC vs full"] = pd.NA
    view["Selected"] = view["feature_count"].eq(selected_count).map({True: "✓", False: ""})
    view = view.rename(
        columns={
            "feature_count": "Feature count",
            "accuracy_cv": "Accuracy CV",
            "pr_auc_cv": "PR-AUC CV",
        }
    )
    for column in ["Accuracy CV", "PR-AUC CV", "Δ Accuracy vs full", "Δ PR-AUC vs full"]:
        view[column] = pd.to_numeric(view[column], errors="coerce").round(3)
    return view[
        [
            "Selected",
            "Feature count",
            "Accuracy CV",
            "PR-AUC CV",
            "Δ Accuracy vs full",
            "Δ PR-AUC vs full",
        ]
    ].sort_values("Feature count")


def feature_selection_stage_matrix(metadata: dict, selected_count: int) -> pd.DataFrame:
    """Show exactly which stored columns survive at each feature-count stage."""

    stages: dict[int, list[str]] = {}
    for stage in metadata.get("feature_selection", []):
        try:
            feature_count = int(stage.get("feature_count"))
        except (TypeError, ValueError):
            continue
        features = stage.get("features")
        if feature_count < selected_count or not isinstance(features, list):
            continue
        normalized = [str(feature) for feature in features if str(feature).strip()]
        if normalized:
            stages[feature_count] = normalized

    if not stages or selected_count not in stages:
        return pd.DataFrame()

    stage_counts = sorted(stages, reverse=True)
    ranked_features = stages[stage_counts[0]]
    rows = []
    for rank, feature in enumerate(ranked_features, start=1):
        row: dict[str, object] = {"Rank": rank, "Column": feature}
        for feature_count in stage_counts:
            row[f"Stage {feature_count}"] = feature in stages[feature_count]
        rows.append(row)
    return pd.DataFrame(rows)


def imbalance_summary_table(metadata: dict) -> pd.DataFrame:
    """Build a compact comparison of the stored imbalance strategies."""

    comparison = pd.DataFrame(metadata.get("imbalance_comparison", []))
    required = {"method", "accuracy", "pr_auc", "recall", "f2", "brier"}
    if comparison.empty or not required.issubset(comparison.columns):
        return pd.DataFrame()
    selected_method = metadata.get("selected_imbalance_method")
    columns = ["method", "accuracy", "pr_auc", "recall", "f2", "brier"]
    view = comparison[columns].copy()
    view["Selected"] = view["method"].eq(selected_method).map({True: "✓", False: ""})
    view = view.rename(
        columns={
            "method": "Method",
            "accuracy": "Accuracy",
            "pr_auc": "PR-AUC",
            "recall": "Recall",
            "f2": "F2",
            "brier": "Brier",
        }
    )
    for column in ["Accuracy", "PR-AUC", "Recall", "F2", "Brier"]:
        view[column] = pd.to_numeric(view[column], errors="coerce").round(3)
    return view[["Selected", "Method", "Accuracy", "PR-AUC", "Recall", "F2", "Brier"]]


def benchmark_summary_table(metadata: dict) -> pd.DataFrame:
    """Build a ranked candidate-model table from the active artifact."""

    benchmark = pd.DataFrame(metadata.get("benchmark", []))
    required = {"model", "pr_auc_cv", "roc_auc_cv", "f1", "recall"}
    if benchmark.empty or not required.issubset(benchmark.columns):
        return pd.DataFrame()
    view = benchmark[["model", "pr_auc_cv", "roc_auc_cv", "f1", "recall"]].copy()
    view["Selected"] = view["model"].eq("XGBoost").map({True: "✓", False: ""})
    view = view.rename(
        columns={
            "model": "Model",
            "pr_auc_cv": "PR-AUC CV",
            "roc_auc_cv": "ROC-AUC CV",
            "f1": "F1",
            "recall": "Recall",
        }
    )
    for column in ["PR-AUC CV", "ROC-AUC CV", "F1", "Recall"]:
        view[column] = pd.to_numeric(view[column], errors="coerce").round(3)
    return view[["Selected", "Model", "PR-AUC CV", "ROC-AUC CV", "F1", "Recall"]].sort_values(
        "PR-AUC CV",
        ascending=False,
    )


def threshold_scenario_table(metadata: dict, selected_threshold: float) -> pd.DataFrame:
    """Compare recall-oriented, requested, selected, and default thresholds."""

    threshold_selection = metadata.get("threshold_selection", {})
    sweep = pd.DataFrame(threshold_selection.get("table", []))
    required = {
        "threshold",
        "precision",
        "recall",
        "specificity",
        "f2",
        "false_negatives",
        "false_positives",
    }
    if sweep.empty or not required.issubset(sweep.columns):
        return pd.DataFrame()
    for column in required:
        sweep[column] = pd.to_numeric(sweep[column], errors="coerce")
    sweep = sweep.dropna(subset=["threshold"])

    def nearest_sweep_row(threshold: float) -> dict:
        index = (sweep["threshold"] - threshold).abs().idxmin()
        return sweep.loc[index].to_dict()

    minimum_false_negatives = sweep["false_negatives"].min()
    recall_candidates = sweep[sweep["false_negatives"].eq(minimum_false_negatives)]
    recall_row = recall_candidates.sort_values(["false_positives", "threshold"]).iloc[0].to_dict()
    selected_metrics = metadata.get("validation_metrics", {})
    selected_row = {
        "threshold": selected_threshold,
        **{column: selected_metrics.get(column) for column in required if column != "threshold"},
    }
    scenarios = [
        ("Recall-priority candidate", recall_row),
        ("Requested ≈0.20 example", nearest_sweep_row(0.20)),
        ("Selected by validation F2", selected_row),
        ("Default 0.50", nearest_sweep_row(0.50)),
    ]
    rows = []
    for scenario, values in scenarios:
        rows.append(
            {
                "Scenario": scenario,
                "Threshold": values.get("threshold"),
                "Precision": values.get("precision"),
                "Recall": values.get("recall"),
                "Specificity": values.get("specificity"),
                "F2": values.get("f2"),
                "False negatives": values.get("false_negatives"),
                "False positives": values.get("false_positives"),
            }
        )
    view = pd.DataFrame(rows)
    for column in ["Threshold", "Precision", "Recall", "Specificity", "F2"]:
        view[column] = pd.to_numeric(view[column], errors="coerce").round(3)
    for column in ["False negatives", "False positives"]:
        view[column] = pd.to_numeric(view[column], errors="coerce").astype("Int64")
    return view


def selected_feature_evidence_table(metadata: dict, selected_features: list[str]) -> pd.DataFrame:
    """Build evidence for the exact columns used by the final model."""

    roles = {
        "MMSE": "Cognitive assessment",
        "FunctionalAssessment": "Functional assessment",
        "ADL": "Daily-living function",
        "MemoryComplaints": "Memory symptom flag",
        "BehavioralProblems": "Behaviour symptom flag",
    }
    evidence = pd.DataFrame({"feature": selected_features})
    ranking = pd.DataFrame(metadata.get("feature_ranking", []))
    if not ranking.empty and "feature" in ranking.columns:
        ranking = ranking[[column for column in ["feature", "importance_mean", "importance_std"] if column in ranking.columns]]
        evidence = evidence.merge(ranking, on="feature", how="left")
    effects = pd.DataFrame(metadata.get("eda", {}).get("cohens_d_top15", []))
    if not effects.empty and {"feature", "cohens_d"}.issubset(effects.columns):
        evidence = evidence.merge(effects[["feature", "cohens_d"]], on="feature", how="left")
    means = metadata.get("eda", {}).get("selected_feature_means_by_diagnosis", {})
    evidence["role"] = evidence["feature"].map(roles).fillna("Final V3 model feature")
    evidence["Diagnosis=0 mean"] = evidence["feature"].map(lambda feature: means.get(feature, {}).get("0"))
    evidence["Diagnosis=1 mean"] = evidence["feature"].map(lambda feature: means.get(feature, {}).get("1"))
    evidence = evidence.rename(
        columns={
            "feature": "Column",
            "role": "Role",
            "importance_mean": "Permutation importance",
            "importance_std": "Importance std",
            "cohens_d": "Cohen's d",
        }
    )
    ordered = [
        "Column",
        "Role",
        "Permutation importance",
        "Importance std",
        "Cohen's d",
        "Diagnosis=0 mean",
        "Diagnosis=1 mean",
    ]
    for column in ordered:
        if column not in evidence.columns:
            evidence[column] = pd.NA
    for column in ordered[2:]:
        evidence[column] = pd.to_numeric(evidence[column], errors="coerce").round(4)
    return evidence[ordered]
