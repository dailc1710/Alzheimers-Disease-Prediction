"""Synchronize the V3 Word report with the active model artifact.

The source report is never overwritten. Numeric tables, key narrative claims,
and model-evidence figures are refreshed in a new DOCX while preserving the
original document structure and formatting as closely as possible.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from sklearn.metrics import auc, roc_curve

plt.switch_backend("Agg")


FIGURE_SPECS = {
    "word/media/image8.png": (1200, 900),
    "word/media/image9.png": (1200, 750),
    "word/media/image10.png": (1650, 750),
    "word/media/image11.png": (1800, 900),
    "word/media/image12.png": (900, 825),
    "word/media/image13.png": (750, 675),
    "word/media/image14.png": (825, 750),
    "word/media/image15.png": (1050, 600),
}


def _vn(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


def _set_paragraph_text(paragraph: Any, text: str) -> None:
    """Replace paragraph text while keeping the first run's formatting."""

    if paragraph.runs:
        paragraph.runs[0].text = text
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.add_run(text)


def _set_cell_text(cell: Any, text: str) -> None:
    paragraph = cell.paragraphs[0]
    _set_paragraph_text(paragraph, text)
    for extra_paragraph in cell.paragraphs[1:]:
        _set_paragraph_text(extra_paragraph, "")


def _table_row_by_prefix(table: Any, prefix: str) -> Any:
    for row in table.rows[1:]:
        if row.cells[0].text.strip().startswith(prefix):
            return row
    raise KeyError(f"Table row not found: {prefix}")


def _figure(width_px: int, height_px: int) -> tuple[Any, Any]:
    dpi = 150
    return plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)


def _save_figure(figure: Any, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)


def _plot_feature_ranking(metadata: dict[str, Any], path: Path) -> None:
    ranking = pd.DataFrame(metadata["feature_ranking"]).sort_values("importance_mean")
    figure, axis = _figure(*FIGURE_SPECS["word/media/image8.png"])
    axis.barh(
        ranking["feature"],
        ranking["importance_mean"] * 100,
        xerr=ranking["importance_std"] * 100,
        color="#6C9BC3",
        alpha=0.9,
    )
    axis.axvline(0, color="#475569", linewidth=0.8)
    axis.set_title("Validation permutation importance — all 32 predictors")
    axis.set_xlabel("Mean accuracy decrease (percentage points)")
    axis.tick_params(axis="y", labelsize=6.5)
    axis.grid(axis="x", color="#D9E0E8", linewidth=0.6, alpha=0.8)
    _save_figure(figure, path)


def _plot_feature_counts(metadata: dict[str, Any], path: Path) -> None:
    frame = pd.DataFrame(metadata["feature_selection"]).sort_values("feature_count")
    figure, axis = _figure(*FIGURE_SPECS["word/media/image9.png"])
    axis.plot(
        frame["feature_count"],
        frame["accuracy_cv"],
        marker="o",
        linewidth=2.2,
        color="#6C9BC3",
        label="Accuracy CV",
    )
    axis.plot(
        frame["feature_count"],
        frame["pr_auc_cv"],
        marker="s",
        linewidth=2.2,
        color="#2F7F73",
        label="PR-AUC CV",
    )
    selected = frame.loc[frame["feature_count"].eq(5)].iloc[0]
    axis.scatter([5, 5], [selected["accuracy_cv"], selected["pr_auc_cv"]], color="#F2A900", s=70, zorder=4)
    axis.axvline(5, color="#475569", linestyle="--", linewidth=1.0, label="Selected: 5 features")
    axis.set_title("Repeated-CV performance by retained feature count")
    axis.set_xlabel("Number of retained features")
    axis.set_ylabel("Score")
    axis.set_ylim(0.75, 0.97)
    axis.set_xticks(frame["feature_count"])
    axis.grid(color="#D9E0E8", linewidth=0.6, alpha=0.8)
    axis.legend(loc="lower right")
    _save_figure(figure, path)


def _plot_imbalance(metadata: dict[str, Any], path: Path) -> None:
    frame = pd.DataFrame(metadata["imbalance_comparison"])
    width_px, height_px = FIGURE_SPECS["word/media/image10.png"]
    dpi = 150
    figure, axes = plt.subplots(1, 2, figsize=(width_px / dpi, height_px / dpi), dpi=dpi, gridspec_kw={"width_ratios": [3.2, 1]})
    metrics = ["accuracy", "pr_auc", "f1", "recall"]
    labels = ["Accuracy", "PR-AUC", "F1", "Recall"]
    positions = np.arange(len(frame))
    bar_width = 0.18
    colors = ["#6C9BC3", "#2F7F73", "#F2A900", "#9B8AFB"]
    for index, (metric, label, color) in enumerate(zip(metrics, labels, colors, strict=True)):
        axes[0].bar(positions + (index - 1.5) * bar_width, frame[metric], bar_width, label=label, color=color)
    axes[0].set_title("Class-imbalance strategy comparison")
    axes[0].set_xticks(positions, frame["method"])
    axes[0].set_ylim(0.88, 0.97)
    axes[0].set_ylabel("Repeated-CV score")
    axes[0].grid(axis="y", color="#D9E0E8", linewidth=0.6)
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].bar(frame["method"], frame["brier"], color=["#6C9BC3", "#2F7F73", "#D28C63"])
    axes[1].set_title("Brier (lower is better)")
    axes[1].set_ylim(0, max(frame["brier"]) * 1.25)
    axes[1].tick_params(axis="x", rotation=25, labelsize=8)
    axes[1].grid(axis="y", color="#D9E0E8", linewidth=0.6)
    _save_figure(figure, path)


def _plot_benchmark(metadata: dict[str, Any], path: Path) -> None:
    frame = pd.DataFrame(metadata["benchmark"])
    short_names = {
        "DummyClassifier": "Dummy",
        "Logistic Regression": "Logistic",
        "Gradient Boosting": "GradBoost",
        "Random Forest": "RandomForest",
        "SVM (RBF)": "SVM",
    }
    frame["label"] = frame["model"].map(short_names).fillna(frame["model"])
    figure, axis = _figure(*FIGURE_SPECS["word/media/image11.png"])
    metrics = ["pr_auc_cv", "roc_auc_cv", "f1", "recall"]
    labels = ["PR-AUC CV", "ROC-AUC CV", "F1", "Recall"]
    positions = np.arange(len(frame))
    bar_width = 0.19
    colors = ["#2F7F73", "#6C9BC3", "#F2A900", "#9B8AFB"]
    for index, (metric, label, color) in enumerate(zip(metrics, labels, colors, strict=True)):
        axis.bar(positions + (index - 1.5) * bar_width, frame[metric], bar_width, label=label, color=color)
    axis.set_title("Two baselines and five candidate models — repeated stratified CV")
    axis.set_xticks(positions, frame["label"], rotation=18, ha="right")
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Score")
    axis.grid(axis="y", color="#D9E0E8", linewidth=0.6)
    axis.legend(ncol=4, loc="upper left", fontsize=8)
    _save_figure(figure, path)


def _plot_calibration(metadata: dict[str, Any], path: Path) -> None:
    calibration = metadata["calibration"]
    before = calibration["before"]
    after = calibration["after"]
    figure, axis = _figure(*FIGURE_SPECS["word/media/image12.png"])
    axis.plot([0, 1], [0, 1], "--", color="#0F172A", label="Perfect calibration")
    axis.plot(
        before["mean_predicted"],
        before["fraction_positive"],
        marker="o",
        color="#6C9BC3",
        label=f"Raw score (Brier={before['brier']:.4f})",
    )
    axis.plot(
        after["mean_predicted"],
        after["fraction_positive"],
        marker="s",
        color="#D97706",
        label=f"Calibrated - Platt (Brier={after['brier']:.4f})",
    )
    axis.set_title("Calibration curve — active artifact validation set")
    axis.set_xlabel("Mean predicted score")
    axis.set_ylabel("Observed positive rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.05)
    axis.grid(color="#D9E0E8", linewidth=0.6)
    axis.legend(loc="upper left", fontsize=8)
    _save_figure(figure, path)


def _locked_test_predictions(payload: dict[str, Any], locked_test_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(locked_test_path)
    features = list(payload["features"])
    actual = pd.to_numeric(frame["Diagnosis"], errors="raise").astype(int).to_numpy()
    probabilities = np.asarray(payload["model"].predict_proba(frame[features]))[:, 1]
    predictions = (probabilities >= float(payload["threshold"])).astype(int)
    return actual, probabilities, predictions


def _plot_confusion(metadata: dict[str, Any], path: Path) -> None:
    counts = np.asarray(metadata["test_confusion_matrix"]["counts"], dtype=int)
    figure, axis = _figure(*FIGURE_SPECS["word/media/image13.png"])
    image = axis.imshow(counts, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(counts[row, column]), ha="center", va="center", fontsize=15, color="#0F172A")
    axis.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
    axis.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    axis.set_title(f"Locked-test confusion matrix — threshold {metadata['threshold']:.3f}")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    _save_figure(figure, path)


def _plot_roc(actual: np.ndarray, probabilities: np.ndarray, path: Path) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(actual, probabilities)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    figure, axis = _figure(*FIGURE_SPECS["word/media/image14.png"])
    axis.plot(false_positive_rate, true_positive_rate, color="#D97706", linewidth=2.2, label=f"XGBoost (AUC={roc_auc:.4f})")
    axis.plot([0, 1], [0, 1], "--", color="#475569", label="Random")
    axis.set_title("ROC curve — active artifact on locked test")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(color="#D9E0E8", linewidth=0.6)
    axis.legend(loc="lower right")
    _save_figure(figure, path)


def _plot_model_importance(payload: dict[str, Any], path: Path) -> None:
    importances = []
    for calibrated in payload["model"].calibrated_classifiers_:
        estimator = calibrated.estimator
        model = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
        importances.append(np.asarray(model.feature_importances_, dtype=float))
    mean_importance = np.mean(importances, axis=0)
    frame = pd.DataFrame({"feature": payload["features"], "importance": mean_importance}).sort_values("importance")
    figure, axis = _figure(*FIGURE_SPECS["word/media/image15.png"])
    axis.barh(frame["feature"], frame["importance"], color="#2F7F73")
    axis.set_title("Final XGBoost — feature importance (fold mean)", fontsize=12)
    axis.set_xlabel("XGBoost feature importance")
    axis.grid(axis="x", color="#D9E0E8", linewidth=0.6)
    _save_figure(figure, path)


def _build_figures(metadata: dict[str, Any], payload: dict[str, Any], locked_test_path: Path, output_dir: Path) -> dict[str, Path]:
    paths = {name: output_dir / Path(name).name for name in FIGURE_SPECS}
    actual, probabilities, _ = _locked_test_predictions(payload, locked_test_path)
    _plot_feature_ranking(metadata, paths["word/media/image8.png"])
    _plot_feature_counts(metadata, paths["word/media/image9.png"])
    _plot_imbalance(metadata, paths["word/media/image10.png"])
    _plot_benchmark(metadata, paths["word/media/image11.png"])
    _plot_calibration(metadata, paths["word/media/image12.png"])
    _plot_confusion(metadata, paths["word/media/image13.png"])
    _plot_roc(actual, probabilities, paths["word/media/image14.png"])
    _plot_model_importance(payload, paths["word/media/image15.png"])
    return paths


def _sync_tables(document: Document, metadata: dict[str, Any]) -> None:
    selection = {int(item["feature_count"]): item for item in metadata["feature_selection"]}
    feature_table = document.tables[4]
    for row in feature_table.rows[1:]:
        count = int(row.cells[0].text.split()[0])
        _set_cell_text(row.cells[1], _vn(selection[count]["accuracy_cv"]))
        _set_cell_text(row.cells[2], _vn(selection[count]["pr_auc_cv"]))

    imbalance = {item["method"]: item for item in metadata["imbalance_comparison"]}
    imbalance_table = document.tables[5]
    method_rows = {
        "Baseline": _table_row_by_prefix(imbalance_table, "Baseline"),
        "scale_pos_weight": _table_row_by_prefix(imbalance_table, "scale_pos_weight"),
        "SMOTENC": _table_row_by_prefix(imbalance_table, "SMOTENC"),
    }
    for method, row in method_rows.items():
        values = imbalance[method]
        for column_index, metric in enumerate(["accuracy", "pr_auc", "f1", "recall", "brier"], start=1):
            _set_cell_text(row.cells[column_index], _vn(values[metric]))

    benchmark = {item["model"]: item for item in metadata["benchmark"]}
    benchmark_table = document.tables[7]
    report_rows = {
        "DummyClassifier": "DummyClassifier",
        "Logistic Regression": "Logistic Regression",
        "KNN": "KNN",
        "SVM (RBF)": "SVM",
        "Random Forest": "Random Forest",
        "Gradient Boosting": "Gradient Boosting",
        "XGBoost": "XGBoost",
    }
    for model, prefix in report_rows.items():
        row = _table_row_by_prefix(benchmark_table, prefix)
        values = benchmark[model]
        for column_index, metric in enumerate(["pr_auc_cv", "roc_auc_cv", "f1", "recall"], start=1):
            _set_cell_text(row.cells[column_index], _vn(values[metric]))

    recall_table = document.tables[8]
    recall_rows = {
        "KNN": "KNN",
        "SVM (RBF)": "SVM",
        "Logistic Regression": "Logistic Regression",
        "Random Forest": "Random Forest",
        "Gradient Boosting": "Gradient Boosting",
        "XGBoost": "XGBoost",
    }
    for model, prefix in recall_rows.items():
        row = _table_row_by_prefix(recall_table, prefix)
        _set_cell_text(row.cells[2], _vn(benchmark[model]["recall"]))

    tuning_table = document.tables[9]
    tuning_values = {
        "learning_rate": _vn(metadata["hyperparameters"]["learning_rate"], 2),
        "max_depth": str(metadata["hyperparameters"]["max_depth"]),
        "n_estimators": str(metadata["hyperparameters"]["n_estimators"]),
        "PR-AUC": _vn(metadata["tuning_best_pr_auc"]),
    }
    for prefix, value in tuning_values.items():
        row = _table_row_by_prefix(tuning_table, prefix)
        _set_cell_text(row.cells[1], value)

    calibration_table = document.tables[10]
    before = metadata["calibration"]["before"]
    after = metadata["calibration"]["after"]
    brier_row = _table_row_by_prefix(calibration_table, "Brier")
    ece_row = _table_row_by_prefix(calibration_table, "ECE")
    _set_cell_text(brier_row.cells[1], _vn(before["brier"]))
    _set_cell_text(brier_row.cells[2], _vn(after["brier"]))
    _set_cell_text(ece_row.cells[1], _vn(before["ece"]))
    _set_cell_text(ece_row.cells[2], _vn(after["ece"]))

    test_table = document.tables[11]
    test_metrics = metadata["test_metrics"]
    confidence_intervals = metadata["bootstrap_95_ci"]
    test_rows = {
        "Accuracy": (test_metrics["accuracy"], None),
        "ROC-AUC": (test_metrics["roc_auc"], confidence_intervals["roc_auc"]),
        "PR-AUC": (test_metrics["pr_auc"], confidence_intervals["pr_auc"]),
        "F1": (test_metrics["f1"], None),
        "F2": (test_metrics["f2"], None),
        "Precision": (test_metrics["precision"], confidence_intervals["precision"]),
        "Recall": (test_metrics["recall"], confidence_intervals["recall"]),
    }
    for metric, (value, interval) in test_rows.items():
        row = _table_row_by_prefix(test_table, metric)
        _set_cell_text(row.cells[1], _vn(value))
        if interval is not None:
            _set_cell_text(row.cells[2], f"[{_vn(interval[0])} ; {_vn(interval[1])}]")


def _sync_paragraphs(document: Document, metadata: dict[str, Any]) -> None:
    selection = {int(item["feature_count"]): item for item in metadata["feature_selection"]}
    test = metadata["test_metrics"]
    benchmark = {item["model"]: item for item in metadata["benchmark"]}
    before = metadata["calibration"]["before"]
    after = metadata["calibration"]["after"]
    replacements = {
        93: (
            "Giữ 5 feature (MMSE, FunctionalAssessment, ADL, MemoryComplaints, BehavioralProblems) "
            f"cho Accuracy {_vn(selection[5]['accuracy_cv'])} và PR-AUC {_vn(selection[5]['pr_auc_cv'])}; "
            f"bộ 32 feature đạt Accuracy {_vn(selection[32]['accuracy_cv'])} và PR-AUC {_vn(selection[32]['pr_auc_cv'])}. "
            "Bộ 5 feature được chọn vì tinh gọn 84,4% đầu vào nhưng vẫn duy trì hiệu năng cross-validation tương đương hoặc tốt hơn."
        ),
        103: (
            "Không có phương pháp nào thắng ở mọi chỉ số. Baseline có Accuracy/F1 cao nhất và Brier thấp nhất; "
            "SMOTENC nhỉnh nhất về PR-AUC; scale_pos_weight đạt Recall và F2 cao nhất. "
            "Theo policy của đề tài, scale_pos_weight được chọn vì PR-AUC nằm trong 0,005 so với mức tốt nhất, "
            "sau đó ưu tiên F2, Recall, Brier và phương pháp đơn giản hơn."
        ),
        115: (
            f"KẾT LUẬN — chọn XGBoost vì: (1) PR-AUC cao nhất ({_vn(benchmark['XGBoost']['pr_auc_cv'])}) "
            "trong bảy mô hình; "
            f"(2) Recall đạt {_vn(benchmark['XGBoost']['recall'])}, cao nhất trong benchmark; "
            "(3) scale_pos_weight xử lý mất cân bằng trực tiếp trong quá trình huấn luyện."
        ),
        121: (
            "GridSearchCV lặp qua 27 tổ hợp tham số x 5-fold = 135 lần huấn luyện và chọn tổ hợp có PR-AUC trung bình cao nhất. "
            f"Sau tối ưu, PR-AUC của XGBoost đạt {_vn(metadata['tuning_best_pr_auc'])} (5-fold CV) trước khi refit mô hình cuối."
        ),
        126: (
            f"Sau Platt scaling, Brier Score cải thiện từ {_vn(before['brier'])} xuống {_vn(after['brier'])} "
            f"và ECE cải thiện từ {_vn(before['ece'])} xuống {_vn(after['ece'])}. "
            f"Ngưỡng phân loại được chọn bằng F2-score trên validation là {_vn(metadata['threshold'], 3)} trên score đã hiệu chỉnh."
        ),
        132: (
            f"Confusion matrix tại ngưỡng {_vn(metadata['threshold'], 3)}: TN=242, FP=12, FN=13, TP=126. "
            "Mô hình chỉ dùng 5 feature nhưng duy trì hiệu năng tốt trên tập test khóa."
        ),
        138: "App được triển khai với 4 không gian làm việc chính:",
        139: "Tab 1 — Dự đoán: nhập 5 chỉ số cho một ca hoặc tải CSV để sàng lọc hàng loạt; kết quả gồm calibrated score và tín hiệu dương/âm.",
        140: "Tab 2 — Dataset dự án: kiểm tra schema, phát hiện lỗi cấp dòng/ô, xử lý dữ liệu và xuất CSV đã làm sạch; tab này không dự đoán hoặc tái huấn luyện.",
        142: "Tab 3 — Thông tin mô hình: trình bày quy trình, EDA, học không giám sát, lựa chọn đặc trưng, mất cân bằng, benchmark, calibration, ngưỡng và feedback sau khám.",
        143: "Tab 4 — Cập nhật mô hình: nhận CSV 5 đặc trưng có nhãn hoặc feedback bác sĩ đã xác minh, so sánh champion/challenger và chỉ promote khi đạt policy.",
        160: (
            "Mô hình cuối cùng sử dụng 5/32 feature và trên tập test khóa đạt "
            f"Accuracy {test['accuracy']:.2%}, ROC-AUC {_vn(test['roc_auc'])}, PR-AUC {_vn(test['pr_auc'])}, "
            f"Recall {test['recall']:.2%} và Specificity {test['specificity']:.2%}. "
            "XGBoost được chọn vì đạt PR-AUC cao nhất trong benchmark đồng thời hỗ trợ trọng số lớp tường minh."
        ),
    }
    for index, text in replacements.items():
        _set_paragraph_text(document.paragraphs[index], text)


def _replace_media(docx_path: Path, replacements: dict[str, Path]) -> None:
    temporary_path = docx_path.with_suffix(".media-update.docx")
    with zipfile.ZipFile(docx_path, "r") as source, zipfile.ZipFile(
        temporary_path, "w", zipfile.ZIP_DEFLATED
    ) as target:
        for item in source.infolist():
            replacement = replacements.get(item.filename)
            data = replacement.read_bytes() if replacement else source.read(item.filename)
            target.writestr(item, data)
    os.replace(temporary_path, docx_path)


def synchronize_report(
    source_report: Path,
    output_report: Path,
    metadata_path: Path,
    model_path: Path,
    locked_test_path: Path,
) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload = joblib.load(model_path)
    document = Document(source_report)
    _sync_tables(document, metadata)
    _sync_paragraphs(document, metadata)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_report)

    with tempfile.TemporaryDirectory(prefix="alz-report-sync-") as temporary_directory:
        figure_paths = _build_figures(
            metadata,
            payload,
            locked_test_path,
            Path(temporary_directory),
        )
        _replace_media(output_report, figure_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_report", type=Path)
    parser.add_argument("output_report", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--locked-test", type=Path, required=True)
    args = parser.parse_args()
    synchronize_report(
        args.source_report,
        args.output_report,
        args.metadata,
        args.model,
        args.locked_test,
    )
    print(args.output_report.resolve())


if __name__ == "__main__":
    main()
