"""CSV safety, schema, and export helpers for the Streamlit data workflow."""

from __future__ import annotations

import csv
from io import StringIO
from typing import Sequence

import pandas as pd

from ml_pipeline import (
    ALL_FEATURES,
    BINARY_COLUMNS,
    CATEGORICAL_ALLOWED_VALUES,
    IDENTIFIER_COLUMNS,
    RANGE_RULES,
    SELECTED_FEATURES,
    TARGET_COLUMN,
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_ROWS = 10_000
MAX_UPLOAD_COLUMNS = 100
MAX_TEXT_LENGTH = 2_048
SCREENING_DISCLAIMER = (
    "Educational screening only; not a diagnosis and not a substitute for a clinician's assessment."
)


def neutralize_csv_formulas(frame: pd.DataFrame) -> pd.DataFrame:
    """Prevent spreadsheet formula execution in exported text cells."""

    exported = frame.copy()
    formula_prefixes = ("=", "+", "-", "@")
    for column in exported.select_dtypes(include=["object", "string"]).columns:
        exported[column] = exported[column].map(
            lambda value: ("'" + value)
            if isinstance(value, str) and value.startswith(formula_prefixes)
            else value
        )
    return exported


def screening_export_frame(
    frame: pd.DataFrame,
    metadata: dict,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Add provenance and safety context to every screening export."""

    exported = neutralize_csv_formulas(frame)
    exported["model_version"] = metadata.get("version", "unknown")
    exported["threshold"] = float(metadata.get("threshold", 0.5) if threshold is None else threshold)
    exported["screening_only"] = True
    exported["disclaimer"] = SCREENING_DISCLAIMER
    return exported


def read_uploaded_csv(
    uploaded,
    required_columns: Sequence[str] | None = None,
    language: str = "en",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read a CSV with visible encoding, delimiter, and parser diagnostics."""

    raw_bytes = uploaded.getvalue()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Uploaded CSV exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB safety limit.")
    encoding = None
    decoded = None
    for candidate in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            decoded = raw_bytes.decode(candidate)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    if decoded is None or encoding is None:
        raise ValueError("Could not decode the uploaded CSV as UTF-8/Windows-1252/Latin-1.")

    try:
        delimiter = csv.Sniffer().sniff(decoded[:10000], delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","

    try:
        frame = pd.read_csv(
            StringIO(decoded),
            sep=delimiter,
            keep_default_na=True,
            na_filter=True,
            on_bad_lines="error",
        )
    except Exception as exc:
        raise ValueError(f"CSV parse error (encoding={encoding}, delimiter={delimiter!r}): {exc}") from exc

    if len(frame) > MAX_UPLOAD_ROWS:
        missing = [column for column in (required_columns or []) if column not in frame.columns]
        if missing:
            if language == "vi":
                raise ValueError(
                    f"CSV có {len(frame):,} dòng (giới hạn {MAX_UPLOAD_ROWS:,}) và thiếu "
                    f"{len(missing)} cột bắt buộc của mô hình: {', '.join(missing)}. "
                    "Đây là bộ dữ liệu khác mẫu; không thể tự đổi tên chỉ số khác thành đầu vào mô hình. "
                    "Hãy dùng mẫu CSV của tab này với các chỉ số được đo đúng."
                )
            raise ValueError(
                f"CSV has {len(frame):,} rows (limit {MAX_UPLOAD_ROWS:,}) and is missing "
                f"{len(missing)} required model columns: {', '.join(missing)}. "
                "This is a different schema; unrelated measures cannot be renamed into model inputs. "
                "Use this tab's CSV template with the actual measured indicators."
            )
        if language == "vi":
            raise ValueError(
                f"CSV có {len(frame):,} dòng, vượt giới hạn {MAX_UPLOAD_ROWS:,} dòng mỗi lần xử lý. "
                f"Hãy chia thành các file tối đa {MAX_UPLOAD_ROWS:,} dòng, giữ nguyên tiêu đề; chưa dòng nào được dự đoán."
            )
        raise ValueError(
            f"CSV has {len(frame):,} rows, above the {MAX_UPLOAD_ROWS:,}-row limit per batch. "
            f"Split it into files of at most {MAX_UPLOAD_ROWS:,} rows with the same header; no rows were scored."
        )
    if len(frame.columns) > MAX_UPLOAD_COLUMNS:
        raise ValueError(f"Uploaded CSV exceeds the {MAX_UPLOAD_COLUMNS}-column safety limit.")
    text_columns = frame.select_dtypes(include=["object", "string"])
    max_text_length = (
        text_columns.astype("string").apply(lambda column: column.str.len()).max().max()
        if not text_columns.empty
        else 0
    )
    if pd.notna(max_text_length) and int(max_text_length) > MAX_TEXT_LENGTH:
        raise ValueError(f"Uploaded text fields exceed the {MAX_TEXT_LENGTH}-character safety limit.")

    return frame, {
        "source": "Uploaded CSV",
        "filename": uploaded.name,
        "size_bytes": uploaded.size,
        "encoding": encoding,
        "delimiter": repr(delimiter),
        "parser": "pandas.read_csv; on_bad_lines=error",
    }


def template_columns(purpose: str, schema: str) -> list[str]:
    if schema == "Five-feature scoring CSV":
        return [*SELECTED_FEATURES]
    if schema == "Five-feature retraining CSV":
        return [*SELECTED_FEATURES, TARGET_COLUMN]
    return [
        *IDENTIFIER_COLUMNS,
        *ALL_FEATURES,
        *([TARGET_COLUMN] if purpose == "Training/retraining audit" else []),
    ]


def detect_v3_schema(columns: Sequence[str]) -> str:
    """Choose the available V3 cleaning path from CSV headers, without guessing fields."""

    available = set(columns)
    if set(ALL_FEATURES).issubset(available):
        return "Full V3 scoring CSV"
    if set(SELECTED_FEATURES).issubset(available):
        return "Five-feature scoring CSV"
    missing = [column for column in SELECTED_FEATURES if column not in available]
    raise ValueError(
        "CSV is missing required model columns: " + ", ".join(missing)
    )


def validate_five_feature_scoring_columns(
    columns: Sequence[str], *, language: str = "vi"
) -> None:
    """Keep batch prediction distinct from Full V3 data processing and retraining."""

    available = set(columns)
    missing = [column for column in SELECTED_FEATURES if column not in available]
    if missing:
        prefix = (
            "CSV dự đoán thiếu 5 chỉ số bắt buộc: "
            if language == "vi"
            else "Prediction CSV is missing required model inputs: "
        )
        raise ValueError(prefix + ", ".join(missing))

    extra = [column for column in columns if column not in {*SELECTED_FEATURES, "PatientID"}]
    if not extra:
        return
    if set(ALL_FEATURES).issubset(available):
        message = (
            "CSV Full V3 thuộc tab Dataset dự án. Tab Dự đoán hàng loạt chỉ nhận 5 chỉ số mô hình và PatientID tùy chọn."
            if language == "vi"
            else "Use the Project dataset tab for Full V3 CSVs. Batch prediction accepts only five model inputs and optional PatientID."
        )
    elif TARGET_COLUMN in extra:
        message = (
            "CSV dự đoán không nhận Diagnosis. Nếu có nhãn thực tế, dùng tab Cập nhật mô hình; tab này chỉ nhận 5 chỉ số và PatientID tùy chọn."
            if language == "vi"
            else "Prediction CSV must not include Diagnosis. Use Model update for labelled cases; this tab accepts only five inputs and optional PatientID."
        )
    else:
        message = (
            "CSV dự đoán chỉ nhận 5 chỉ số mô hình và PatientID tùy chọn. Cột khác: "
            if language == "vi"
            else "Prediction CSV accepts only five model inputs and optional PatientID. Extra columns: "
        ) + ", ".join(extra[:8])
    raise ValueError(message)


def cleaning_removed_rows(
    raw: pd.DataFrame, cleaned: pd.DataFrame, *, full_schema: bool
) -> pd.DataFrame:
    """Return every pre-validation removal with its original row and reason."""

    if "_source_row" not in raw or "_source_row" not in cleaned:
        raise ValueError("Source row numbers are required for a cleaning audit.")
    duplicate_mask = raw.drop(columns=["_source_row"]).duplicated()
    removed = [
        raw.loc[duplicate_mask].assign(_removal_reason="duplicate row")
    ]
    if full_schema:
        after_duplicates = raw.loc[~duplicate_mask]
        systolic = pd.to_numeric(after_duplicates["SystolicBP"], errors="coerce")
        diastolic = pd.to_numeric(after_duplicates["DiastolicBP"], errors="coerce")
        invalid_bp = systolic.notna() & diastolic.notna() & (systolic <= diastolic)
        removed.append(
            after_duplicates.loc[invalid_bp].assign(
                _removal_reason="SystolicBP <= DiastolicBP"
            )
        )
    result = pd.concat(removed, ignore_index=True).sort_values("_source_row")
    if len(result) != len(raw) - len(cleaned):
        raise ValueError("Cleaning audit does not match the number of removed rows.")
    return result.reset_index(drop=True)


def build_error_review_table(
    raw: pd.DataFrame,
    valid: pd.DataFrame,
    issues: pd.DataFrame,
    *,
    language: str = "vi",
) -> pd.DataFrame:
    """Compare only problematic input cells with their actual processing result."""

    columns = ["row", "field", "original", "issue", "processed", "status"]
    if "_source_row" not in raw or "_row_number" not in valid:
        raise ValueError("Source row numbers are required for error review.")
    source = raw.set_index("_source_row", drop=False)
    accepted = valid.set_index("_row_number", drop=False)
    accepted_rows = set(accepted.index)
    records: list[dict[str, object]] = []
    represented: set[tuple[int, str]] = set()

    def value_text(value: object) -> str:
        if pd.isna(value):
            return "Thiếu" if language == "vi" else "Missing"
        return str(value)[:120]

    def original_text(row: int, field: str) -> str:
        if field in source.columns:
            return value_text(source.at[row, field])
        if field == "SystolicBP/DiastolicBP":
            return " / ".join(
                value_text(source.at[row, column])
                for column in ("SystolicBP", "DiastolicBP")
            )
        if field == "model_features":
            return " / ".join(value_text(source.at[row, name]) for name in SELECTED_FEATURES)
        return "—"

    def issue_text(value: str) -> str:
        if language != "vi":
            return value
        translations = {
            "duplicate row": "Dòng trùng",
            "SystolicBP <= DiastolicBP": "Huyết áp tâm thu ≤ tâm trương",
            "missing value": "Thiếu giá trị",
            "invalid format": "Sai định dạng",
            "missing or invalid target": "Thiếu/sai nhãn Diagnosis",
            "target must be 0 or 1": "Diagnosis phải là 0 hoặc 1",
            "unknown binary code": "Mã nhị phân không hợp lệ",
            "unknown categorical code": "Mã phân loại không hợp lệ",
            "conflicting duplicate PatientID": "PatientID trùng/mâu thuẫn",
            "all selected model features missing": "Thiếu cả 5 đặc trưng mô hình",
            "SystolicBP must be greater than DiastolicBP": "Huyết áp tâm thu phải lớn hơn tâm trương",
        }
        if value.startswith("negative value") or value.startswith("negative target"):
            return "Giá trị âm ngoài miền hợp lệ"
        if value.startswith("outside allowed range"):
            return "Ngoài khoảng hợp lệ"
        return translations.get(value, value)

    if not issues.empty:
        for (row_number, field), group in issues.groupby(["row", "field"], sort=False):
            row = int(row_number)
            field = str(field)
            represented.add((row, field))
            if field == "model_features":
                represented.update((row, name) for name in SELECTED_FEATURES)
            elif field == "SystolicBP/DiastolicBP":
                represented.update((row, name) for name in ("SystolicBP", "DiastolicBP"))
            if row not in source.index:
                continue
            issue_labels = list(dict.fromkeys(issue_text(str(value)) for value in group["issue"]))
            corrected = row in accepted_rows and group["action"].astype(str).str.contains("imputed").any()
            status = "corrected" if corrected else "rejected" if row not in accepted_rows else "review"
            records.append(
                {
                    "row": row,
                    "field": field,
                    "original": original_text(row, field),
                    "issue": "; ".join(issue_labels),
                    "processed": value_text(accepted.at[row, field]) if corrected and field in accepted.columns else "—",
                    "status": status,
                }
            )

    for field in raw.columns:
        if field == "_source_row":
            continue
        numeric = pd.to_numeric(raw[field], errors="coerce")
        missing = raw[field].isna()
        negative = numeric.lt(0).fillna(False)
        for index in raw.index[missing | negative]:
            row = int(raw.at[index, "_source_row"])
            if (row, field) in represented:
                continue
            represented.add((row, field))
            if missing.at[index]:
                issue = (
                    "Thiếu giá trị chưa được quy tắc V3 xử lý"
                    if language == "vi"
                    else "Missing value not covered by V3 rules"
                )
            else:
                issue = (
                    "Giá trị âm chưa được quy tắc V3 xử lý"
                    if language == "vi"
                    else "Negative value not covered by V3 rules"
                )
            records.append(
                {
                    "row": row,
                    "field": field,
                    "original": original_text(row, field),
                    "issue": issue,
                    "processed": "—",
                    "status": "rejected" if row not in accepted_rows else "review",
                }
            )

    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records, columns=columns).sort_values(
        ["row", "field"], kind="stable"
    ).reset_index(drop=True)


def retraining_template_bytes() -> bytes:
    """Return five model inputs plus the required observed diagnosis label."""

    output = StringIO()
    csv.writer(output, lineterminator="\n").writerow(
        template_columns("Training/retraining audit", "Five-feature retraining CSV")
    )
    return output.getvalue().encode("utf-8-sig")


def schema_guide(columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for column in columns:
        if column in IDENTIFIER_COLUMNS:
            rule = "identifier; preserve for traceability"
            kind = "identifier"
        elif column in RANGE_RULES:
            lower, upper = RANGE_RULES[column]
            rule = f"numeric range [{lower:g}, {upper:g}]"
            kind = "numeric"
        elif column in CATEGORICAL_ALLOWED_VALUES:
            rule = f"allowed codes {list(CATEGORICAL_ALLOWED_VALUES[column])}"
            kind = "categorical"
        elif column in BINARY_COLUMNS or column == TARGET_COLUMN:
            rule = "allowed codes [0, 1]"
            kind = "binary"
        else:
            rule = "numeric or categorical value according to V3 schema"
            kind = "feature"
        records.append({"column": column, "type": kind, "rule": rule})
    return pd.DataFrame(records)


def validation_rule_summary(language: str = "en", *, five_feature_only: bool = False) -> pd.DataFrame:
    """Summarize the rules relevant to the active CSV processing path."""

    if five_feature_only:
        rows = [
            ("One input missing", "MMSE is blank", "Impute from saved training statistics", "Thiếu một chỉ số", "MMSE để trống", "Điền bằng thống kê đã lưu từ tập huấn luyện"),
            ("All five inputs missing", "All five model fields are blank", "Reject the row", "Thiếu cả 5 chỉ số", "Cả 5 cột mô hình đều trống", "Loại dòng"),
            ("Negative/out of range", "MMSE=-1 or ADL>10", "Reject the row", "Giá trị âm/ngoài khoảng", "MMSE=-1 hoặc ADL>10", "Loại dòng"),
            ("Invalid numeric format", "MMSE='abc'", "Treat as missing, then impute if recoverable", "Sai định dạng số", "MMSE='abc'", "Chuyển thành thiếu rồi điền nếu còn đủ dữ liệu"),
            ("Invalid binary code", "MemoryComplaints=2", "Reject the row", "Mã nhị phân sai", "MemoryComplaints=2", "Loại dòng"),
            ("Conflicting PatientID", "Same PatientID on different rows", "Reject conflicting rows", "PatientID mâu thuẫn", "Một mã trên nhiều dòng khác nhau", "Loại các dòng mâu thuẫn"),
        ]
        if language == "vi":
            return pd.DataFrame(
                [(case_vi, example_vi, action_vi) for _, _, _, case_vi, example_vi, action_vi in rows],
                columns=["Trường hợp", "Ví dụ", "Cách xử lý"],
            )
        return pd.DataFrame(
            [(case, example, action) for case, example, action, _, _, _ in rows],
            columns=["Case", "Example", "Validator action"],
        )

    rows = [
            {
                "Case": "Missing numeric value",
                "Example": "MMSE or BMI is blank",
                "Validator action": "Impute with the saved training median and keep the row",
                "Trường hợp": "Thiếu giá trị số",
                "Ví dụ": "MMSE hoặc BMI để trống",
                "Cách xử lý": "Điền median đã lưu từ tập huấn luyện và giữ dòng",
            },
            {
                "Case": "Physiologically impossible negative",
                "Example": "Age=-25 or Cholesterol=-100",
                "Validator action": "Reject the row",
                "Trường hợp": "Giá trị âm phi sinh lý",
                "Ví dụ": "Age=-25 hoặc Cholesterol=-100",
                "Cách xử lý": "Loại dòng",
            },
            {
                "Case": "Invalid numeric format",
                "Example": "SystolicBP='one hundred' or MMSE='N/A'",
                "Validator action": "Convert to missing, then impute when the row remains recoverable",
                "Trường hợp": "Sai định dạng số",
                "Ví dụ": "SystolicBP='một trăm' hoặc MMSE='N/A'",
                "Cách xử lý": "Chuyển thành giá trị thiếu, sau đó điền nếu dòng còn có thể phục hồi",
            },
            {
                "Case": "Outside the V3 range",
                "Example": "Age>120 or MMSE>30",
                "Validator action": "Reject the row",
                "Trường hợp": "Ngoài khoảng V3",
                "Ví dụ": "Age>120 hoặc MMSE>30",
                "Cách xử lý": "Loại dòng",
            },
            {
                "Case": "Unknown categorical code",
                "Example": "Gender>3 or Ethnicity=-1",
                "Validator action": "Reject the row",
                "Trường hợp": "Mã phân loại không hợp lệ",
                "Ví dụ": "Gender>3 hoặc Ethnicity=-1",
                "Cách xử lý": "Loại dòng",
            },
            {
                "Case": "Invalid blood-pressure ordering",
                "Example": "SystolicBP <= DiastolicBP",
                "Validator action": "Reject the row when both BP fields are supplied",
                "Trường hợp": "Sai thứ tự huyết áp",
                "Ví dụ": "SystolicBP <= DiastolicBP",
                "Cách xử lý": "Loại dòng khi có đủ hai cột huyết áp",
            },
        ]
    view = pd.DataFrame(rows)
    if language == "vi":
        return view[["Trường hợp", "Ví dụ", "Cách xử lý"]]
    return view[["Case", "Example", "Validator action"]]


__all__ = [
    "MAX_UPLOAD_BYTES",
    "MAX_UPLOAD_COLUMNS",
    "MAX_UPLOAD_ROWS",
    "MAX_TEXT_LENGTH",
    "SCREENING_DISCLAIMER",
    "neutralize_csv_formulas",
    "read_uploaded_csv",
    "retraining_template_bytes",
    "schema_guide",
    "screening_export_frame",
    "template_columns",
    "validate_five_feature_scoring_columns",
    "validation_rule_summary",
]
