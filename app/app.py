"""Four-tab Streamlit interface for the V3 Alzheimer's screening model."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import streamlit as st

from auth import (
    AuthUser,
    authenticate,
    authentication_disabled,
    create_user,
    list_users,
    user_count,
)
from feedback_store import (
    attach_case_ids,
    feedback_training_frame,
    get_retraining_run,
    list_cases,
    list_retraining_runs,
    record_prediction_cases,
    record_retraining_run,
    submit_feedback,
)
from ml_pipeline import (
    ALL_FEATURES,
    BINARY_COLUMNS,
    CATEGORICAL_ALLOWED_VALUES,
    DATASET_COLUMNS,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_LOCKED_TEST_PATH,
    IDENTIFIER_COLUMNS,
    RANGE_RULES,
    SELECTED_FEATURES,
    TARGET_COLUMN,
    VALIDATION_META_COLUMNS,
    _accepted_retraining_rows,
    _locked_test_frame,
    clean_dataset,
    fit_imputation_statistics,
    load_artifacts,
    monitor_prediction_batch,
    predict_dataframe,
    prepare_retraining_rows,
    retrain_with_new_data,
    summarize_prediction_batch,
    validate_dataframe,
)
from ui import data_processing as data_processing_ui
from ui import layout as layout_ui
from ui import model_info as model_info_ui

SCREENING_DISCLAIMER = data_processing_ui.SCREENING_DISCLAIMER
_read_uploaded_csv = data_processing_ui.read_uploaded_csv
_detect_v3_schema = data_processing_ui.detect_v3_schema
_validate_five_feature_scoring_columns = data_processing_ui.validate_five_feature_scoring_columns
_cleaning_removed_rows = data_processing_ui.cleaning_removed_rows
_build_error_review_table = data_processing_ui.build_error_review_table
_retraining_template_bytes = data_processing_ui.retraining_template_bytes
_schema_guide = data_processing_ui.schema_guide
_screening_export_frame = data_processing_ui.screening_export_frame
_template_columns = data_processing_ui.template_columns
_validation_rule_summary = data_processing_ui.validation_rule_summary
_format_model_value = model_info_ui.format_model_value
_feature_selection_stage_matrix = model_info_ui.feature_selection_stage_matrix
_feature_selection_summary_table = model_info_ui.feature_selection_summary_table
_imbalance_summary_table = model_info_ui.imbalance_summary_table
_benchmark_summary_table = model_info_ui.benchmark_summary_table
_model_influence_table = model_info_ui.model_influence_table
_model_metric_table = model_info_ui.model_metric_table
_retraining_metric_comparison = model_info_ui.retraining_metric_comparison
_selected_feature_evidence_table = model_info_ui.selected_feature_evidence_table
_threshold_scenario_table = model_info_ui.threshold_scenario_table

st.set_page_config(page_title="Alzheimer Screening", page_icon="🧠", layout="wide")


def _inject_app_styles() -> None:
    """Apply the shared visual system for the dashboard."""

    layout_ui.inject_app_styles()


def _render_sidebar_brand() -> None:
    """Render the persistent product identity above navigation."""

    layout_ui.render_sidebar_brand()


def _render_sidebar_status(version: str, feature_count: int, threshold: float) -> None:
    """Show the active artifact context without crowding the main page."""

    layout_ui.render_sidebar_status(version, feature_count, threshold, _t)



# Values synchronized into the updated V3 report. Keep this snapshot separate
# from future artifacts so later retraining does not rewrite submitted evidence.
REPORT_V3_MODEL_REFERENCE = {
    "model": "XGBoost (scale_pos_weight)",
    "benchmark": {
        "pr_auc_cv": 0.9394,
        "roc_auc_cv": 0.9588,
        "f1": 0.9337,
        "recall": 0.9347,
    },
    "feature_selection": {
        5: {"accuracy_cv": 0.9530, "pr_auc_cv": 0.9394},
        32: {"accuracy_cv": 0.9427, "pr_auc_cv": 0.9361},
    },
    "test_metrics": {
        "accuracy": 0.9364,
        "roc_auc": 0.9416,
        "pr_auc": 0.9221,
        "f1": 0.9097,
        "f2": 0.9078,
        "precision": 0.9130,
        "recall": 0.9065,
    },
    "bootstrap_ci": {
        "roc_auc": [0.9080, 0.9672],
        "pr_auc": [0.8790, 0.9550],
        "precision": [0.8648, 0.9572],
        "recall": [0.8540, 0.9477],
    },
    "confusion_matrix": {"TN": 242, "FP": 12, "FN": 13, "TP": 126, "n": 393},
    "threshold": 0.425,
    "hyperparameters": {
        "learning_rate": 0.03,
        "max_depth": 3,
        "n_estimators": 300,
    },
    "calibration": {
        "brier_before": 0.0696,
        "brier_after": 0.0642,
        "ece_before": 0.0735,
        "ece_after": 0.0480,
    },
}

def _language_selector() -> str:
    """Render the app language selector and return the active language code."""

    options = {"Tiếng Việt": "vi", "English": "en"}
    current = st.session_state.get("language", "vi")
    default_index = 0 if current == "vi" else 1
    selected = st.sidebar.selectbox(
        "Ngôn ngữ / Language",
        list(options),
        index=default_index,
        key="language_choice",
    )
    language = options[selected]
    st.session_state["language"] = language
    return language


def _t(english: str, vietnamese: str) -> str:
    """Return the selected-language UI string; technical values stay unchanged."""

    return vietnamese if st.session_state.get("language", "vi") == "vi" else english


def _session_user() -> AuthUser | None:
    """Read the authenticated identity from Streamlit session state."""

    value = st.session_state.get("auth_user")
    if not isinstance(value, dict):
        return None
    username = str(value.get("username", "")).strip()
    role = str(value.get("role", "")).strip()
    return AuthUser(username=username, role=role) if username and role else None


def _set_session_user(user: AuthUser) -> None:
    st.session_state["auth_user"] = {"username": user.username, "role": user.role}


def _require_authentication() -> AuthUser:
    """Render first-admin setup/login and stop until a user is authenticated."""

    if authentication_disabled():
        return AuthUser(username="test-admin", role="admin")

    current = _session_user()
    if current is not None:
        return current

    first_run = user_count() == 0
    with st.container(key="auth_page"):
        intro_column, form_column = st.columns(
            [1.05, 0.95], gap="large", vertical_alignment="center"
        )
        with intro_column:
            layout_ui.render_auth_intro(_t)
        with form_column:
            with st.container(key="auth_panel"):
                layout_ui.render_auth_heading(_t, first_run=first_run)
                if first_run:
                    with st.form("bootstrap_admin_form", border=False):
                        username = st.text_input(_t("Administrator username", "Tên đăng nhập quản trị"))
                        password = st.text_input(_t("Password", "Mật khẩu"), type="password")
                        confirmation = st.text_input(
                            _t("Confirm password", "Nhập lại mật khẩu"), type="password"
                        )
                        submitted = st.form_submit_button(
                            _t("Create administrator", "Tạo tài khoản quản trị"),
                            type="primary",
                            use_container_width=True,
                        )
                    st.caption(_t(
                        "Your password is stored locally as a salted hash, never as plain text.",
                        "Mật khẩu được lưu cục bộ dưới dạng băm có muối, không lưu dạng văn bản thuần.",
                    ))
                    if submitted:
                        if password != confirmation:
                            st.error(_t("Passwords do not match.", "Hai mật khẩu không khớp."))
                        else:
                            try:
                                user = create_user(
                                    username, password, "admin", created_by="first-run-setup"
                                )
                            except ValueError as exc:
                                st.error(str(exc))
                            else:
                                _set_session_user(user)
                                st.rerun()
                else:
                    with st.form("login_form", border=False):
                        username = st.text_input(_t("Username", "Tên đăng nhập"))
                        password = st.text_input(_t("Password", "Mật khẩu"), type="password")
                        submitted = st.form_submit_button(
                            _t("Sign in", "Đăng nhập"),
                            type="primary",
                            use_container_width=True,
                        )
                    if submitted:
                        user = authenticate(username, password)
                        if user is None:
                            st.error(_t("Incorrect username or password.", "Sai tên đăng nhập hoặc mật khẩu."))
                        else:
                            _set_session_user(user)
                            st.rerun()
    st.stop()


def _render_account_sidebar(user: AuthUser) -> None:
    """Show the current account, logout action, and admin account creation."""

    st.sidebar.markdown("---")
    st.sidebar.caption(_t("Signed in", "Đang đăng nhập"))
    st.sidebar.markdown(f"**{user.username}** · `{user.role}`")
    if st.sidebar.button(_t("Sign out", "Đăng xuất"), key="logout_button"):
        st.session_state.pop("auth_user", None)
        st.rerun()

    if user.role != "admin" or authentication_disabled():
        return
    with st.sidebar.expander(_t("Account management", "Quản lý tài khoản"), expanded=False):
        accounts = pd.DataFrame(list_users())
        if not accounts.empty:
            st.dataframe(accounts[["username", "role"]], width="stretch", hide_index=True)
        with st.form("create_account_form"):
            username = st.text_input(_t("New username", "Tên đăng nhập mới"))
            role = st.selectbox(
                _t("Role", "Vai trò"),
                ["viewer", "doctor", "admin"],
                format_func=lambda value: {
                    "viewer": _t("Viewer", "Người dùng"),
                    "doctor": _t("Doctor", "Bác sĩ"),
                    "admin": _t("Administrator", "Quản trị"),
                }[value],
            )
            password = st.text_input(_t("Temporary password", "Mật khẩu ban đầu"), type="password")
            submitted = st.form_submit_button(_t("Create account", "Tạo tài khoản"))
        if submitted:
            try:
                create_user(
                    username,
                    password,
                    role,
                    created_by=user.username,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success(_t("Account created.", "Đã tạo tài khoản."))


@st.cache_resource(show_spinner=False)
def get_runtime(artifact_cache_key: int = 0) -> tuple[dict, dict]:
    """Load the active pair and invalidate the cache when the model changes."""

    del artifact_cache_key
    return load_artifacts(DEFAULT_ARTIFACTS_DIR)


def _active_artifact_cache_key() -> int:
    model_path = DEFAULT_ARTIFACTS_DIR / "active_model.joblib"
    metadata_path = DEFAULT_ARTIFACTS_DIR / "active_metadata.json"
    mtimes = [path.stat().st_mtime_ns for path in (model_path, metadata_path) if path.exists()]
    return hash(tuple(mtimes))


@st.cache_data(show_spinner=False)
def get_reference_data() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_DATA_PATH)


def _binary_label(value: int) -> str:
    if st.session_state.get("language", "vi") == "vi":
        return "Có" if int(value) == 1 else "Không"
    return "Yes" if int(value) == 1 else "No"


def _single_case_manual_form(
    payload: dict,
    metadata: dict,
    threshold: float,
    user: AuthUser,
) -> None:
    """Render the one-case form without mixing it with CSV batch controls."""

    st.caption(_t("Enter the five features used by the final calibrated model.", "Nhập 5 đặc trưng được sử dụng bởi mô hình đã hiệu chuẩn."))
    with st.form("single_case_form"):
        left, right = st.columns(2)
        with left:
            mmse = st.slider("MMSE", min_value=0.0, max_value=30.0, value=20.0, step=0.1)
            functional = st.slider(
                "Functional assessment", min_value=0.0, max_value=10.0, value=5.0, step=0.1
            )
            adl = st.slider("ADL", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        with right:
            memory = st.radio(_t("Memory complaints", "Than phiền về trí nhớ"), [0, 1], format_func=_binary_label, horizontal=True)
            behavioral = st.radio(
                _t("Behavioral problems", "Vấn đề hành vi"), [0, 1], format_func=_binary_label, horizontal=True
            )
        save_for_feedback = st.checkbox(
            _t(
                "Save this case locally so its real examination result can be added later",
                "Lưu ca này cục bộ để cập nhật kết quả khám thực tế sau này",
            ),
            value=True,
        )
        submitted = st.form_submit_button(_t("Screen case", "Sàng lọc trường hợp"), type="primary")

    if submitted:
        row = pd.DataFrame(
            [
                {
                    "MMSE": mmse,
                    "FunctionalAssessment": functional,
                    "ADL": adl,
                    "MemoryComplaints": memory,
                    "BehavioralProblems": behavioral,
                }
            ]
        )
        prediction = predict_dataframe(payload, row).iloc[0]
        score = float(prediction["calibrated_score"])
        predicted_label = int(prediction["predicted_label"])
        confidence = float(prediction["prediction_confidence"])
        predicted_text = _t(
            "Disease indicated" if predicted_label else "No disease indicated",
            "Có bệnh" if predicted_label else "Không bệnh",
        )
        result_message = _t(
            f"Model prediction: {predicted_text}",
            f"Dự đoán mô hình: {predicted_text.upper()}",
        )
        if predicted_label:
            st.error(result_message)
        else:
            st.success(result_message)
        result_metrics = st.columns(2)
        result_metrics[0].metric(
            _t("Confidence", "Độ tin cậy"),
            f"{confidence * 100:.1f}%",
        )
        result_metrics[1].metric(
            _t("Alzheimer score", "Xác suất Alzheimer"),
            f"{score * 100:.1f}%",
            _t(f"threshold {threshold:.3f}", f"ngưỡng {threshold:.3f}"),
        )
        if predicted_label:
            st.error(_t("Screening signal: higher", "Tín hiệu sàng lọc: cao"))
            st.warning(_t("Consider discussing this result with a qualified healthcare professional. No treatment decision should be based on this model.", "Hãy trao đổi kết quả với nhân viên y tế có chuyên môn. Không sử dụng riêng mô hình này để quyết định điều trị."))
        else:
            st.success(_t("Screening signal: lower", "Tín hiệu sàng lọc: thấp"))
            st.info(_t("A lower screening signal does not rule out Alzheimer's disease or any other condition.", "Tín hiệu sàng lọc thấp không loại trừ bệnh Alzheimer hoặc bất kỳ tình trạng nào khác."))
        if save_for_feedback:
            try:
                case_ids = record_prediction_cases(
                    pd.DataFrame([prediction]),
                    actor=user.username,
                    source="single_case",
                    model_version=str(metadata.get("version", payload.get("version", "unknown"))),
                    threshold=threshold,
                )
            except Exception as exc:
                st.warning(
                    _t(
                        f"Prediction completed, but the case could not be saved: {exc}",
                        f"Đã dự đoán xong nhưng không thể lưu ca: {exc}",
                    )
                )
            else:
                st.success(
                    _t(
                        f"Case saved for follow-up: {case_ids[0]}",
                        f"Đã lưu ca để cập nhật sau khám: {case_ids[0]}",
                    )
                )
                st.code(case_ids[0])
        st.caption(SCREENING_DISCLAIMER)


def _single_case_tab(payload: dict, metadata: dict, user: AuthUser) -> None:
    """Render report tab 1 without duplicating the batch CSV workflow."""

    st.subheader(_t("Predict one case", "Dự đoán 1 ca"))
    threshold = float(metadata.get("threshold", payload.get("threshold", 0.5)))
    version = metadata.get("version", payload.get("version", "unknown"))
    st.caption(_t(f"Model version: {version} · Screening threshold: {threshold:.3f}", f"Phiên bản mô hình: {version} · Ngưỡng sàng lọc: {threshold:.3f}"))
    st.caption(
        _t(
            "Enter the five inputs used by the final XGBoost model. The result is a calibrated screening score and a positive/negative screening signal, not a diagnosis.",
            "Nhập 5 chỉ số của mô hình XGBoost cuối. Kết quả gồm điểm sàng lọc đã hiệu chỉnh và tín hiệu dương/âm, không phải chẩn đoán.",
        )
    )
    _single_case_manual_form(payload, metadata, threshold, user)


def _eda_allowed_code_rules() -> dict[str, tuple[int, ...]]:
    return {
        **{column: (0, 1) for column in BINARY_COLUMNS},
        TARGET_COLUMN: (0, 1),
        **CATEGORICAL_ALLOWED_VALUES,
    }


def _eda_schema_label(frame: pd.DataFrame) -> str:
    columns = set(frame.columns)
    if set(ALL_FEATURES).issubset(columns):
        return "Full V3 labelled CSV" if TARGET_COLUMN in columns else "Full V3 scoring CSV"
    if set(SELECTED_FEATURES).issubset(columns):
        return "Five-feature scoring CSV"
    return "Unknown or partial schema"


def _eda_v3_analysis_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the report-aligned analysis view without changing the raw CSV."""

    if not {"SystolicBP", "DiastolicBP"}.issubset(frame.columns):
        return frame.copy()
    systolic = pd.to_numeric(frame["SystolicBP"], errors="coerce")
    diastolic = pd.to_numeric(frame["DiastolicBP"], errors="coerce")
    bp_violation = systolic.notna() & diastolic.notna() & (systolic <= diastolic)
    return frame.loc[~bp_violation].copy()


def _eda_quality_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a field-level exploratory quality profile without modifying the data."""

    allowed_code_rules = _eda_allowed_code_rules()
    records: list[dict[str, object]] = []
    for column in frame.columns:
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        missing = int(series.isna().sum())
        invalid_format = int(
            (series.notna() & numeric.isna()).sum()
            if column in RANGE_RULES
            else 0
        )
        negative = int((numeric < 0).sum()) if column in ALL_FEATURES else 0
        out_of_range = 0
        rule = "No V3 domain rule"
        if column in RANGE_RULES:
            lower, upper = RANGE_RULES[column]
            out_of_range = int((series.notna() & numeric.notna() & ((numeric < lower) | (numeric > upper))).sum())
            rule = f"numeric range [{lower:g}, {upper:g}]"
        elif column in allowed_code_rules:
            allowed = allowed_code_rules[column]
            rule = f"allowed codes {list(allowed)}"
        elif column in IDENTIFIER_COLUMNS:
            rule = "identifier; not a model feature"
        invalid_codes = 0
        if column in allowed_code_rules:
            allowed = allowed_code_rules[column]
            invalid_codes = int((series.notna() & ~numeric.isin(list(allowed))).sum())
        unique = int(series.nunique(dropna=True))
        if unique <= 1 and len(frame) > 0:
            rule = f"constant/near-constant column; {rule}"
        if column in IDENTIFIER_COLUMNS:
            action = "Keep for traceability; do not use as a model feature."
        elif column == TARGET_COLUMN:
            action = "Ignore for screening; require valid 0/1 labels for retraining."
        elif invalid_codes or out_of_range or invalid_format or negative:
            action = "Review in EDA, then reject according to Batch V3 rules."
        elif missing:
            action = "Review in EDA, then impute/reject in Batch V3 rules."
        else:
            action = "No observed quality issue."
        records.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing": missing,
                "missing_pct": round((missing / len(frame)) * 100, 2) if len(frame) else 0.0,
                "unique": unique,
                "negative_cells": negative,
                "invalid_format_cells": invalid_format,
                "out_of_range_cells": out_of_range,
                "invalid_code_cells": invalid_codes,
                "v3_rule": rule,
                "recommended_action": action,
            }
        )
    return pd.DataFrame(records)


def _eda_issue_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return row/cell-level issues for audit and download; never mutate ``frame``."""

    working = frame.reset_index(drop=True)
    allowed_code_rules = _eda_allowed_code_rules()
    issues: list[dict[str, object]] = []

    def add_issue(row_index: int, column: str, issue_type: str, value: object, rule: str, action: str) -> None:
        issues.append(
            {
                "source_row": row_index + 1,
                "PatientID": working.iloc[row_index].get("PatientID", ""),
                "column": column,
                "issue_type": issue_type,
                "observed_value": "<missing>" if pd.isna(value) else str(value),
                "rule": rule,
                "recommended_action": action,
            }
        )

    for column in working.columns:
        series = working[column]
        numeric = pd.to_numeric(series, errors="coerce")
        for row_index in series[series.isna()].index:
            action = "Batch: impute eligible feature or reject if all five model features are missing."
            if column == TARGET_COLUMN:
                action = "Screening: ignore label; retraining: reject row."
            add_issue(row_index, column, "missing", series.loc[row_index], "value is missing", action)
        if column in RANGE_RULES:
            invalid_format_mask = series.notna() & numeric.isna()
            lower, upper = RANGE_RULES[column]
            range_mask = series.notna() & numeric.notna() & ((numeric < lower) | (numeric > upper))
            for row_index in series[invalid_format_mask].index:
                add_issue(
                    row_index,
                    column,
                    "invalid_format",
                    series.loc[row_index],
                    "must be numeric",
                    "Batch: convert to missing, then apply V3 impute/reject rule.",
                )
            for row_index in series[range_mask].index:
                add_issue(
                    row_index,
                    column,
                    "out_of_range",
                    series.loc[row_index],
                    f"must be in [{lower:g}, {upper:g}]",
                    "Batch: reject row and preserve the rule in the processing log.",
                )
        if column in ALL_FEATURES:
            negative_mask = series.notna() & numeric.notna() & (numeric < 0)
            for row_index in series[negative_mask].index:
                add_issue(
                    row_index,
                    column,
                    "negative_value",
                    series.loc[row_index],
                    "feature value must not be negative",
                    "Batch: reject row; do not clip silently.",
                )
        if column in allowed_code_rules:
            allowed = allowed_code_rules[column]
            invalid_code_mask = series.notna() & ~numeric.isin(list(allowed))
            for row_index in series[invalid_code_mask].index:
                add_issue(
                    row_index,
                    column,
                    "invalid_code",
                    series.loc[row_index],
                    f"allowed codes are {list(allowed)}",
                    "Batch: reject row and record the observed code.",
                )

    duplicate_mask = working.duplicated(keep=False)
    for row_index in working.index[duplicate_mask]:
        add_issue(
            row_index,
            "<row>",
            "exact_duplicate",
            f"source_row={row_index + 1}",
            "entire row is duplicated",
            "Batch: remove exact duplicate rows.",
        )

    if "PatientID" in working.columns:
        patient_ids = working["PatientID"].astype("string").str.strip()
        compare_columns = [column for column in [*ALL_FEATURES, TARGET_COLUMN] if column in working.columns]
        for patient_id in patient_ids.dropna().unique():
            if not str(patient_id).strip():
                continue
            row_indices = patient_ids[patient_ids == patient_id].index.tolist()
            if len(row_indices) < 2:
                continue
            signatures = working.loc[row_indices, compare_columns].astype("string").fillna("<missing>").drop_duplicates()
            issue_type = "conflicting_patient_id" if len(signatures) > 1 else "duplicate_patient_id"
            action = (
                "Manual review: reject all rows for this PatientID until the conflict is resolved."
                if issue_type == "conflicting_patient_id"
                else "Review duplicate PatientID rows before downstream use."
            )
            for row_index in row_indices:
                add_issue(
                    row_index,
                    "PatientID",
                    issue_type,
                    patient_id,
                    "PatientID appears more than once",
                    action,
                )

    if {"SystolicBP", "DiastolicBP"}.issubset(working.columns):
        systolic = pd.to_numeric(working["SystolicBP"], errors="coerce")
        diastolic = pd.to_numeric(working["DiastolicBP"], errors="coerce")
        bp_mask = systolic.notna() & diastolic.notna() & (systolic <= diastolic)
        for row_index in working.index[bp_mask]:
            add_issue(
                row_index,
                "SystolicBP/DiastolicBP",
                "bp_order_violation",
                f"{systolic.loc[row_index]} <= {diastolic.loc[row_index]}",
                "SystolicBP must be greater than DiastolicBP",
                "Batch: remove row before validation/modeling.",
            )

    columns = ["source_row", "PatientID", "column", "issue_type", "observed_value", "rule", "recommended_action"]
    if not issues:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(issues, columns=columns).sort_values(
        ["source_row", "issue_type", "column"]
    ).reset_index(drop=True)


def _cohens_d_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate standardized mean differences against Diagnosis for V3 EDA."""

    columns = ["feature", "n_no_diagnosis", "n_diagnosis", "mean_no_diagnosis", "mean_diagnosis", "cohens_d", "abs_cohens_d", "magnitude"]
    if TARGET_COLUMN not in frame.columns:
        return pd.DataFrame(columns=columns)
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    rows: list[dict[str, object]] = []
    for feature in ALL_FEATURES:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        group_zero = values[target.eq(0)].dropna()
        group_one = values[target.eq(1)].dropna()
        if len(group_zero) < 2 or len(group_one) < 2:
            continue
        pooled_variance = (
            ((len(group_zero) - 1) * group_zero.var(ddof=1) + (len(group_one) - 1) * group_one.var(ddof=1))
            / (len(group_zero) + len(group_one) - 2)
        )
        pooled_std = pooled_variance ** 0.5
        d_value = (group_one.mean() - group_zero.mean()) / pooled_std if pooled_std > 0 else 0.0
        magnitude = abs(d_value)
        if magnitude < 0.2:
            interpretation = "negligible"
        elif magnitude < 0.5:
            interpretation = "small"
        elif magnitude < 0.8:
            interpretation = "moderate"
        else:
            interpretation = "large"
        rows.append(
            {
                "feature": feature,
                "n_no_diagnosis": len(group_zero),
                "n_diagnosis": len(group_one),
                "mean_no_diagnosis": round(float(group_zero.mean()), 4),
                "mean_diagnosis": round(float(group_one.mean()), 4),
                "cohens_d": round(float(d_value), 4),
                "abs_cohens_d": round(float(magnitude), 4),
                "magnitude": interpretation,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("abs_cohens_d", ascending=False).reset_index(drop=True)


def _show_eda_figure(figure: Any) -> None:
    """Render EDA figures at their designed size instead of full-page width."""

    st.pyplot(figure, clear_figure=True, width="content")


def _plot_eda_heatmap(correlation: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        st.warning("Heatmap unavailable because matplotlib/seaborn is not installed.")
        return
    fig, ax = plt.subplots(figsize=(8.8, 6.8), dpi=110)
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        linewidths=0.15,
        cbar_kws={"shrink": 0.72},
    )
    ax.set_title("Pearson correlation matrix — numeric V3 features", fontsize=11, pad=10)
    ax.tick_params(axis="both", labelsize=7)
    fig.tight_layout()
    _show_eda_figure(fig)
    plt.close(fig)


def _plot_eda_boxplots(frame: pd.DataFrame) -> None:
    if TARGET_COLUMN not in frame.columns:
        st.info("Boxplots by Diagnosis require the Diagnosis column.")
        return
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        st.warning("Boxplots unavailable because matplotlib/seaborn is not installed.")
        return
    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    available = [feature for feature in SELECTED_FEATURES if feature in frame.columns]
    if not available:
        return
    column_count = min(3, len(available))
    row_count = (len(available) + column_count - 1) // column_count
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(9.2, 3.0 * row_count),
        dpi=110,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for index, feature in enumerate(available):
        plot_data = pd.DataFrame(
            {
                "Diagnosis": target,
                feature: pd.to_numeric(frame[feature], errors="coerce"),
            }
        ).dropna()
        plot_data = plot_data[plot_data["Diagnosis"].isin([0, 1])]
        axis = flat_axes[index]
        sns.boxplot(data=plot_data, x="Diagnosis", y=feature, ax=axis, color="#6C9BC3", showfliers=False)
        axis.set_title(feature, fontsize=10)
        axis.set_xlabel("Diagnosis", fontsize=9)
        axis.tick_params(labelsize=8)
    for axis in flat_axes[len(available) :]:
        axis.remove()
    fig.suptitle("Final model features by Diagnosis", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _show_eda_figure(fig)
    plt.close(fig)


def _plot_eda_cohens_d(effect_table: pd.DataFrame) -> None:
    """Render the top Cohen's d values horizontally so feature names stay readable."""

    chart_data = effect_table.head(15).sort_values("cohens_d")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.bar_chart(chart_data.set_index("feature")[["cohens_d"]], y="cohens_d", height=280)
        return
    fig, ax = plt.subplots(figsize=(7.5, max(3.2, 0.25 * len(chart_data))), dpi=110)
    ax.barh(chart_data["feature"], chart_data["cohens_d"], color="#6C9BC3")
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_title("Top feature separation by Diagnosis")
    ax.set_xlabel("Cohen's d (Diagnosis 1 − Diagnosis 0)")
    ax.set_ylabel("Feature")
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    _show_eda_figure(fig)
    plt.close(fig)


def _plot_eda_bp_scatter(frame: pd.DataFrame) -> None:
    """Plot BP values with the V3 validity boundary, SystolicBP = DiastolicBP."""

    bp_frame = frame[["SystolicBP", "DiastolicBP"]].apply(pd.to_numeric, errors="coerce")
    bp_frame = bp_frame.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if len(bp_frame) < 2:
        st.info("At least two numeric blood-pressure rows are required for the scatter plot.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.scatter_chart(bp_frame, x="SystolicBP", y="DiastolicBP", height=300)
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=110)
    ax.scatter(
        bp_frame["SystolicBP"],
        bp_frame["DiastolicBP"],
        s=16,
        alpha=0.45,
        color="#6C9BC3",
        edgecolors="none",
    )
    low = float(min(bp_frame["SystolicBP"].min(), bp_frame["DiastolicBP"].min()))
    high = float(max(bp_frame["SystolicBP"].max(), bp_frame["DiastolicBP"].max()))
    if low == high:
        low -= 1
        high += 1
    ax.plot([low, high], [low, high], linestyle="--", color="#555555", linewidth=1.0, label="SystolicBP = DiastolicBP")
    ax.set_title("Systolic vs diastolic blood pressure — raw CSV")
    ax.set_xlabel("SystolicBP")
    ax.set_ylabel("DiastolicBP")
    ax.legend(loc="best")
    fig.tight_layout()
    _show_eda_figure(fig)
    plt.close(fig)


def _plot_eda_field_issues(profile: pd.DataFrame) -> None:
    """Show only columns with issues as a readable stacked horizontal chart."""

    issue_columns = [
        "missing",
        "invalid_format_cells",
        "negative_cells",
        "out_of_range_cells",
        "invalid_code_cells",
    ]
    labels = {
        "missing": "Missing",
        "invalid_format_cells": "Invalid format",
        "negative_cells": "Negative",
        "out_of_range_cells": "Out of range",
        "invalid_code_cells": "Invalid code",
    }
    chart_data = profile.set_index("column")[issue_columns].rename(columns=labels)
    chart_data["total"] = chart_data.sum(axis=1)
    chart_data = chart_data.loc[chart_data["total"] > 0].sort_values("total").tail(20)
    if chart_data.empty:
        st.success("No field-level quality issue was detected under the V3 rules.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.bar_chart(chart_data.drop(columns="total"), height=300)
        return
    fig, ax = plt.subplots(figsize=(8.0, max(3.2, 0.24 * len(chart_data))), dpi=110)
    chart_data.drop(columns="total").plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=["#7A9ECA", "#D28C63", "#9A9A9A", "#C47D8C", "#B5A36A"],
        width=0.75,
    )
    ax.set_title("CSV issues by column — top 20 columns")
    ax.set_xlabel("Issue occurrences")
    ax.set_ylabel("Column")
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(title="Issue type", loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    _show_eda_figure(fig)
    plt.close(fig)


def _eda_quality_tab() -> None:
    """Explore and audit the selected CSV before cleaning, validation, or modeling."""

    st.subheader(_t("Exploratory Data Analysis (EDA)", "Phân tích dữ liệu khám phá (EDA)"))
    st.caption(
        _t(
            "EDA reads raw data only to answer: where is the CSV wrong, how serious is it, and what processing is needed. "
            "This tab does not impute, delete rows, or create predictions.",
            "EDA chỉ đọc dữ liệu thô để trả lời: CSV sai ở đâu, mức độ nào và cần xử lý bước nào. "
            "Tab này không điền thiếu, không xóa dòng và không tạo dự đoán.",
        )
    )

    source = st.radio(
        _t("EDA data source", "Nguồn dữ liệu EDA"),
        [_t("Project dataset", "Dataset của dự án"), _t("Upload CSV", "Tải CSV lên")],
        horizontal=True,
        key="eda_source",
    )
    uploaded = None
    if source == _t("Upload CSV", "Tải CSV lên"):
        uploaded = st.file_uploader(_t("Choose CSV for exploratory analysis", "Chọn CSV để phân tích khám phá"), type=["csv"], key="eda_csv")
        if uploaded is None:
            st.info(_t("Upload a CSV to start EDA.", "Tải CSV lên để bắt đầu EDA."))
            return

    try:
        if uploaded is None:
            frame = get_reference_data().copy()
            file_info = {
                "source": "Project dataset",
                "filename": DEFAULT_DATA_PATH.name,
                "size_bytes": DEFAULT_DATA_PATH.stat().st_size,
                "encoding": "project file",
                "delimiter": "','",
                "parser": "pandas.read_csv",
            }
        else:
            frame, file_info = _read_uploaded_csv(uploaded)
    except Exception as exc:
        st.error(str(exc))
        return

    if frame.empty:
        st.warning(_t("The selected CSV contains no data rows.", "CSV đã chọn không có dòng dữ liệu."))
        return
    duplicate_headers = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_headers:
        st.error("Duplicate header names detected: " + ", ".join(map(str, duplicate_headers)))
        return

    schema_label = _eda_schema_label(frame)
    profile = _eda_quality_table(frame)
    issue_table = _eda_issue_table(frame)
    analysis_frame = _eda_v3_analysis_frame(frame)
    missing_expected = [column for column in DATASET_COLUMNS if column not in frame.columns]
    unexpected = [column for column in frame.columns if column not in DATASET_COLUMNS]

    with st.expander("CSV intake and schema audit", expanded=True):
        st.dataframe(pd.DataFrame([file_info]), width="stretch", hide_index=True)
        context = pd.DataFrame(
            [
                {
                    "schema_status": schema_label,
                    "rows": len(frame),
                    "columns": len(frame.columns),
                    "missing_full_v3_columns": len(missing_expected),
                    "extra_columns": len(unexpected),
                }
            ]
        )
        st.dataframe(context, width="stretch", hide_index=True)
        if missing_expected:
            st.warning("Missing Full V3 columns: " + ", ".join(missing_expected))
        else:
            st.success("All Full V3 columns are present.")
        if unexpected:
            st.info("Extra columns are visible in EDA but are not part of V3: " + ", ".join(unexpected))
        constant_columns = profile.loc[profile["unique"] <= 1, "column"].tolist()
        if constant_columns:
            st.warning("Constant/near-constant columns: " + ", ".join(constant_columns))
        identifier_columns = [column for column in IDENTIFIER_COLUMNS if column in frame.columns]
        if identifier_columns:
            st.caption("Identifier columns kept for traceability: " + ", ".join(identifier_columns))

    numeric_features = [column for column in ALL_FEATURES if column in frame.columns]
    feature_numeric = frame[numeric_features].apply(pd.to_numeric, errors="coerce")
    missing_cells = int(frame.isna().sum().sum())
    duplicate_rows = int(frame.duplicated().sum())
    negative_cells = int((feature_numeric < 0).sum().sum())
    out_of_range_cells = int(profile["out_of_range_cells"].sum())
    invalid_code_cells = int(profile["invalid_code_cells"].sum())
    invalid_format_cells = int(profile["invalid_format_cells"].sum())
    issue_rows = int(issue_table["source_row"].nunique()) if not issue_table.empty else 0
    bp_invalid_rows = int((issue_table["issue_type"] == "bp_order_violation").sum()) if not issue_table.empty else 0
    issue_rate = (issue_rows / len(frame) * 100) if len(frame) else 0.0
    duplicate_rate = (duplicate_rows / len(frame) * 100) if len(frame) else 0.0

    st.markdown(f"#### {_t('Raw CSV quality overview', 'Tổng quan chất lượng CSV thô')}")
    metrics = st.columns(8)
    metrics[0].metric("Rows", len(frame))
    metrics[1].metric("Columns", len(frame.columns))
    metrics[2].metric("Missing cells", missing_cells)
    metrics[3].metric("Exact duplicates", duplicate_rows)
    metrics[4].metric("Negative cells", negative_cells)
    metrics[5].metric("Rows with issues", issue_rows)
    metrics[6].metric("BP violations", bp_invalid_rows)
    metrics[7].metric("V3 EDA rows", len(analysis_frame))
    st.caption(
        f"V3 range violations: {out_of_range_cells} cells · invalid codes: {invalid_code_cells} cells · "
        f"invalid numeric formats: {invalid_format_cells} cells · issue rate: {issue_rate:.2f}% · "
        f"exact duplicate rate: {duplicate_rate:.2f}%. EDA did not modify any row."
    )
    if bp_invalid_rows:
        st.info(
            f"V3 analysis charts use {len(analysis_frame)} rows after excluding {bp_invalid_rows} "
            "SystolicBP/DiastolicBP violations. The raw audit above still shows every original row."
        )
    st.warning("EDA dùng cho kiểm tra dữ liệu nghiên cứu, không phải chẩn đoán y khoa.")
    st.caption(
        "Chỉ BP violations được loại khỏi các biểu đồ phân tích theo quy ước V3; các lỗi khác vẫn được giữ để EDA không tự ý làm sạch dữ liệu."
    )

    st.markdown(f"#### {_t('Where are the CSV problems?', 'CSV đang có vấn đề ở đâu?')}")
    _plot_eda_field_issues(profile)
    with st.expander("View field-level details", expanded=False):
        st.dataframe(profile, width="stretch", hide_index=True)
        st.download_button(
            "Download field quality profile",
            data=profile.to_csv(index=False).encode("utf-8"),
            file_name="eda_field_quality_profile.csv",
            mime="text/csv",
            key="eda_profile_download",
        )

    st.markdown(f"#### {_t('Row/cell-level CSV issues', 'Lỗi ở cấp dòng/ô trong CSV')}")
    if issue_table.empty:
        st.success("No observed row/cell issue under the V3 audit rules.")
    else:
        issue_counts = issue_table["issue_type"].value_counts().rename("issues").to_frame()
        st.bar_chart(issue_counts, y="issues", x_label="Issue type", y_label="Occurrences", height=280)
        st.caption("One row can have multiple issue types, so occurrence count can exceed the number of affected rows.")
        with st.expander("View row/cell issue details", expanded=False):
            st.dataframe(issue_table.head(500), width="stretch", hide_index=True)
            if len(issue_table) > 500:
                st.caption(f"Showing first 500 of {len(issue_table)} issue records; the download contains all records.")
            st.download_button(
                "Download CSV issue audit",
                data=issue_table.to_csv(index=False).encode("utf-8"),
                file_name="eda_quality_issues.csv",
                mime="text/csv",
                key="eda_issues_download",
            )

    with st.expander("View raw data preview", expanded=False):
        st.dataframe(frame.head(100), width="stretch", hide_index=True)

    if TARGET_COLUMN in frame.columns:
        st.markdown(f"#### {_t('Target balance', 'Cân bằng biến mục tiêu')}")
        target = pd.to_numeric(analysis_frame[TARGET_COLUMN], errors="coerce")
        diagnosis_counts = (
            target.map({0: "0 — No diagnosis", 1: "1 — Diagnosis"})
            .fillna("Missing/invalid")
            .value_counts()
            .rename("rows")
            .to_frame()
        )
        st.bar_chart(diagnosis_counts, y="rows", x_label="Diagnosis", y_label="Rows", height=260)
        st.caption(
            f"n={len(analysis_frame)} V3 analysis rows after BP exclusions. Diagnosis is descriptive here "
            "and is ignored during screening prediction."
        )
    else:
        st.info("Diagnosis is not present; target balance and group comparisons are unavailable for this CSV.")

    st.markdown("#### " + _t("Cohen's d feature separation", "Mức tách biệt đặc trưng theo Cohen's d"))
    effect_table = _cohens_d_table(analysis_frame)
    if effect_table.empty:
        st.info("Cohen's d requires a valid binary Diagnosis column and at least two observations per group.")
    else:
        _plot_eda_cohens_d(effect_table)
        n_no_diagnosis = int(effect_table.iloc[0]["n_no_diagnosis"])
        n_diagnosis = int(effect_table.iloc[0]["n_diagnosis"])
        st.caption(
            f"n0={n_no_diagnosis}, n1={n_diagnosis}, analysis rows={len(analysis_frame)}. "
            "Magnitude guide: |d| < 0.2 negligible; 0.2–0.5 small; 0.5–0.8 moderate; > 0.8 large."
        )
        with st.expander("View Cohen's d values", expanded=False):
            st.dataframe(effect_table.head(15), width="stretch", hide_index=True)
            st.download_button(
                "Download Cohen's d table",
                data=effect_table.to_csv(index=False).encode("utf-8"),
                file_name="eda_cohens_d.csv",
                mime="text/csv",
                key="eda_cohens_download",
            )

    st.markdown(f"#### {_t('Pearson correlation matrix', 'Ma trận tương quan Pearson')}")
    correlation_columns = [column for column in ALL_FEATURES if column in frame.columns]
    if len(correlation_columns) >= 2:
        correlation = analysis_frame[correlation_columns].apply(pd.to_numeric, errors="coerce").corr(min_periods=2).round(3)
        _plot_eda_heatmap(correlation)
        st.caption(f"Pearson correlation on n={len(analysis_frame)} V3 analysis rows after BP exclusions.")
        with st.expander("View correlation values", expanded=False):
            st.dataframe(correlation, width="stretch")
        st.download_button(
            "Download correlation matrix",
            data=correlation.to_csv().encode("utf-8"),
            file_name="eda_pearson_correlation.csv",
            mime="text/csv",
            key="eda_correlation_download",
        )
    else:
        st.info("At least two V3 features are required for a correlation matrix.")

    st.markdown(f"#### {_t('Final five features by Diagnosis', '5 đặc trưng cuối theo Diagnosis')}")
    _plot_eda_boxplots(analysis_frame)
    if TARGET_COLUMN in analysis_frame.columns:
        boxplot_target = pd.to_numeric(analysis_frame[TARGET_COLUMN], errors="coerce")
        st.caption(
            f"Boxplots use n0={int(boxplot_target.eq(0).sum())} and n1={int(boxplot_target.eq(1).sum())} "
            f"valid labels from {len(analysis_frame)} BP-valid analysis rows."
        )

    if {"SystolicBP", "DiastolicBP"}.issubset(frame.columns):
        st.markdown(f"#### {_t('Systolic vs diastolic blood pressure', 'Huyết áp tâm thu và tâm trương')}")
        _plot_eda_bp_scatter(frame)
        st.caption("Rows on or below the diagonal violate the V3 rule SystolicBP > DiastolicBP.")


@st.cache_data(show_spinner=False)
def _unsupervised_learning_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    """Fit PCA and K-Means on the 32 predictors without using Diagnosis."""

    missing_features = [feature for feature in ALL_FEATURES if feature not in frame.columns]
    if missing_features:
        raise ValueError(
            "Unsupervised analysis requires all 32 V3 features. Missing: "
            + ", ".join(missing_features)
        )

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    feature_frame = frame[ALL_FEATURES].apply(pd.to_numeric, errors="coerce")
    empty_features = [feature for feature in ALL_FEATURES if feature_frame[feature].notna().sum() == 0]
    if empty_features:
        raise ValueError(
            "These features contain no usable numeric values: " + ", ".join(empty_features)
        )

    imputed_values = SimpleImputer(strategy="median").fit_transform(feature_frame)
    standardized_values = StandardScaler().fit_transform(imputed_values)
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(standardized_values)

    diagnostics: list[dict[str, float | int]] = []
    labels_by_k: dict[int, Any] = {}
    for cluster_count in range(2, 7):
        estimator = KMeans(n_clusters=cluster_count, random_state=42, n_init=20)
        labels = estimator.fit_predict(standardized_values)
        labels_by_k[cluster_count] = labels
        diagnostics.append(
            {
                "k": cluster_count,
                "inertia": float(estimator.inertia_),
                "silhouette": float(silhouette_score(standardized_values, labels)),
            }
        )

    diagnostics_frame = pd.DataFrame(diagnostics)
    best_row = diagnostics_frame.sort_values(
        ["silhouette", "k"], ascending=[False, True]
    ).iloc[0]
    best_k = int(best_row["k"])
    best_labels = labels_by_k[best_k]
    cluster_names = pd.Series(best_labels, index=frame.index).map(
        lambda value: f"Cụm {int(value) + 1}"
    )

    pca_frame = pd.DataFrame(
        {
            "PCA 1": coordinates[:, 0],
            "PCA 2": coordinates[:, 1],
            "Cụm": cluster_names.to_numpy(),
        },
        index=frame.index,
    )
    imputed_frame = pd.DataFrame(imputed_values, columns=ALL_FEATURES, index=frame.index)
    profile_frame = imputed_frame.assign(Cụm=cluster_names)
    cluster_sizes = profile_frame.groupby("Cụm", sort=True).size().rename("Số bệnh nhân")
    cluster_profile = (
        profile_frame.groupby("Cụm", sort=True)[ALL_FEATURES]
        .mean()
        .round(3)
        .join(cluster_sizes)
        .reset_index()
    )
    profile_columns = ["Cụm", "Số bệnh nhân", *ALL_FEATURES]
    cluster_profile = cluster_profile[profile_columns]

    diagnosis_comparison = pd.DataFrame()
    if TARGET_COLUMN in frame.columns:
        diagnosis = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        comparison = pd.DataFrame(
            {"Cụm": cluster_names, TARGET_COLUMN: diagnosis}, index=frame.index
        )
        comparison = comparison[comparison[TARGET_COLUMN].isin([0, 1])]
        if not comparison.empty:
            diagnosis_comparison = (
                comparison.groupby("Cụm", sort=True)[TARGET_COLUMN]
                .agg(**{"Số bệnh nhân": "size", "Ca Alzheimer": "sum", "Tỷ lệ Alzheimer": "mean"})
                .reset_index()
            )
            diagnosis_comparison["Ca Alzheimer"] = diagnosis_comparison[
                "Ca Alzheimer"
            ].astype(int)
            diagnosis_comparison["Tỷ lệ Alzheimer"] = diagnosis_comparison[
                "Tỷ lệ Alzheimer"
            ].astype(float)

    return {
        "feature_count": len(ALL_FEATURES),
        "rows": len(frame),
        "imputed_cells": int(feature_frame.isna().sum().sum()),
        "diagnosis_used_for_fit": False,
        "diagnostics": diagnostics_frame,
        "best_k": best_k,
        "pca_explained_variance": pca.explained_variance_ratio_,
        "pca_frame": pca_frame,
        "cluster_profile": cluster_profile,
        "diagnosis_comparison": diagnosis_comparison,
    }


def _plot_unsupervised_diagnostics(diagnostics: pd.DataFrame, best_k: int) -> None:
    """Show the elbow and silhouette evidence used to compare k=2 through k=6."""

    try:
        import altair as alt
    except ImportError:
        left, right = st.columns(2)
        left.line_chart(diagnostics, x="k", y="inertia", height=300)
        right.line_chart(diagnostics, x="k", y="silhouette", height=300)
        return

    best_reference = pd.DataFrame({"k": [best_k]})

    def line_chart(metric: str, title: str, y_title: str, color: str) -> Any:
        line = (
            alt.Chart(diagnostics)
            .mark_line(point=alt.OverlayMarkDef(filled=True, size=70), color=color, strokeWidth=2.5)
            .encode(
                x=alt.X("k:O", title="Số cụm k", sort=[2, 3, 4, 5, 6]),
                y=alt.Y(f"{metric}:Q", title=y_title, scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("k:O", title="k"),
                    alt.Tooltip(f"{metric}:Q", title=y_title, format=".4f"),
                ],
            )
        )
        reference = (
            alt.Chart(best_reference)
            .mark_rule(color="#475569", strokeDash=[5, 4])
            .encode(x="k:O")
        )
        return (line + reference).properties(title=title, height=285)

    left, right = st.columns(2)
    with left:
        st.altair_chart(
            line_chart("inertia", "Elbow — độ nén trong cụm", "Inertia", "#6C9BC3"),
            width="stretch",
        )
    with right:
        st.altair_chart(
            line_chart(
                "silhouette",
                "Silhouette — độ tách biệt giữa các cụm",
                "Silhouette Score",
                "#2F7F73",
            ),
            width="stretch",
        )


def _plot_pca_clusters(pca_frame: pd.DataFrame) -> None:
    """Plot the two-dimensional PCA projection colored only by K-Means cluster."""

    try:
        import altair as alt
    except ImportError:
        st.scatter_chart(pca_frame, x="PCA 1", y="PCA 2", color="Cụm", height=480)
        return

    palette = ["#2F7F73", "#6C9BC3", "#F2A900", "#9B8AFB", "#D28C63", "#6B7280"]
    chart = (
        alt.Chart(pca_frame.reset_index(drop=True))
        .mark_circle(size=48, opacity=0.62, stroke="#FFFFFF", strokeWidth=0.25)
        .encode(
            x=alt.X("PCA 1:Q", title="Thành phần chính 1"),
            y=alt.Y("PCA 2:Q", title="Thành phần chính 2"),
            color=alt.Color(
                "Cụm:N",
                title="Cụm K-Means",
                scale=alt.Scale(range=palette),
            ),
            tooltip=[
                alt.Tooltip("Cụm:N"),
                alt.Tooltip("PCA 1:Q", format=".3f"),
                alt.Tooltip("PCA 2:Q", format=".3f"),
            ],
        )
        .properties(
            title="PCA hai chiều theo cụm K-Means — không dùng Diagnosis khi huấn luyện",
            height=470,
        )
        .interactive()
    )
    st.altair_chart(chart, width="stretch")


def _plot_cluster_diagnosis_comparison(comparison: pd.DataFrame) -> None:
    """Compare learned clusters with Diagnosis only after unsupervised fitting."""

    try:
        import altair as alt
    except ImportError:
        st.bar_chart(comparison, x="Cụm", y="Tỷ lệ Alzheimer", height=300)
        return

    bars = (
        alt.Chart(comparison)
        .mark_bar(color="#2F7F73", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Cụm:N", title="Cụm K-Means", sort=None),
            y=alt.Y(
                "Tỷ lệ Alzheimer:Q",
                title="Tỷ lệ Diagnosis = 1",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                alt.Tooltip("Cụm:N"),
                alt.Tooltip("Số bệnh nhân:Q", format="d"),
                alt.Tooltip("Ca Alzheimer:Q", format="d"),
                alt.Tooltip("Tỷ lệ Alzheimer:Q", format=".1%"),
            ],
        )
    )
    labels = bars.mark_text(dy=-9, color="#0F172A").encode(
        text=alt.Text("Tỷ lệ Alzheimer:Q", format=".1%")
    )
    st.altair_chart(
        (bars + labels).properties(
            title="Đối chiếu cụm với nhãn Diagnosis sau phân cụm",
            height=310,
        ),
        width="stretch",
    )


def _unsupervised_learning_tab() -> None:
    """Render PCA and K-Means evidence on the cleaned full V3 dataset."""

    st.subheader(_t("Unsupervised learning", "Học không giám sát"))
    st.caption(
        _t(
            "Explore latent patient groups with standardized predictors, PCA, and K-Means. Diagnosis is excluded from every fitting step.",
            "Khám phá các nhóm bệnh nhân tiềm ẩn bằng dữ liệu chuẩn hóa, PCA và K-Means. Diagnosis bị loại khỏi toàn bộ quá trình huấn luyện.",
        )
    )
    st.info(
        _t(
            "Method boundary: the clusters support exploratory interpretation only. They do not replace XGBoost, create a medical diagnosis, or change the deployed model.",
            "Ranh giới phương pháp: các cụm chỉ phục vụ phân tích khám phá. Chúng không thay thế XGBoost, không tạo chẩn đoán y khoa và không làm thay đổi mô hình đang triển khai.",
        )
    )

    try:
        raw_frame = get_reference_data().copy()
        cleaned_frame, cleaning_report = clean_dataset(raw_frame)
        analysis = _unsupervised_learning_analysis(cleaned_frame)
    except Exception as exc:
        st.error(str(exc))
        return

    summary = st.columns(4)
    summary[0].metric(_t("Clean rows", "Dòng đã làm sạch"), analysis["rows"])
    summary[1].metric(_t("Predictors", "Đặc trưng đầu vào"), analysis["feature_count"])
    summary[2].metric(_t("Compared k", "Số k đã thử"), "2–6")
    summary[3].metric(_t("Best silhouette k", "k tốt nhất theo Silhouette"), analysis["best_k"])
    st.caption(
        _t(
            f"Source: {DEFAULT_DATA_PATH.name}. Removed {cleaning_report['invalid_blood_pressure_rows_removed']} BP-order violations and {cleaning_report['duplicates_removed']} duplicates; median-imputed {analysis['imputed_cells']} non-numeric/missing feature cells before standardization.",
            f"Nguồn: {DEFAULT_DATA_PATH.name}. Đã loại {cleaning_report['invalid_blood_pressure_rows_removed']} dòng sai thứ tự huyết áp và {cleaning_report['duplicates_removed']} dòng trùng; điền median cho {analysis['imputed_cells']} ô đặc trưng thiếu/không phải số trước khi chuẩn hóa.",
        )
    )

    st.markdown("#### " + _t("1. Standardize 32 predictors", "1. Chuẩn hóa 32 đặc trưng"))
    st.write(
        _t(
            "PatientID, DoctorInCharge, and Diagnosis are excluded. The 32 numeric/coded predictors are median-imputed when needed and standardized to mean 0 and standard deviation 1.",
            "PatientID, DoctorInCharge và Diagnosis được loại bỏ. 32 đặc trưng dạng số/mã được điền median khi cần, sau đó chuẩn hóa về trung bình 0 và độ lệch chuẩn 1.",
        )
    )
    st.success(
        _t(
            "Leakage check passed: Diagnosis was not used by StandardScaler, PCA, K-Means, or the k-selection metrics.",
            "Kiểm tra rò rỉ nhãn đạt yêu cầu: Diagnosis không được dùng trong StandardScaler, PCA, K-Means hoặc chỉ số chọn k.",
        )
    )

    st.markdown("#### " + _t("2. Select the number of clusters", "2. Chọn số lượng cụm"))
    _plot_unsupervised_diagnostics(analysis["diagnostics"], analysis["best_k"])
    st.dataframe(
        analysis["diagnostics"].rename(
            columns={"k": "Số cụm k", "inertia": "Inertia", "silhouette": "Silhouette Score"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        _t(
            "Elbow shows within-cluster compression; the highest Silhouette Score is used as the explicit selection rule.",
            "Elbow thể hiện độ nén trong cụm; Silhouette Score cao nhất được dùng làm quy tắc chọn k minh bạch.",
        )
    )
    best_silhouette = float(
        analysis["diagnostics"].loc[
            analysis["diagnostics"]["k"].eq(analysis["best_k"]), "silhouette"
        ].iloc[0]
    )
    if best_silhouette < 0.25:
        st.warning(
            _t(
                f"The best Silhouette Score is only {best_silhouette:.3f}, indicating substantial cluster overlap. Treat k={analysis['best_k']} as the best exploratory option among k=2–6, not as evidence of stable clinical subtypes.",
                f"Silhouette Score tốt nhất chỉ đạt {best_silhouette:.3f}, cho thấy các cụm còn chồng lấn nhiều. Chỉ xem k={analysis['best_k']} là phương án khám phá tốt nhất trong k=2–6, không phải bằng chứng về các phân nhóm lâm sàng ổn định.",
            )
        )

    st.markdown("#### " + _t("3. PCA two-dimensional projection", "3. Trực quan hóa PCA hai chiều"))
    _plot_pca_clusters(analysis["pca_frame"])
    explained = analysis["pca_explained_variance"]
    st.caption(
        _t(
            f"PCA 1 explains {float(explained[0]):.1%} and PCA 2 explains {float(explained[1]):.1%} of standardized-feature variance. Point color represents only the learned cluster.",
            f"PCA 1 giải thích {float(explained[0]):.1%} và PCA 2 giải thích {float(explained[1]):.1%} phương sai của dữ liệu đã chuẩn hóa. Màu điểm chỉ thể hiện cụm đã học.",
        )
    )

    st.markdown("#### " + _t("4. Cluster profile", "4. Đặc điểm trung bình của từng cụm"))
    st.dataframe(analysis["cluster_profile"], width="stretch", hide_index=True)
    st.caption(
        _t(
            "Values are cluster means on the original feature scales after median imputation; use them to describe each cluster, not to assign a diagnosis.",
            "Các giá trị là trung bình theo cụm trên thang đo gốc sau khi điền median; dùng để mô tả cụm, không dùng để gán chẩn đoán.",
        )
    )

    st.markdown(
        "#### "
        + _t(
            "5. Post-hoc comparison with Diagnosis",
            "5. Đối chiếu với Diagnosis sau khi phân cụm",
        )
    )
    diagnosis_comparison = analysis["diagnosis_comparison"]
    if diagnosis_comparison.empty:
        st.info(
            _t(
                "A valid binary Diagnosis column is unavailable, so the post-hoc comparison cannot be shown.",
                "Không có cột Diagnosis nhị phân hợp lệ nên chưa thể thực hiện đối chiếu sau phân cụm.",
            )
        )
    else:
        _plot_cluster_diagnosis_comparison(diagnosis_comparison)
        display_comparison = diagnosis_comparison.copy()
        display_comparison["Tỷ lệ Alzheimer"] = display_comparison[
            "Tỷ lệ Alzheimer"
        ].map(lambda value: f"{float(value):.1%}")
        st.dataframe(display_comparison, width="stretch", hide_index=True)
        st.warning(
            _t(
                "Diagnosis is introduced only in this final comparison. Any difference between clusters is descriptive and does not prove clinical validity or causality.",
                "Diagnosis chỉ được đưa vào ở bước đối chiếu cuối cùng này. Khác biệt giữa các cụm chỉ mang tính mô tả, không chứng minh giá trị lâm sàng hoặc quan hệ nhân quả.",
            )
        )


def _eda_tab() -> None:
    """Keep supervised EDA evidence and unsupervised exploration separate."""

    evidence_tab, unsupervised_tab = st.tabs(
        [
            _t("EDA and data quality", "EDA & chất lượng dữ liệu"),
            _t("Unsupervised learning", "Học không giám sát"),
        ]
    )
    with evidence_tab:
        _eda_quality_tab()
    with unsupervised_tab:
        _unsupervised_learning_tab()


def _batch_processing_tab(
    payload: dict, metadata: dict, user: AuthUser, *, use_project_dataset: bool = False
) -> None:
    """Clean, validate, audit, and screen an upload or the project dataset."""

    uploaded = None
    if use_project_dataset:
        st.subheader(_t("Project dataset", "Dataset dự án"))
        st.caption(
            _t(
                f"Use the bundled {DEFAULT_DATA_PATH.name}, or upload another CSV. This tab checks, cleans, and exports data; it does not predict or retrain.",
                f"Dùng file có sẵn {DEFAULT_DATA_PATH.name}, hoặc tải CSV khác lên. Tab này kiểm tra, làm sạch và xuất dữ liệu; không dự đoán hay huấn luyện lại.",
            )
        )
        uploaded = st.file_uploader(
            _t("Upload CSV input for data processing", "Tải CSV đầu vào để xử lý dữ liệu"),
            type=["csv"],
            key="project_dataset_csv",
        )
        st.caption(_t(
            "Leave this empty to use the bundled dataset. Uploaded CSV limit: 10,000 rows and 10 MB.",
            "Để trống để dùng dataset có sẵn. CSV tải lên tối đa 10.000 dòng và 10 MB.",
        ))
        if uploaded is None:
            st.info(_t(
                "This is the project's existing data, not an independent external validation set. Results here are for workflow inspection only.",
                "Đây là dữ liệu có sẵn của dự án, không phải tập kiểm định bên ngoài độc lập. Kết quả chỉ dùng để xem quy trình xử lý.",
            ))
        else:
            st.info(_t(
                "The uploaded CSV replaces the bundled dataset for this processing run. If Diagnosis is present, its labels must be 0 or 1; no prediction is made here.",
                "CSV vừa tải lên thay cho dataset có sẵn ở lần xử lý này. Nếu có cột Diagnosis, nhãn phải là 0 hoặc 1; tab này không chạy dự đoán.",
            ))
    else:
        st.subheader(_t("Batch prediction from CSV", "Dự đoán hàng loạt từ CSV"))
        st.caption(
            _t(
                "Upload one case per row with the five selected model inputs. PatientID is optional; Full V3 files belong in Project dataset.",
                "Mỗi dòng là một ca với đúng 5 chỉ số mô hình. PatientID có thể thêm; CSV Full V3 dùng ở tab Dataset dự án.",
            )
        )
        uploaded = st.file_uploader(
            _t("Upload CSV for batch prediction", "Tải CSV lên để dự đoán hàng loạt"),
            type=["csv"],
            key="processing_csv",
            help=_t("The extension check is only a best-effort check; the parser diagnostics below are authoritative.", "Kiểm tra phần mở rộng chỉ mang tính tham khảo; thông tin parser bên dưới mới là kết quả chính thức."),
        )
        st.caption(_t(
            "App limit: 10,000 rows and 10 MB per CSV. The uploader's 200 MB label is a Streamlit transfer limit, not this app's processing limit.",
            "Giới hạn xử lý của app: 10.000 dòng và 10 MB mỗi CSV. Dòng 200 MB ở ô tải file là giới hạn truyền file của Streamlit, không phải giới hạn xử lý của app.",
        ))
    with st.expander(
        _t("V3 validation rules used in this tab", "Quy tắc kiểm tra V3 dùng trong tab này"),
        expanded=False,
    ):
        language = st.session_state.get("language", "vi")
        st.dataframe(
            _validation_rule_summary(language, five_feature_only=not use_project_dataset),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            _t(
                "Missing values are imputed only when the row still contains usable model evidence; a row missing all five model inputs is rejected.",
                "Chỉ điền giá trị thiếu khi dòng vẫn còn tín hiệu mô hình có thể sử dụng; dòng thiếu cả 5 đầu vào sẽ bị loại.",
            )
        )

    schema = None
    template_columns = None
    if not use_project_dataset:
        schema = "Five-feature scoring CSV"
        template_columns = _template_columns("Screening prediction", schema)
        with st.expander(_t("CSV template and allowed-value dictionary", "Mẫu CSV và từ điển giá trị hợp lệ"), expanded=False):
            template = pd.DataFrame([{column: "" for column in template_columns}])
            st.download_button(
                _t("Download CSV template", "Tải mẫu CSV"),
                data=template.to_csv(index=False).encode("utf-8"),
                file_name="alzheimer_csv_template.csv",
                mime="text/csv",
                key="processing_template_download",
            )
            st.dataframe(
                _schema_guide(template_columns),
                width="stretch",
                hide_index=True,
            )
            st.caption(_t(
                "PatientID may be added for row tracking; it is not a model input. Do not include Diagnosis or other Full V3 columns here.",
                "Có thể thêm PatientID để theo dõi dòng; đây không phải đầu vào mô hình. Không thêm Diagnosis hoặc cột Full V3 khác ở tab này.",
            ))

    if uploaded is None and not use_project_dataset:
        st.info(_t("Upload a CSV to start processing.", "Tải CSV lên để bắt đầu xử lý."))
        return

    try:
        if uploaded is None:
            raw = get_reference_data().copy()
            file_info = {
                "source": "Project dataset",
                "filename": DEFAULT_DATA_PATH.name,
                "size_bytes": DEFAULT_DATA_PATH.stat().st_size,
                "encoding": "project file",
                "delimiter": "','",
                "parser": "pandas.read_csv",
            }
        else:
            raw, file_info = _read_uploaded_csv(
                uploaded,
                required_columns=SELECTED_FEATURES if use_project_dataset else template_columns,
                language=st.session_state.get("language", "vi"),
            )
    except Exception as exc:
        st.error(str(exc))
        return

    if use_project_dataset:
        try:
            schema = _detect_v3_schema(raw.columns)
        except ValueError as exc:
            st.error(str(exc))
            return
        template_columns = ALL_FEATURES if schema == "Full V3 scoring CSV" else SELECTED_FEATURES
        st.success(_t(
            f"Detected input structure: {'Full V3 (32 clinical features)' if schema == 'Full V3 scoring CSV' else '5 model features'}. No manual schema selection is needed.",
            f"Đã nhận diện cấu trúc đầu vào: {'Full V3 (32 chỉ số)' if schema == 'Full V3 scoring CSV' else '5 đặc trưng mô hình'}. Không cần chọn schema thủ công.",
        ))
    else:
        try:
            _validate_five_feature_scoring_columns(
                raw.columns, language=st.session_state.get("language", "vi")
            )
        except ValueError as exc:
            st.error(str(exc))
            return

    required_columns = template_columns
    missing_columns = [column for column in required_columns if column not in raw.columns]
    allowed_columns = set(
        DATASET_COLUMNS
        if use_project_dataset or schema == "Full V3 scoring CSV"
        else [*template_columns, "PatientID"]
    )
    unexpected_columns = [column for column in raw.columns if column not in allowed_columns]
    with st.expander(_t("File and schema diagnostics", "Chẩn đoán file và schema"), expanded=True):
        st.dataframe(pd.DataFrame([file_info]), width="stretch", hide_index=True)
        if missing_columns:
            st.error(_t("Missing required columns: ", "Thiếu các cột bắt buộc: ") + ", ".join(missing_columns))
        else:
            st.success(_t(f"Schema check passed: {len(required_columns)} required columns found.", f"Kiểm tra schema đạt: tìm thấy {len(required_columns)} cột bắt buộc."))
        if unexpected_columns:
            if use_project_dataset:
                st.info(
                    _t(
                        "Additional columns are retained in the processed CSV; V3 rules apply only to recognized fields: ",
                        "Các cột bổ sung được giữ trong CSV đầu ra; quy tắc V3 chỉ áp dụng cho trường đã nhận diện: ",
                    )
                    + ", ".join(unexpected_columns)
                )
            else:
                st.warning(
                    _t(
                        "Unexpected columns are preserved for traceability but ignored by the model: ",
                        "Các cột không dự kiến được giữ để truy vết nhưng bị mô hình bỏ qua: ",
                    )
                    + ", ".join(unexpected_columns)
                )
    if missing_columns:
        return

    if use_project_dataset:
        with st.expander(_t("CSV input preview", "Xem trước CSV đầu vào"), expanded=False):
            st.dataframe(raw.head(100), width="stretch", hide_index=True)
            st.caption(_t(
                "This is the original input before duplicate removal, validation, or imputation.",
                "Đây là dữ liệu gốc trước khi loại dòng trùng, kiểm tra hoặc điền giá trị thiếu.",
            ))

    raw.insert(0, "_source_row", range(1, len(raw) + 1))
    raw_values = raw.drop(columns=["_source_row"])
    reference = get_reference_data()
    imputation_statistics = payload.get("imputation_statistics") or metadata.get(
        "imputation_statistics"
    )
    imputation_source = "active model artifact"
    if not imputation_statistics:
        reference_clean, _ = clean_dataset(reference)
        imputation_statistics = fit_imputation_statistics(reference_clean)
        imputation_source = "fallback from cleaned project reference data; retrain artifact to persist it"

    raw_rows = len(raw)
    raw_columns = len(raw_values.columns)
    missing_cells = int(raw_values.isna().sum().sum())
    duplicate_rows = int(raw_values.duplicated().sum())
    numeric_raw = raw_values.apply(pd.to_numeric, errors="coerce")
    negative_cells = int((numeric_raw < 0).sum().sum())
    duplicate_patient_ids = 0
    if "PatientID" in raw.columns:
        patient_ids = raw["PatientID"].astype("string").str.strip()
        duplicate_patient_ids = int(
            (patient_ids.notna() & patient_ids.ne("") & patient_ids.duplicated(keep=False)).sum()
        )

    try:
        is_full_schema = schema == "Full V3 scoring CSV"
        validation_mode = (
            "training" if use_project_dataset and TARGET_COLUMN in raw.columns else "screening"
        )
        if is_full_schema:
            cleaned, clean_report = clean_dataset(
                raw,
                require_target=validation_mode == "training",
            )
            processing_mode = _t(
                "Full V3 schema: duplicate and impossible BP rows removed before validation.",
                "Schema Full V3: dòng trùng và dòng vi phạm quy tắc huyết áp được loại trước khi kiểm tra.",
            )
        else:
            duplicate_mask = raw.drop(columns=["_source_row"]).duplicated()
            cleaned = raw.loc[~duplicate_mask].reset_index(drop=True)
            clean_report = {
                "rows_before": raw_rows,
                "rows_after": len(cleaned),
                "duplicates_removed": duplicate_rows,
                "invalid_blood_pressure_rows_removed": 0,
                "missing_cells_after_cleaning": missing_cells,
            }
            processing_mode = _t(
                "Five-feature path: exact duplicate rows removed before validation.",
                "Luồng 5 đặc trưng: loại dòng trùng hoàn toàn trước khi kiểm tra dữ liệu.",
            )
        validation = validate_dataframe(
            cleaned,
            reference_data=reference,
            mode=validation_mode,
            imputation_statistics=imputation_statistics,
        )
        cleaning_removed = _cleaning_removed_rows(
            raw, cleaned, full_schema=is_full_schema
        )
    except Exception as exc:
        st.error(_t(f"Could not process this CSV: {exc}", f"Không thể xử lý CSV này: {exc}"))
        return

    cleaning_issues = pd.DataFrame(
        [
            {
                "row": int(row["_source_row"]),
                "field": (
                    "SystolicBP/DiastolicBP"
                    if row["_removal_reason"] == "SystolicBP <= DiastolicBP"
                    else "row"
                ),
                "issue": row["_removal_reason"],
                "action": "removed before validation",
            }
            for _, row in cleaning_removed.iterrows()
        ],
        columns=["row", "field", "issue", "action"],
    )
    processing_issues = pd.concat(
        [cleaning_issues, validation.issues], ignore_index=True
    )
    validation_rejected = validation.rejected.drop(
        columns=[column for column in VALIDATION_META_COLUMNS if column != "_row_number"],
        errors="ignore",
    ).copy()
    validation_rejected["_removal_reason"] = "validation rule (see log)"
    rejected_download = pd.concat(
        [
            cleaning_removed.rename(columns={"_source_row": "_row_number"}),
            validation_rejected,
        ],
        ignore_index=True,
        sort=False,
    ).sort_values("_row_number").rename(
        columns={"_row_number": "source_row", "_removal_reason": "reason"}
    )
    total_rejected_rows = len(rejected_download)
    if raw_rows != len(validation.valid) + total_rejected_rows:
        st.error(_t(
            "Input/output row counts do not reconcile; exports were stopped.",
            "Số dòng đầu vào và đầu ra không khớp; hệ thống đã dừng xuất file.",
        ))
        return

    st.info(processing_mode)
    imputation_source_vi = (
        "artifact mô hình đang dùng"
        if imputation_source == "active model artifact"
        else "dự phòng từ dataset dự án đã làm sạch; hãy huấn luyện lại để lưu vào artifact"
    )
    st.caption(_t(f"Imputation source: {imputation_source}", f"Nguồn điền thiếu: {imputation_source_vi}"))
    rows_after_dedup = raw_rows - int(clean_report["duplicates_removed"])
    format_error_count = int(
        validation.issues["issue"].eq("invalid format").sum()
        if not validation.issues.empty
        else 0
    )
    flow_step = _t("Step", "Bước")
    flow_rows_after = _t("Rows after step", "Số dòng sau bước")
    flow_action = _t("Action", "Hành động")
    with st.container(border=True):
        st.markdown("#### " + _t("Processing flow", "Luồng xử lý"))
        st.caption(_t("The table traces the input CSV through cleaning, validation, and output.", "Bảng đối chiếu CSV đầu vào qua từng bước làm sạch, kiểm tra và xuất đầu ra."))
        flow_rows = [
            {
                flow_step: _t("1. Load CSV", "1. Đọc CSV"),
                flow_rows_after: raw_rows,
                flow_action: _t(
                    f"Loaded {raw_rows} rows and {raw_columns} columns",
                    f"Đã đọc {raw_rows} dòng và {raw_columns} cột",
                ),
            },
            {
                flow_step: _t("2. Profile", "2. Lập hồ sơ dữ liệu"),
                flow_rows_after: raw_rows,
                flow_action: _t(
                    (
                        f"Found {missing_cells} parser-missing cells, {negative_cells} negative cells, "
                        f"{duplicate_rows} exact duplicates, and {duplicate_patient_ids} duplicate PatientID cells"
                    ),
                    (
                        f"Phát hiện {missing_cells} ô thiếu do parser, {negative_cells} ô âm, "
                        f"{duplicate_rows} dòng trùng hoàn toàn và {duplicate_patient_ids} ô PatientID bị trùng"
                    ),
                ),
            },
            {
                flow_step: _t("3. Remove duplicates", "3. Loại dòng trùng"),
                flow_rows_after: rows_after_dedup,
                flow_action: _t(
                    f"Removed {clean_report['duplicates_removed']} exact duplicate rows",
                    f"Đã loại {clean_report['duplicates_removed']} dòng trùng hoàn toàn",
                ),
            },
        ]
        if is_full_schema:
            bp_removed = int(clean_report["invalid_blood_pressure_rows_removed"])
            flow_rows.append(
                {
                    flow_step: _t("4. V3 BP rule", "4. Quy tắc huyết áp V3"),
                    flow_rows_after: clean_report["rows_after"],
                    flow_action: _t(
                        f"Removed {bp_removed} impossible BP rows",
                        f"Đã loại {bp_removed} dòng huyết áp không hợp lệ",
                    ),
                }
            )
        flow_rows.extend(
            [
                {
                    flow_step: _t(
                        f"{5 if is_full_schema else 4}. Validate + impute",
                        f"{5 if is_full_schema else 4}. Kiểm tra và điền thiếu",
                    ),
                    flow_rows_after: validation.summary["valid_rows"],
                    flow_action: _t(
                        (
                            f"Imputed {validation.summary['imputed_issue_count']} cells in "
                            f"{validation.summary['imputed_rows']} rows; parsed {format_error_count} format errors; "
                            f"rejected {validation.summary['rejected_rows']} rows"
                        ),
                        (
                            f"Đã điền {validation.summary['imputed_issue_count']} ô ở {validation.summary['imputed_rows']} dòng; "
                            f"phát hiện {format_error_count} lỗi định dạng; loại {validation.summary['rejected_rows']} dòng"
                        ),
                    ),
                },
                {
                    flow_step: _t(
                        f"{6 if is_full_schema else 5}. Export audit outputs",
                        f"{6 if is_full_schema else 5}. Xuất kết quả kiểm tra",
                    ),
                    flow_rows_after: validation.summary["valid_rows"],
                    flow_action: _t(
                        (
                            "Cleaned CSV, complete log, and rejected rows are available below"
                            if use_project_dataset
                            else "Cleaned CSV, complete log, rejected rows, and predictions are available below"
                        ),
                        (
                            "Có thể tải CSV sạch, nhật ký đầy đủ và dòng bị loại ở bên dưới"
                            if use_project_dataset
                            else "Có thể tải CSV sạch, nhật ký đầy đủ, dòng bị loại và kết quả dự đoán ở bên dưới"
                        ),
                    ),
                },
            ]
        )
        st.dataframe(
            pd.DataFrame(flow_rows),
            width="stretch",
            hide_index=True,
        )

    first_row = st.columns(5)
    first_row[0].metric(_t("Input rows", "Dòng đầu vào"), raw_rows)
    first_row[1].metric(_t("Input columns", "Cột đầu vào"), raw_columns)
    first_row[2].metric(_t("Missing cells", "Ô bị thiếu"), missing_cells)
    first_row[3].metric(_t("Duplicate rows", "Dòng trùng"), duplicate_rows)
    first_row[4].metric(_t("Negative cells", "Ô có giá trị âm"), negative_cells)

    second_row = st.columns(5 if is_full_schema else 4)
    metric_index = 0
    if is_full_schema:
        second_row[metric_index].metric(
            _t("BP rows removed", "Dòng huyết áp bị loại"),
            clean_report["invalid_blood_pressure_rows_removed"],
        )
        metric_index += 1
    second_row[metric_index].metric(_t("Rows after cleaning", "Dòng sau làm sạch"), clean_report["rows_after"])
    second_row[metric_index + 1].metric(_t("Imputed rows", "Dòng được điền thiếu"), validation.summary["imputed_rows"])
    second_row[metric_index + 2].metric(_t("Valid rows", "Dòng hợp lệ"), validation.summary["valid_rows"])
    second_row[metric_index + 3].metric(_t("Rejected rows", "Dòng bị loại"), total_rejected_rows)
    st.caption(_t(
        f"Row check: {raw_rows} input = {validation.summary['valid_rows']} valid + {total_rejected_rows} removed/rejected.",
        f"Đối chiếu dòng: {raw_rows} đầu vào = {validation.summary['valid_rows']} hợp lệ + {total_rejected_rows} bị loại.",
    ))

    if use_project_dataset:
        st.markdown("#### " + _t("Errors and corrections only", "Bảng lỗi và kết quả xử lý"))
        review = _build_error_review_table(
            raw,
            validation.valid,
            processing_issues,
            language=st.session_state.get("language", "vi"),
        )
        if review.empty:
            st.success(_t("No problematic cells or rows were found.", "Không phát hiện ô hoặc dòng có lỗi."))
        else:
            labels = {
                "row": _t("Data row", "Dòng dữ liệu"),
                "field": _t("Column", "Cột"),
                "original": _t("Input value", "Giá trị đầu vào"),
                "issue": _t("Problem", "Lỗi"),
                "processed": _t("After processing", "Sau xử lý"),
                "status": _t("Status", "Trạng thái"),
            }
            status_labels = {
                "corrected": _t("Corrected", "Đã xử lý"),
                "rejected": _t("Row rejected", "Đã loại dòng"),
                "review": _t("Needs review", "Cần xem lại"),
            }
            display = review.rename(columns=labels).copy()
            display[labels["status"]] = review["status"].map(status_labels)

            def highlight_error_row(row: pd.Series) -> list[str]:
                styles = [""] * len(row)
                styles[row.index.get_loc(labels["original"])] = (
                    "background-color:#fee2e2;color:#991b1b;font-weight:600"
                )
                styles[row.index.get_loc(labels["issue"])] = "color:#991b1b"
                status = row[labels["status"]]
                if status == status_labels["corrected"]:
                    green = "background-color:#dcfce7;color:#166534;font-weight:600"
                    styles[row.index.get_loc(labels["processed"])] = green
                    styles[row.index.get_loc(labels["status"])] = green
                elif status == status_labels["rejected"]:
                    styles[row.index.get_loc(labels["status"])] = (
                        "background-color:#fee2e2;color:#991b1b;font-weight:600"
                    )
                else:
                    styles[row.index.get_loc(labels["status"])] = (
                        "background-color:#fef3c7;color:#92400e;font-weight:600"
                    )
                return styles

            st.caption(_t(
                "Red = problematic input or rejected row; green = actual corrected value; amber = not automatically handled. Only rows with problems are shown.",
                "Đỏ = đầu vào lỗi hoặc dòng bị loại; xanh = giá trị đã sửa thực tế; vàng = chưa được tự động xử lý. Bảng chỉ hiện chỗ có lỗi.",
            ))
            needs_review = int(review["status"].eq("review").sum())
            if needs_review:
                st.warning(_t(
                    f"{needs_review} field(s) need manual review. They may still be present in the cleaned CSV; no correction was invented.",
                    f"Có {needs_review} trường cần kiểm tra thủ công. Chúng có thể vẫn nằm trong CSV đầu ra; hệ thống không tự tạo giá trị để sửa.",
                ))
            st.dataframe(
                display.head(500).style.apply(highlight_error_row, axis=1),
                width="stretch",
                hide_index=True,
            )
            if len(display) > 500:
                st.caption(_t(
                    f"Showing the first 500 of {len(display)} problem entries; download the full table below.",
                    f"Đang hiện 500/{len(display)} mục lỗi; tải bảng đầy đủ ở dưới.",
                ))
            st.download_button(
                _t("Download full error review CSV", "Tải bảng lỗi đầy đủ"),
                data=data_processing_ui.neutralize_csv_formulas(display).to_csv(index=False).encode("utf-8"),
                file_name="alzheimer_error_review.csv",
                mime="text/csv",
                key="dataset_error_review_download",
            )

    condition_column = _t("CSV condition", "Điều kiện CSV")
    action_column = _t("V3 action", "Xử lý theo V3")
    with st.expander(_t("Error handling rules", "Quy tắc xử lý lỗi"), expanded=True):
        error_rules = [
            {
                condition_column: _t("Negative or out-of-range value", "Giá trị âm hoặc ngoài miền cho phép"),
                action_column: _t("Reject row and record the field/value rule", "Loại dòng và ghi lại quy tắc của trường/giá trị"),
            },
            {
                condition_column: _t("One missing/format-invalid feature", "Một đặc trưng bị thiếu/sai định dạng"),
                action_column: _t("Impute with saved training median/mode and mark row as imputed", "Điền bằng median/mode đã lưu khi huấn luyện và đánh dấu dòng đã điền"),
            },
            {
                condition_column: _t("All five selected model features missing", "Thiếu cả năm đặc trưng được chọn"),
                action_column: _t("Reject row; do not create a prediction from five imputed values", "Loại dòng; không tạo dự đoán từ năm giá trị được điền"),
            },
        ]
        if is_full_schema:
            error_rules.append(
                {
                    condition_column: "SystolicBP <= DiastolicBP",
                    action_column: _t("Reject row before modeling and record the BP rule violation", "Loại dòng trước khi chạy model và ghi nhận vi phạm quy tắc huyết áp"),
                }
            )
        error_rules.append(
            {
                condition_column: _t("Conflicting duplicate PatientID", "PatientID trùng nhưng dữ liệu mâu thuẫn"),
                action_column: _t("Reject all rows for that ID and send them to manual review", "Loại toàn bộ dòng của ID đó và chuyển kiểm tra thủ công"),
            }
        )
        st.dataframe(
            pd.DataFrame(error_rules),
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### " + _t("CSV outputs", "CSV đầu ra"))
    st.write("**" + _t("Cleaned data preview", "Xem trước dữ liệu đã làm sạch") + "**")
    valid_download = validation.valid.drop(columns=VALIDATION_META_COLUMNS, errors="ignore")
    if valid_download.empty:
        st.warning(_t("No valid rows remain after processing.", "Không còn dòng hợp lệ sau khi xử lý."))
    else:
        st.dataframe(valid_download.head(100), width="stretch", hide_index=True)
    if use_project_dataset or not valid_download.empty:
        cleaned_export = (
            data_processing_ui.neutralize_csv_formulas(valid_download)
            if use_project_dataset
            else _screening_export_frame(valid_download, metadata)
        )
        st.download_button(
            _t("Download cleaned CSV", "Tải CSV đã làm sạch"),
            data=cleaned_export.to_csv(index=False).encode("utf-8"),
            file_name=(
                "alzheimer_processed_dataset.csv"
                if use_project_dataset else "alzheimer_cleaned_data.csv"
            ),
            mime="text/csv",
            key="processing_cleaned_download",
        )

    if use_project_dataset or not processing_issues.empty:
        with st.expander(_t("Validation and imputation log", "Nhật ký kiểm tra và điền thiếu"), expanded=False):
            if processing_issues.empty:
                st.caption(_t("No cleaning or validation issues.", "Không có lỗi làm sạch hoặc kiểm tra dữ liệu."))
            else:
                st.dataframe(processing_issues, width="stretch", hide_index=True)
            issues_export = (
                data_processing_ui.neutralize_csv_formulas(processing_issues)
                if use_project_dataset
                else _screening_export_frame(processing_issues, metadata)
            )
            st.download_button(
                _t("Download processing log", "Tải nhật ký xử lý"),
                data=issues_export.to_csv(index=False).encode("utf-8"),
                file_name="alzheimer_processing_log.csv",
                mime="text/csv",
                key="processing_issues_download",
            )
    if use_project_dataset or not rejected_download.empty:
        with st.expander(_t("Rejected rows", "Các dòng bị loại"), expanded=False):
            if rejected_download.empty:
                st.caption(_t("No rows were removed or rejected.", "Không có dòng nào bị loại."))
            else:
                st.dataframe(rejected_download, width="stretch", hide_index=True)
            rejected_export = (
                data_processing_ui.neutralize_csv_formulas(rejected_download)
                if use_project_dataset
                else _screening_export_frame(rejected_download, metadata)
            )
            st.download_button(
                _t("Download rejected rows", "Tải các dòng bị loại"),
                data=rejected_export.to_csv(index=False).encode("utf-8"),
                file_name="alzheimer_rejected_rows.csv",
                mime="text/csv",
                key="processing_rejected_download",
            )

    if use_project_dataset:
        st.success(_t(
            "Data processing is complete. Download the cleaned data, full log, or rejected rows above; no predictions were generated.",
            "Đã xử lý dữ liệu. Tải CSV sạch, nhật ký đầy đủ hoặc các dòng bị loại ở trên; tab này không tạo dự đoán.",
        ))
        return

    st.divider()
    st.subheader(_t("Batch screening result", "Kết quả sàng lọc batch"))
    st.warning(
        _t(
            "This is an educational screening result, not a medical diagnosis. CSVs with an actual Diagnosis label belong in Model update.",
            "Đây là kết quả sàng lọc phục vụ học tập, không phải chẩn đoán y khoa. CSV có nhãn Diagnosis thực tế cần đưa vào tab Cập nhật mô hình.",
        )
    )
    save_batch_for_feedback = False
    if uploaded is not None:
        save_batch_for_feedback = st.checkbox(
            _t(
                "Save valid uploaded cases locally for later clinical feedback",
                "Lưu cục bộ các ca hợp lệ đã tải lên để cập nhật kết quả khám sau này",
            ),
            value=False,
            key="save_batch_for_feedback",
            help=_t(
                "PatientID is not stored in the feedback database; the app creates non-identifying CASE codes.",
                "PatientID không được lưu trong cơ sở dữ liệu feedback; ứng dụng tạo mã CASE không định danh.",
            ),
        )
    if validation.valid.empty:
        st.warning(_t("No valid rows remain for prediction.", "Không còn dòng hợp lệ để dự đoán."))
    else:
        prediction_input = validation.valid.drop(columns=[TARGET_COLUMN], errors="ignore")
        predicted = predict_dataframe(payload, prediction_input)
        predicted = predicted.rename(
            columns={
                "_row_number": "source_row",
                "_quality_status": "quality_status",
                "_imputed_fields": "imputed_fields",
                "_issue_count": "quality_issue_count",
            }
        )
        predicted["predicted_diagnosis"] = predicted["predicted_label"].map(
            {0: "No disease indicated", 1: "Disease indicated"}
        )
        predicted["confidence_percent"] = (
            predicted["prediction_confidence"] * 100
        ).round(1)
        if save_batch_for_feedback and uploaded is not None:
            batch_key = hashlib.sha256(uploaded.getvalue()).hexdigest()
            try:
                case_ids = record_prediction_cases(
                    predicted,
                    actor=user.username,
                    source="batch_upload",
                    model_version=str(metadata.get("version", payload.get("version", "unknown"))),
                    threshold=float(metadata.get("threshold", payload.get("threshold", 0.5))),
                    batch_key=batch_key,
                )
            except Exception as exc:
                st.warning(
                    _t(
                        f"Predictions completed, but follow-up cases could not be saved: {exc}",
                        f"Đã dự đoán xong nhưng không thể lưu các ca theo dõi: {exc}",
                    )
                )
            else:
                predicted = attach_case_ids(predicted, case_ids)
                st.success(
                    _t(
                        f"Saved {len(case_ids)} cases. After a doctor verifies their labels, they become eligible for five-feature retraining.",
                        f"Đã lưu {len(case_ids)} ca. Sau khi bác sĩ xác minh nhãn thực tế, các ca này có thể dùng để huấn luyện lại mô hình 5 chỉ số.",
                    )
                )
        predicted = _screening_export_frame(
            predicted,
            metadata,
            threshold=float(metadata.get("threshold", payload.get("threshold", 0.5))),
        )
        st.caption(_t("Privacy-minimized monitoring summary (no PatientID or raw feature values are logged):", "Tóm tắt giám sát tối thiểu dữ liệu riêng tư (không ghi PatientID hoặc giá trị đặc trưng thô):"))
        batch_summary = summarize_prediction_batch(predicted, validation.summary, metadata)
        st.json(batch_summary)
        monitoring = monitor_prediction_batch(prediction_input, validation.summary, metadata)
        st.markdown("#### " + _t("Data quality and drift alerts", "Cảnh báo chất lượng dữ liệu và drift"))
        if monitoring["alerts"]:
            for alert in monitoring["alerts"]:
                message = str(alert.get("message", alert.get("metric", "Monitoring alert")))
                if alert.get("type") == "configuration":
                    st.info(_t(message, "Artifact chưa lưu baseline giám sát; hãy huấn luyện lại để bật cảnh báo drift."))
                else:
                    st.warning(_t(message, {
                        "Rejected-row rate is above the configured data-quality threshold.": "Tỷ lệ dòng bị loại cao hơn ngưỡng chất lượng dữ liệu.",
                        "Imputation rate is above the configured data-quality threshold.": "Tỷ lệ dòng phải điền thiếu cao hơn ngưỡng chất lượng dữ liệu.",
                    }.get(message, message)))
        else:
            st.success(_t("No configured quality or feature-drift alert was triggered for this batch.", "Batch này không kích hoạt cảnh báo chất lượng hoặc drift đặc trưng."))
        with st.expander(_t("View monitoring details", "Xem chi tiết giám sát"), expanded=False):
            st.json(monitoring)
        st.dataframe(predicted, width="stretch", hide_index=True)
        st.download_button(
            _t("Download predictions", "Tải kết quả dự đoán"),
            data=predicted.to_csv(index=False).encode("utf-8"),
            file_name="alzheimer_screening_predictions.csv",
            mime="text/csv",
            key="processing_predictions_download",
        )


def _feedback_tab(user: AuthUser) -> None:
    """Capture actual outcomes while preserving label revisions for audit."""

    st.subheader(_t("Clinical feedback loop", "Vòng lặp phản hồi thực tế"))
    st.caption(
        _t(
            "Return after the real examination, select the saved CASE code, and record the actual outcome. Doctor/admin labels are verified; viewer labels remain pending until a clinician confirms them.",
            "Sau khi có kết quả khám thực tế, chọn mã CASE đã lưu và cập nhật kết quả thật. Nhãn của bác sĩ/admin được xác minh; nhãn do người dùng nhập sẽ chờ bác sĩ xác nhận.",
        )
    )
    visible_cases = list_cases(actor=user.username if user.role == "viewer" else None)
    if visible_cases.empty:
        st.info(
            _t(
                "No saved cases are available. Save a single prediction or an uploaded batch first.",
                "Chưa có ca đã lưu. Hãy lưu một dự đoán cá nhân hoặc batch đã tải lên trước.",
            )
        )
        return

    case_options = visible_cases["case_id"].astype(str).tolist()
    case_lookup = visible_cases.set_index("case_id")

    st.markdown("#### " + _t("1 · Select a saved screening case", "1 · Chọn ca đã sàng lọc"))
    selected_case = st.selectbox(
        _t("Case code", "Mã ca"), case_options, key="clinical_case_selection"
    )
    selected_row = case_lookup.loc[selected_case]
    predicted_label = int(selected_row["predicted_label"])
    predicted_text = _t(
        "Disease indicated" if predicted_label else "No disease indicated",
        "Có tín hiệu bệnh" if predicted_label else "Không có tín hiệu bệnh",
    )
    score_percent = float(selected_row["calibrated_score"]) * 100
    st.markdown("#### " + _t("2 · AI screening result (read-only)", "2 · Kết quả AI (chỉ xem)"))
    st.info(
        _t(
            f"Model screening: {predicted_text} · score {score_percent:.1f}%",
            f"AI sàng lọc: {predicted_text} · điểm {score_percent:.1f}%",
        )
    )
    st.caption(_t(
        "This score is not a doctor's diagnosis or a certainty of disease.",
        "Điểm này không phải kết luận của bác sĩ hay xác suất chắc chắn mắc bệnh.",
    ))

    st.markdown("#### " + _t("3 · Doctor's examination conclusion", "3 · Kết luận khám của bác sĩ"))
    verified_status = st.empty()
    is_verifier = user.role in {"doctor", "admin"}
    if is_verifier:
        st.caption(_t(
            "A doctor/administrator account can verify or correct the examination conclusion. The account and time are recorded.",
            "Tài khoản bác sĩ/quản trị có thể xác minh hoặc sửa kết luận khám. Hệ thống lưu tài khoản và thời điểm xác minh.",
        ))
    else:
        st.caption(_t(
            "Enter the conclusion written by the doctor. Your report stays pending until a doctor/administrator verifies it.",
            "Nhập đúng kết luận bác sĩ ghi sau khám. Kết quả bạn gửi sẽ chờ bác sĩ/quản trị xác minh.",
        ))

    feedback_revision = selected_row["feedback_revision"]
    revision_key = 0 if pd.isna(feedback_revision) else int(feedback_revision)
    diagnosis = st.radio(
        _t("Doctor's conclusion after examination", "Bác sĩ kết luận sau khám"),
        [0, 1],
        index=None,
        format_func=lambda value: _t(
            "No Alzheimer's diagnosis" if value == 0 else "Alzheimer's diagnosis",
            "Không mắc Alzheimer" if value == 0 else "Mắc Alzheimer",
        ),
        horizontal=True,
        key=f"clinical_diagnosis_{selected_case}_{revision_key}",
    )
    confirmed = st.checkbox(
        _t(
            "I checked this choice against the actual doctor's conclusion, not the AI result",
            "Tôi đã đối chiếu lựa chọn này với kết luận khám của bác sĩ, không phải kết quả AI",
        ),
        value=False,
        key=f"clinical_confirmation_{selected_case}_{revision_key}",
    )
    submitted = st.button(
        _t(
            "Save verified clinical conclusion" if is_verifier else "Send for doctor verification",
            "Lưu kết luận khám đã xác minh" if is_verifier else "Gửi kết quả để bác sĩ xác minh",
        ),
        type="primary",
        disabled=diagnosis is None or not confirmed,
        key="clinical_feedback_save",
    )
    if submitted:
        try:
            result = submit_feedback(
                selected_case,
                diagnosis,
                submitted_by=user.username,
                verified=is_verifier,
                allowed_owner=user.username if user.role == "viewer" else None,
            )
        except (PermissionError, ValueError) as exc:
            st.error(str(exc))
        else:
            if is_verifier:
                st.success(
                    _t(
                        f"Verified label saved as revision {result['revision']}.",
                        f"Đã lưu nhãn xác minh ở phiên bản {result['revision']}.",
                    )
                )
                if result["verified_model_mismatch"]:
                    st.warning(
                        _t(
                            f"Model error warning for {result['case_id']}: the screening prediction differs from the verified clinical result. This case is recorded for review and can be considered at the next retraining run.",
                            f"Cảnh báo mô hình dự đoán sai ở {result['case_id']}: kết quả sàng lọc khác kết quả khám đã xác minh. Ca này được lưu để đối chiếu và xem xét trong lần huấn luyện lại tiếp theo.",
                        )
                    )
            else:
                st.info(
                    _t(
                        f"Feedback saved as revision {result['revision']} and is waiting for doctor verification.",
                        f"Đã lưu phản hồi ở phiên bản {result['revision']} và đang chờ bác sĩ xác minh.",
                    )
                )
            visible_cases = list_cases(actor=user.username if user.role == "viewer" else None)

    current_row = visible_cases.set_index("case_id").loc[selected_case]
    with verified_status.container():
        if pd.notna(current_row["verified_diagnosis"]):
            verified_label = int(current_row["verified_diagnosis"])
            verified_text = _t(
                "No Alzheimer's diagnosis" if verified_label == 0 else "Alzheimer's diagnosis",
                "Không mắc Alzheimer" if verified_label == 0 else "Mắc Alzheimer",
            )
            if verified_label != predicted_label:
                st.warning(_t(
                    f"Verified examination conclusion: {verified_text}. This differs from the AI screening result.",
                    f"Kết luận khám đã xác minh: {verified_text}. Kết quả này khác dự đoán AI — ghi nhận ca mô hình dự đoán sai.",
                ))
            else:
                st.success(_t(
                    f"Verified examination conclusion: {verified_text}. This matches the AI screening result.",
                    f"Kết luận khám đã xác minh: {verified_text}. Kết quả này khớp dự đoán AI.",
                ))
            verified_at = pd.to_datetime(current_row["verified_at_utc"], utc=True)
            verified_at = verified_at.tz_convert("Asia/Bangkok").strftime("%d/%m/%Y %H:%M")
            st.caption(_t(
                f"Verified by account {current_row['verified_by']} · {verified_at} (ICT) · revision {int(current_row['verified_revision'])}.",
                f"Tài khoản xác minh: {current_row['verified_by']} · {verified_at} (giờ VN) · phiên bản {int(current_row['verified_revision'])}.",
            ))
        else:
            st.info(_t(
                "No verified doctor's examination conclusion has been recorded for this case yet.",
                "Ca này chưa có kết luận khám được bác sĩ/quản trị xác minh.",
            ))
        if current_row["latest_verified"] == 0:
            pending_text = _t(
                "No Alzheimer's diagnosis" if int(current_row["latest_diagnosis"]) == 0 else "Alzheimer's diagnosis",
                "Không mắc Alzheimer" if int(current_row["latest_diagnosis"]) == 0 else "Mắc Alzheimer",
            )
            st.caption(_t(
                f"Newest submitted report: {pending_text} (pending verification).",
                f"Phản hồi mới nhất: {pending_text} (đang chờ xác minh).",
            ))

    st.markdown("#### " + _t("Case and feedback status", "Trạng thái ca và phản hồi"))
    display = visible_cases.copy()
    has_verified_label = display["verified_revision"].notna()
    model_mismatch = has_verified_label & display["verified_diagnosis"].ne(display["predicted_label"])
    pending_feedback = display["latest_verified"].eq(0)
    summary = st.columns(4)
    summary[0].metric(_t("Saved cases", "Ca đã lưu"), len(display))
    summary[1].metric(_t("Pending review", "Chờ xác minh"), int(pending_feedback.sum()))
    summary[2].metric(_t("Verified model mismatches", "Ca mô hình sai đã xác minh"), int(model_mismatch.sum()))
    summary[3].metric(_t("Retrain candidates", "Ca có nhãn để retrain"), int(has_verified_label.sum()))
    display["model_prediction"] = display["predicted_label"].map(
        {0: _t("No disease", "Không bệnh"), 1: _t("Disease", "Có bệnh")}
    )
    display["score_percent"] = (display["calibrated_score"] * 100).round(1)
    display["actual_result"] = display["latest_diagnosis"].map(
        {0.0: _t("No disease", "Không bệnh"), 1.0: _t("Disease", "Có bệnh")}
    ).fillna(_t("Not provided", "Chưa cập nhật"))
    display["verified_result"] = display["verified_diagnosis"].map(
        {0.0: _t("No disease", "Không bệnh"), 1.0: _t("Disease", "Có bệnh")}
    ).fillna(_t("Not verified", "Chưa xác minh"))
    display["label_status"] = display["latest_verified"].map(
        {0.0: _t("Pending doctor", "Chờ bác sĩ"), 1.0: _t("Verified", "Đã xác minh")}
    ).fillna(_t("No feedback", "Chưa có phản hồi"))
    display.loc[pending_feedback & has_verified_label, "label_status"] = _t(
        "New feedback pending; prior verified label retained",
        "Phản hồi mới chờ xác minh; giữ nhãn đã xác minh trước",
    )
    display["model_review"] = _t("Not verified", "Chưa xác minh")
    display.loc[has_verified_label & ~model_mismatch, "model_review"] = _t(
        "Matches verified result", "Khớp kết quả đã xác minh"
    )
    display.loc[model_mismatch, "model_review"] = _t(
        "Verified model error", "Mô hình dự đoán sai"
    )
    display["retrain_ready"] = has_verified_label.map(
        {True: _t("Yes", "Có"), False: _t("No", "Không")}
    )
    st.dataframe(
        display[
            [
                "case_id",
                "source",
                "model_prediction",
                "score_percent",
                "actual_result",
                "verified_result",
                "label_status",
                "model_review",
                "feedback_revision",
                "retrain_ready",
            ]
        ].rename(columns={
            "case_id": _t("Case code", "Mã ca"),
            "source": _t("Source", "Nguồn"),
            "model_prediction": _t("Model prediction", "Dự đoán mô hình"),
            "score_percent": _t("Screening score (%)", "Điểm sàng lọc (%)"),
            "actual_result": _t("Latest feedback", "Phản hồi mới nhất"),
            "verified_result": _t("Verified clinical result", "Kết quả khám đã xác minh"),
            "label_status": _t("Label status", "Trạng thái nhãn"),
            "model_review": _t("Model comparison", "Đối chiếu mô hình"),
            "feedback_revision": _t("Feedback revision", "Phiên bản phản hồi"),
            "retrain_ready": _t("Retrain candidate", "Ứng viên retrain"),
        }),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        _t(
            "A verified clinical label makes a five-feature case a retraining candidate, not an automatic model update. The administrator must start a run; validation and model-quality checks decide whether it is accepted. Pending feedback never replaces the last verified label.",
            "Ca 5 chỉ số có nhãn bác sĩ/admin xác minh là ứng viên huấn luyện lại, không tự cập nhật mô hình. Quản trị viên cần khởi chạy; kiểm tra dữ liệu và chất lượng mô hình quyết định có tiếp nhận hay không. Phản hồi chờ xác minh không thay nhãn đã xác minh trước đó.",
        )
    )


def _show_retraining_metric_comparison(checks: list[dict]) -> None:
    """Show signed metric changes; a lower Brier score is an improvement."""

    comparison = _retraining_metric_comparison(checks)
    if comparison.empty:
        st.caption(_t("No comparable metric checks were saved for this run.", "Phiên này chưa lưu đủ chỉ số để so sánh."))
        return

    st.markdown("#### " + _t("Current vs new model", "So sánh mô hình hiện tại và mô hình mới"))
    st.caption(_t(
        "Change = new minus current, evaluated on the same locked test cohort. Higher PR-AUC, Recall and F2 are better; lower Brier is better.",
        "Thay đổi = mô hình mới trừ mô hình hiện tại, đo trên cùng tập kiểm tra cố định. PR-AUC, Recall và F2 tăng là tốt; Brier giảm là tốt.",
    ))
    direction_labels = {
        "up": _t("Increased", "Tăng"),
        "down": _t("Decreased", "Giảm"),
        "same": _t("Unchanged", "Không đổi"),
    }
    outcome_labels = {
        "better": _t("Better", "Tốt hơn"),
        "worse": _t("Worse", "Kém hơn"),
        "same": _t("Unchanged", "Không đổi"),
    }
    chart_data = comparison.copy()
    chart_data["outcome_label"] = chart_data["outcome"].map(outcome_labels)
    chart_data["direction_label"] = chart_data["direction"].map(direction_labels)
    chart_data["policy_label"] = chart_data["policy_passed"].map({
        True: _t("Pass", "Đạt"), False: _t("Fail", "Không đạt"),
    })
    try:
        import altair as alt
    except ImportError:
        st.bar_chart(chart_data.set_index("label")["delta"], height=230)
    else:
        better, worse, same = (outcome_labels[key] for key in ("better", "worse", "same"))
        extent = max(float(chart_data["delta"].abs().max()) * 1.5, 0.001)
        shared = {
            "x": alt.X(
                "delta:Q",
                title=_t("Change (new − current)", "Thay đổi (mới − hiện tại)"),
                scale=alt.Scale(domain=[-extent, extent]),
                axis=alt.Axis(format="+.3f"),
            ),
            "y": alt.Y("label:N", sort=comparison["label"].tolist(), title=None),
            "tooltip": [
                alt.Tooltip("label:N", title=_t("Metric", "Chỉ số")),
                alt.Tooltip("old:Q", title=_t("Current", "Hiện tại"), format=".4f"),
                alt.Tooltip("new:Q", title=_t("New", "Mới"), format=".4f"),
                alt.Tooltip("delta:Q", title=_t("Change", "Thay đổi"), format="+.4f"),
                alt.Tooltip("direction_label:N", title=_t("Direction", "Chiều")),
                alt.Tooltip("outcome_label:N", title=_t("Assessment", "Đánh giá")),
                alt.Tooltip("policy_label:N", title=_t("Policy", "Điều kiện")),
            ],
        }
        color_scale = alt.Scale(
            domain=[better, worse, same],
            range=["#2563EB", "#D97706", "#64748B"],
        )
        color = alt.Color(
            "outcome_label:N",
            title=_t("Assessment", "Đánh giá"),
            scale=color_scale,
        )
        bars = alt.Chart(chart_data).mark_bar(size=24).encode(**shared, color=color)
        points = alt.Chart(chart_data).mark_point(filled=True, size=75).encode(
            **shared, color=alt.Color("outcome_label:N", scale=color_scale, legend=None)
        )
        zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color="#475569").encode(x="zero:Q")
        st.altair_chart((zero + bars + points).properties(height=220), width="stretch")

    display = pd.DataFrame({
        _t("Metric", "Chỉ số"): comparison["label"],
        _t("Current", "Hiện tại"): comparison["old"].map(lambda value: f"{value:.4f}"),
        _t("New", "Mới"): comparison["new"].map(lambda value: f"{value:.4f}"),
        _t("Change", "Thay đổi"): comparison["delta"].map(lambda value: f"{value:+.4f}"),
        _t("Direction", "Chiều"): chart_data["direction_label"],
        _t("Assessment", "Đánh giá"): chart_data["outcome_label"],
        _t("Policy", "Điều kiện"): chart_data["policy_label"],
    })
    st.dataframe(display, width="stretch", hide_index=True)


def _retrain_tab(payload: dict, metadata: dict, user: AuthUser) -> None:
    """Validate labelled five-feature cases and expose the retrain loop."""

    st.subheader(_t("Retrain / Model update", "Huấn luyện lại / Cập nhật mô hình"))
    st.caption(
        _t(
            "Use verified clinical feedback or upload a labelled five-feature CSV. New rows are validated, the original dataset is preserved, "
            "a versioned challenger is trained, and it is promoted only after a same-cohort policy check.",
            "Dùng feedback lâm sàng đã xác minh hoặc tải CSV gồm 5 chỉ số và nhãn thực tế. Các dòng mới sẽ được kiểm tra, dữ liệu gốc được giữ nguyên, "
            "mô hình thử thách có phiên bản sẽ được huấn luyện và chỉ được đưa vào sử dụng sau khi đạt kiểm tra chính sách cùng nhóm dữ liệu.",
        )
    )
    st.warning(
        _t(
            "Persistent retraining can change model artifacts and may take several minutes. Administrator access and Diagnosis are required; "
            "missing or invalid labels are rejected and never imputed.",
            "Huấn luyện lưu phiên bản có thể thay đổi artifact mô hình và mất vài phút. Cần quyền quản trị và cột Diagnosis; "
            "nhãn thiếu hoặc không hợp lệ sẽ bị loại và không bao giờ được điền tự động.",
        )
    )

    authorized = user.role == "admin"
    if authorized:
        st.success(
            _t(
                "Administrator access verified. A passed challenger is promoted automatically; a failed challenger never replaces the active model.",
                "Đã xác thực quyền quản trị. Model mới đạt toàn bộ policy sẽ tự động được đưa vào sử dụng; model không đạt sẽ không thay thế model hiện tại.",
            )
        )
    else:
        st.info(
            _t(
                "Only an administrator can start retraining. Doctors and viewers can review the session log.",
                "Chỉ quản trị viên được chạy huấn luyện lại. Bác sĩ và người dùng vẫn có thể xem nhật ký phiên.",
            )
        )

    source_labels = {
        "Labelled five-feature CSV": "CSV 5 chỉ số có nhãn",
        "Verified clinical feedback": "Feedback đã được bác sĩ xác minh",
    }
    if st.session_state.get("retraining_source") not in {None, *source_labels}:
        st.session_state["retraining_source"] = "Labelled five-feature CSV"
    retrain_source = st.radio(
        _t("New labelled data source", "Nguồn dữ liệu mới có nhãn"),
        list(source_labels),
        format_func=lambda value: _t(value, source_labels[value]),
        horizontal=True,
        key="retraining_source",
    )

    retrain_columns = _template_columns("Training/retraining audit", "Five-feature retraining CSV")
    uploaded = None
    if retrain_source == "Labelled five-feature CSV":
        st.markdown("#### " + _t("Update from a labelled CSV", "Cập nhật từ CSV có nhãn"))
        st.caption(_t(
            "Use this route for new cases measured outside the app. Each row needs the five model inputs and a clinician-verified Diagnosis.",
            "Dùng luồng này cho ca mới được ghi nhận ngoài ứng dụng. Mỗi dòng cần đủ 5 chỉ số mô hình và Diagnosis đã được bác sĩ xác minh.",
        ))
        with st.expander(_t("Retraining CSV template", "Mẫu CSV huấn luyện lại"), expanded=False):
            st.download_button(
                _t("Download retraining template", "Tải mẫu huấn luyện lại"),
                data=_retraining_template_bytes(),
                file_name="alzheimer_retraining_template.csv",
                mime="text/csv",
                key="retraining_template_download",
            )
            st.caption(_t(
                "The template has five model inputs and one clinician-verified Diagnosis (0 or 1). An optional PatientID helps distinguish repeat uploads; it is not a model feature. Do not invent missing clinical values.",
                "Mẫu gồm 5 chỉ số mô hình và 1 cột Diagnosis (0 hoặc 1) do bác sĩ xác minh. Có thể thêm PatientID để nhận biết ca trùng; đây không phải đặc trưng mô hình. Không tự tạo giá trị lâm sàng còn thiếu.",
            ))
            st.dataframe(_schema_guide(retrain_columns), width="stretch", hide_index=True)

        uploaded = st.file_uploader(
            _t("Upload labelled five-feature CSV for model update", "Tải CSV 5 chỉ số có nhãn để cập nhật mô hình"),
            type=["csv"],
            key="retraining_csv",
        )
        st.caption(_t(
            "App limit: 10,000 rows and 10 MB per CSV; five measured model inputs plus Diagnosis are required.",
            "Giới hạn xử lý của app: 10.000 dòng và 10 MB mỗi CSV; cần đủ 5 chỉ số mô hình được đo thực tế và Diagnosis.",
        ))
    else:
        st.markdown("#### " + _t("Update from verified clinical feedback", "Cập nhật từ feedback đã xác minh"))
        st.info(_t(
            "This route uses cases already saved in the app and labelled by a doctor or administrator. No CSV upload is needed.",
            "Luồng này lấy các ca đã lưu trong ứng dụng và được bác sĩ hoặc quản trị viên xác minh nhãn thực tế. Không cần tải CSV.",
        ))
    history = list_retraining_runs()
    with st.expander(_t("Retraining session log", "Nhật ký phiên huấn luyện lại"), expanded=False):
        if history.empty:
            st.caption(_t("No retraining session has been recorded.", "Chưa có phiên huấn luyện lại nào."))
        else:
            st.dataframe(history, width="stretch", hide_index=True)
            selected_run_id = st.selectbox(
                _t("View metric comparison for session", "Xem so sánh chỉ số của phiên"),
                history["run_id"].tolist(),
                key="retraining_history_selection",
            )
            saved_run = get_retraining_run(selected_run_id)
            if saved_run:
                saved_checks = saved_run["details"].get("promotion_decision", {}).get("checks", [])
                _show_retraining_metric_comparison(saved_checks)
    if retrain_source == "Labelled five-feature CSV":
        if uploaded is None:
            st.info(_t(
                "Upload a labelled CSV to inspect it before retraining.",
                "Tải CSV có nhãn lên để kiểm tra trước khi huấn luyện lại.",
            ))
            return
        try:
            raw, file_info = _read_uploaded_csv(
                uploaded,
                required_columns=retrain_columns,
                language=st.session_state.get("language", "vi"),
            )
        except Exception as exc:
            st.error(str(exc))
            return
    else:
        raw = feedback_training_frame()
        file_info = {
            "source": "Verified clinical feedback",
            "rows": len(raw),
            "schema": "5 features + Diagnosis",
        }
        if raw.empty:
            st.info(
                _t(
                    "No retrain-ready feedback exists yet. Save a prediction and have a doctor/admin verify its actual label.",
                    "Chưa có feedback sẵn sàng huấn luyện lại. Hãy lưu ca dự đoán và để bác sĩ/admin xác minh nhãn thực tế.",
                )
            )
            return
        st.success(
            _t(
                f"Loaded {len(raw)} cases with verified clinical labels.",
                f"Đã nạp {len(raw)} ca có nhãn lâm sàng được xác minh.",
            )
        )
        label_counts = raw["Diagnosis"].value_counts()
        feedback_summary = st.columns(3)
        feedback_summary[0].metric(_t("Verified cases", "Ca đã xác minh"), len(raw))
        feedback_summary[1].metric(_t("Negative labels", "Nhãn âm"), int(label_counts.get(0, 0)))
        feedback_summary[2].metric(_t("Positive labels", "Nhãn dương"), int(label_counts.get(1, 0)))

    st.divider()
    st.markdown("#### " + _t("Shared validation and retraining", "Kiểm tra và huấn luyện chung"))
    st.caption(_t(
        "After the source is selected, both routes use the same five-feature validation and model-promotion policy.",
        "Sau khi chọn nguồn, cả hai luồng dùng cùng quy tắc kiểm tra 5 chỉ số và chính sách cập nhật mô hình.",
    ))
    missing_columns = [column for column in retrain_columns if column not in raw.columns]
    unexpected_columns = [column for column in raw.columns if column not in set(retrain_columns + ["PatientID"])]
    with st.expander(_t("Retraining file diagnostics", "Chẩn đoán file huấn luyện lại"), expanded=True):
        st.dataframe(pd.DataFrame([file_info]), width="stretch", hide_index=True)
        if missing_columns:
            st.error(_t(
                "Missing required retraining columns: " + ", ".join(missing_columns),
                "Thiếu các cột bắt buộc để huấn luyện lại: " + ", ".join(missing_columns),
            ))
        else:
            st.success(_t(
                f"Five-feature retraining schema passed: {len(retrain_columns)} required columns found.",
                f"Đúng mẫu cập nhật: đủ {len(retrain_columns)} cột bắt buộc (5 chỉ số + Diagnosis).",
            ))
        if unexpected_columns:
            st.warning(_t(
                "Unexpected columns will be ignored: " + ", ".join(unexpected_columns),
                "Các cột không dự kiến sẽ bị bỏ qua: " + ", ".join(unexpected_columns),
            ))
    if missing_columns:
        return
    if raw.empty:
        st.warning(_t(
            "The CSV contains headers but no patient rows. Add at least one new case with a verified Diagnosis.",
            "CSV mới có tiêu đề, chưa có dòng bệnh nhân. Hãy thêm ít nhất một ca mới có Diagnosis đã xác minh.",
        ))
        return

    reference = get_reference_data()
    reference_clean, _ = clean_dataset(reference)
    imputation_statistics = payload.get("imputation_statistics") or metadata.get(
        "imputation_statistics"
    )
    if not imputation_statistics:
        imputation_statistics = fit_imputation_statistics(reference_clean)
        st.caption(_t(
            "Preview uses fallback imputation statistics; a new artifact will persist them.",
            "Bản xem trước dùng thống kê điền thiếu dự phòng; artifact mới sẽ lưu các thống kê này.",
        ))

    try:
        locked_test, _ = _locked_test_frame(reference, DEFAULT_LOCKED_TEST_PATH)
        accepted = _accepted_retraining_rows(DEFAULT_ARTIFACTS_DIR)
        validation, candidate_rows, skip_counts = prepare_retraining_rows(
            raw, reference_clean, accepted, locked_test, imputation_statistics,
        )
        repeated_rows = (
            skip_counts["existing_patient_id_rows_skipped"]
            + skip_counts["previous_rows_skipped"]
        )
        if repeated_rows:
            st.warning(_t(
                f"Skipped {repeated_rows} previously seen cases. Use a new PatientID for distinct patients with identical measurements.",
                f"Đã bỏ qua {repeated_rows} ca đã có. Nếu hai bệnh nhân có cùng 5 chỉ số, hãy thêm PatientID khác nhau để phân biệt.",
            ))
        if skip_counts["locked_feature_rows_skipped"]:
            st.warning(_t(
                f"Held out {skip_counts['locked_feature_rows_skipped']} rows matching the locked test cohort.",
                f"Đã giữ ngoài huấn luyện {skip_counts['locked_feature_rows_skipped']} dòng trùng 5 chỉ số với tập kiểm tra cố định.",
            ))
        if skip_counts["corrected_rows"]:
            st.info(_t(
                f"{skip_counts['corrected_rows']} previously accepted cases have corrected values or labels and will replace their earlier training copies if promoted.",
                f"{skip_counts['corrected_rows']} ca đã lưu có giá trị hoặc nhãn được sửa; bản cũ trong tập huấn luyện sẽ được thay nếu mô hình mới đạt điều kiện cập nhật.",
            ))
    except Exception as exc:
        st.error(_t(
            f"Could not validate retraining data: {exc}",
            f"Không thể kiểm tra dữ liệu huấn luyện lại: {exc}",
        ))
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(_t("Input rows", "Dòng đầu vào"), len(raw))
    c2.metric(_t("New/updated valid rows", "Dòng mới/sửa hợp lệ"), len(candidate_rows))
    c3.metric(_t("Rejected new rows", "Dòng mới bị loại"), validation.summary["rejected_rows"])
    c4.metric(_t("Existing rows skipped", "Dòng cũ bỏ qua"), repeated_rows)
    c5.metric(_t("Locked rows held out", "Dòng test giữ ngoài"), skip_counts["locked_feature_rows_skipped"])

    if not validation.issues.empty:
        with st.expander(_t("Retraining validation log", "Nhật ký kiểm tra huấn luyện lại"), expanded=False):
            st.dataframe(validation.issues, width="stretch", hide_index=True)
            st.download_button(
                _t("Download retraining validation log", "Tải nhật ký kiểm tra huấn luyện lại"),
                data=validation.issues.to_csv(index=False).encode("utf-8"),
                file_name="alzheimer_retraining_validation_log.csv",
                mime="text/csv",
                key="retraining_log_download",
            )
    if not validation.rejected.empty:
        with st.expander(_t("Rejected retraining rows", "Các dòng huấn luyện lại bị loại"), expanded=False):
            rejected = validation.rejected.drop(columns=VALIDATION_META_COLUMNS, errors="ignore")
            st.dataframe(rejected, width="stretch", hide_index=True)
            st.download_button(
                _t("Download rejected retraining rows", "Tải các dòng huấn luyện lại bị loại"),
                data=rejected.to_csv(index=False).encode("utf-8"),
                file_name="alzheimer_retraining_rejected_rows.csv",
                mime="text/csv",
                key="retraining_rejected_download",
            )

    if candidate_rows.empty:
        if repeated_rows or skip_counts["locked_feature_rows_skipped"]:
            st.error(_t(
                "Retraining is blocked because no new cases remain after repeat and locked-test checks. Upload genuinely new labelled cases.",
                "Đã chặn huấn luyện lại vì không còn ca mới sau khi lọc ca trùng và ca thuộc tập kiểm tra. Hãy tải ca mới có nhãn thực tế.",
            ))
        else:
            st.error(_t(
                "Retraining is blocked because no valid labelled rows remain.",
                "Đã chặn huấn luyện lại vì không còn dòng có nhãn hợp lệ.",
            ))
        return

    st.caption(
        _t(
            "Policy: PR-AUC must not decrease; recall may fall by at most 0.010; "
            "Brier may worsen by at most 0.005; F2 must improve by 0.005 or have a positive paired 95% CI; "
            "every common estimable subgroup is checked for recall/PR-AUC degradation up to 0.050.",
            "Chính sách: PR-AUC không được giảm; recall chỉ được giảm tối đa 0.010; "
            "Brier chỉ được xấu hơn tối đa 0.005; F2 phải tăng 0.005 hoặc có khoảng tin cậy ghép 95% dương; "
            "mọi subgroup chung có thể ước lượng đều được kiểm tra giảm recall/PR-AUC tối đa 0.050.",
        )
    )
    st.caption(
        _t(
            "One click starts the full loop: validate → enrich training data → train challenger → compare on the locked cohort → promote only when every policy condition passes → record the session.",
            "Một lần bấm chạy toàn bộ vòng lặp: kiểm tra → làm giàu dữ liệu → huấn luyện model mới → so sánh trên tập cố định → chỉ cập nhật khi đạt mọi policy → lưu nhật ký phiên.",
        )
    )
    run_requested = st.button(
        _t("Run automatic retraining loop", "Chạy vòng lặp huấn luyện lại tự động"),
        type="primary",
        key="run_retraining",
        disabled=not authorized,
    )
    if run_requested:
        with st.spinner(_t(
            "Training and evaluating the new model version...",
            "Đang huấn luyện và đánh giá phiên bản model mới...",
        )):
            try:
                result = retrain_with_new_data(
                    raw,
                    existing_data_path=DEFAULT_DATA_PATH,
                    artifacts_dir=DEFAULT_ARTIFACTS_DIR,
                    approve_promotion=True,
                )
            except Exception as exc:
                record_retraining_run(
                    started_by=user.username,
                    source=retrain_source,
                    source_rows=len(raw),
                    status="failed",
                    promoted=False,
                    champion_version=str(metadata.get("version", "unknown")),
                    details={"error": str(exc)},
                )
                st.error(_t(f"Retraining failed: {exc}", f"Huấn luyện lại thất bại: {exc}"))
                return

        retrain = result["retrain"]
        run_id = record_retraining_run(
            started_by=user.username,
            source=retrain_source,
            source_rows=len(raw),
            status="promoted" if retrain.get("promoted") else "retained_champion",
            promoted=bool(retrain.get("promoted")),
            policy_passed=bool(retrain.get("policy_passed")),
            champion_version=str(metadata.get("version", "unknown")),
            challenger_version=str(result.get("metadata", {}).get("version", "unknown")),
            old_pr_auc=retrain.get("old_pr_auc"),
            new_pr_auc=retrain.get("new_pr_auc"),
            details={
                "promotion_decision": retrain.get("promotion_decision", {}),
                "validated_new_rows": retrain.get("validated_new_rows"),
                "rejected_new_rows": retrain.get("rejected_new_rows"),
                "locked_test_sha256": retrain.get("locked_test_sha256"),
            },
        )
        champion_metrics = retrain.get("champion_metrics") or {}
        challenger_metrics = retrain.get("challenger_metrics") or {}
        comparison = pd.DataFrame(
            [
                {"cohort": "locked_test", "model": "Champion", **champion_metrics},
                {"cohort": "locked_test", "model": "Challenger", **challenger_metrics},
            ]
        )
        _show_retraining_metric_comparison(retrain.get("promotion_decision", {}).get("checks", []))
        st.dataframe(comparison, width="stretch", hide_index=True)
        checks = pd.DataFrame(retrain.get("promotion_decision", {}).get("checks", []))
        if not checks.empty:
            st.markdown("#### " + _t("Promotion policy checks", "Kiểm tra chính sách đưa model vào sử dụng"))
            st.dataframe(
                checks[["condition", "metric", "old", "new", "delta", "required", "status"]],
                width="stretch",
                hide_index=True,
            )
        st.caption(_t(
            f"Locked-test SHA-256: {retrain.get('locked_test_sha256', 'not recorded')}",
            f"SHA-256 của tập test cố định: {retrain.get('locked_test_sha256', 'chưa ghi nhận')}",
        ))
        st.caption(_t(f"Session log: {run_id}", f"Mã phiên nhật ký: {run_id}"))
        if retrain["promoted"]:
            st.success(_t(
                "The challenger passed every policy condition and became active. Verified new or corrected five-feature rows were saved for future retraining.",
                "Model mới đạt mọi điều kiện và trở thành model đang dùng. Các ca 5 chỉ số mới hoặc đã sửa được lưu cho những lần huấn luyện sau.",
            ))
        else:
            st.warning(_t(
                "The challenger did not pass every policy condition. The current model and training dataset were kept unchanged; this session was logged.",
                "Model mới chưa đạt mọi điều kiện policy. Model hiện tại và tập huấn luyện được giữ nguyên; phiên này đã được lưu nhật ký.",
            ))
        get_runtime.clear()
        st.info(_t(
            "Runtime cache cleared. Refresh the app to load the selected active artifact.",
            "Đã xóa bộ nhớ đệm runtime. Hãy tải lại app để nạp artifact đang được chọn.",
        ))


def _plot_model_feature_ranking(metadata: dict) -> None:
    """Show the stored permutation-importance ranking without hiding it in a table."""

    ranking = pd.DataFrame(metadata.get("feature_ranking", []))
    required = {"feature", "importance_mean"}
    if ranking.empty or not required.issubset(ranking.columns):
        st.info("Feature-ranking diagnostics are not available in this artifact.")
        return
    chart_data = ranking[["feature", "importance_mean"]].copy()
    chart_data["importance_mean"] = pd.to_numeric(chart_data["importance_mean"], errors="coerce")
    chart_data = chart_data.dropna().sort_values("importance_mean").tail(10)
    if chart_data.empty:
        st.info("Feature-ranking diagnostics are not available in this artifact.")
        return
    try:
        import altair as alt
    except ImportError:
        alt = None
    if alt is not None:
        plot_data = chart_data.rename(
            columns={"feature": "Feature", "importance_mean": "Permutation importance"}
        )
        feature_order = plot_data["Feature"].astype(str).tolist()
        chart = (
            alt.Chart(plot_data)
            .mark_bar(color="#6C9BC3")
            .encode(
                x=alt.X("Permutation importance:Q", title="Mean decrease in validation accuracy after shuffling"),
                y=alt.Y("Feature:N", title="Feature", sort=feature_order),
                tooltip=[
                    alt.Tooltip("Feature:N", title="Feature"),
                    alt.Tooltip("Permutation importance:Q", title="Importance", format=".4f"),
                ],
            )
            .properties(title="Top validation permutation-importance features", height=max(220, 34 * len(plot_data)))
        )
        st.altair_chart(chart, width="stretch")
        st.caption("Hover a bar to see the exact permutation-importance value.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.bar_chart(chart_data.set_index("feature"), y="importance_mean")
        return
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(chart_data))))
    ax.barh(chart_data["feature"], chart_data["importance_mean"], color="#6C9BC3")
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_title("Top validation permutation-importance features")
    ax.set_xlabel("Mean decrease in validation accuracy after shuffling")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _plot_benchmark_altair(benchmark: pd.DataFrame, metrics: list[tuple[str, str]], title: str) -> bool:
    """Render a benchmark chart with native hover tooltips when Altair is available."""

    try:
        import altair as alt
    except ImportError:
        return False
    available = [(column, label) for column, label in metrics if column in benchmark.columns]
    if benchmark.empty or "model" not in benchmark.columns or not available:
        return False
    rows = []
    for _, row in benchmark.iterrows():
        for column, label in available:
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.notna(value):
                rows.append({"Model": row.get("model"), "Metric": label, "Score": float(value)})
    chart_data = pd.DataFrame(rows)
    if chart_data.empty:
        return False
    metric_order = [label for _, label in available]
    model_order = list(benchmark["model"].astype(str))
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("Score:Q", title="Score (0–1)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Model:N", title="Candidate model", sort=model_order),
            color=alt.condition(
                alt.datum.Model == "XGBoost",
                alt.value("#F2A900"),
                alt.value("#6C9BC3"),
            ),
            column=alt.Column("Metric:N", title=None, sort=metric_order),
            tooltip=[
                alt.Tooltip("Model:N", title="Model"),
                alt.Tooltip("Metric:N", title="Metric"),
                alt.Tooltip("Score:Q", title="Score", format=".3f"),
            ],
        )
        .properties(title=title, height=max(180, 34 * len(model_order)))
    )
    st.altair_chart(chart, width="stretch")
    return True


def _plot_model_benchmark(benchmark: pd.DataFrame) -> None:
    """Compare candidate models using the three metrics shown in report Figure 8."""

    required = {"model", "roc_auc_cv", "pr_auc_cv", "recall"}
    if benchmark.empty or not required.issubset(benchmark.columns):
        st.info("Model benchmark diagnostics are not available in this artifact.")
        return
    chart_data = benchmark[["model", "roc_auc_cv", "pr_auc_cv", "recall"]].copy()
    for column in ["roc_auc_cv", "pr_auc_cv", "recall"]:
        chart_data[column] = pd.to_numeric(chart_data[column], errors="coerce")
    chart_data = chart_data.dropna().sort_values("pr_auc_cv")
    if chart_data.empty:
        st.info("Model benchmark diagnostics are not available in this artifact.")
        return

    if _plot_benchmark_altair(
        chart_data,
        [("roc_auc_cv", "ROC-AUC CV"), ("pr_auc_cv", "PR-AUC CV"), ("recall", "Recall")],
        "Candidate-model benchmark",
    ):
        st.caption("XGBoost is highlighted in gold. The three panels match the V3 report benchmark figure; hover a bar to see the exact score.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.bar_chart(chart_data.set_index("model"), y=["roc_auc_cv", "pr_auc_cv", "recall"])
        return

    colors = ["#F2A900" if name == "XGBoost" else "#6C9BC3" for name in chart_data["model"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, max(4.5, 0.38 * len(chart_data))), sharey=True)
    for axis, column, title in [
        (axes[0], "roc_auc_cv", "ROC-AUC CV"),
        (axes[1], "pr_auc_cv", "PR-AUC CV"),
        (axes[2], "recall", "Recall"),
    ]:
        axis.barh(chart_data["model"], chart_data[column], color=colors)
        axis.set_xlim(0, 1)
        axis.set_title(title)
        axis.set_xlabel("Score (0–1)")
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_ylabel("Candidate model")
    fig.suptitle("Candidate-model benchmark", y=1.02)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption("XGBoost is highlighted in gold. The three panels match the V3 report benchmark figure.")


def _plot_benchmark_audit(benchmark: pd.DataFrame) -> None:
    """Show the full candidate-model audit as small-multiple charts."""

    metric_labels = {
        "pr_auc_cv": "PR-AUC CV",
        "roc_auc_cv": "ROC-AUC CV",
        "f1": "F1",
        "recall": "Recall",
    }
    available = [column for column in metric_labels if column in benchmark.columns]
    if benchmark.empty or "model" not in benchmark.columns or not available:
        st.info("Full benchmark audit values are not available in this artifact.")
        return
    if _plot_benchmark_altair(benchmark, list(metric_labels.items()), "Full candidate-model benchmark audit"):
        st.caption("Gold identifies XGBoost. Hover a bar to see the metric value and model.")
        return
    chart_data = benchmark[["model", *available]].copy()
    for column in available:
        chart_data[column] = pd.to_numeric(chart_data[column], errors="coerce")
    sort_column = "pr_auc_cv" if "pr_auc_cv" in available else available[0]
    chart_data = chart_data.dropna(subset=["model"]).sort_values(sort_column)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.bar_chart(chart_data.set_index("model"), y=available)
        return
    colors = ["#F2A900" if name == "XGBoost" else "#6C9BC3" for name in chart_data["model"]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharey=True)
    axes = axes.flatten()
    for axis, column in zip(axes, available):
        bars = axis.barh(chart_data["model"], chart_data[column], color=colors)
        axis.set_xlim(0, 1)
        axis.set_title(metric_labels[column])
        axis.set_xlabel("Score (0–1)")
        axis.grid(axis="x", alpha=0.2)
        for bar, value in zip(bars, chart_data[column]):
            if pd.notna(value):
                axis.text(float(value) + 0.015, bar.get_y() + bar.get_height() / 2, f"{float(value):.3f}", va="center", fontsize=8)
    for axis in axes[len(available):]:
        axis.axis("off")
    axes[0].set_ylabel("Candidate model")
    fig.suptitle("Full candidate-model benchmark audit", y=1.02)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption("Gold identifies XGBoost. PR-AUC is the primary ranking metric; recall shows the screening trade-off.")


def _plot_validation_test_metrics(metadata: dict) -> None:
    """Show validation/test performance as a compact grouped chart."""

    validation = metadata.get("validation_metrics", {})
    test = metadata.get("test_metrics", {})
    metric_labels = {
        "accuracy": "Accuracy",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "f1": "F1",
        "f2": "F2",
        "precision": "Precision",
        "recall": "Recall",
    }
    rows = []
    for key, label in metric_labels.items():
        if key in validation or key in test:
            rows.append(
                {
                    "Metric": label,
                    "Validation": pd.to_numeric(validation.get(key), errors="coerce"),
                    "Test": pd.to_numeric(test.get(key), errors="coerce"),
                }
            )
    chart_data = pd.DataFrame(rows).dropna(subset=["Validation", "Test"], how="all")
    if chart_data.empty:
        st.info("Validation/test metrics are not available in this artifact.")
        return
    try:
        import altair as alt
    except ImportError:
        alt = None
    if alt is not None:
        plot_rows = []
        for _, row in chart_data.iterrows():
            for split, source_column in [("Validation", "Validation"), ("Locked test", "Test")]:
                value = pd.to_numeric(row[source_column], errors="coerce")
                if pd.notna(value):
                    plot_rows.append({"Metric": row["Metric"], "Split": split, "Score": float(value)})
        plot_data = pd.DataFrame(plot_rows)
        metric_order = chart_data["Metric"].astype(str).tolist()
        chart = (
            alt.Chart(plot_data)
            .mark_bar()
            .encode(
                x=alt.X("Metric:N", title="Metric", sort=metric_order, axis=alt.Axis(labelAngle=-25)),
                y=alt.Y("Score:Q", title="Score (0–1)", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "Split:N",
                    title=None,
                    scale=alt.Scale(domain=["Validation", "Locked test"], range=["#6C9BC3", "#2F7F73"]),
                ),
                xOffset=alt.XOffset("Split:N"),
                tooltip=[
                    alt.Tooltip("Metric:N", title="Metric"),
                    alt.Tooltip("Split:N", title="Split"),
                    alt.Tooltip("Score:Q", title="Score", format=".3f"),
                ],
            )
            .properties(title="Validation versus locked-test performance", height=320)
        )
        st.altair_chart(chart, width="stretch")
        st.caption("Higher is better for every displayed metric. Hover a bar for the exact split and score; Brier score is shown separately because lower is better.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.bar_chart(chart_data.set_index("Metric"), y=["Validation", "Test"])
        return

    positions = list(range(len(chart_data)))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 5.2))
    validation_values = chart_data["Validation"].fillna(0).tolist()
    test_values = chart_data["Test"].fillna(0).tolist()
    axis.bar([position - width / 2 for position in positions], validation_values, width, label="Validation", color="#6C9BC3")
    axis.bar([position + width / 2 for position in positions], test_values, width, label="Locked test", color="#2F7F73")
    axis.set_xticks(positions, chart_data["Metric"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score (0–1)")
    axis.set_title("Validation versus locked-test performance")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption("Higher is better for every displayed metric. Brier score is shown separately because lower is better.")


def _plot_bootstrap_ci(metadata: dict) -> None:
    """Show uncertainty around locked-test metrics without a large CI table."""

    ci = metadata.get("bootstrap_95_ci", {})
    test_metrics = metadata.get("test_metrics", {})
    records = []
    for name, values in ci.items():
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            continue
        estimate = pd.to_numeric(test_metrics.get(name), errors="coerce")
        lower = pd.to_numeric(values[0], errors="coerce")
        upper = pd.to_numeric(values[1], errors="coerce")
        if pd.isna(estimate) or pd.isna(lower) or pd.isna(upper):
            continue
        records.append({"Metric": name.replace("_", " ").upper(), "Estimate": estimate, "Lower": lower, "Upper": upper})
    if not records:
        return
    chart_data = pd.DataFrame(records).sort_values("Estimate")
    try:
        import altair as alt
    except ImportError:
        alt = None
    if alt is not None:
        interval = (
            alt.Chart(chart_data)
            .mark_rule(color="#6C9BC3", size=5)
            .encode(
                y=alt.Y("Metric:N", title="Metric", sort=chart_data["Metric"].tolist()),
                x=alt.X("Lower:Q", title="Score (0–1)", scale=alt.Scale(domain=[0, 1])),
                x2="Upper:Q",
            )
        )
        points = (
            alt.Chart(chart_data)
            .mark_point(color="#2F7F73", filled=True, size=90)
            .encode(
                y=alt.Y("Metric:N", sort=chart_data["Metric"].tolist()),
                x=alt.X("Estimate:Q"),
                tooltip=[
                    alt.Tooltip("Metric:N", title="Metric"),
                    alt.Tooltip("Lower:Q", title="95% CI lower", format=".3f"),
                    alt.Tooltip("Estimate:Q", title="Estimate", format=".3f"),
                    alt.Tooltip("Upper:Q", title="95% CI upper", format=".3f"),
                ],
            )
        )
        st.altair_chart(
            (interval + points).properties(
                title="Locked-test estimate with bootstrap 95% CI",
                height=max(220, 34 * len(chart_data)),
            ),
            width="stretch",
        )
        st.caption("Hover a point to see the estimate and both bootstrap 95% confidence-interval bounds.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.bar_chart(chart_data.set_index("Metric")[["Lower", "Estimate", "Upper"]])
        st.caption("Lower, estimate and upper lines show the bootstrap 95% interval when advanced plotting is unavailable.")
        return

    y_positions = list(range(len(chart_data)))
    fig, axis = plt.subplots(figsize=(9, max(3.5, 0.42 * len(chart_data))))
    axis.errorbar(
        chart_data["Estimate"],
        y_positions,
        xerr=[chart_data["Estimate"] - chart_data["Lower"], chart_data["Upper"] - chart_data["Estimate"]],
        fmt="o",
        color="#2F7F73",
        ecolor="#6C9BC3",
        capsize=4,
    )
    axis.set_yticks(y_positions, chart_data["Metric"])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Score (0–1)")
    axis.set_title("Locked-test estimate with bootstrap 95% CI")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption("The dot is the stored test estimate; the horizontal line is its bootstrap 95% confidence interval.")


def _plot_report_active_test_metrics(active_test: dict) -> None:
    """Compare the supplied V3 test evidence with the currently loaded artifact."""

    labels = {
        "accuracy": "Accuracy",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "f1": "F1",
        "f2": "F2",
        "precision": "Precision",
        "recall": "Recall",
    }
    rows = []
    for key, label in labels.items():
        report_value = REPORT_V3_MODEL_REFERENCE["test_metrics"].get(key)
        active_value = pd.to_numeric(active_test.get(key), errors="coerce")
        if report_value is not None or not pd.isna(active_value):
            rows.append({"Metric": label, "V3 report": report_value, "Active artifact": active_value})
    chart_data = pd.DataFrame(rows).dropna(subset=["V3 report", "Active artifact"], how="all")
    if chart_data.empty:
        return
    try:
        import altair as alt
    except ImportError:
        alt = None
    if alt is not None:
        plot_rows = []
        for _, row in chart_data.iterrows():
            for source, column in [("V3 report", "V3 report"), ("Active artifact", "Active artifact")]:
                value = pd.to_numeric(row[column], errors="coerce")
                if pd.notna(value):
                    plot_rows.append({"Metric": row["Metric"], "Source": source, "Score": float(value)})
        plot_data = pd.DataFrame(plot_rows)
        metric_order = chart_data["Metric"].astype(str).tolist()
        chart = (
            alt.Chart(plot_data)
            .mark_bar()
            .encode(
                x=alt.X("Metric:N", title="Metric", sort=metric_order, axis=alt.Axis(labelAngle=-25)),
                y=alt.Y("Score:Q", title="Score (0–1)", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "Source:N",
                    title=None,
                    scale=alt.Scale(domain=["V3 report", "Active artifact"], range=["#9B8AFB", "#2F7F73"]),
                ),
                xOffset=alt.XOffset("Source:N"),
                tooltip=[
                    alt.Tooltip("Metric:N", title="Metric"),
                    alt.Tooltip("Source:N", title="Source"),
                    alt.Tooltip("Score:Q", title="Score", format=".3f"),
                ],
            )
            .properties(title="Reported V3 test evidence versus active artifact", height=320)
        )
        st.altair_chart(chart, width="stretch")
        st.caption("Hover a bar to compare the exact V3-report value with the active-artifact value. This is an audit comparison, not an automatic reproduction claim.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.bar_chart(chart_data.set_index("Metric"), y=["V3 report", "Active artifact"])
        return

    positions = list(range(len(chart_data)))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 5.2))
    axis.bar([position - width / 2 for position in positions], chart_data["V3 report"].fillna(0), width, label="V3 report", color="#9B8AFB")
    axis.bar([position + width / 2 for position in positions], chart_data["Active artifact"].fillna(0), width, label="Active artifact", color="#2F7F73")
    axis.set_xticks(positions, chart_data["Metric"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score (0–1)")
    axis.set_title("Reported V3 test evidence versus active artifact")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption("This is an audit comparison: the active artifact may be a later run and is not automatically an exact reproduction of V3.")


def _plot_threshold_tradeoff(metadata: dict, selected_threshold: float) -> None:
    """Plot the validation-set operating trade-off across candidate thresholds."""

    sweep = pd.DataFrame(metadata.get("threshold_selection", {}).get("table", []))
    required = {"threshold", "precision", "recall", "specificity", "f2"}
    if sweep.empty or not required.issubset(sweep.columns):
        st.info(
            _t(
                "Threshold-sweep evidence is not available in this artifact.",
                "Artifact này không có dữ liệu quét ngưỡng.",
            )
        )
        return
    for column in required:
        sweep[column] = pd.to_numeric(sweep[column], errors="coerce")
    sweep = sweep.dropna(subset=["threshold"]).sort_values("threshold")
    try:
        import altair as alt
    except ImportError:
        st.line_chart(
            sweep.set_index("threshold")[["precision", "recall", "specificity", "f2"]],
            height=300,
        )
        return

    plot_data = sweep.melt(
        id_vars=["threshold"],
        value_vars=["precision", "recall", "specificity", "f2"],
        var_name="Metric",
        value_name="Score",
    )
    plot_data["Metric"] = plot_data["Metric"].map(
        {
            "precision": "Precision",
            "recall": "Recall",
            "specificity": "Specificity",
            "f2": "F2",
        }
    )
    curves = (
        alt.Chart(plot_data)
        .mark_line()
        .encode(
            x=alt.X("threshold:Q", title="Decision threshold"),
            y=alt.Y("Score:Q", title="Validation score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "Metric:N",
                title=None,
                scale=alt.Scale(
                    domain=["Precision", "Recall", "Specificity", "F2"],
                    range=["#4C72B0", "#C44E52", "#55A868", "#8172B2"],
                ),
            ),
            tooltip=[
                alt.Tooltip("threshold:Q", title="Threshold", format=".2f"),
                alt.Tooltip("Metric:N", title="Metric"),
                alt.Tooltip("Score:Q", title="Score", format=".3f"),
            ],
        )
    )
    references = pd.DataFrame(
        [
            {"Threshold": 0.20, "Reference": "Requested ≈0.20"},
            {"Threshold": selected_threshold, "Reference": "Selected by F2"},
        ]
    )
    rules = (
        alt.Chart(references)
        .mark_rule(strokeDash=[6, 4], size=2)
        .encode(
            x=alt.X("Threshold:Q"),
            color=alt.Color(
                "Reference:N",
                title="Reference",
                scale=alt.Scale(
                    domain=["Requested ≈0.20", "Selected by F2"],
                    range=["#F2A900", "#2F7F73"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Reference:N", title="Reference"),
                alt.Tooltip("Threshold:Q", title="Threshold", format=".3f"),
            ],
        )
    )
    chart = alt.layer(curves, rules).resolve_scale(color="independent").properties(
        title="Validation threshold trade-off",
        height=300,
    )
    st.altair_chart(chart, width="stretch")


def _plot_feature_reduction_funnel(selection: pd.DataFrame, selected_count: int) -> None:
    """Show the stored reduction path from the full set to the selected set."""

    required = {"feature_count", "accuracy_cv", "pr_auc_cv"}
    if selection.empty or not required.issubset(selection.columns):
        return
    chart_data = selection[["feature_count", "accuracy_cv", "pr_auc_cv"]].copy()
    for column in required:
        chart_data[column] = pd.to_numeric(chart_data[column], errors="coerce")
    chart_data = chart_data.dropna().loc[lambda frame: frame["feature_count"] >= selected_count]
    chart_data = chart_data.sort_values("feature_count", ascending=False)
    if chart_data.empty:
        return
    chart_data["Stage"] = chart_data["feature_count"].astype(int).map(
        lambda count: _t(
            f"{count} features" + (" · selected" if count == selected_count else ""),
            f"{count} đặc trưng" + (" · được chọn" if count == selected_count else ""),
        )
    )
    chart_data["Selected"] = chart_data["feature_count"].eq(selected_count)
    try:
        import altair as alt
    except ImportError:
        st.bar_chart(chart_data.set_index("Stage")[["feature_count"]], height=280)
        return

    stage_order = chart_data["Stage"].tolist()
    maximum = float(chart_data["feature_count"].max())
    bars = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X(
                "feature_count:Q",
                title=_t("Retained feature count", "Số đặc trưng được giữ"),
                scale=alt.Scale(domain=[0, maximum * 1.15]),
            ),
            y=alt.Y("Stage:N", title=None, sort=stage_order),
            color=alt.condition(
                alt.datum.Selected,
                alt.value("#F2A900"),
                alt.value("#6C9BC3"),
            ),
            tooltip=[
                alt.Tooltip("feature_count:Q", title="Features", format="d"),
                alt.Tooltip("accuracy_cv:Q", title="Accuracy CV", format=".3f"),
                alt.Tooltip("pr_auc_cv:Q", title="PR-AUC CV", format=".3f"),
            ],
        )
    )
    labels = (
        alt.Chart(chart_data)
        .mark_text(align="left", baseline="middle", dx=5, fontWeight="bold")
        .encode(
            x=alt.X("feature_count:Q"),
            y=alt.Y("Stage:N", sort=stage_order),
            text=alt.Text("feature_count:Q", format=".0f"),
        )
    )
    chart = (bars + labels).properties(
        title=_t("Feature-reduction path: 32 → 5", "Quá trình tinh gọn: 32 → 5"),
        height=max(230, 42 * len(chart_data)),
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        _t(
            "Gold marks the final selected set. Hover a stage to inspect its cross-validation scores.",
            "Màu vàng là bộ cuối được chọn. Di chuột lên từng bước để xem điểm cross-validation.",
        )
    )


def _plot_feature_selection(selection: pd.DataFrame, selected_count: int) -> None:
    """Show the accuracy/PR-AUC trade-off as feature count changes."""

    required = {"feature_count", "accuracy_cv", "pr_auc_cv"}
    if selection.empty or not required.issubset(selection.columns):
        st.info("Feature-count comparison is not available in this artifact.")
        return
    chart_data = selection[["feature_count", "accuracy_cv", "pr_auc_cv"]].copy()
    for column in ["feature_count", "accuracy_cv", "pr_auc_cv"]:
        chart_data[column] = pd.to_numeric(chart_data[column], errors="coerce")
    chart_data = chart_data.dropna().sort_values("feature_count")
    if chart_data.empty:
        st.info("Feature-count comparison is not available in this artifact.")
        return
    try:
        import altair as alt
    except ImportError:
        alt = None
    if alt is not None:
        plot_data = chart_data.melt(
            id_vars=["feature_count"],
            value_vars=["accuracy_cv", "pr_auc_cv"],
            var_name="Metric",
            value_name="Score",
        )
        plot_data["Metric"] = plot_data["Metric"].map({"accuracy_cv": "Accuracy CV", "pr_auc_cv": "PR-AUC CV"})
        chart = (
            alt.Chart(plot_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("feature_count:Q", title="Number of features", scale=alt.Scale(zero=False)),
                y=alt.Y("Score:Q", title="Cross-validation score (0–1)", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Metric:N", title=None, scale=alt.Scale(domain=["Accuracy CV", "PR-AUC CV"], range=["#6C9BC3", "#2F7F73"])),
                tooltip=[
                    alt.Tooltip("feature_count:Q", title="Features", format="d"),
                    alt.Tooltip("Metric:N", title="Metric"),
                    alt.Tooltip("Score:Q", title="Score", format=".3f"),
                ],
            )
            .properties(title="Compactness versus cross-validation performance", height=300)
        )
        selected = chart_data[chart_data["feature_count"].eq(selected_count)]
        if not selected.empty:
            selected_point = (
                alt.Chart(selected)
                .mark_point(color="#F2A900", filled=True, size=150, stroke="#333333", strokeWidth=1)
                .encode(
                    x=alt.X("feature_count:Q"),
                    y=alt.Y("pr_auc_cv:Q"),
                    tooltip=[
                        alt.Tooltip("feature_count:Q", title="Selected features", format="d"),
                        alt.Tooltip("accuracy_cv:Q", title="Accuracy CV", format=".3f"),
                        alt.Tooltip("pr_auc_cv:Q", title="PR-AUC CV", format=".3f"),
                    ],
                )
            )
            chart = chart + selected_point
        st.altair_chart(chart, width="stretch")
        st.caption("The selected point is highlighted in gold. Hover a point to see the exact feature count and cross-validation score.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.line_chart(chart_data.set_index("feature_count"), y=["accuracy_cv", "pr_auc_cv"])
        return

    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(chart_data["feature_count"], chart_data["accuracy_cv"], marker="o", label="Accuracy CV", color="#6C9BC3")
    axis.plot(chart_data["feature_count"], chart_data["pr_auc_cv"], marker="o", label="PR-AUC CV", color="#2F7F73")
    selected = chart_data[chart_data["feature_count"].eq(selected_count)]
    if not selected.empty:
        axis.scatter(selected["feature_count"], selected["pr_auc_cv"], s=120, color="#F2A900", edgecolor="#333333", zorder=4, label=f"Selected: {selected_count} features")
    axis.set_xticks(chart_data["feature_count"].astype(int).tolist())
    axis.set_ylim(0, 1)
    axis.set_xlabel("Number of features")
    axis.set_ylabel("Cross-validation score (0–1)")
    axis.set_title("Compactness versus cross-validation performance")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption("The selected point is highlighted; a smaller set is preferred only when performance remains comparable.")


def _plot_imbalance_comparison(imbalance: pd.DataFrame, selected_method: str) -> None:
    """Compare imbalance strategies using the metrics shown in report Figure 7."""

    required = {"method", "accuracy", "pr_auc", "f1", "recall", "brier"}
    if imbalance.empty or not required.issubset(imbalance.columns):
        st.info("Imbalance-method comparison is not available in this artifact.")
        return
    chart_data = imbalance.copy()
    for column in ["accuracy", "pr_auc", "f1", "recall", "brier"]:
        chart_data[column] = pd.to_numeric(chart_data[column], errors="coerce")
    chart_data = chart_data.dropna(subset=["accuracy", "pr_auc", "f1", "recall", "brier"])
    if chart_data.empty:
        st.info("Imbalance-method comparison is not available in this artifact.")
        return
    try:
        import altair as alt
    except ImportError:
        alt = None
    if alt is not None:
        screening_data = chart_data.melt(
            id_vars=["method"],
            value_vars=["accuracy", "pr_auc", "f1", "recall"],
            var_name="Metric",
            value_name="Score",
        )
        screening_data["Metric"] = screening_data["Metric"].map(
            {"accuracy": "Accuracy", "pr_auc": "PR-AUC", "f1": "F1", "recall": "Recall"}
        )
        metric_order = ["Accuracy", "PR-AUC", "F1", "Recall"]
        method_order = chart_data["method"].astype(str).tolist()
        category_axis = alt.Axis(
            labelAngle=0,
            labelLimit=180,
            labelOverlap=False,
            labelPadding=8,
            titlePadding=8,
        )
        screening_chart = (
            alt.Chart(screening_data)
            .mark_bar()
            .encode(
                x=alt.X("method:N", title="Method", sort=method_order, axis=category_axis),
                y=alt.Y(
                    "Score:Q",
                    title="Score",
                    scale=alt.Scale(domain=[0.8, 1], zero=False),
                ),
                # The report compares scores in the 0.80–1.00 band.  Set the
                # bar baseline explicitly so Vega-Lite draws the truncated
                # bars instead of treating the off-screen zero baseline as
                # the second endpoint (which produces height 0).
                y2=alt.datum(0.8),
                color=alt.Color(
                    "Metric:N",
                    title=None,
                    sort=metric_order,
                    scale=alt.Scale(
                        domain=metric_order,
                        range=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
                    ),
                ),
                xOffset=alt.XOffset("Metric:N", sort=metric_order),
                tooltip=[
                    alt.Tooltip("method:N", title="Method"),
                    alt.Tooltip("Metric:N", title="Metric"),
                    alt.Tooltip("Score:Q", title="Score", format=".3f"),
                ],
            )
            .properties(title="Screening metrics", width="container", height=300)
        )
        brier_chart = (
            alt.Chart(chart_data)
            .mark_bar(color="#8172B2")
            .encode(
                x=alt.X("method:N", title="Method", sort=method_order, axis=category_axis),
                y=alt.Y("brier:Q", title="Brier", scale=alt.Scale(domain=[0, 0.06])),
                tooltip=[
                    alt.Tooltip("method:N", title="Method"),
                    alt.Tooltip("brier:Q", title="Brier score", format=".4f"),
                ],
            )
            .properties(title="Brier Score (thấp hơn = tốt hơn)", width="container", height=300)
        )
        # Keep the two plots full-width and vertically stacked.  A horizontal
        # concat is too narrow on the Streamlit viewport, so long method names
        # (especially ``scale_pos_weight``) overlap the plot edge or get cut.
        st.altair_chart(screening_chart, width="stretch")
        st.altair_chart(brier_chart, width="stretch")
        st.caption(
            f"Configured method: {selected_method}. The screening panel includes Accuracy, PR-AUC, F1 and Recall; the Brier panel uses the lower-is-better scale from the V3 report. Hover a bar for the exact value."
        )
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.bar_chart(chart_data.set_index("method"), y=["accuracy", "pr_auc", "f1", "recall", "brier"])
        return

    positions = list(range(len(chart_data)))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2.4, 1]})
    for offset, column, label, color in [
        (-1.5 * width, "accuracy", "ACCURACY", "#4C72B0"),
        (-0.5 * width, "pr_auc", "PR-AUC", "#DD8452"),
        (0.5 * width, "f1", "F1", "#55A868"),
        (1.5 * width, "recall", "RECALL", "#C44E52"),
    ]:
        axes[0].bar([position + offset for position in positions], chart_data[column], width, label=label, color=color)
    axes[0].set_xticks(positions, chart_data["method"])
    axes[0].set_ylim(0.8, 1.0)
    axes[0].set_ylabel("Score (0–1)")
    axes[0].set_title("Screening metrics")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend()
    axes[1].bar(chart_data["method"], chart_data["brier"], color="#8172B2")
    axes[1].set_title("Brier Score (thấp hơn = tốt hơn)")
    axes[1].set_ylabel("Brier")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Class-imbalance strategy comparison", y=1.02)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption("The screening panel includes Accuracy, PR-AUC, F1 and Recall; Brier rewards probability calibration. The configured method is recorded as: " + str(selected_method) + ".")


def _confusion_counts(confusion: dict | None) -> tuple[int, int, int, int] | None:
    """Normalize a stored or report confusion matrix to TN, FP, FN, TP."""

    if not confusion:
        return None
    if all(key in confusion for key in ["TN", "FP", "FN", "TP"]):
        return tuple(int(confusion[key]) for key in ["TN", "FP", "FN", "TP"])
    counts = confusion.get("counts")
    if isinstance(counts, list) and len(counts) == 2 and all(isinstance(row, list) and len(row) == 2 for row in counts):
        return int(counts[0][0]), int(counts[0][1]), int(counts[1][0]), int(counts[1][1])
    return None


def _plot_confusion_matrix(confusion: dict, title: str) -> tuple[int, int, int, int] | None:
    """Show a confusion matrix as a heatmap, with a bar fallback for minimal installs."""

    counts = _confusion_counts(confusion)
    if counts is None:
        st.info("Confusion-matrix counts are not available in this artifact.")
        return None
    tn, fp, fn, tp = counts
    matrix = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual negative", "Actual positive"],
        columns=["Predicted negative", "Predicted positive"],
    )
    try:
        import altair as alt
    except ImportError:
        alt = None
    if alt is not None:
        chart_data = matrix.rename_axis("Observed label").reset_index().melt(
            id_vars=["Observed label"],
            var_name="Prediction",
            value_name="Patients",
        )
        heatmap = (
            alt.Chart(chart_data)
            .mark_rect()
            .encode(
                x=alt.X("Prediction:N", title="Prediction"),
                y=alt.Y("Observed label:N", title="Observed label"),
                color=alt.Color("Patients:Q", title="Patients"),
                tooltip=[
                    alt.Tooltip("Observed label:N", title="Observed label"),
                    alt.Tooltip("Prediction:N", title="Prediction"),
                    alt.Tooltip("Patients:Q", title="Patients", format=","),
                ],
            )
        )
        labels = heatmap.mark_text(color="#111111", fontSize=16).encode(text=alt.Text("Patients:Q", format=","))
        st.altair_chart(
            (heatmap + labels).properties(title=title, height=260),
            width="stretch",
        )
        st.caption("Hover a cell to see the exact TN/FP/FN/TP count.")
        return counts
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.bar_chart(pd.DataFrame({"Count": [tn, fp, fn, tp]}, index=["TN", "FP", "FN", "TP"]))
        st.caption("TN=true negative, FP=false positive, FN=false negative, TP=true positive.")
        return counts

    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    image = axis.imshow(matrix.to_numpy(), cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, int(matrix.iloc[row, column]), ha="center", va="center", fontsize=16, color="#111111")
    axis.set_xticks([0, 1], matrix.columns)
    axis.set_yticks([0, 1], matrix.index)
    axis.set_xlabel("Prediction")
    axis.set_ylabel("Observed label")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, shrink=0.82, label="Patients")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    return counts


def _confusion_operating_metrics(counts: tuple[int, int, int, int]) -> dict[str, float]:
    tn, fp, fn, tp = counts
    return {
        "Sensitivity / Recall": tp / (tp + fn) if tp + fn else 0.0,
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "PPV / Precision": tp / (tp + fp) if tp + fp else 0.0,
        "NPV": tn / (tn + fn) if tn + fn else 0.0,
    }


def _plot_calibration_curve(metadata: dict[str, Any] | None = None) -> None:
    """Render the validation calibration curve used by report Figure 9."""

    calibration = (metadata or {}).get("calibration", {})
    before = calibration.get("before", {})
    after = calibration.get("after", {})
    required = {"mean_predicted", "fraction_positive", "brier"}
    if not required.issubset(before) or not required.issubset(after):
        return
    if not before["mean_predicted"] or not after["mean_predicted"]:
        return

    rows = []
    for stage, evidence in [
        (f"Raw score (Brier={float(before['brier']):.4f})", before),
        (f"Calibrated - Platt (Brier={float(after['brier']):.4f})", after),
    ]:
        for mean_predicted, fraction_positive in zip(
            evidence["mean_predicted"], evidence["fraction_positive"]
        ):
            rows.append(
                {
                    "Mean predicted score": float(mean_predicted),
                    "Observed positive rate": float(fraction_positive),
                    "Stage": stage,
                }
            )
    curve_data = pd.DataFrame(rows)
    if curve_data.empty:
        return

    try:
        import altair as alt
    except ImportError:
        alt = None
    if alt is not None:
        reference = pd.DataFrame(
            {
                "Mean predicted score": [0.0, 1.0],
                "Observed positive rate": [0.0, 1.0],
            }
        )
        reference_chart = (
            alt.Chart(reference)
            .mark_line(strokeDash=[5, 5], color="#222222")
            .encode(
                x=alt.X("Mean predicted score:Q", title="Mean predicted score", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Observed positive rate:Q", title="Observed positive rate", scale=alt.Scale(domain=[0, 1])),
            )
        )
        curve_chart = (
            alt.Chart(curve_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("Mean predicted score:Q", title="Mean predicted score", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Observed positive rate:Q", title="Observed positive rate", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "Stage:N",
                    title=None,
                    scale=alt.Scale(
                        domain=[
                            f"Raw score (Brier={float(before['brier']):.4f})",
                            f"Calibrated - Platt (Brier={float(after['brier']):.4f})",
                        ],
                        range=["#1f77b4", "#ff7f0e"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("Stage:N", title="Stage"),
                    alt.Tooltip("Mean predicted score:Q", title="Mean predicted score", format=".4f"),
                    alt.Tooltip("Observed positive rate:Q", title="Observed positive rate", format=".4f"),
                ],
            )
        )
        st.altair_chart(
            alt.layer(reference_chart, curve_chart).properties(
                title="Calibration Curve (validation set)", height=320
            ),
            width="stretch",
        )
        st.caption("Dashed line = perfect calibration. The curve uses the saved validation bins from the active artifact.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.plot([0, 1], [0, 1], "k--", label="Calibraton hoàn hảo")
    axis.plot(before["mean_predicted"], before["fraction_positive"], "o-", label=f"Raw score (Brier={float(before['brier']):.4f})")
    axis.plot(after["mean_predicted"], after["fraction_positive"], "s-", label=f"Calibrated - Platt (Brier={float(after['brier']):.4f})")
    axis.set_xlabel("Mean predicted score")
    axis.set_ylabel("Observed positive rate")
    axis.set_title("Calibration Curve (validation set)")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)
    st.caption("Dashed line = perfect calibration. The curve uses the saved validation bins from the active artifact.")


def _plot_calibration_evidence(metadata: dict[str, Any] | None = None) -> None:
    """Show active-artifact calibration evidence, with V3 fallback for legacy artifacts."""

    active_calibration = (metadata or {}).get("calibration", {})
    before = active_calibration.get("before", {})
    after = active_calibration.get("after", {})
    using_active_evidence = all(
        key in before and key in after for key in ("brier", "ece")
    )
    if using_active_evidence:
        calibration = {
            "brier_before": float(before["brier"]),
            "brier_after": float(after["brier"]),
            "ece_before": float(before["ece"]),
            "ece_after": float(after["ece"]),
        }
    else:
        calibration = REPORT_V3_MODEL_REFERENCE["calibration"]
    chart_data = pd.DataFrame(
        {
            "Metric": ["Brier score", "ECE"],
            "Before sigmoid/Platt": [calibration["brier_before"], calibration["ece_before"]],
            "After sigmoid/Platt": [calibration["brier_after"], calibration["ece_after"]],
        }
    )
    try:
        import altair as alt
    except ImportError:
        alt = None
    if alt is not None:
        plot_data = chart_data.melt(id_vars=["Metric"], var_name="Stage", value_name="Error")
        chart = (
            alt.Chart(plot_data)
            .mark_bar()
            .encode(
                x=alt.X("Metric:N", title="Metric"),
                y=alt.Y("Error:Q", title="Error (lower is better)"),
                color=alt.Color("Stage:N", title=None, scale=alt.Scale(domain=["Before sigmoid/Platt", "After sigmoid/Platt"], range=["#9B8AFB", "#2F7F73"])),
                xOffset=alt.XOffset("Stage:N"),
                tooltip=[
                    alt.Tooltip("Metric:N", title="Metric"),
                    alt.Tooltip("Stage:N", title="Stage"),
                    alt.Tooltip("Error:Q", title="Error", format=".4f"),
                ],
            )
            .properties(title="Active-artifact calibration evidence" if using_active_evidence else "V3 calibration evidence", height=300)
        )
        st.altair_chart(chart, width="stretch")
        st.caption(
            "Hover a bar to see the exact before/after calibration error. Lower is better for both metrics."
            if using_active_evidence
            else "Hover a bar to see the supplied V3 report's before/after calibration error. Lower is better for both metrics."
        )
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.bar_chart(chart_data.set_index("Metric"), y=["Before sigmoid/Platt", "After sigmoid/Platt"])
        st.caption(
            "Lower is better for both calibration metrics."
            if using_active_evidence
            else "Lower is better for both calibration metrics. Values are the supplied V3 report evidence."
        )
        return

    positions = list(range(len(chart_data)))
    width = 0.36
    fig, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.bar([position - width / 2 for position in positions], chart_data["Before sigmoid/Platt"], width, label="Before", color="#9B8AFB")
    axis.bar([position + width / 2 for position in positions], chart_data["After sigmoid/Platt"], width, label="After", color="#2F7F73")
    axis.set_xticks(positions, chart_data["Metric"])
    axis.set_ylabel("Error (lower is better)")
    axis.set_title("Active-artifact calibration evidence" if using_active_evidence else "V3 calibration evidence")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption(
        (
            "The active artifact records "
            f"Brier {calibration['brier_before']:.4f} → {calibration['brier_after']:.4f} and "
            f"ECE {calibration['ece_before']:.4f} → {calibration['ece_after']:.4f} after sigmoid/Platt calibration."
        )
        if using_active_evidence
        else (
            "The supplied V3 report reference records "
            f"Brier {REPORT_V3_MODEL_REFERENCE['calibration']['brier_before']:.4f} → "
            f"{REPORT_V3_MODEL_REFERENCE['calibration']['brier_after']:.4f} and ECE "
            f"{REPORT_V3_MODEL_REFERENCE['calibration']['ece_before']:.4f} → "
            f"{REPORT_V3_MODEL_REFERENCE['calibration']['ece_after']:.4f} after sigmoid/Platt calibration."
        )
    )


def _plot_roc_curve(payload: dict[str, Any]) -> None:
    """Render the locked-test ROC curve used by report Figure 11."""

    locked_path = DEFAULT_ARTIFACTS_DIR.parent / "data" / "locked_test.csv"
    if not locked_path.exists():
        return
    try:
        from sklearn.metrics import auc, roc_curve

        locked = pd.read_csv(locked_path)
        if TARGET_COLUMN not in locked.columns:
            return
        scored = predict_dataframe(payload, locked)
        false_positive_rate, true_positive_rate, _ = roc_curve(
            locked[TARGET_COLUMN].astype(int), scored["calibrated_score"]
        )
        area = float(auc(false_positive_rate, true_positive_rate))
    except Exception:
        return

    curve_data = pd.DataFrame(
        {
            "False Positive Rate": list(false_positive_rate) + [0.0, 1.0],
            "True Positive Rate": list(true_positive_rate) + [0.0, 1.0],
            "Series": [f"XGBoost (AUC={area:.3f})"] * len(false_positive_rate)
            + ["Random (AUC=0.5)", "Random (AUC=0.5)"],
        }
    )
    try:
        import altair as alt
    except ImportError:
        alt = None
    if alt is not None:
        chart = (
            alt.Chart(curve_data)
            .mark_line()
            .encode(
                x=alt.X("False Positive Rate:Q", title="False Positive Rate", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("True Positive Rate:Q", title="True Positive Rate", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "Series:N",
                    title=None,
                    scale=alt.Scale(
                        domain=[f"XGBoost (AUC={area:.3f})", "Random (AUC=0.5)"],
                        range=["#DD8452", "#222222"],
                    ),
                ),
                tooltip=[alt.Tooltip("Series:N", title="Curve")],
            )
            .properties(title="ROC Curve - Model cuối cùng (test set)", height=320)
        )
        st.altair_chart(chart, width="stretch")
        st.caption("The curve is calculated on the locked internal test cohort; it is not used to select the model or threshold.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    axis.plot(false_positive_rate, true_positive_rate, color="#DD8452", label=f"XGBoost (AUC={area:.3f})")
    axis.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve - Model cuối cùng (test set)")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)
    st.caption("The curve is calculated on the locked internal test cohort; it is not used to select the model or threshold.")


def _plot_selected_feature_evidence(evidence: pd.DataFrame) -> None:
    """Show the five final input columns and their validation permutation importance."""

    if evidence.empty or "Permutation importance" not in evidence.columns:
        st.info("Selected-feature evidence is not available in this artifact.")
        return
    chart_data = evidence[["Column", "Permutation importance"]].copy()
    chart_data["Permutation importance"] = pd.to_numeric(chart_data["Permutation importance"], errors="coerce")
    chart_data = chart_data.dropna().sort_values("Permutation importance")
    if chart_data.empty:
        st.info("Selected-feature evidence is not available in this artifact.")
        return
    try:
        import altair as alt
    except ImportError:
        alt = None
    if alt is not None:
        feature_order = chart_data["Column"].astype(str).tolist()
        chart = (
            alt.Chart(chart_data)
            .mark_bar(color="#2F7F73")
            .encode(
                x=alt.X("Permutation importance:Q", title="Mean decrease in validation accuracy after shuffling"),
                y=alt.Y("Column:N", title="Final model column", sort=feature_order),
                tooltip=[
                    alt.Tooltip("Column:N", title="Column"),
                    alt.Tooltip("Permutation importance:Q", title="Permutation importance", format=".4f"),
                ],
            )
            .properties(title="Evidence for the five columns used by XGBoost", height=max(220, 38 * len(chart_data)))
        )
        st.altair_chart(chart, width="stretch")
        st.caption("Hover a bar to see the exact importance. This is model dependence, not a causal medical effect.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.bar_chart(chart_data.set_index("Column"), y="Permutation importance")
        return
    fig, axis = plt.subplots(figsize=(9, max(3.6, 0.55 * len(chart_data))))
    axis.barh(chart_data["Column"], chart_data["Permutation importance"], color="#2F7F73")
    axis.set_xlabel("Mean decrease in validation accuracy after shuffling")
    axis.set_ylabel("Final model column")
    axis.set_title("Evidence for the five columns used by XGBoost")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption("Higher permutation importance means stronger dependence of this fitted model on the validation score; it is not a causal effect.")


def _plot_training_split(split_sizes: dict) -> None:
    """Show the train/validation/test composition as a compact visual."""

    labels = {
        "train": "Train",
        "validation": "Validation",
        "train_validation": "Train + validation",
        "test": "Locked test",
    }
    records = [
        {"Split": label, "Rows": pd.to_numeric(split_sizes.get(key), errors="coerce")}
        for key, label in labels.items()
    ]
    chart_data = pd.DataFrame(records).dropna()
    if chart_data.empty:
        st.info("Training split sizes are not available in this artifact.")
        return
    try:
        import altair as alt
    except ImportError:
        alt = None
    if alt is not None:
        split_order = chart_data["Split"].tolist()
        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X("Split:N", title="Split", sort=split_order),
                y=alt.Y("Rows:Q", title="Rows"),
                color=alt.Color("Split:N", legend=None, scale=alt.Scale(domain=split_order, range=["#6C9BC3", "#9B8AFB", "#2F7F73", "#F2A900"])),
                tooltip=[alt.Tooltip("Split:N", title="Split"), alt.Tooltip("Rows:Q", title="Rows", format=",")],
            )
            .properties(title="Data split used by the active artifact", height=300)
        )
        st.altair_chart(chart, width="stretch")
        st.caption("The locked test split is kept separate from model selection, calibration and threshold selection. Hover a bar to see the row count.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.bar_chart(chart_data.set_index("Split"), y="Rows")
        return
    colors = ["#6C9BC3", "#9B8AFB", "#2F7F73", "#F2A900"]
    fig, axis = plt.subplots(figsize=(7.5, 3.8))
    axis.bar(chart_data["Split"], chart_data["Rows"], color=colors[: len(chart_data)])
    axis.set_ylabel("Rows")
    axis.set_title("Data split used by the active artifact")
    axis.grid(axis="y", alpha=0.2)
    for index, value in enumerate(chart_data["Rows"]):
        axis.text(index, value, f"{int(value)}", ha="center", va="bottom")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption("The locked test split is kept separate from model selection, calibration and threshold selection.")


def _plot_input_coverage(selected_features: list[str], all_features: list[str]) -> None:
    """Show how compact the final V3 input set is compared with all candidates."""

    selected_count = len(selected_features)
    candidate_count = max(len(all_features), selected_count)
    excluded_count = max(candidate_count - selected_count, 0)
    chart_data = pd.DataFrame(
        {
            "Group": ["Final V3 inputs", "Excluded candidates"],
            "Columns": [selected_count, excluded_count],
        }
    )
    try:
        import altair as alt
    except ImportError:
        alt = None
    if alt is not None:
        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X("Columns:Q", title="Number of candidate columns"),
                y=alt.Y("Group:N", title="", sort=["Final V3 inputs", "Excluded candidates"]),
                color=alt.Color("Group:N", legend=None, scale=alt.Scale(domain=["Final V3 inputs", "Excluded candidates"], range=["#2F7F73", "#A6AEBB"])),
                tooltip=[alt.Tooltip("Group:N", title="Group"), alt.Tooltip("Columns:Q", title="Columns", format=",")],
            )
            .properties(title="Final model input coverage", height=220)
        )
        st.altair_chart(chart, width="stretch")
        st.caption("Hover a bar to see the exact number of columns. The detailed feature names remain available in the expandable audit list.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.bar_chart(chart_data.set_index("Group"), y="Columns")
        return
    fig, axis = plt.subplots(figsize=(7.5, 3.8))
    bars = axis.barh(chart_data["Group"], chart_data["Columns"], color=["#2F7F73", "#A6AEBB"])
    axis.set_xlabel("Number of candidate columns")
    axis.set_ylabel("")
    axis.set_title("Final model input coverage")
    axis.grid(axis="x", alpha=0.2)
    axis.invert_yaxis()
    for bar, value in zip(bars, chart_data["Columns"]):
        axis.text(value + max(candidate_count * 0.02, 0.15), bar.get_y() + bar.get_height() / 2, str(int(value)), va="center")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption("The active model uses five V3 columns; the exact names remain available in the expandable audit list.")


def _plot_cleaning_audit(cleaning: dict, metadata: dict, split_sizes: dict) -> None:
    """Show cleaning flow and the locked train/validation/test split."""

    raw_rows = pd.to_numeric(metadata.get("rows_raw"), errors="coerce")
    rows_used = pd.to_numeric(metadata.get("rows_clean"), errors="coerce")
    bp_removed = pd.to_numeric(cleaning.get("invalid_blood_pressure_rows_removed"), errors="coerce")
    duplicates_removed = pd.to_numeric(cleaning.get("duplicates_removed"), errors="coerce")
    missing_after = pd.to_numeric(cleaning.get("missing_cells_after_cleaning"), errors="coerce")
    raw_rows = 0 if pd.isna(raw_rows) else int(raw_rows)
    rows_used = 0 if pd.isna(rows_used) else int(rows_used)
    bp_removed = 0 if pd.isna(bp_removed) else int(bp_removed)
    duplicates_removed = 0 if pd.isna(duplicates_removed) else int(duplicates_removed)
    missing_after = 0 if pd.isna(missing_after) else int(missing_after)
    after_bp = max(raw_rows - bp_removed, 0)

    row_flow = pd.DataFrame(
        {
            "Stage": ["Raw rows", "After BP check", "Rows used"],
            "Rows": [raw_rows, after_bp, rows_used],
        }
    )
    split_data = pd.DataFrame(
        {
            "Split": ["Train", "Validation", "Train + validation", "Locked test"],
            "Rows": [
                pd.to_numeric(split_sizes.get("train"), errors="coerce"),
                pd.to_numeric(split_sizes.get("validation"), errors="coerce"),
                pd.to_numeric(split_sizes.get("train_validation"), errors="coerce"),
                pd.to_numeric(split_sizes.get("test"), errors="coerce"),
            ],
        }
    ).dropna()
    try:
        import altair as alt
    except ImportError:
        alt = None
    if alt is not None and not split_data.empty:
        row_panel = row_flow.rename(columns={"Stage": "Category"}).assign(Panel="Row flow")
        split_panel = split_data.rename(columns={"Split": "Category"}).assign(Panel="V3 model data split")
        audit_data = pd.concat([row_panel[["Panel", "Category", "Rows"]], split_panel[["Panel", "Category", "Rows"]]], ignore_index=True)
        panel_order = ["Row flow", "V3 model data split"]
        chart = (
            alt.Chart(audit_data)
            .mark_bar()
            .encode(
                x=alt.X("Category:N", title="", axis=alt.Axis(labelAngle=-20)),
                y=alt.Y("Rows:Q", title="Rows"),
                color=alt.Color("Category:N", legend=None),
                column=alt.Column("Panel:N", title=None, sort=panel_order),
                tooltip=[alt.Tooltip("Panel:N", title="Chart"), alt.Tooltip("Category:N", title="Category"), alt.Tooltip("Rows:Q", title="Rows", format=",")],
            )
            .properties(title="Dataset cleaning and V3 split audit", height=220)
        )
        st.altair_chart(chart, width="stretch")
        positive_rate = _format_model_value(cleaning.get("positive_rate_after_cleaning"), "percent")
        train_validation_rows = pd.to_numeric(split_sizes.get("train_validation"), errors="coerce")
        if pd.isna(train_validation_rows):
            train_validation_rows = pd.to_numeric(split_sizes.get("train"), errors="coerce") + pd.to_numeric(split_sizes.get("validation"), errors="coerce")
        test_rows = pd.to_numeric(split_sizes.get("test"), errors="coerce")
        st.caption(
            f"Train + validation = {int(train_validation_rows) if pd.notna(train_validation_rows) else '—'} rows; "
            f"locked test = {int(test_rows) if pd.notna(test_rows) else '—'} rows. "
            f"Cleaning checks: BP removed {bp_removed}, duplicates {duplicates_removed}, missing after {missing_after}; "
            f"positive rate {positive_rate}. Hover bars for exact values."
        )
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.bar_chart(row_flow.set_index("Stage"), y="Rows")
        if not split_data.empty:
            st.bar_chart(split_data.set_index("Split"), y="Rows")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    row_bars = axes[0].bar(row_flow["Stage"], row_flow["Rows"], color=["#6C9BC3", "#9B8AFB", "#2F7F73"])
    axes[0].set_title("Row flow")
    axes[0].set_ylabel("Rows")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.2)
    for bar, value in zip(row_bars, row_flow["Rows"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{int(value)}", ha="center", va="bottom")

    split_bars = axes[1].bar(split_data["Split"], split_data["Rows"], color=["#6C9BC3", "#9B8AFB", "#2F7F73", "#F2A900"])
    axes[1].set_title("V3 model data split")
    axes[1].set_ylabel("Rows")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.2)
    for bar, value in zip(split_bars, split_data["Rows"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{int(value)}", ha="center", va="bottom")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    positive_rate = _format_model_value(cleaning.get("positive_rate_after_cleaning"), "percent")
    train_validation_rows = pd.to_numeric(split_sizes.get("train_validation"), errors="coerce")
    if pd.isna(train_validation_rows):
        train_validation_rows = pd.to_numeric(split_sizes.get("train"), errors="coerce") + pd.to_numeric(
            split_sizes.get("validation"), errors="coerce"
        )
    test_rows = pd.to_numeric(split_sizes.get("test"), errors="coerce")
    st.caption(
        f"Train + validation = {int(train_validation_rows) if pd.notna(train_validation_rows) else '—'} rows; "
        f"locked test = {int(test_rows) if pd.notna(test_rows) else '—'} rows. "
        f"Cleaning checks: BP removed {bp_removed}, duplicates {duplicates_removed}, missing after {missing_after}; "
        f"positive rate {positive_rate}."
    )


def _model_info_tab(payload: dict, metadata: dict) -> None:
    """Render a readable model card from the active artifact and its metadata."""

    st.subheader(_t("Model information", "Thông tin mô hình"))
    st.warning(
        _t(
            "This is an educational/research screening model, not a medical diagnosis. "
            "Do not use its score as the sole basis for a clinical decision.",
            "Đây là mô hình sàng lọc phục vụ học tập/nghiên cứu, không phải công cụ chẩn đoán y khoa. "
            "Không dùng điểm số này làm cơ sở duy nhất cho quyết định lâm sàng.",
        )
    )

    threshold = float(metadata.get("threshold", payload.get("threshold", 0.5)))
    version = metadata.get("version", payload.get("version", "unknown"))
    selected_features = list(metadata.get("selected_features", payload.get("features", SELECTED_FEATURES)))
    split_sizes = metadata.get("split_sizes", {})

    overview = st.columns(4)
    overview[0].metric(_t("Model", "Mô hình"), "XGBoost")
    overview[1].metric(_t("Screening threshold", "Ngưỡng sàng lọc"), f"{threshold:.3f}")
    overview[2].metric(_t("Calibration", "Hiệu chuẩn"), "Sigmoid")
    overview[3].metric(_t("Features", "Đặc trưng"), len(selected_features))
    st.caption(_t(f"Active artifact version: {version} · calibration method: sigmoid (Platt scaling)", f"Phiên bản artifact đang dùng: {version} · phương pháp hiệu chuẩn: sigmoid (Platt scaling)"))
    cohort = metadata.get("evaluation_cohort", {})
    st.caption(
        _t(
            f"Evaluation scope: {metadata.get('validation_scope', 'not recorded')} · cohort: {cohort.get('name', 'not recorded')} ({cohort.get('rows', split_sizes.get('test', '—'))} rows) · external validation: {metadata.get('external_validation_status', 'not recorded')}",
            f"Phạm vi đánh giá: {metadata.get('validation_scope', 'chưa ghi nhận')} · nhóm dữ liệu: {cohort.get('name', 'chưa ghi nhận')} ({cohort.get('rows', split_sizes.get('test', '—'))} dòng) · thẩm định bên ngoài: {metadata.get('external_validation_status', 'chưa ghi nhận')}",
        )
    )
    st.caption(
        _t(
            "The threshold is selected on the validation set for F2, then applied unchanged to the locked test set and new screening rows.",
            "Ngưỡng được chọn trên tập validation theo F2, sau đó áp dụng giữ nguyên cho tập test khóa và các dòng sàng lọc mới.",
        )
    )

    st.markdown("#### " + _t("Model card summary", "Tóm tắt model card"))
    st.markdown("**" + _t("Model workflow", "Quy trình mô hình") + "**")
    workflow = st.columns(4)
    workflow[0].metric(_t("Inputs", "Đầu vào"), _t(f"{len(selected_features)} columns", f"{len(selected_features)} cột"))
    workflow[1].metric(_t("Model", "Mô hình"), "XGBoost")
    workflow[2].metric(_t("Calibration", "Hiệu chuẩn"), "Sigmoid")
    workflow[3].metric(_t("Decision", "Quyết định"), _t(f"Threshold {threshold:.3f}", f"Ngưỡng {threshold:.3f}"))
    st.caption(
        _t(
            "Five selected columns → XGBoost gradient boosting → sigmoid/Platt calibration → screening score → Positive/Negative screen.",
            "5 cột được chọn → XGBoost gradient boosting → hiệu chuẩn sigmoid/Platt → điểm sàng lọc → kết quả Dương tính/Âm tính.",
        )
    )

    split_left, split_right = st.columns([1.35, 1])
    with split_left:
        _plot_training_split(split_sizes)
    with split_right:
        st.markdown("**" + _t("Reproducibility audit", "Kiểm tra khả năng tái lập") + "**")
        audit = st.columns(2)
        audit[0].metric(_t("Random state", "Trạng thái ngẫu nhiên"), metadata.get("random_state", "—"))
        audit[1].metric("GridSearch", metadata.get("grid_search_candidates", "—"))
        audit[0].metric(_t("Artifact version", "Phiên bản artifact"), str(version)[-12:])
        audit[1].metric(_t("Raw rows", "Số dòng thô"), metadata.get("rows_raw", "—"))
        st.caption(
            _t(
                "The locked test set is not used for model selection, calibration or threshold tuning. A new site or population requires external validation.",
                "Tập test khóa không được dùng để chọn mô hình, hiệu chuẩn hoặc tinh chỉnh ngưỡng. Dữ liệu từ cơ sở hoặc quần thể mới cần được thẩm định bên ngoài.",
            )
        )
    st.info(
        _t(
            "Model type: XGBoost gradient-boosting classifier for structured/tabular data. This project does not use a text-generating LLM for the numeric screening prediction; the model choice follows the V3 tabular-data experiment.",
            "Loại mô hình: bộ phân loại XGBoost gradient boosting cho dữ liệu có cấu trúc dạng bảng. Dự án không dùng LLM tạo văn bản cho dự đoán số; lựa chọn mô hình tuân theo thí nghiệm dữ liệu dạng bảng V3.",
        )
    )

    st.markdown(
        "#### "
        + _t(
            "Preprocessing completion & model optimization",
            "Hoàn thiện Tiền xử lý & Tối ưu Mô hình",
        )
    )
    st.caption(
        _t(
            "This section is generated from the active artifact: feature-count experiments, class-imbalance comparisons, repeated-CV benchmarks, tuned XGBoost parameters, and the validation threshold sweep.",
            "Phần này được tạo từ artifact đang dùng: thí nghiệm số lượng đặc trưng, so sánh xử lý mất cân bằng, benchmark cross-validation lặp lại, tham số XGBoost đã tinh chỉnh và kết quả quét ngưỡng trên validation.",
        )
    )
    optimization_tabs = st.tabs(
        [
            _t("1 · Feature selection", "1 · Lựa chọn đặc trưng"),
            _t("2 · Class imbalance", "2 · Mất cân bằng"),
            _t("3 · Model & tuning", "3 · Mô hình & tham số"),
            _t("4 · Threshold adjustment", "4 · Chỉnh ngưỡng"),
        ]
    )

    selection = pd.DataFrame(metadata.get("feature_selection", []))
    imbalance = pd.DataFrame(metadata.get("imbalance_comparison", []))

    with optimization_tabs[0]:
        if metadata.get("selection_mode") == "fixed_five":
            st.caption(_t(
                "The 32-to-5 selection evidence below comes from the earlier full V3 training run; the current update retrained only the fixed five inputs.",
                "Bằng chứng chọn 32 → 5 bên dưới là từ lần huấn luyện Full V3 trước đó; lần cập nhật hiện tại chỉ huấn luyện lại trên 5 chỉ số đã chốt.",
            ))
        st.markdown(
            "**"
            + _t(
                f"Feature-count experiment: {len(metadata.get('all_features', ALL_FEATURES))} → {len(selected_features)} features",
                f"Quá trình lựa chọn: {len(metadata.get('all_features', ALL_FEATURES))} → {len(selected_features)} đặc trưng",
            )
            + "**"
        )
        st.write(
            _t(
                "Candidate columns are ranked on the development data, then several feature counts are compared using the same repeated cross-validation protocol. The compact five-feature set is selected only after checking that its stored Accuracy and PR-AUC remain comparable to the full set.",
                "Các cột ứng viên được xếp hạng trên dữ liệu development, sau đó nhiều số lượng đặc trưng được so sánh bằng cùng quy trình cross-validation lặp lại. Bộ 5 đặc trưng chỉ được chọn sau khi kiểm tra Accuracy và PR-AUC đã lưu vẫn tương đương bộ đầy đủ.",
            )
        )
        if selection.empty:
            st.info(
                _t(
                    "Feature-selection evidence is not available in this artifact.",
                    "Artifact này không có bằng chứng lựa chọn đặc trưng.",
                )
            )
        else:
            funnel_column, performance_column = st.columns([0.9, 1.1])
            with funnel_column:
                _plot_feature_reduction_funnel(selection, len(selected_features))
            with performance_column:
                _plot_feature_selection(selection, len(selected_features))
            stage_matrix = _feature_selection_stage_matrix(metadata, len(selected_features))
            if not stage_matrix.empty:
                with st.expander(
                    _t(
                        "Show/hide retained columns (32 → 20 → 12 → 8 → 5)",
                        "Hiện/ẩn danh sách cột (32 → 20 → 12 → 8 → 5)",
                    ),
                    expanded=False,
                ):
                    st.caption(
                        _t(
                            "Each row is one original input column, ordered by stored validation permutation importance. A checked box means that column remains in the candidate set at that step.",
                            "Mỗi dòng là một cột đầu vào ban đầu, xếp theo permutation importance đã lưu trên validation. Dấu chọn nghĩa là cột đó vẫn được giữ ở bước tương ứng.",
                        )
                    )
                    column_config: dict[str, object] = {
                        "Rank": st.column_config.NumberColumn(
                            _t("Importance rank", "Hạng quan trọng"),
                            format="%d",
                            width="small",
                        ),
                        "Column": st.column_config.TextColumn(
                            _t("Source column name", "Tên cột gốc"),
                            width="medium",
                        ),
                    }
                    for stage_column in stage_matrix.columns[2:]:
                        feature_count = stage_column.removeprefix("Stage ")
                        column_config[stage_column] = st.column_config.CheckboxColumn(
                            _t(f"Keep at {feature_count}", f"Mốc {feature_count} cột"),
                            disabled=True,
                            width="small",
                        )
                    st.dataframe(
                        stage_matrix,
                        width="stretch",
                        height=620,
                        hide_index=True,
                        column_config=column_config,
                    )
                    final_columns = stage_matrix.loc[
                        stage_matrix[f"Stage {len(selected_features)}"],
                        "Column",
                    ].tolist()
                    st.success(
                        _t(
                            "Final model inputs: " + ", ".join(final_columns) + ".",
                            "5 cột cuối đưa vào mô hình: " + ", ".join(final_columns) + ".",
                        )
                    )
            st.markdown(
                "**"
                + _t(
                    "Performance audit at every retained feature count",
                    "Bảng đối chiếu hiệu năng theo từng số lượng đặc trưng",
                )
                + "**"
            )
            st.dataframe(
                _feature_selection_summary_table(metadata, len(selected_features)),
                width="stretch",
                hide_index=True,
            )
            selected_row = selection[selection["feature_count"].eq(len(selected_features))]
            full_row = selection[
                selection["feature_count"].eq(len(metadata.get("all_features", ALL_FEATURES)))
            ]
            if not selected_row.empty and not full_row.empty:
                selected_values = selected_row.iloc[0]
                full_values = full_row.iloc[0]
                selected_count = len(selected_features)
                full_count = len(metadata.get("all_features", ALL_FEATURES))
                reduction = 1 - (selected_count / full_count)
                test_specificity = metadata.get("test_metrics", {}).get("specificity")
                summary_metrics = st.columns(4)
                summary_metrics[0].metric(
                    _t("Input reduction", "Giảm số đầu vào"),
                    f"{reduction * 100:.1f}%",
                    f"{full_count} → {selected_count}",
                )
                summary_metrics[1].metric(
                    "Accuracy CV · 5 features",
                    f"{float(selected_values['accuracy_cv']):.3f}",
                    f"{float(selected_values['accuracy_cv']) - float(full_values['accuracy_cv']):+.3f} vs 32",
                )
                summary_metrics[2].metric(
                    "PR-AUC CV · 5 features",
                    f"{float(selected_values['pr_auc_cv']):.3f}",
                    f"{float(selected_values['pr_auc_cv']) - float(full_values['pr_auc_cv']):+.3f} vs 32",
                )
                summary_metrics[3].metric(
                    _t("Locked-test specificity", "Specificity trên test khóa"),
                    f"{float(test_specificity):.3f}" if test_specificity is not None else "—",
                    f"threshold {threshold:.3f}",
                )
                st.success(
                    _t(
                        f"Selected {len(selected_features)} features: Accuracy CV {float(selected_values['accuracy_cv']):.3f} and PR-AUC CV {float(selected_values['pr_auc_cv']):.3f}, versus {float(full_values['accuracy_cv']):.3f} and {float(full_values['pr_auc_cv']):.3f} with all {len(metadata.get('all_features', ALL_FEATURES))} features.",
                        f"Bộ {len(selected_features)} đặc trưng được chọn: Accuracy CV {float(selected_values['accuracy_cv']):.3f} và PR-AUC CV {float(selected_values['pr_auc_cv']):.3f}, so với {float(full_values['accuracy_cv']):.3f} và {float(full_values['pr_auc_cv']):.3f} khi dùng đủ {len(metadata.get('all_features', ALL_FEATURES))} đặc trưng.",
                    )
                )
        st.markdown(
            "**"
            + _t(
                "Final five features, ranked by validation permutation importance",
                "5 đặc trưng cuối, xếp theo permutation importance trên validation",
            )
            + "**"
        )
        feature_table = _selected_feature_evidence_table(metadata, selected_features)
        feature_table = feature_table.sort_values(
            "Permutation importance",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)
        feature_table.insert(0, "Rank", range(1, len(feature_table) + 1))
        st.dataframe(
            feature_table[
                [
                    "Rank",
                    "Column",
                    "Role",
                    "Permutation importance",
                    "Importance std",
                    "Cohen's d",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
        test_specificity = metadata.get("test_metrics", {}).get("specificity")
        if test_specificity is not None:
            st.caption(
                _t(
                    f"Metric boundary: the feature-count experiment stores cross-validation Accuracy and PR-AUC, not specificity for every count. Specificity {float(test_specificity):.3f} is reported separately for the final selected model on the untouched locked test set at threshold {threshold:.3f}.",
                    f"Ranh giới đánh giá: thí nghiệm số lượng đặc trưng chỉ lưu Accuracy và PR-AUC qua cross-validation, không lưu specificity cho từng mốc. Specificity {float(test_specificity):.3f} được báo cáo riêng cho mô hình cuối trên tập test khóa chưa đụng tới, tại ngưỡng {threshold:.3f}.",
                )
            )
        st.markdown("#### " + _t("Feature influence and excluded columns", "Mức ảnh hưởng và các cột bị loại"))
        st.caption(_t(
            "Validation permutation importance measures model dependence, not a causal or clinical effect.",
            "Permutation importance trên validation đo mức phụ thuộc của mô hình, không chứng minh tác động nhân quả hay ý nghĩa lâm sàng.",
        ))
        _plot_model_feature_ranking(metadata)
        influence_table = _model_influence_table(metadata)
        if not influence_table.empty:
            with st.expander(_t("View influence values", "Xem giá trị ảnh hưởng"), expanded=False):
                st.dataframe(influence_table, width="stretch", hide_index=True)
        excluded_features = [
            feature for feature in metadata.get("all_features", ALL_FEATURES)
            if feature not in selected_features
        ]
        with st.expander(_t("Columns excluded from the final model", "Các cột bị loại khỏi mô hình cuối"), expanded=False):
            st.write(", ".join(excluded_features) if excluded_features else _t("None", "Không có"))

    with optimization_tabs[1]:
        st.markdown("#### " + _t("Class imbalance and weighting", "Mất cân bằng lớp và gán trọng số"))
        st.caption(_t(
            "Compare the stored training strategies here; the final model uses class weights without adding synthetic patient records.",
            "So sánh các cách xử lý đã lưu; mô hình cuối dùng trọng số lớp, không thêm bản ghi bệnh nhân tổng hợp.",
        ))
        positive_rate = pd.to_numeric(metadata.get("eda", {}).get("positive_rate"), errors="coerce")
        negative_rate = 1.0 - positive_rate if pd.notna(positive_rate) else pd.NA
        hyperparameters = metadata.get("hyperparameters", {})
        class_weight = pd.to_numeric(hyperparameters.get("scale_pos_weight"), errors="coerce")
        imbalance_metrics = st.columns(4)
        imbalance_metrics[0].metric(
            _t("Negative class", "Lớp âm"),
            f"{float(negative_rate) * 100:.1f}%" if pd.notna(negative_rate) else "—",
        )
        imbalance_metrics[1].metric(
            _t("Positive class", "Lớp dương"),
            f"{float(positive_rate) * 100:.1f}%" if pd.notna(positive_rate) else "—",
        )
        imbalance_metrics[2].metric(
            "scale_pos_weight",
            f"{float(class_weight):.3f}" if pd.notna(class_weight) else "—",
        )
        imbalance_metrics[3].metric(
            _t("Selected method", "Phương pháp chọn"),
            metadata.get("selected_imbalance_method", "—"),
        )
        imbalance_table = _imbalance_summary_table(metadata)
        if imbalance_table.empty:
            st.info(
                _t(
                    "Imbalance comparison is not available in this artifact.",
                    "Artifact này không có bảng so sánh xử lý mất cân bằng.",
                )
            )
        else:
            st.dataframe(imbalance_table, width="stretch", hide_index=True)
            _plot_imbalance_comparison(imbalance, metadata.get("selected_imbalance_method", "not recorded"))
        st.success(
            _t(
                "The active XGBoost model keeps class weighting (scale_pos_weight) instead of resampling. SMOTENC is retained as an audit comparison only, so the final training set does not contain synthetic patient rows.",
                "Mô hình XGBoost đang dùng giữ phương pháp gán trọng số lớp (scale_pos_weight), không resample. SMOTENC chỉ được giữ làm phương án so sánh kiểm toán, vì vậy tập huấn luyện cuối không chứa các dòng bệnh nhân tổng hợp.",
            )
        )
        st.caption(
            _t(
                "The observed development prevalence is approximately 65% negative / 35% positive: a mild imbalance. Weighting preserves the original records while increasing the loss contribution of the minority class.",
                "Tỷ lệ quan sát trên development xấp xỉ 65% âm / 35% dương: mất cân bằng nhẹ. Gán trọng số giữ nguyên các bản ghi gốc đồng thời tăng đóng góp vào hàm loss của lớp thiểu số.",
            )
        )

    with optimization_tabs[2]:
        st.markdown("**" + _t("Repeated-CV candidate benchmark", "Benchmark mô hình bằng repeated CV") + "**")
        benchmark_table = _benchmark_summary_table(metadata)
        if benchmark_table.empty:
            st.info(
                _t(
                    "Candidate-model benchmark is not available in this artifact.",
                    "Artifact này không có benchmark mô hình ứng viên.",
                )
            )
        else:
            st.dataframe(benchmark_table, width="stretch", hide_index=True)
            xgb_row = benchmark_table[benchmark_table["Model"].eq("XGBoost")]
            if not xgb_row.empty:
                st.success(
                    _t(
                        f"XGBoost is selected with stored PR-AUC CV {float(xgb_row.iloc[0]['PR-AUC CV']):.3f} and Recall {float(xgb_row.iloc[0]['Recall']):.3f}.",
                        f"XGBoost được chọn với PR-AUC CV đã lưu {float(xgb_row.iloc[0]['PR-AUC CV']):.3f} và Recall {float(xgb_row.iloc[0]['Recall']):.3f}.",
                    )
                )
        params = metadata.get("hyperparameters", {})
        tuning_table = pd.DataFrame(
            [
                {
                    "Parameter": "learning_rate",
                    "Selected value": params.get("learning_rate", "—"),
                    "Role": _t("Boosting step size", "Tốc độ cập nhật boosting"),
                },
                {
                    "Parameter": "max_depth",
                    "Selected value": params.get("max_depth", "—"),
                    "Role": _t("Maximum tree depth", "Độ sâu tối đa của cây"),
                },
                {
                    "Parameter": "n_estimators",
                    "Selected value": params.get("n_estimators", "—"),
                    "Role": _t("Number of boosting trees", "Số cây boosting"),
                },
                {
                    "Parameter": "scale_pos_weight",
                    "Selected value": params.get("scale_pos_weight", "—"),
                    "Role": _t("Minority-class loss weight", "Trọng số loss của lớp thiểu số"),
                },
            ]
        )
        st.markdown("**" + _t("Selected hyperparameters", "Tham số sau tinh chỉnh") + "**")
        st.dataframe(tuning_table, width="stretch", hide_index=True)
        tuning_metrics = st.columns(2)
        tuning_metrics[0].metric(
            _t("Grid-search candidates", "Cấu hình GridSearch"),
            metadata.get("grid_search_candidates", "—"),
        )
        tuning_metrics[1].metric(
            _t("Best tuning PR-AUC", "PR-AUC tuning tốt nhất"),
            _format_model_value(metadata.get("tuning_best_pr_auc")),
        )
        st.caption(
            _t(
                "Hyperparameters are selected on development cross-validation. The locked test set is not used for model-family or parameter selection.",
                "Hyperparameter được chọn bằng cross-validation trên development. Tập test khóa không được dùng để chọn họ mô hình hoặc tham số.",
            )
        )
        _render_model_detail_evidence(payload, metadata)

    with optimization_tabs[3]:
        st.markdown("**" + _t("Validation threshold sweep", "Quét ngưỡng trên validation") + "**")
        st.write(
            _t(
                "Lowering the threshold can increase Recall and reduce false negatives, but it can also create more false positives. The active threshold is selected by F2 on validation and is then frozen before locked-test evaluation.",
                "Hạ ngưỡng có thể tăng Recall và giảm False Negative, nhưng cũng có thể tạo thêm False Positive. Ngưỡng active được chọn theo F2 trên validation rồi được cố định trước khi đánh giá test khóa.",
            )
        )
        _plot_threshold_tradeoff(metadata, threshold)
        threshold_table = _threshold_scenario_table(metadata, threshold)
        if threshold_table.empty:
            st.info(
                _t(
                    "Threshold-scenario values are not available in this artifact.",
                    "Artifact này không có giá trị so sánh các kịch bản ngưỡng.",
                )
            )
        else:
            validation_rows = metadata.get("split_sizes", {}).get("validation")
            validation_label = (
                f"validation (n={int(validation_rows):,})"
                if validation_rows is not None
                else "validation"
            )
            st.caption(
                _t(
                    f"Every row in this table is calculated only on {validation_label}. "
                    "These values are used to select the threshold and must not be compared "
                    "row-for-row with the locked-test counts below.",
                    f"Tất cả dòng trong bảng này chỉ được tính trên tập {validation_label}. "
                    "Các số liệu này dùng để chọn ngưỡng, không được so trực tiếp từng dòng "
                    "với số đếm của tập test khóa ở phía dưới.",
                )
            )
            display_threshold_table = threshold_table.copy()
            display_threshold_table["Scenario"] = display_threshold_table["Scenario"].replace(
                {
                    "Recall-priority candidate": _t(
                        "Recall-priority candidate", "Ứng viên ưu tiên Recall"
                    ),
                    "Requested ≈0.20 example": _t(
                        "Requested ≈0.20 example", "Ví dụ ngưỡng ≈0,20"
                    ),
                    "Selected by validation F2": _t(
                        "Selected by validation F2", "Được chọn theo F2 validation"
                    ),
                    "Default 0.50": _t("Default 0.50", "Mặc định 0,50"),
                }
            )
            display_threshold_table = display_threshold_table.rename(
                columns={
                    "Cohort": _t("Cohort", "Tập dữ liệu"),
                    "Scenario": _t("Scenario", "Kịch bản"),
                    "Threshold": _t("Threshold", "Ngưỡng"),
                    "False negatives": _t("False negatives", "Bỏ sót (FN)"),
                    "False positives": _t("False positives", "Báo động giả (FP)"),
                }
            )
            st.dataframe(display_threshold_table, width="stretch", hide_index=True)
            requested_row = threshold_table[
                threshold_table["Scenario"].eq("Requested ≈0.20 example")
            ]
            selected_threshold_row = threshold_table[
                threshold_table["Scenario"].eq("Selected by validation F2")
            ]
            if not requested_row.empty and not selected_threshold_row.empty:
                requested = requested_row.iloc[0]
                selected = selected_threshold_row.iloc[0]
                if float(requested["Recall"]) <= float(selected["Recall"]):
                    st.info(
                        _t(
                            f"In this validation cohort, threshold {float(requested['Threshold']):.2f} does not improve Recall over the selected {float(selected['Threshold']):.3f}: both are {float(selected['Recall']):.3f}. The selected threshold has fewer false positives ({int(selected['False positives'])} vs {int(requested['False positives'])}).",
                            f"Trên validation hiện tại, ngưỡng {float(requested['Threshold']):.2f} không tăng Recall so với ngưỡng được chọn {float(selected['Threshold']):.3f}: cả hai đều là {float(selected['Recall']):.3f}. Ngưỡng được chọn có ít False Positive hơn ({int(selected['False positives'])} so với {int(requested['False positives'])}).",
                        )
                    )
        st.warning(
            _t(
                "A lower threshold must not be chosen from the locked-test result. Any clinical operating threshold requires external validation, explicit harm/cost analysis, and prospective review.",
                "Không được chọn ngưỡng thấp hơn dựa trên kết quả test khóa. Mọi ngưỡng vận hành lâm sàng cần thẩm định ngoài, phân tích rõ tác hại/chi phí và đánh giá tiền cứu.",
            )
        )
        _render_threshold_operating_point(metadata)

def _render_threshold_operating_point(metadata: dict) -> None:
    """Keep threshold-dependent locked-test results in the threshold tab."""

    st.markdown("#### " + _t("Locked-test operating point", "Điểm vận hành trên test khóa"))
    active_confusion = metadata.get("test_confusion_matrix")
    if active_confusion:
        operating_counts = _plot_confusion_matrix(
            active_confusion, "Active artifact locked-test confusion matrix"
        )
        st.caption(_t(
            "Counts are stored by the active artifact at its current screening threshold.",
            "Các số đếm được lưu trong artifact đang dùng tại ngưỡng sàng lọc hiện tại.",
        ))
    else:
        operating_counts = _plot_confusion_matrix(
            REPORT_V3_MODEL_REFERENCE["confusion_matrix"],
            "V3 report locked-test confusion matrix (n=393)",
        )
        st.caption(_t(
            "The active artifact does not store confusion-matrix counts. This chart is V3 report evidence, not the current artifact's result.",
            "Artifact đang dùng không lưu số đếm ma trận nhầm lẫn. Biểu đồ này là bằng chứng báo cáo V3, không phải kết quả của artifact hiện tại.",
        ))
    if operating_counts:
        tn, fp, fn, tp = operating_counts
        operating_metrics = _confusion_operating_metrics(operating_counts)
        precision = operating_metrics["PPV / Precision"]
        recall = operating_metrics["Sensitivity / Recall"]
        specificity = operating_metrics["Specificity"]
        f2 = (
            5 * precision * recall / (4 * precision + recall)
            if 4 * precision + recall
            else 0.0
        )
        test_rows = tn + fp + fn + tp
        threshold = float(metadata.get("threshold", REPORT_V3_MODEL_REFERENCE["threshold"]))
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        _t("Cohort", "Tập dữ liệu"): _t(
                            f"Locked test (n={test_rows:,})",
                            f"Test khóa (n={test_rows:,})",
                        ),
                        _t("Threshold", "Ngưỡng"): round(threshold, 3),
                        "TN": tn,
                        "FP": fp,
                        "FN": fn,
                        "TP": tp,
                        "Precision": round(precision, 3),
                        "Recall": round(recall, 3),
                        "Specificity": round(specificity, 3),
                        "F2": round(f2, 3),
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            _t(
                "This is a one-time evaluation on the locked test cohort after the threshold "
                "was fixed on validation; it is not used to choose or retune the threshold.",
                "Đây là lần đánh giá duy nhất trên tập test khóa sau khi ngưỡng đã được cố định "
                "từ validation; số liệu này không dùng để chọn hoặc chỉnh lại ngưỡng.",
            )
        )
        for column, (label, value) in zip(st.columns(4), operating_metrics.items()):
            column.metric(label, f"{value * 100:.1f}%")
        st.caption(_t(
            "These operating-point metrics depend on the chosen threshold, unlike ROC-AUC and PR-AUC.",
            "Các chỉ số tại điểm vận hành phụ thuộc ngưỡng đã chọn, khác với ROC-AUC và PR-AUC.",
        ))


def _render_model_detail_evidence(payload: dict, metadata: dict) -> None:
    """Show evaluation and governance evidence only in the model/tuning tab."""

    threshold = float(metadata.get("threshold", payload.get("threshold", 0.5)))
    calibration = metadata.get("calibration", {})
    cleaning = metadata.get("cleaning", {})
    split_sizes = metadata.get("split_sizes", {})
    benchmark = pd.DataFrame(metadata.get("benchmark", []))
    st.markdown("#### " + _t("Model comparison", "So sánh mô hình"))
    st.caption(
        _t(
            "Compare all candidate models on repeated stratified cross-validation. PR-AUC is the primary ranking metric for this imbalanced screening task; recall shows the screening trade-off.",
            "So sánh tất cả mô hình ứng viên bằng cross-validation phân tầng lặp lại. PR-AUC là tiêu chí xếp hạng chính cho bài toán sàng lọc mất cân bằng; recall thể hiện đánh đổi khi sàng lọc.",
        )
    )
    if not benchmark.empty and "model" in benchmark.columns:
        _plot_model_benchmark(benchmark)
        active_best = benchmark.sort_values("pr_auc_cv", ascending=False).iloc[0]
        active_xgb = benchmark[benchmark["model"].eq("XGBoost")]
        if not active_xgb.empty and active_best["model"] == "XGBoost":
            st.success(_t(f"Selected model: XGBoost · highest stored PR-AUC CV: {float(active_xgb.iloc[0]['pr_auc_cv']):.3f}", f"Mô hình được chọn: XGBoost · PR-AUC CV cao nhất đã lưu: {float(active_xgb.iloc[0]['pr_auc_cv']):.3f}"))
        with st.expander(_t("View exact model comparison values", "Xem giá trị so sánh mô hình chính xác"), expanded=False):
            _plot_benchmark_audit(benchmark)
    else:
        st.info(_t("Model comparison diagnostics are not available in this artifact.", "Artifact này không có thông tin chẩn đoán so sánh mô hình."))

    with st.expander(_t("Dataset and preprocessing audit", "Kiểm tra dataset và tiền xử lý"), expanded=False):
        _plot_cleaning_audit(cleaning, metadata, split_sizes)
        st.caption(_t(
            f"Training dataset: {metadata.get('dataset', 'not recorded')}",
            f"Dataset huấn luyện: {metadata.get('dataset', 'chưa ghi nhận')}",
        ))
        saved_imputer = payload.get("imputation_statistics") or metadata.get("imputation_statistics")
        if saved_imputer:
            st.success(_t(
                "Training imputation statistics are saved in the active artifact.",
                "Thống kê điền thiếu từ tập huấn luyện đã được lưu trong artifact đang dùng.",
            ))
        else:
            st.warning(_t(
                "This artifact has no saved imputation statistics; batch processing uses the cleaned project-reference fallback.",
                "Artifact này chưa lưu thống kê điền thiếu; xử lý batch dùng thống kê dự phòng từ dataset dự án đã làm sạch.",
            ))

    st.markdown("#### " + _t("Validation and test performance", "Hiệu năng validation và test"))
    metric_table = _model_metric_table(metadata)
    if metric_table.empty:
        st.info(_t("Validation/test metrics are not available in this artifact.", "Artifact này không có metric validation/test."))
    else:
        _plot_validation_test_metrics(metadata)
        brier = metadata.get("test_metrics", {}).get("brier")
        if brier is not None:
            st.metric(_t("Locked-test Brier score", "Brier score trên test khóa"), f"{float(brier):.3f}", help=_t("Lower is better for calibration quality.", "Giá trị thấp hơn thể hiện hiệu chuẩn tốt hơn."))
        with st.expander(_t("View exact validation/test values", "Xem giá trị validation/test chính xác"), expanded=False):
            st.dataframe(metric_table, width="stretch", hide_index=True)

    st.markdown("#### " + _t("Calibration curve", "Đường cong hiệu chuẩn"))
    _plot_calibration_curve(metadata)

    st.markdown("#### " + _t("Calibration evidence", "Bằng chứng hiệu chuẩn"))
    _plot_calibration_evidence(metadata)
    active_brier = metadata.get("test_metrics", {}).get("brier")
    if active_brier is not None:
        st.caption(
            _t(
                f"Current active artifact locked-test Brier score: {float(active_brier):.3f}.",
                f"Brier score test khóa của artifact hiện tại: {float(active_brier):.3f}.",
            )
        )

    st.markdown("#### " + _t("ROC curve", "Đường cong ROC"))
    _plot_roc_curve(payload)

    ci = metadata.get("bootstrap_95_ci", {})
    if ci:
        _plot_bootstrap_ci(metadata)
        ci_table = pd.DataFrame(
            [
                {
                    "Metric": name.replace("_", " ").upper(),
                    "Lower 95%": f"{float(values[0]):.3f}",
                    "Upper 95%": f"{float(values[1]):.3f}",
                }
                for name, values in ci.items()
                if isinstance(values, (list, tuple)) and len(values) >= 2
            ]
        )
        with st.expander(_t("View exact bootstrap 95% confidence intervals", "Xem khoảng tin cậy bootstrap 95% chính xác"), expanded=False):
            st.dataframe(ci_table, width="stretch", hide_index=True)

    with st.expander(_t("Training split and calibration details", "Chi tiết chia tập huấn luyện và hiệu chuẩn"), expanded=False):
        _plot_training_split(split_sizes)
        validation_calibration = calibration.get("validation_summary", {})
        calibration_metrics = st.columns(6)
        calibration_metrics[0].metric(_t("Calibration", "Hiệu chuẩn"), "Platt / sigmoid")
        calibration_metrics[1].metric(_t("Validation F2", "F2 validation"), _format_model_value(calibration.get("validation_f2")))
        calibration_metrics[2].metric(_t("Threshold", "Ngưỡng"), f"{threshold:.3f}")
        calibration_metrics[3].metric(_t("Calibration intercept", "Intercept hiệu chuẩn"), _format_model_value(validation_calibration.get("intercept")))
        calibration_metrics[4].metric(_t("Calibration slope", "Slope hiệu chuẩn"), _format_model_value(validation_calibration.get("slope")))
        calibration_metrics[5].metric(_t("Imbalance", "Mất cân bằng"), metadata.get("selected_imbalance_method", "not recorded"))
        params = metadata.get("hyperparameters", {})
        st.caption(
            "Hyperparameters: "
            f"learning_rate={params.get('learning_rate', '—')}, "
            f"max_depth={params.get('max_depth', '—')}, "
            f"n_estimators={params.get('n_estimators', '—')}, "
            f"scale_pos_weight={params.get('scale_pos_weight', '—')}"
        )

    st.markdown("#### " + _t("Why XGBoost is selected", "Vì sao chọn XGBoost"))
    st.write(
        _t(
            "V3 compares two baseline models and five candidates with repeated stratified cross-validation. PR-AUC is the primary ranking metric because Diagnosis is imbalanced; recall is also important for a screening use case. The code currently retains XGBoost as the official final family after this benchmark, rather than silently swapping models during scoring.",
            "V3 so sánh 2 mô hình cơ sở và 5 mô hình ứng viên bằng cross-validation phân tầng lặp lại. PR-AUC là tiêu chí xếp hạng chính vì Diagnosis bị mất cân bằng; recall cũng quan trọng cho mục tiêu sàng lọc. Code hiện giữ XGBoost là họ mô hình chính thức sau benchmark, không âm thầm đổi mô hình khi dự đoán.",
        )
    )
    if not benchmark.empty and "model" in benchmark.columns:
        with st.expander(_t("View benchmark audit chart", "Xem biểu đồ kiểm tra benchmark"), expanded=False):
            _plot_benchmark_audit(benchmark)
        active_best = benchmark.sort_values("pr_auc_cv", ascending=False).iloc[0]
        active_xgb = benchmark[benchmark["model"].eq("XGBoost")]
        if not active_xgb.empty and active_best["model"] != "XGBoost":
            st.warning(
                _t(
                    f"The stored benchmark ranks {active_best['model']} above XGBoost on PR-AUC but the artifact is still labelled XGBoost. Verify the selection policy before presenting this as an automatic winner.",
                    f"Benchmark đã lưu xếp {active_best['model']} cao hơn XGBoost theo PR-AUC nhưng artifact vẫn mang nhãn XGBoost. Hãy kiểm tra chính sách chọn mô hình trước khi trình bày đây là mô hình thắng tự động.",
                )
            )
    else:
        st.info(_t("Model benchmark diagnostics are not available in this artifact.", "Artifact này không có thông tin benchmark mô hình."))

    active_test = metadata.get("test_metrics", {})
    active_params = metadata.get("hyperparameters", {})
    if abs(threshold - REPORT_V3_MODEL_REFERENCE["threshold"]) > 0.001:
        st.warning(
            _t(
                f"Threshold mismatch: the supplied V3 report records {REPORT_V3_MODEL_REFERENCE['threshold']:.3f}, while the active artifact uses {threshold:.3f}. This tab labels the active artifact as a later/different run, not as an exact reproduction of the report.",
                f"Ngưỡng khác nhau: báo cáo V3 ghi nhận {REPORT_V3_MODEL_REFERENCE['threshold']:.3f}, còn artifact đang dùng là {threshold:.3f}. Tab này đánh dấu artifact hiện tại là lần chạy sau/khác, không phải bản tái lập chính xác báo cáo.",
            )
        )
    differing_params = [
        name for name, expected in REPORT_V3_MODEL_REFERENCE["hyperparameters"].items()
        if active_params.get(name) is not None and float(active_params[name]) != float(expected)
    ]
    if differing_params:
        st.warning(_t("Hyperparameter mismatch with the V3 report: ", "Hyperparameter khác với báo cáo V3: ") + ", ".join(differing_params) + ".")

    st.markdown("#### " + _t("V3 report versus active artifact", "Báo cáo V3 và artifact đang dùng"))
    _plot_report_active_test_metrics(active_test)
    with st.expander("View exact V3 evidence and consistency audit", expanded=False):
        report_benchmark = REPORT_V3_MODEL_REFERENCE["benchmark"]
        active_xgb = benchmark[benchmark["model"].eq("XGBoost")] if not benchmark.empty and "model" in benchmark.columns else pd.DataFrame()
        active_benchmark = active_xgb.iloc[0].to_dict() if not active_xgb.empty else {}
        evidence_table = pd.DataFrame(
            [
                {"Evidence": "XGBoost PR-AUC (CV)", "V3 report": f"{report_benchmark['pr_auc_cv']:.4f}", "Active artifact": _format_model_value(active_benchmark.get("pr_auc_cv"))},
                {"Evidence": "XGBoost ROC-AUC (CV)", "V3 report": f"{report_benchmark['roc_auc_cv']:.4f}", "Active artifact": _format_model_value(active_benchmark.get("roc_auc_cv"))},
                {"Evidence": "XGBoost recall (CV)", "V3 report": f"{report_benchmark['recall']:.4f}", "Active artifact": _format_model_value(active_benchmark.get("recall"))},
                {"Evidence": "Threshold", "V3 report": f"{REPORT_V3_MODEL_REFERENCE['threshold']:.3f}", "Active artifact": f"{threshold:.3f}"},
            ]
        )
        st.dataframe(evidence_table, width="stretch", hide_index=True)
        report_test = REPORT_V3_MODEL_REFERENCE["test_metrics"]
        test_evidence = pd.DataFrame(
            [
                {"Metric": name.replace("_", " ").upper(), "V3 report": f"{value:.3f}", "Active artifact": _format_model_value(active_test.get(name))}
                for name, value in report_test.items()
            ]
        )
        st.write("**Final test metrics**")
        st.dataframe(test_evidence, width="stretch", hide_index=True)
        report_ci = REPORT_V3_MODEL_REFERENCE["bootstrap_ci"]
        active_ci = metadata.get("bootstrap_95_ci", {})
        ci_evidence = pd.DataFrame(
            [
                {
                    "Metric": name.replace("_", " ").upper(),
                    "V3 report 95% CI": f"[{values[0]:.4f}; {values[1]:.4f}]",
                    "Active artifact 95% CI": (
                        f"[{float(active_ci[name][0]):.4f}; {float(active_ci[name][1]):.4f}]"
                        if name in active_ci and len(active_ci[name]) >= 2
                        else "—"
                    ),
                }
                for name, values in report_ci.items()
            ]
        )
        st.write("**Bootstrap 95% confidence intervals**")
        st.dataframe(ci_evidence, width="stretch", hide_index=True)
        active_selection = pd.DataFrame(metadata.get("feature_selection", []))
        report_selection = REPORT_V3_MODEL_REFERENCE["feature_selection"]
        selection_evidence = pd.DataFrame(
            [
                {
                    "Feature count": count,
                    "V3 accuracy": f"{values['accuracy_cv']:.4f}",
                    "Active accuracy": _format_model_value(
                        active_selection.loc[active_selection["feature_count"].eq(count), "accuracy_cv"].iloc[0]
                        if not active_selection.empty and not active_selection.loc[active_selection["feature_count"].eq(count)].empty
                        else None
                    ),
                    "V3 PR-AUC": f"{values['pr_auc_cv']:.4f}",
                    "Active PR-AUC": _format_model_value(
                        active_selection.loc[active_selection["feature_count"].eq(count), "pr_auc_cv"].iloc[0]
                        if not active_selection.empty and not active_selection.loc[active_selection["feature_count"].eq(count)].empty
                        else None
                    ),
                }
                for count, values in report_selection.items()
            ]
        )
        st.write("**Feature-selection evidence**")
        st.dataframe(selection_evidence, width="stretch", hide_index=True)
        report_calibration = REPORT_V3_MODEL_REFERENCE["calibration"]
        st.write(
            "**Calibration evidence from V3:** "
            f"Brier {report_calibration['brier_before']:.4f} → {report_calibration['brier_after']:.4f}; "
            f"ECE {report_calibration['ece_before']:.4f} → {report_calibration['ece_after']:.4f}. "
            "The active-artifact chart above uses the saved validation before/after pair when available."
        )
        st.caption(
            "Source evidence: supplied V3 report, sections 6, 8, 10 and 11; "
            "XGBoost scale_pos_weight documentation; scikit-learn calibration and permutation-importance guidance."
        )

    with st.expander(_t("Fairness, external validation, and monitoring status", "Công bằng, thẩm định bên ngoài và giám sát"), expanded=False):
        subgroup = metadata.get("subgroup_metrics", [])
        if subgroup:
            st.info(_t("Subgroup metrics are stored with sample size, estimability status, and bootstrap intervals. Small or one-class groups are marked not estimable.", "Metric theo nhóm được lưu cùng cỡ mẫu, trạng thái có thể ước lượng và khoảng bootstrap. Nhóm quá nhỏ hoặc chỉ có một lớp được đánh dấu là không thể ước lượng."))
            subgroup_frame = pd.DataFrame(subgroup)
            st.dataframe(
                subgroup_frame[[column for column in ["group", "level", "n", "prevalence", "recall", "recall_ci_low", "recall_ci_high", "status"] if column in subgroup_frame.columns]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.warning(_t("Subgroup metrics are not stored in this artifact; do not infer fairness from overall metrics.", "Artifact này không lưu metric theo nhóm; không suy luận tính công bằng từ metric tổng thể."))
        st.write(
            _t(
                "Before any real-world use, evaluate the model on an external dataset, report performance and calibration by relevant subgroups, check data drift over time, and keep an auditable model/data version history. The current app supports artifact versioning and retraining comparison, but it does not constitute clinical validation or post-deployment monitoring.",
                "Trước khi sử dụng thực tế, cần đánh giá mô hình trên dataset bên ngoài, báo cáo hiệu năng và hiệu chuẩn theo các nhóm liên quan, kiểm tra drift theo thời gian và lưu lịch sử phiên bản model/data có thể kiểm toán. App hỗ trợ quản lý phiên bản artifact và so sánh huấn luyện lại, nhưng không thay thế thẩm định lâm sàng hoặc giám sát sau triển khai.",
            )
        )

    warnings = metadata.get("warnings", [])
    if warnings:
        with st.expander(_t("Artifact warnings and limitations", "Cảnh báo và giới hạn của artifact"), expanded=True):
            for warning in warnings:
                if isinstance(warning, dict):
                    st.info(warning.get("message", str(warning)))
                else:
                    st.warning(str(warning))


def _project_workflow_overview() -> None:
    """Summarize the nine main stages used to build and deploy the V3 model."""

    st.subheader(_t("Project workflow overview", "Quy trình thực hiện tổng quan"))
    st.caption(
        _t(
            "The project was completed through the following nine main stages, from data quality review to Streamlit deployment.",
            "Dưới đây là trình tự 9 bước chính đã thực hiện xuyên suốt đề tài, từ kiểm tra dữ liệu đến triển khai Streamlit.",
        )
    )

    workflow_steps = [
        (
            "Data cleaning",
            "Làm sạch dữ liệu",
            "Check missing values and duplicates, then remove 186 rows that violate the physiological rule SystolicBP ≤ DiastolicBP.",
            "Kiểm tra dữ liệu thiếu, dòng trùng lặp và lọc 186 dòng vi phạm luật sinh lý SystolicBP ≤ DiastolicBP.",
        ),
        (
            "Exploratory data analysis (EDA)",
            "Phân tích khám phá dữ liệu (EDA)",
            "Examine the label distribution, Cohen's d, correlations, and boxplots.",
            "Phân tích phân bố nhãn, Cohen's d, tương quan và boxplot.",
        ),
        (
            "Feature selection",
            "Chọn lọc đặc trưng (Feature Selection)",
            "Use permutation importance to measure each feature's contribution to accuracy and reduce 32 inputs to the five most important features.",
            "Dùng permutation importance để đo mức đóng góp vào Accuracy và rút gọn từ 32 xuống 5 đặc trưng quan trọng nhất.",
        ),
        (
            "Imbalance-treatment comparison",
            "So sánh xử lý mất cân bằng",
            "Compare class weighting (scale_pos_weight) with SMOTENC, which correctly handles categorical features.",
            "So sánh trọng số lớp (scale_pos_weight) với SMOTENC, phương pháp xử lý đúng các đặc trưng phân loại.",
        ),
        (
            "Model training and comparison",
            "Huấn luyện và so sánh mô hình",
            "Evaluate the baseline and five candidate models with repeated stratified cross-validation (5 folds × 2 repeats), ranked by PR-AUC.",
            "Đánh giá mô hình cơ sở và 5 mô hình ứng viên bằng Repeated Stratified CV (5-fold × 2 lần lặp), xếp hạng theo PR-AUC.",
        ),
        (
            "Hyperparameter tuning",
            "Tối ưu Hyperparameter",
            "Run GridSearchCV over 27 parameter combinations and five folds, for 135 training runs.",
            "Chạy GridSearchCV với 27 tổ hợp tham số × 5-fold, tương đương 135 lần huấn luyện.",
        ),
        (
            "Probability calibration and threshold selection",
            "Hiệu chỉnh xác suất và chọn ngưỡng",
            "Apply Platt scaling and choose the F2-optimal decision threshold on the validation set.",
            "Dùng Platt scaling để hiệu chỉnh xác suất và chọn ngưỡng tối ưu F2 trên tập validation.",
        ),
        (
            "Final training and test evaluation",
            "Huấn luyện mô hình cuối và đánh giá Test",
            "Refit on train + validation, evaluate once on the locked test set, and report bootstrap 95% confidence intervals.",
            "Refit trên train + validation, đánh giá một lần trên tập test khóa và báo cáo Bootstrap 95% CI.",
        ),
        (
            "Packaging and Streamlit deployment",
            "Đóng gói và triển khai ứng dụng Streamlit",
            "Package the versioned model artifact and deploy single-case screening, batch CSV prediction, model evidence, clinical feedback, and controlled retraining.",
            "Đóng gói artifact có phiên bản và triển khai dự đoán 1 ca, dự đoán CSV hàng loạt, bằng chứng mô hình, phản hồi bác sĩ và tái huấn luyện có kiểm soát.",
        ),
    ]

    for row_start in range(0, len(workflow_steps), 3):
        columns = st.columns(3, gap="medium")
        for offset, (title_en, title_vi, description_en, description_vi) in enumerate(
            workflow_steps[row_start : row_start + 3]
        ):
            step_number = row_start + offset + 1
            with columns[offset].container(border=True):
                st.markdown(
                    f"**{_t(f'Step {step_number}', f'Bước {step_number}')} · "
                    f"{_t(title_en, title_vi)}**"
                )
                st.write(_t(description_en, description_vi))

    st.info(
        _t(
            "Open Model overview for detailed evidence and metrics. Operational feedback and retraining continue in the Clinical feedback and Model update areas.",
            "Mở Tổng quan & tối ưu mô hình để xem bằng chứng và chỉ số chi tiết. Phản hồi sau khám và tái huấn luyện được tiếp tục quản lý tại mục Phản hồi sau khám và tab Cập nhật mô hình.",
        )
    )


def _model_workspace_tab(payload: dict, metadata: dict, user: AuthUser) -> None:
    """Keep model evidence and clinical feedback in one workspace."""

    view_labels = {
        "Project workflow": "Quy trình tổng quan",
        "Model overview": "Tổng quan & tối ưu mô hình",
        "EDA evidence": "EDA & bằng chứng dữ liệu",
        "Clinical feedback": "Phản hồi sau khám",
    }
    if st.session_state.get("model_information_view") not in {None, *view_labels}:
        st.session_state["model_information_view"] = "Project workflow"
    selected_view = st.radio(
        _t("Model-information view", "Nội dung Thông tin mô hình"),
        list(view_labels),
        format_func=lambda value: _t(value, view_labels[value]),
        horizontal=True,
        key="model_information_view",
    )
    st.caption(
        _t(
            "The project workflow, model evidence, EDA, and clinical feedback are organized here. Use the separate Model update tab to upload labelled CSV data.",
            "Quy trình đề tài, bằng chứng mô hình, EDA và phản hồi sau khám được sắp xếp tại đây. Dùng tab Cập nhật mô hình riêng để tải CSV có nhãn thực tế.",
        )
    )
    if selected_view == "Project workflow":
        _project_workflow_overview()
    elif selected_view == "Model overview":
        _model_info_tab(payload, metadata)
    elif selected_view == "EDA evidence":
        _eda_tab()
    elif selected_view == "Clinical feedback":
        _feedback_tab(user)


def _prediction_workspace_tab(payload: dict, metadata: dict, user: AuthUser) -> None:
    """Offer individual and CSV screening within one main prediction tab."""

    single_tab, batch_tab = st.tabs(
        [
            _t("Single-case prediction", "Dự đoán 1 ca"),
            _t("Batch prediction from CSV", "Tải CSV dự đoán hàng loạt"),
        ]
    )
    with single_tab:
        _single_case_tab(payload, metadata, user)
    with batch_tab:
        _batch_processing_tab(payload, metadata, user)


def main() -> None:
    _inject_app_styles()
    _render_sidebar_brand()
    _language_selector()
    user = _require_authentication()
    st.title(_t("Alzheimer's Disease Screening", "Sàng lọc bệnh Alzheimer"))
    _render_account_sidebar(user)
    try:
        payload, metadata = get_runtime(_active_artifact_cache_key())
    except Exception as exc:
        st.error(str(exc))
        st.code("python train_pipeline.py")
        st.stop()

    active_threshold = float(metadata.get("threshold", payload.get("threshold", 0.5)))
    active_version = metadata.get("version", payload.get("version", "unknown"))
    active_features = list(
        metadata.get("selected_features", payload.get("features", SELECTED_FEATURES))
    )
    _render_sidebar_status(active_version, len(active_features), active_threshold)

    st.markdown(
        f"""
        <div class="app-hero">
            <div>
                <div class="hero-kicker">NeuroScreen · V3 clinical research UI</div>
                <div class="hero-copy">
                    {_t(
                        "A calm workspace for transparent Alzheimer screening, data quality review and model governance.",
                        "Không gian làm việc trực quan cho sàng lọc Alzheimer, kiểm tra chất lượng dữ liệu và quản trị mô hình.",
                    )}
                </div>
            </div>
            <div class="hero-meta">
                <div class="hero-meta-label">{_t("Active release", "Bản phát hành")}</div>
                <div class="hero-meta-value">{active_version}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(_t("Educational/research screening model; not a substitute for a clinician's assessment.", "Mô hình sàng lọc phục vụ học tập/nghiên cứu; không thay thế đánh giá của bác sĩ."))

    overview = st.columns(3)
    overview[0].metric(_t("Active model", "Mô hình đang dùng"), "XGBoost")
    overview[1].metric(_t("Model inputs", "Đầu vào mô hình"), len(active_features), _t("selected features", "đặc trưng đã chọn"))
    overview[2].metric(_t("Screening threshold", "Ngưỡng sàng lọc"), f"{active_threshold:.3f}", _t("calibrated score", "điểm đã hiệu chuẩn"))
    st.markdown(
        '<div class="section-kicker">'
        + _t("Application workspace", "Không gian làm việc")
        + "</div>",
        unsafe_allow_html=True,
    )

    navigation_labels = {
        "Prediction": "1 · Dự đoán",
        "Project dataset": "2 · Dataset dự án",
        "Model information": "3 · Thông tin mô hình",
        "Model update": "4 · Cập nhật mô hình (CSV)",
    }
    navigation_options = list(navigation_labels)
    legacy_section = st.session_state.get("active_section", "Prediction")
    legacy_processing_source = st.session_state.pop("processing_source", None)
    if legacy_section in {"Single case", "Batch CSV"}:
        current_section = "Prediction"
    elif legacy_section == "Prediction" and legacy_processing_source == "Project dataset":
        current_section = "Project dataset"
    elif legacy_section == "EDA":
        current_section = "Model information"
    elif legacy_section == "Retrain / Model update":
        current_section = "Model update"
    elif legacy_section == "Model information" and st.session_state.get("model_information_view") == "Retrain / Model update":
        current_section = "Model update"
    else:
        current_section = legacy_section
    if current_section not in navigation_options:
        current_section = navigation_options[0]
    if st.session_state.get("active_section") != current_section:
        st.session_state["active_section"] = current_section
    active_section = st.radio(
        _t("Main application tabs", "Các tab chính của ứng dụng"),
        navigation_options,
        index=navigation_options.index(current_section),
        format_func=lambda value: _t(value, navigation_labels[value]),
        horizontal=True,
        key="active_section",
        label_visibility="collapsed",
    )
    if active_section == "Prediction":
        _prediction_workspace_tab(payload, metadata, user)
    elif active_section == "Project dataset":
        _batch_processing_tab(payload, metadata, user, use_project_dataset=True)
    elif active_section == "Model information":
        _model_workspace_tab(payload, metadata, user)
    else:
        _retrain_tab(payload, metadata, user)


if __name__ == "__main__":
    main()
