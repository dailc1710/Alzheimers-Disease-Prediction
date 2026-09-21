import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from feedback_store import (
    feedback_training_frame,
    list_cases,
    record_prediction_cases,
    record_retraining_run,
)
from ml_pipeline import ALL_FEATURES

os.environ.setdefault("ALZHEIMER_AUTH_DISABLED", "1")


class AppSmokeTests(unittest.TestCase):
    def test_login_page_uses_auth_layout_without_loading_the_dashboard(self):
        with patch("auth.authentication_disabled", return_value=False), patch(
            "auth.user_count", return_value=1
        ):
            app_test = AppTest.from_file("app.py").run(timeout=60)
        self.assertEqual([], list(app_test.exception))
        self.assertTrue(any("Chào mừng trở lại" in str(item.value) for item in app_test.markdown))
        self.assertEqual(["Tên đăng nhập", "Mật khẩu"], [item.label for item in app_test.text_input])
        self.assertEqual([], list(app_test.title))

    def test_first_admin_setup_uses_same_auth_layout(self):
        with patch("auth.authentication_disabled", return_value=False), patch(
            "auth.user_count", return_value=0
        ):
            app_test = AppTest.from_file("app.py").run(timeout=60)
        self.assertEqual([], list(app_test.exception))
        self.assertTrue(
            any("Thiết lập tài khoản đầu tiên" in str(item.value) for item in app_test.markdown)
        )
        self.assertEqual(
            ["Tên đăng nhập quản trị", "Mật khẩu", "Nhập lại mật khẩu"],
            [item.label for item in app_test.text_input],
        )

    def test_streamlit_ui_runs_without_exceptions(self):
        app_test = AppTest.from_file("app.py").run(timeout=60)
        self.assertEqual([], list(app_test.exception))
        self.assertGreaterEqual(len(app_test.title), 1)

    def test_streamlit_app_imports_without_starting_a_server(self):
        app = importlib.import_module("app")
        self.assertTrue(callable(app.main))
        self.assertTrue(callable(app._require_authentication))
        self.assertTrue(callable(app._feedback_tab))
        self.assertEqual(0.425, app.REPORT_V3_MODEL_REFERENCE["threshold"])
        self.assertEqual(
            0.9394,
            app.REPORT_V3_MODEL_REFERENCE["benchmark"]["pr_auc_cv"],
        )

    def test_primary_navigation_separates_prediction_and_model_update_csv(self):
        app_test = AppTest.from_file("app.py").run(timeout=60)
        navigation = app_test.radio(key="active_section")
        self.assertEqual(
            [
                "1 · Tổng quan, dữ liệu & EDA",
                "2 · Xây dựng & đánh giá mô hình",
                "3 · Dự đoán",
                "4 · Vận hành & cập nhật",
            ],
            list(navigation.options),
        )
        self.assertEqual("Prediction", navigation.value)
        self.assertEqual(
            ["Dự đoán 1 ca", "Tải CSV dự đoán hàng loạt"],
            [tab.label for tab in app_test.get("tab")],
        )
        self.assertEqual(1, len(app_test.get("file_uploader")))
        self.assertIn("dự đoán hàng loạt", app_test.get("file_uploader")[0].label)
        self.assertTrue(any("Dự đoán 1 ca" in item.value for item in app_test.subheader))

    def test_batch_csv_tab_shows_uploader_and_v3_validation_contract(self):
        app_test = AppTest.from_file("app.py").run(timeout=60)
        self.assertEqual([], list(app_test.exception))
        self.assertEqual("Prediction", app_test.radio(key="active_section").value)
        self.assertEqual(1, len(app_test.get("file_uploader")))
        self.assertIn("dự đoán hàng loạt", app_test.get("file_uploader")[0].label)
        self.assertTrue(
            any(
                "Quy tắc kiểm tra V3" in item.label
                for item in app_test.expander
            )
        )
        self.assertFalse(any(item.label == "Schema đầu vào" for item in app_test.radio))

    def test_batch_prediction_redirects_full_v3_csv_to_project_dataset(self):
        columns = ["PatientID", *ALL_FEATURES]
        csv_bytes = (
            ",".join(columns) + "\n" + ",".join(["1"] * len(columns)) + "\n"
        ).encode("utf-8")

        class UploadedCSV:
            name = "full_v3.csv"
            size = len(csv_bytes)

            def getvalue(self):
                return csv_bytes

        with patch("streamlit.file_uploader", return_value=UploadedCSV()):
            app_test = AppTest.from_file("app.py").run(timeout=60)
        self.assertEqual([], list(app_test.exception))
        self.assertTrue(
            any("CSV Full V3 thuộc tab Dữ liệu & EDA" in item.value for item in app_test.error)
        )

    def test_batch_prediction_accepts_five_feature_csv(self):
        csv_bytes = (
            b"MMSE,FunctionalAssessment,ADL,MemoryComplaints,BehavioralProblems\n"
            b"20,5,5,0,0\n"
        )

        class UploadedCSV:
            name = "five_features.csv"
            size = len(csv_bytes)

            def getvalue(self):
                return csv_bytes

        with patch("streamlit.file_uploader", return_value=UploadedCSV()):
            app_test = AppTest.from_file("app.py").run(timeout=60)
        self.assertEqual([], list(app_test.exception))
        self.assertEqual([], list(app_test.error))
        self.assertTrue(any("Kết quả sàng lọc batch" in item.value for item in app_test.subheader))

    def test_project_dataset_has_its_own_csv_uploader_and_bundled_fallback(self):
        app_test = AppTest.from_file("app.py").run(timeout=60)
        app_test.radio(key="active_section").set_value("Project dataset").run(timeout=60)
        app_test.radio(key="data_workspace_view").set_value("Dataset and cleaning").run(timeout=60)
        self.assertEqual([], list(app_test.exception))
        self.assertEqual(1, len(app_test.get("file_uploader")))
        self.assertIn("CSV đầu vào", app_test.get("file_uploader")[0].label)
        self.assertEqual(0, len(app_test.get("tab")))
        self.assertTrue(any("Dataset dự án" in item.value for item in app_test.subheader))
        self.assertFalse(any("Kết quả sàng lọc batch" in item.value for item in app_test.subheader))
        self.assertFalse(any("Schema đầu vào" == item.label for item in app_test.radio))
        self.assertTrue(any("CSV đầu ra" in str(item.value) for item in app_test.markdown))
        self.assertTrue(any("file có sẵn" in str(item.value) for item in app_test.caption))
        self.assertEqual("35", next(item.value for item in app_test.metric if item.label == "Cột đầu vào"))
        self.assertEqual("186", next(item.value for item in app_test.metric if item.label == "Dòng bị loại"))

    def test_project_dataset_upload_auto_detects_and_reconciles_rows(self):
        csv_bytes = (
            b"MMSE,FunctionalAssessment,ADL,MemoryComplaints,BehavioralProblems,Diagnosis\n"
            b"20,5,5,0,0,1\n"
            b"20,5,5,0,0,1\n"
            b"-1,5,5,0,0,0\n"
            b",5,5,0,0,1\n"
            b"18,5,5,0,0,2\n"
        )

        class UploadedCSV:
            name = "sample.csv"
            size = len(csv_bytes)

            def getvalue(self):
                return csv_bytes

        app_test = AppTest.from_file("app.py").run(timeout=60)
        with patch("streamlit.file_uploader", return_value=UploadedCSV()):
            app_test.radio(key="active_section").set_value("Project dataset").run(timeout=60)
            app_test.radio(key="data_workspace_view").set_value("Dataset and cleaning").run(timeout=60)
        self.assertEqual([], list(app_test.exception))
        self.assertEqual([], list(app_test.error))
        self.assertTrue(any("5 đặc trưng mô hình" in item.value for item in app_test.success))
        self.assertEqual("5", next(item.value for item in app_test.metric if item.label == "Dòng đầu vào"))
        self.assertEqual("6", next(item.value for item in app_test.metric if item.label == "Cột đầu vào"))
        self.assertEqual("1", next(item.value for item in app_test.metric if item.label == "Dòng trùng"))
        self.assertEqual("2", next(item.value for item in app_test.metric if item.label == "Dòng hợp lệ"))
        self.assertEqual("3", next(item.value for item in app_test.metric if item.label == "Dòng bị loại"))
        self.assertTrue(any("Bảng lỗi và kết quả xử lý" in str(item.value) for item in app_test.markdown))
        self.assertTrue(any(
            {"Dòng dữ liệu", "Giá trị đầu vào", "Sau xử lý", "Trạng thái"}.issubset(item.value.columns)
            for item in app_test.dataframe
        ))
        self.assertFalse(any("Kết quả sàng lọc batch" in item.value for item in app_test.subheader))

    def test_eda_tab_renders_compact_charts_without_exceptions(self):
        app_test = AppTest.from_file("app.py").run(timeout=60)
        app_test.radio(key="active_section").set_value("Project dataset").run(timeout=60)
        app_test.radio(key="data_workspace_view").set_value("EDA evidence").run(timeout=60)
        self.assertEqual([], list(app_test.exception))
        self.assertTrue(any("Pearson" in str(item.value) for item in app_test.markdown))
        self.assertTrue(
            any("Học không giám sát" in str(item.value) for item in app_test.subheader)
        )

    def test_unsupervised_analysis_uses_32_features_without_diagnosis(self):
        app = importlib.import_module("app")
        cleaned, _ = app.clean_dataset(app.get_reference_data())
        analysis = app._unsupervised_learning_analysis(cleaned)
        self.assertEqual(32, analysis["feature_count"])
        self.assertEqual(list(range(2, 7)), analysis["diagnostics"]["k"].tolist())
        self.assertFalse(analysis["diagnosis_used_for_fit"])
        self.assertNotIn("Diagnosis", analysis["cluster_profile"].columns)
        self.assertFalse(analysis["diagnosis_comparison"].empty)
        self.assertEqual(len(cleaned), int(analysis["cluster_profile"]["Số bệnh nhân"].sum()))

    def test_model_information_shows_optimization_evidence(self):
        app = importlib.import_module("app")
        app_test = AppTest.from_file("app.py").run(timeout=60)
        app_test.radio(key="active_section").set_value("Model information").run(timeout=60)
        self.assertEqual([], list(app_test.exception))
        self.assertTrue(
            any(
                "Hoàn thiện Tiền xử lý & Tối ưu Mô hình" in str(item.value)
                for item in app_test.markdown
            )
        )
        self.assertTrue(
            any(
                "Hiện/ẩn danh sách cột" in item.label
                for item in app_test.expander
            )
        )
        _, metadata = app.load_artifacts(app.DEFAULT_ARTIFACTS_DIR)
        stage_matrix = app._feature_selection_stage_matrix(metadata, len(app.SELECTED_FEATURES))
        self.assertEqual(32, len(stage_matrix))
        for feature_count in [32, 20, 12, 8, 5]:
            self.assertEqual(feature_count, int(stage_matrix[f"Stage {feature_count}"].sum()))
        self.assertEqual(
            set(app.SELECTED_FEATURES),
            set(stage_matrix.loc[stage_matrix["Stage 5"], "Column"]),
        )
        self.assertTrue(
            any(
                "5 đặc trưng cuối" in str(item.value)
                for item in app_test.markdown
            )
        )

    def test_optimization_tabs_keep_evidence_in_its_own_tab(self):
        app = importlib.import_module("app")
        app_test = AppTest.from_file("app.py").run(timeout=60)
        app_test.radio(key="active_section").set_value("Model information").run(timeout=60)
        self.assertEqual([], list(app_test.exception))
        tabs = app_test.get("tab")
        self.assertEqual(4, len(tabs))
        tab_text = [" ".join(str(item.value) for item in tab.markdown) for tab in tabs]
        self.assertIn("Quá trình lựa chọn", tab_text[0])
        self.assertIn("Mất cân bằng lớp và gán trọng số", tab_text[1])
        self.assertIn("Benchmark mô hình", tab_text[2])
        self.assertIn("Quét ngưỡng trên validation", tab_text[3])
        self.assertNotIn("So sánh mô hình", tab_text[0])
        self.assertNotIn("Quá trình lựa chọn", tab_text[2])
        self.assertNotIn("Điểm vận hành trên test khóa", tab_text[2])
        self.assertIn("Điểm vận hành trên test khóa", tab_text[3])
        threshold_captions = " ".join(str(item.value) for item in tabs[3].caption)
        self.assertIn(
            "Tất cả dòng trong bảng này chỉ được tính trên tập validation",
            threshold_captions,
        )

        _, metadata = app.load_artifacts(app.DEFAULT_ARTIFACTS_DIR)
        threshold_table = app._threshold_scenario_table(metadata, metadata["threshold"])
        self.assertEqual({"Validation (n=314)"}, set(threshold_table["Cohort"]))
        selected = threshold_table.loc[
            threshold_table["Scenario"].eq("Selected by validation F2")
        ].iloc[0]
        self.assertEqual(16, int(selected["False negatives"]))
        self.assertEqual(6, int(selected["False positives"]))
        locked_test_tables = [
            item.value
            for item in tabs[3].dataframe
            if {"TN", "FP", "FN", "TP"}.issubset(item.value.columns)
        ]
        self.assertEqual(1, len(locked_test_tables))
        self.assertEqual(
            {"TN": 242, "FP": 12, "FN": 13, "TP": 126},
            locked_test_tables[0].iloc[0][["TN", "FP", "FN", "TP"]].to_dict(),
        )

    def test_project_workflow_precedes_dataset_views_and_shows_ten_steps(self):
        app_test = AppTest.from_file("app.py").run(timeout=60)
        app_test.radio(key="active_section").set_value("Project dataset").run(timeout=60)
        self.assertEqual([], list(app_test.exception))
        workflow_navigation = app_test.radio(key="data_workspace_view")
        self.assertEqual(
            [
                "Quy trình tổng quan",
                "Dataset & làm sạch",
                "EDA & bằng chứng dữ liệu",
            ],
            list(workflow_navigation.options),
        )
        self.assertEqual("Project workflow", workflow_navigation.value)
        self.assertTrue(
            any(
                "Quy trình thực hiện tổng quan" in str(item.value)
                for item in app_test.subheader
            )
        )
        rendered_markdown = " ".join(str(item.value) for item in app_test.markdown)
        for step_number in range(1, 11):
            self.assertIn(f"Bước {step_number}", rendered_markdown)

    def test_retraining_tab_exposes_feedback_and_csv_sources(self):
        with patch.dict(os.environ, {"ALZHEIMER_AUTH_DISABLED": "1"}, clear=False):
            app_test = AppTest.from_file("app.py").run(timeout=60)
            app_test.radio(key="active_section").set_value("Model update").run(timeout=60)
            app_test.radio(key="operations_workspace_view").set_value("Retrain / Model update").run(timeout=60)
            self.assertEqual("Labelled five-feature CSV", app_test.radio(key="retraining_source").value)
            uploaders = app_test.get("file_uploader")
            self.assertEqual(1, len(uploaders))
            self.assertIn("có nhãn", uploaders[0].label.lower())
            self.assertNotIn("dự đoán hàng loạt", uploaders[0].label.lower())
            self.assertTrue(any("quyền quản trị" in item.value.lower() for item in app_test.success))
            self.assertTrue(any("Mẫu CSV huấn luyện lại" in item.label for item in app_test.expander))
            with patch("streamlit.file_uploader") as uploader:
                app_test.radio(key="retraining_source").set_value("Verified clinical feedback").run(timeout=60)
                uploader.assert_not_called()
            self.assertEqual([], list(app_test.exception))
            self.assertEqual(0, len(app_test.get("file_uploader")))
            self.assertFalse(any("Mẫu CSV huấn luyện lại" in item.label for item in app_test.expander))
            self.assertTrue(any(
                "Cập nhật từ feedback đã xác minh" in str(item.value)
                for item in app_test.markdown
            ))

    def test_retraining_comparison_marks_brier_decrease_as_better(self):
        app = importlib.import_module("app")
        comparison = app._retraining_metric_comparison([
            {"metric": "pr_auc", "old": 0.9221, "new": 0.9198, "passed": False},
            {"metric": "recall", "old": 0.9065, "new": 0.9065, "passed": True},
            {"metric": "f2", "old": 0.9078, "new": 0.9065, "passed": False},
            {"metric": "brier", "old": 0.0550, "new": 0.0535, "passed": True},
        ])
        self.assertEqual(["PR-AUC", "Recall", "F2", "Brier"], comparison["label"].tolist())
        self.assertEqual(["worse", "same", "worse", "better"], comparison["outcome"].tolist())
        self.assertEqual(["down", "same", "down", "down"], comparison["direction"].tolist())

    def test_saved_retraining_comparison_renders_without_a_new_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "state.sqlite3"
            run_id = record_retraining_run(
                started_by="admin",
                source="Labelled Full V3 CSV",
                source_rows=10,
                status="retained_champion",
                promoted=False,
                policy_passed=False,
                details={
                    "champion_threshold": 0.425,
                    "challenger_threshold": 0.380,
                    "active_threshold_after": 0.425,
                    "selected_features": [
                        "MMSE",
                        "FunctionalAssessment",
                        "ADL",
                        "MemoryComplaints",
                        "BehavioralProblems",
                    ],
                    "champion_feature_importance": [
                        {"feature": "MMSE", "percent": 18.36},
                        {"feature": "FunctionalAssessment", "percent": 19.39},
                        {"feature": "ADL", "percent": 18.42},
                        {"feature": "MemoryComplaints", "percent": 25.90},
                        {"feature": "BehavioralProblems", "percent": 17.93},
                    ],
                    "challenger_feature_importance": [
                        {"feature": "MMSE", "percent": 19.00},
                        {"feature": "FunctionalAssessment", "percent": 20.00},
                        {"feature": "ADL", "percent": 18.00},
                        {"feature": "MemoryComplaints", "percent": 25.00},
                        {"feature": "BehavioralProblems", "percent": 18.00},
                    ],
                    "promotion_decision": {"checks": [
                        {"metric": "pr_auc", "old": 0.9221, "new": 0.9198, "passed": False},
                        {"metric": "recall", "old": 0.9065, "new": 0.9065, "passed": True},
                        {"metric": "f2", "old": 0.9078, "new": 0.9065, "passed": False},
                        {"metric": "brier", "old": 0.0550, "new": 0.0535, "passed": True},
                    ]},
                },
                database_path=database,
            )
            with patch.dict(os.environ, {
                "ALZHEIMER_AUTH_DISABLED": "1",
                "ALZHEIMER_STATE_DB": str(database),
            }, clear=False):
                app_test = AppTest.from_file("app.py").run(timeout=60)
                app_test.radio(key="active_section").set_value("Model update").run(timeout=60)
                app_test.radio(key="operations_workspace_view").set_value("Retrain / Model update").run(timeout=60)
                self.assertEqual([], list(app_test.exception))
                self.assertEqual(run_id, app_test.selectbox(key="retraining_history_selection").value)
                self.assertTrue(any("So sánh mô hình hiện tại" in str(item.value) for item in app_test.markdown))
                metric_values = {item.label: item.value for item in app_test.metric}
                self.assertEqual("0.425", metric_values["Ngưỡng Champion"])
                self.assertEqual("0.380", metric_values["Ngưỡng Challenger"])
                self.assertEqual("0.425", metric_values["Ngưỡng active sau phiên"])
                importance_tables = [
                    item.value
                    for item in app_test.dataframe
                    if {
                        "Đặc trưng",
                        "Tỷ trọng Champion",
                        "Tỷ trọng Challenger",
                        "Thay đổi",
                    }.issubset(item.value.columns)
                ]
                self.assertEqual(1, len(importance_tables))
                self.assertEqual(5, len(importance_tables[0]))
                self.assertEqual("+0.64 điểm %", importance_tables[0].iloc[0]["Thay đổi"])

    def test_feedback_view_renders_inside_operations_workspace(self):
        app_test = AppTest.from_file("app.py").run(timeout=60)
        app_test.radio(key="active_section").set_value("Model update").run(timeout=60)
        app_test.radio(key="operations_workspace_view").set_value("Clinical feedback").run(timeout=60)
        self.assertEqual([], list(app_test.exception))
        self.assertTrue(any("Vòng lặp phản hồi" in item.value for item in app_test.subheader))

    def test_verified_five_feature_feedback_warns_and_refreshes_case_status(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "state.sqlite3"
            frame = pd.DataFrame([{
                "MMSE": 20,
                "FunctionalAssessment": 5,
                "ADL": 5,
                "MemoryComplaints": 0,
                "BehavioralProblems": 0,
                "calibrated_score": 0.3,
                "predicted_label": 0,
            }])
            record_prediction_cases(
                frame, actor="test-admin", source="single_case", model_version="v1",
                threshold=0.5, database_path=database,
            )
            with patch.dict(os.environ, {
                "ALZHEIMER_AUTH_DISABLED": "1",
                "ALZHEIMER_STATE_DB": str(database),
            }, clear=False):
                app_test = AppTest.from_file("app.py").run(timeout=60)
                app_test.radio(key="active_section").set_value("Model update").run(timeout=60)
                app_test.radio(key="operations_workspace_view").set_value("Clinical feedback").run(timeout=60)
                self.assertTrue(any("Kết quả AI (chỉ xem)" in item.value for item in app_test.markdown))
                self.assertTrue(any("Kết luận khám của bác sĩ" in item.value for item in app_test.markdown))
                result_input = next(item for item in app_test.radio if item.label == "Bác sĩ kết luận sau khám")
                self.assertIsNone(result_input.value)
                save_button = next(item for item in app_test.button if item.label == "Lưu kết luận khám đã xác minh")
                self.assertTrue(save_button.disabled)
                result_input.set_value(1).run(timeout=60)
                self.assertTrue(next(item for item in app_test.button if item.label == "Lưu kết luận khám đã xác minh").disabled)
                next(item for item in app_test.checkbox if item.label.startswith("Tôi đã đối chiếu")).set_value(True).run(timeout=60)
                self.assertFalse(next(item for item in app_test.button if item.label == "Lưu kết luận khám đã xác minh").disabled)
                next(item for item in app_test.button if item.label == "Lưu kết luận khám đã xác minh").click().run(timeout=60)

            self.assertEqual([], list(app_test.exception))
            self.assertTrue(any("Cảnh báo mô hình dự đoán sai" in item.value for item in app_test.warning))
            self.assertTrue(any("Tài khoản xác minh: test-admin" in item.value for item in app_test.caption))
            table = next(item.value for item in app_test.dataframe if "Đối chiếu mô hình" in item.value.columns)
            self.assertEqual("Mô hình dự đoán sai", table.iloc[0]["Đối chiếu mô hình"])
            self.assertEqual("Có", table.iloc[0]["Ứng viên retrain"])

    def test_viewer_clinical_result_waits_for_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "state.sqlite3"
            frame = pd.DataFrame([{
                "MMSE": 20,
                "FunctionalAssessment": 5,
                "ADL": 5,
                "MemoryComplaints": 0,
                "BehavioralProblems": 0,
                "calibrated_score": 0.3,
                "predicted_label": 0,
            }])
            record_prediction_cases(
                frame, actor="viewer.one", source="single_case", model_version="v1",
                threshold=0.5, database_path=database,
            )
            with patch.dict(os.environ, {
                "ALZHEIMER_AUTH_DISABLED": "0",
                "ALZHEIMER_STATE_DB": str(database),
            }, clear=False):
                app_test = AppTest.from_file("app.py")
                app_test.session_state["auth_user"] = {"username": "viewer.one", "role": "viewer"}
                app_test.run(timeout=60)
                app_test.radio(key="active_section").set_value("Model update").run(timeout=60)
                app_test.radio(key="operations_workspace_view").set_value("Clinical feedback").run(timeout=60)
                next(item for item in app_test.radio if item.label == "Bác sĩ kết luận sau khám").set_value(1).run(timeout=60)
                next(item for item in app_test.checkbox if item.label.startswith("Tôi đã đối chiếu")).set_value(True).run(timeout=60)
                next(item for item in app_test.button if item.label == "Gửi kết quả để bác sĩ xác minh").click().run(timeout=60)

            self.assertEqual([], list(app_test.exception))
            self.assertTrue(any("đang chờ xác minh" in item.value for item in app_test.caption))
            case = list_cases(database_path=database).iloc[0]
            self.assertEqual(0, int(case["latest_verified"]))
            self.assertTrue(feedback_training_frame(database).empty)


if __name__ == "__main__":
    unittest.main()
