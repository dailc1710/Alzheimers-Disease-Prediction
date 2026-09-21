import csv
import unittest
from io import StringIO
from unittest.mock import patch

from ml_pipeline import SELECTED_FEATURES, TARGET_COLUMN
from ui.data_processing import read_uploaded_csv, retraining_template_bytes


class _UploadedCsv:
    def __init__(self, content: bytes):
        self.content = content
        self.name = "alzheimer_retraining_template.csv"
        self.size = len(content)

    def getvalue(self) -> bytes:
        return self.content


class RetrainingTemplateTests(unittest.TestCase):
    def test_large_foreign_schema_reports_missing_model_inputs_before_batch_limit(self):
        content = (
            "Country,Cognitive Test Score,Alzheimer's Diagnosis\n"
            "Spain,90,No\nSpain,80,Yes\nSpain,70,No\n"
        ).encode("utf-8")
        with patch("ui.data_processing.MAX_UPLOAD_ROWS", 2):
            with self.assertRaises(ValueError) as raised:
                read_uploaded_csv(
                    _UploadedCsv(content),
                    required_columns=SELECTED_FEATURES,
                    language="vi",
                )
        message = str(raised.exception)
        self.assertIn("3 dòng", message)
        self.assertIn("MMSE", message)
        self.assertIn("BehavioralProblems", message)
        self.assertIn("bộ dữ liệu khác mẫu", message)

    def test_large_valid_schema_explains_how_to_split_without_scoring(self):
        header = ",".join(SELECTED_FEATURES)
        row = "20,5,5,0,0"
        content = (header + "\n" + "\n".join([row] * 3) + "\n").encode("utf-8")
        with patch("ui.data_processing.MAX_UPLOAD_ROWS", 2):
            with self.assertRaises(ValueError) as raised:
                read_uploaded_csv(
                    _UploadedCsv(content),
                    required_columns=SELECTED_FEATURES,
                    language="vi",
                )
        self.assertIn("chia thành các file", str(raised.exception))
        self.assertIn("chưa dòng nào được dự đoán", str(raised.exception))

    def test_download_has_five_features_and_diagnosis_without_fake_patient(self):
        content = retraining_template_bytes()
        rows = list(csv.reader(StringIO(content.decode("utf-8-sig"))))
        expected = [*SELECTED_FEATURES, TARGET_COLUMN]
        self.assertEqual([expected], rows)
        self.assertEqual(6, len(rows[0]))

        parsed, _ = read_uploaded_csv(_UploadedCsv(content))
        self.assertEqual(expected, parsed.columns.tolist())
        self.assertTrue(parsed.empty)


if __name__ == "__main__":
    unittest.main()
