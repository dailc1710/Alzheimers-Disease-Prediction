# Thiết kế tab EDA theo báo cáo V3

## Kết luận

Tab EDA hiện tại mới là một profile chất lượng dữ liệu cơ bản. Nó đã có rows/columns, missing, duplicate, giá trị âm, kiểm tra miền V3, phân phối một biến, phân bố `Diagnosis`, tương quan của 5 feature cuối và biểu đồ BP. Tuy nhiên, như vậy chưa tái hiện đầy đủ phần EDA trong báo cáo V3 và chưa giải thích rõ cho người dùng CSV đang sai ở đâu, vì sao sai và bước tiếp theo phải làm gì.

Tab EDA mới nên là một tab **quan sát dữ liệu thô trước xử lý**. Nó không được tự impute, không tự xóa dòng và không chạy prediction. Việc sửa/loại/impute phải tiếp tục nằm trong `Batch CSV + Data processing`; việc dùng dữ liệu có `Diagnosis` để cập nhật model phải nằm trong `Retrain / Model update`.

## Bằng chứng từ báo cáo V3

Nguồn chính là `../../deliverables/reports/I1_2510_E0_-_Nhom_1-_Alzheimer_project_report_V3_revised.docx`, đặc biệt phần 4, phần 5, Table 2 và Table 12.

### EDA phải nói được gì

1. Bộ dữ liệu thô có 2.149 dòng và 35 cột.
2. Có 0 dòng trùng và 0 giá trị thiếu trong dataset gốc.
3. Có 186 dòng vi phạm `SystolicBP <= DiastolicBP`; báo cáo loại các dòng này trước phân tích/model.
4. `DoctorInCharge` chỉ có 1 giá trị duy nhất nên không có thông tin phân biệt; `PatientID` là định danh, không phải feature.
5. Sau lọc BP, nhãn `Diagnosis` có 1.269 dòng 0 và 694 dòng 1, tương ứng 64,6% và 35,4%.
6. EDA của báo cáo gồm phân bố nhãn, Cohen's d, ma trận tương quan Pearson và boxplot 5 biến phân biệt mạnh nhất theo nhóm `Diagnosis`.
7. Năm feature chính thức được chọn là `MMSE`, `FunctionalAssessment`, `ADL`, `MemoryComplaints` và `BehavioralProblems`.

### CSV lỗi phải được giải thích theo loại lỗi

Table 12 của báo cáo yêu cầu kiểm tra các trường hợp: thiếu giá trị, số âm phi lý, sai định dạng, ngoài khoảng hợp lý, mã phân loại không hợp lệ và vi phạm luật `SP < DP`.

| Vấn đề trong CSV | EDA cần hiển thị | Bước xử lý kế tiếp |
|---|---|---|
| Thiếu giá trị | Tên cột, số ô thiếu, tỷ lệ %, các dòng bị ảnh hưởng | Batch impute theo artifact; nếu cả 5 feature model đều thiếu thì reject |
| Số âm phi lý | Cột, số ô âm, ví dụ dòng/giá trị | Reject dòng và ghi rule vào log |
| Sai định dạng | Giá trị gốc không chuyển được sang số và số dòng bị ảnh hưởng | Parser đưa về missing; Batch áp dụng rule impute/reject |
| Ngoài khoảng V3 | Cận dưới/trên, giá trị quan sát, số dòng | Reject dòng, không clip im lặng |
| Mã category/binary lạ | Tập giá trị quan sát và tập giá trị cho phép | Reject dòng |
| `SystolicBP <= DiastolicBP` | Số dòng vi phạm và preview hai cột BP | Loại trước validation/model |
| Dòng trùng hoàn toàn | Số dòng và source row | Batch loại duplicate exact |
| Trùng `PatientID` nhưng nội dung khác | PatientID, số bản ghi và cột khác nhau | Reject/manual review, không tự chọn một bản ghi |
| Thiếu/không hợp lệ `Diagnosis` | Chỉ cảnh báo trong screening; hiển thị rõ label không dùng để predict | Retrain reject; screening bỏ qua label |

## Phạm vi tab EDA mới

### 1. CSV intake và schema audit

Tab cần cho chọn dataset mẫu hoặc upload CSV, sau đó hiển thị ngay:

- filename, kích thước, encoding, delimiter, parser status;
- số dòng/cột sau khi đọc;
- cột thiếu so với Full V3, cột dư và cột bị trùng tên;
- trạng thái schema: Full V3, five-feature scoring hoặc unknown;
- lỗi parser theo dòng nếu CSV malformed.

Pandas `read_csv` hỗ trợ kiểm soát encoding, `na_values`, `keep_default_na`, `na_filter` và `on_bad_lines`; vì vậy phần EDA không nên chỉ nói “upload thất bại” mà phải chỉ ra lỗi đọc CSV và cấu hình đã dùng.^1

### 2. Bảng issue audit theo cột và theo dòng

Đây là phần đang thiếu rõ nhất. EDA cần có một bảng tổng hợp theo cột gồm:

- dtype thực tế và kiểu V3 kỳ vọng;
- non-null, missing count, missing rate;
- unique count và cột constant;
- invalid format count;
- negative count;
- out-of-range count;
- invalid code count;
- rule áp dụng và action đề xuất.

Kèm theo đó là bảng preview các dòng lỗi với `source_row`, `PatientID`, `column`, `observed_value`, `issue_type`, `rule`, `recommended_action`, và nút tải `eda_quality_issues.csv`. EDA chỉ gắn cờ; nó không thay đổi bản ghi. `DataFrame.isna()` chỉ nhận diện giá trị NA, không tự coi chuỗi rỗng là missing nếu parser chưa chuyển chuỗi đó thành NA.^2

Duplicate cũng phải báo rõ đang tính trên toàn dòng hay trên một subset. Pandas mặc định đánh dấu dòng lặp dựa trên toàn bộ cột và có thể dùng `subset` để kiểm tra theo cột.^3

### 3. EDA định lượng theo V3

Tab mới nên có các khu vực sau:

- **Target balance:** bar chart Diagnosis 0/1, count, percentage và cảnh báo nếu thiếu target.
- **Cohen's d:** bảng và horizontal bar top 15 feature; chú giải `|d| < 0,2` không đáng kể, `0,2–0,5` nhỏ, `0,5–0,8` vừa, `> 0,8` lớn.
- **Correlation matrix:** Pearson cho toàn bộ numeric features, không chỉ 5 feature cuối; có thể cho chọn Pearson/Spearman nếu cần. `DataFrame.corr()` tính pairwise correlation và loại NA theo từng cặp.^4
- **Group distributions:** boxplot hoặc violin/strip plot của 5 feature chính, chia theo `Diagnosis`.
- **Feature-selection context:** bảng top feature và đánh dấu 5 feature được pipeline chọn, nhưng không gọi đây là bằng chứng causal.
- **BP sanity plot:** scatter SystolicBP vs DiastolicBP với đường chéo để nhìn các dòng vi phạm.

Mọi biểu đồ phải ghi rõ `n`, nguồn dữ liệu, có/không lọc BP, và cảnh báo rằng đây là dữ liệu tổng hợp/nghiên cứu. Streamlit có chart native cho bar và scatter, nên có thể dùng trực tiếp trong tab mà không thêm runtime biểu đồ mới.^5

### 4. Tách ranh giới với hai tab còn lại

```text
EDA raw CSV
  -> phát hiện và giải thích vấn đề
Batch CSV + Data processing
  -> validate, impute, reject, export log/cleaned/rejected rows
Screening prediction
  -> chỉ dự đoán dòng hợp lệ

Labelled CSV + Diagnosis
  -> Retrain / Model update
```

EDA không được fit imputer, scaler, feature selector hoặc model. Nếu có hiển thị kết quả feature selection/model, phải ghi đó là artifact đã tạo từ pipeline. Scikit-learn khuyến nghị tách train/test trước mọi bước `fit` hoặc `fit_transform` để tránh data leakage.^6

## Tiêu chí nghiệm thu tab EDA mới

- Mở tab không làm thay đổi dataset và không tạo prediction.
- Dataset mẫu hiển thị đúng baseline V3: 2.149 rows, 35 columns, 0 missing, 0 duplicate, 186 BP violations, `DoctorInCharge` constant.
- Upload CSV lỗi phải có thông báo schema/parser cụ thể, không chỉ thông báo chung.
- Một file cố tình chứa NaN, âm, sai định dạng, ngoài khoảng, mã lạ và BP sai phải xuất hiện ở đúng nhóm issue.
- Người dùng nhìn được cả tổng số issue và dòng/cột cụ thể bị lỗi.
- Target balance, Cohen's d, full correlation và boxplot theo Diagnosis phải có sample size và context.
- Có nút tải bảng issue audit; không có nút “clean” trong EDA.
- Cảnh báo rõ: EDA là kiểm tra dữ liệu nghiên cứu, không phải chẩn đoán y khoa.

## Trạng thái triển khai

Tab EDA đã được viết lại trong `app.py`. UI hiện có CSV intake/schema audit, KPI chất lượng, biểu đồ issue theo cột và theo loại lỗi, target balance, Cohen's d top 15, full Pearson correlation heatmap, boxplot 5 feature theo `Diagnosis`, và BP scatter. Các bảng dài chỉ nằm trong expander hoặc file download để màn hình chính dễ đọc hơn. Smoke test trên dataset V3 cho kết quả 2.149 dòng, 35 cột, 0 missing, 0 duplicate và 186 BP violations; smoke test CSV lỗi bắt đủ các nhóm lỗi mà Table 12 yêu cầu. Batch vẫn giữ vai trò xử lý dữ liệu, không gộp logic xử lý vào EDA.

## Sources

1. pandas, [pandas.read_csv documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html?highlight=delimiter+csv), parameters for encoding, missing markers and malformed lines.
2. pandas, [DataFrame.isna documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html), definition and limits of NA detection.
3. pandas, [DataFrame.duplicated documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.duplicated.html?highlight=duplicate), duplicate semantics and subset behavior.
4. pandas, [DataFrame.corr documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html?highlight=corr), pairwise correlation and missing-value behavior.
5. Streamlit, [Chart elements documentation](https://docs.streamlit.io/develop/api-reference/charts), native bar/scatter chart options.
6. scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html), preprocessing and data-leakage safeguards.
7. Project source, `../../deliverables/reports/I1_2510_E0_-_Nhom_1-_Alzheimer_project_report_V3_revised.docx`, sections 4–5, Table 2, Table 12, and implementation section 12.
