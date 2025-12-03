# ĐỒ ÁN KTDL_251: Dự đoán Tuổi Thọ Khi Sinh Sử dụng Machine Learning

## Mô tả Dự án

Dự án này triển khai một pipeline machine learning hoàn chỉnh để dự đoán tuổi thọ khi sinh (Life Expectancy at Birth) dựa trên dữ liệu phát triển thế giới. Sử dụng hai mô hình: Linear Regression và Support Vector Regression (SVR), với quy trình bao gồm:

- Phân tích dữ liệu khám phá (EDA)
- Lựa chọn đặc trưng dựa trên tương quan Pearson
- Xử lý đa cộng tuyến bằng Variance Inflation Factor (VIF)
- Chuẩn hóa dữ liệu
- Huấn luyện và đánh giá mô hình
- Phân tích độ quan trọng của đặc trưng

## Cấu trúc Dự án

```
.
├── code_ver2.py              # Script chính chứa pipeline ML
├── devide.py                 # Script chia dữ liệu (nếu có)
├── world_development_data_imputed.csv  # Dữ liệu gốc
├── data_final/               # Dữ liệu đã chia
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
└── output/                   # Kết quả đầu ra
    ├── linearregression_final_model.joblib
    ├── svr_final_model.joblib
    ├── scaler.joblib
    ├── model_performance_summary.csv
    ├── perm_importance.csv
    ├── pred_vs_actual.csv
    └── vif_summary.csv
```

## Yêu cầu

- Python 3.7+
- Các thư viện: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, joblib

Cài đặt dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels joblib
```

## Cách Chạy

1. Đảm bảo dữ liệu trong thư mục `data_final/` (train.csv, valid.csv, test.csv)
2. Chạy script chính:

```bash
python code_ver2.py
```

3. Kết quả sẽ được lưu trong thư mục `output/`

## Kết quả

- Mô hình được chọn dựa trên R² trên tập validation
- Đánh giá cuối cùng trên tập test
- Phân tích độ quan trọng của đặc trưng bằng coefficients và permutation importance
- Biểu đồ so sánh độ quan trọng

## Tác giả

DatVo975

## Lưu ý

- Dữ liệu gốc từ world_development_data_imputed.csv
- Target: LifeExpBirth (Tuổi thọ khi sinh)
- Top 12 đặc trưng được chọn dựa trên tương quan