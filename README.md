# 📊 Life Expectancy Prediction Project

## 🎯 Mục tiêu

Xác định các yếu tố kinh tế - xã hội ảnh hưởng mạnh nhất đến tuổi thọ trung bình (Life Expectancy at Birth) bằng Machine Learning.

---

## 📁 Cấu trúc thư mục

```
Project_KTDL_251/
│
├── devided_sets/           # Dữ liệu đã chia
│   ├── train.csv           # Tập huấn luyện (60%)
│   ├── valid.csv           # Tập validation (20%)
│   └── test.csv            # Tập kiểm tra (20%)
│
├── output/                 # Kết quả đầu ra
│   ├── *.csv               # Các file kết quả
│   ├── *.joblib            # Models đã train
│   └── *.png               # Biểu đồ
│
├── code_ver2.py            # Code chính
├── devide.py               # Code chia dữ liệu
└── README.md               # File này
```

---

## 🔧 Yêu cầu hệ thống

### Python 3.8+

### Thư viện cần thiết:

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn scipy joblib
```

---

## 📝 Quy trình thực hiện

### **Bước 1: Chia dữ liệu**

```bash
python devide.py
```

**Chức năng:**

- Chia dữ liệu thành 3 tập: train (60%), valid (20%), test (20%)
- Output: `devided_sets/train.csv`, `valid.csv`, `test.csv`

---

### **Bước 2: Chạy pipeline ML**

```bash
python code_ver2.py
```

**Pipeline tự động thực hiện:**

#### **2.1. Load Data**

- Đọc 3 file CSV từ `devided_sets/`
- Loại bỏ cột `GNIAtlas` (nếu có)

#### **2.2. Feature Selection (Correlation-based)**

- Chọn **12 biến** có correlation cao nhất với `LifeExpBirth`
- Phương pháp: Pearson Correlation
- Loại bỏ: `Year` (không dùng làm feature)

#### **2.3. VIF Analysis (Multicollinearity Check)**

- Tính VIF (Variance Inflation Factor) cho 12 biến
- **Tự động loại bỏ** biến có VIF > 10
- Tính lại VIF sau khi loại bỏ
- Output: `vif_summary.csv`

**Ý nghĩa:**

- VIF < 5: Không có vấn đề
- VIF 5-10: Đa cộng tuyến vừa phải
- VIF > 10: Nghiêm trọng (cần loại bỏ)

#### **2.4. Data Scaling**

- Chuẩn hóa dữ liệu bằng `StandardScaler`
- Fit trên train, transform trên valid/test
- Lưu scaler: `scaler.joblib`

#### **2.5. Model Training**

**Train 2 models:**

1. **Linear Regression**
   - Model tuyến tính đơn giản
   - Dễ giải thích hệ số

2. **SVR (Support Vector Regression)**
   - Kernel: RBF
   - Xử lý tốt quan hệ phi tuyến

**Đánh giá trên validation set:**

- Chọn model có R² cao hơn

#### **2.6. Model Selection & Testing**

- Chọn model tốt nhất (LR hoặc SVR)
- Đánh giá trên test set
- Metrics: R², RMSE, MAE

#### **2.7. Feature Importance Analysis**

**2 phương pháp:**

1. **Linear Regression Coefficients** (nếu LR được chọn)
   - Hệ số càng lớn → biến càng quan trọng
   - Output: `linear_top_features.png`

2. **Permutation Importance**
   - Đo tác động thực tế của biến
   - Áp dụng cho cả LR và SVR
   - Output: `perm_importance.csv`

#### **2.8. Residual Analysis**

**So sánh sai số giữa LR và SVR:**

1. **Distribution Plot**
   - Phân phối sai số (KDE plot)
   - Kiểm tra bias, skewness
   - Output: `residual_distribution_lr_vs_svr.png`

2. **Scatter Plot**
   - y_true vs y_pred
   - Đánh giá độ chính xác
   - Output: `y_true_vs_pred_lr_svr.png`

3. **Feature-Error Correlation**
   - Biến nào khiến model sai nhiều?
   - Output: `residuals_feature_correlation.csv`

4. **Statistical Analysis**
   - Mean, Std, Skewness, Kurtosis của residuals
   - Output: `residuals_comparison_lr_svr.csv`

---

## 📊 Kết quả đầu ra (13 files)

### **1. Models & Scaler (3 files)**

| File                                  | Mô tả                   |
| ------------------------------------- | ----------------------- |
| `scaler.joblib`                       | StandardScaler đã fit   |
| `linearregression_final_model.joblib` | Linear Regression model |
| `svr_final_model.joblib`              | SVR model               |

### **2. Performance Metrics (2 files)**

| File                            | Mô tả                            |
| ------------------------------- | -------------------------------- |
| `model_performance_summary.csv` | So sánh LR vs SVR (valid & test) |
| `pred_vs_actual.csv`            | Predictions của model được chọn  |

### **3. Feature Analysis (2 files)**

| File                  | Mô tả                                           |
| --------------------- | ----------------------------------------------- |
| `vif_summary.csv`     | VIF của các biến (kiểm tra multicollinearity)   |
| `perm_importance.csv` | Permutation importance (tầm quan trọng thực tế) |

### **4. Residual Analysis (3 files)**

| File                                | Mô tả                                 |
| ----------------------------------- | ------------------------------------- |
| `residuals_comparison_lr_svr.csv`   | Chi tiết residuals (có Country, Year) |
| `residuals_feature_correlation.csv` | Correlation giữa features và errors   |
| (stats in memory)                   | Mean, Std, Skewness, Kurtosis         |

### **5. Visualizations (3 files)**

| File                                  | Mô tả                              |
| ------------------------------------- | ---------------------------------- |
| `linear_top_features.png`             | LR coefficients (nếu LR được chọn) |
| `residual_distribution_lr_vs_svr.png` | Phân phối sai số                   |
| `y_true_vs_pred_lr_svr.png`           | Scatter plots so sánh 2 models     |

---

## 📈 Giải thích các chỉ số

### **R² (R-squared)**

- Đo mức độ phù hợp của model
- 0.0 - 1.0 (càng cao càng tốt)
- **> 0.85**: Rất tốt

### **RMSE (Root Mean Squared Error)**

- Sai số trung bình (đơn vị: năm)
- Càng thấp càng tốt
- **< 3.0**: Chấp nhận được

### **MAE (Mean Absolute Error)**

- Sai số tuyệt đối trung bình
- Dễ hiểu hơn RMSE
- **< 2.5**: Tốt

### **VIF (Variance Inflation Factor)**

- Đo đa cộng tuyến
- **< 5**: OK
- **5-10**: Cân nhắc
- **> 10**: Loại bỏ

---

## 🔍 Cách đọc kết quả

### **1. Xem model nào được chọn:**

```
Console output:
✅ Pipeline completed. Model selected: SVR
📊 Test R²: 0.8849, RMSE: 2.9035, MAE: 2.0099
```

### **2. Xem biến nào quan trọng nhất:**

Mở `perm_importance.csv`:

```
Feature,Importance
MortRateU5,0.6582        ← Quan trọng nhất!
FertRate,0.1621
AdolFertRate,0.1217
...
```

### **3. Kiểm tra multicollinearity:**

Mở `vif_summary.csv`:

```
Feature,VIF
FertRate,6.47            ← Chấp nhận được
GDP,2.04                 ← Rất tốt
PopDens,1.23             ← Tuyệt vời
```

### **4. Xem predictions:**

Mở `pred_vs_actual.csv`:

```
y_true,y_pred
75.2,74.8                ← Sai 0.4 năm
68.5,69.1                ← Sai 0.6 năm
```

---

## 📚 Kiến thức áp dụng

### **1. Feature Engineering**

- ✅ Correlation-based selection
- ✅ VIF analysis (multicollinearity)
- ✅ Automatic feature removal

### **2. Model Selection**

- ✅ Linear Regression
- ✅ Support Vector Regression (RBF kernel)
- ✅ Validation-based selection

### **3. Model Evaluation**

- ✅ Multiple metrics (R², RMSE, MAE)
- ✅ Cross-validation strategy
- ✅ Residual analysis

### **4. Feature Importance**

- ✅ Coefficients (LR)
- ✅ Permutation importance
- ✅ Comparison between methods

---

## ⚠️ Lưu ý

### **1. Nếu có lỗi "File not found"**

- Kiểm tra thư mục `devided_sets/` đã có 3 file CSV chưa
- Chạy lại `python devide.py`

### **2. Nếu VIF > 10**

- Code tự động loại bỏ biến
- Xem console để biết biến nào bị loại
- File `vif_summary_final.csv` sẽ được tạo

### **3. Nếu muốn thay đổi số biến**

- Sửa `TOP_K = 12` trong code
- Giá trị đề xuất: 10-15

### **4. Nếu muốn thay đổi VIF threshold**

- Sửa `VIF_THRESHOLD = 10` trong code
- Giá trị đề xuất: 5-10

---

## 🎓 Kết luận

Project này trình bày quy trình **hoàn chỉnh** để:

1. ✅ Chọn features hiệu quả (correlation + VIF)
2. ✅ Train & compare models (LR vs SVR)
3. ✅ Đánh giá toàn diện (metrics + residuals)
4. ✅ Giải thích kết quả (importance + visualization)

**Kết quả:** Model dự đoán tuổi thọ với độ chính xác cao (R² ~0.88), xác định được các yếu tố kinh tế-xã hội quan trọng nhất.

---

## 👨‍💻 Tác giả

- **Sinh viên:**
- **MSSV:**
- **Môn học:**
- **Học kỳ:**

---

## 📞 Hỗ trợ

Nếu có vấn đề khi chạy code:

1. Kiểm tra phiên bản Python (>= 3.8)
2. Cài đặt đầy đủ thư viện
3. Xem lại console output để biết lỗi cụ thể
4. Kiểm tra dữ liệu đầu vào

---

**🎉 Good luck with your project!**
