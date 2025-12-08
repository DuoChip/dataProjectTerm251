# -*- coding: utf-8 -*-
# Liner and svr

"""
Full pipeline sử dụng 3 tập dữ liệu (train / valid / test)
----------------------------------------------------------
- Load train, valid, test CSV
- EDA cơ bản (train)
- Feature selection (dựa trên train)
- Scale bằng train
- Huấn luyện trên train
- Đánh giá trên valid (tinh chỉnh mô hình)
- Đánh giá cuối cùng trên test
- Phân tích hệ số, permutation importance, VIF
"""
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import joblib

# ---------- CONFIG ----------
DATA_DIR = 'data_final'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = 'LifeExpBirth'
TOP_K = 12
RANDOM_STATE = 42
PERM_REPEATS = 8

# ---------- UTILS ----------
def metrics(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

# ---------- 1. LOAD DATA ----------
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
valid_df = pd.read_csv(os.path.join(DATA_DIR, 'valid.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# ---------- 2. EDA ----------
print("\n--- EDA (Train) ---")
print(train_df.describe().T.head())
print("Missing values:\n", train_df.isna().sum().sort_values(ascending=False).head(10))

# ---------- 3. FEATURE SELECTION ----------
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
for c in [TARGET, 'Year']:
    if c in numeric_cols:
        numeric_cols.remove(c)

corrs = train_df[numeric_cols + [TARGET]].corr()[TARGET].abs().drop(TARGET).sort_values(ascending=False)
top_features = corrs.head(TOP_K).index.tolist()
print(f"\nTop {TOP_K} features by abs Pearson correlation with {TARGET}:")
print(corrs.head(TOP_K))

# ---------- 4. BUILD MODEL DATA ----------
def prepare(df):
    df = df[top_features + [TARGET]].dropna()
    return df[top_features].values, df[TARGET].values

X_train, y_train = prepare(train_df)
X_valid, y_valid = prepare(valid_df)
X_test,  y_test  = prepare(test_df)

# ---------- 5. MULTICOLLINEARITY (VIF) ----------
def handle_multicollinearity(df, target_col, threshold=10, verbose=True):
    """
    Loại bỏ dần biến có đa cộng tuyến cao (VIF > threshold).
    Giữ lại biến có tương quan mạnh hơn với target.
    In ra VIF của từng vòng nếu verbose=True.
    """
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    corr_with_target = df[numeric_df.columns].corrwith(df[target_col]).abs()

    dropped_features = []
    round_idx = 1

    while True:
        # Tính VIF cho các feature hiện tại
        X_vif = sm.add_constant(numeric_df)
        vif_series = pd.Series(
            [variance_inflation_factor(X_vif.values, i) for i in range(1, X_vif.shape[1])],
            index=numeric_df.columns
        )

        if verbose:
            print(f"\n Vòng {round_idx}: VIF hiện tại")
            print(vif_series.sort_values(ascending=False).round(3))

        max_vif = vif_series.max()
        if max_vif <= threshold:
            if verbose:
                print(f"\n✅ Tất cả VIF ≤ {threshold}, dừng loại bỏ.")
            break

        # Xác định biến có VIF cao nhất
        high_vif_feature = vif_series.idxmax()
        corr_features = numeric_df.corr()[high_vif_feature].abs().sort_values(ascending=False)
        corr_features = corr_features.drop(high_vif_feature)

        # Xác định biến nào nên bỏ (dựa trên tương quan với target)
        if corr_features.empty:
            to_drop = high_vif_feature
        else:
            most_corr_feature = corr_features.index[0]
            if corr_with_target[high_vif_feature] < corr_with_target[most_corr_feature]:
                to_drop = high_vif_feature
            else:
                to_drop = most_corr_feature

        if verbose:
            print(f"\n❌ Loại bỏ biến: {to_drop} "
                  f"(VIF cao nhất = {max_vif:.2f}, tương quan thấp hơn với target)")

        numeric_df = numeric_df.drop(columns=[to_drop])
        dropped_features.append(to_drop)
        round_idx += 1

    # Sau khi loại xong, tính lại VIF cuối cùng
    vif_data = pd.DataFrame({
        "Feature": numeric_df.columns,
        "VIF": [variance_inflation_factor(sm.add_constant(numeric_df).values, i)
                for i in range(1, numeric_df.shape[1] + 1)]
    }).sort_values("VIF", ascending=False)

    return dropped_features, vif_data

dropped_features, vif_data = handle_multicollinearity(train_df[top_features + [TARGET]], TARGET, threshold=10)

print("\n❌ Các feature bị loại do đa cộng tuyến:")
print(dropped_features)

# Loại các cột bị loại khỏi toàn bộ tập dữ liệu
for df in [train_df, valid_df, test_df]:
    df.drop(columns=dropped_features, inplace=True, errors='ignore')

print("\n✅ Loaded datasets:")
print(f"Train: {train_df.shape}, Valid: {valid_df.shape}, Test: {test_df.shape}")

# ---------- 6. SCALING ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))

# ---------- 7. TRAIN LINEAR REGRESSION ----------
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_valid_lr = lr.predict(X_valid_scaled)
metrics_lr_valid = metrics(y_valid, y_pred_valid_lr)

print("\nLinear Regression (validation metrics):", metrics_lr_valid)

# ---------- 8. TRAIN SVR ----------
svr = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
svr.fit(X_train_scaled, y_train)
y_pred_valid_svr = svr.predict(X_valid_scaled)
metrics_svr_valid = metrics(y_valid, y_pred_valid_svr)

print("\nSVR (validation metrics):", metrics_svr_valid)

# ---------- 9. CHỌN MÔ HÌNH TỐT HƠN ----------
chosen_model = lr if metrics_lr_valid['R2'] > metrics_svr_valid['R2'] else svr
model_name = 'LinearRegression' if chosen_model == lr else 'SVR'
print(f"\n✅ Chọn mô hình: {model_name} (R2 validation = {max(metrics_lr_valid['R2'], metrics_svr_valid['R2']):.4f})")

# ---------- 10. ĐÁNH GIÁ CUỐI CÙNG TRÊN TEST ----------
y_pred_test = chosen_model.predict(X_test_scaled)
metrics_test = metrics(y_test, y_pred_test)
print(f"\n📊 Final Evaluation on Test Set ({model_name}):")
for k, v in metrics_test.items():
    print(f"{k}: {v:.4f}")

# ---------- 11. FEATURE IMPORTANCE ----------
if model_name == 'LinearRegression':
    lr_coefs = pd.Series(lr.coef_, index=top_features).sort_values(key=lambda x: abs(x), ascending=False)
    print("\nLinear Coefficients:")
    print(lr_coefs)

    plt.figure(figsize=(8,6))
    sns.barplot(x=lr_coefs.abs(), y=lr_coefs.index, color='skyblue')
    plt.title("Top Features ảnh hưởng mạnh nhất (Linear Regression)")
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'linear_top_features.png'), dpi=150)
    plt.close()

# ---------- 12. PERMUTATION IMPORTANCE ----------
perm = permutation_importance(chosen_model, X_test_scaled, y_test, n_repeats=PERM_REPEATS, random_state=RANDOM_STATE)
perm_series = pd.Series(perm.importances_mean, index=top_features).sort_values(ascending=False)
print("\nTop Permutation Importance (test):")
print(perm_series.head(10))
if model_name == 'LinearRegression':
    coef_series = pd.Series(np.abs(lr.coef_), index=top_features)
    
    # Gom hai loại importance vào cùng DataFrame để so sánh
    compare_df = pd.DataFrame({
        'Linear Coef (abs)': coef_series,
        'Permutation Importance': perm_series
    }).fillna(0)

    # Chuẩn hóa (optional): dễ so sánh nếu chênh lệch thang đo
    compare_df = compare_df / compare_df.max()

    plt.figure(figsize=(10, 6))
    compare_df.sort_values('Permutation Importance', ascending=False).head(10).plot(kind='bar')
    plt.title('So sánh độ quan trọng: Linear Coefficient vs Permutation Importance')
    plt.ylabel('Tầm quan trọng (chuẩn hóa)')
    plt.xlabel('Biến độc lập')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_comparison.png'), dpi=150)
    plt.close()
    print(f"📊 Đã lưu biểu đồ so sánh tại: {os.path.join(OUTPUT_DIR, 'feature_importance_comparison.png')}")

# ---------- 13. LƯU KẾT QUẢ ----------
y_pred_test_lr = lr.predict(X_test_scaled)
metrics_lr_test = metrics(y_test, y_pred_test_lr)
perf_summary = pd.DataFrame([
    {'Model': 'LinearRegression', **metrics_lr_valid, 'Phase': 'Validation'},
    {'Model': 'SVR', **metrics_svr_valid, 'Phase': 'Validation'},
    {'Model': 'LinearRegression', **metrics_lr_test, 'Phase': 'Test'},
    {'Model': model_name, **metrics_test, 'Phase': 'Test'}  # Mô hình được chọn (có thể trùng Linear hoặc SVR)
])
perf_summary.to_csv(os.path.join(OUTPUT_DIR, 'model_performance_summary.csv'), index=False)
vif_data.to_csv(os.path.join(OUTPUT_DIR, 'vif_summary.csv'), index=False)
perm_series.to_csv(os.path.join(OUTPUT_DIR, 'perm_importance.csv'), header=['perm_importance_mean'])
joblib.dump(chosen_model, os.path.join(OUTPUT_DIR, f'{model_name.lower()}_final_model.joblib'))
vif_data.to_csv(os.path.join(OUTPUT_DIR, 'vif_summary.csv'), index=False)
perm_series.to_csv(os.path.join(OUTPUT_DIR, 'perm_importance.csv'), header=['perm_importance_mean'])
joblib.dump(chosen_model, os.path.join(OUTPUT_DIR, f'{model_name.lower()}_final_model.joblib'))
pred_vs_actual_df = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred_test
})
pred_vs_actual_path = os.path.join(OUTPUT_DIR, 'pred_vs_actual.csv')
pred_vs_actual_df.to_csv(pred_vs_actual_path, index=False)

print("\n✅ Saved all outputs to:", OUTPUT_DIR)
print("\nPerformance Summary:")
print(perf_summary)