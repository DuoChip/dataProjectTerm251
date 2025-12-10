# -*- coding: utf-8 -*-
# Full Linear Regression + SVR Pipeline with Extended Residual Analysis
"""
Mục tiêu: Xác định yếu tố kinh tế – xã hội ảnh hưởng mạnh nhất đến tuổi thọ trung bình (LifeExpBirth)
Phương pháp: Linear Regression & SVR
Quy trình:
    - Load train/valid/test
    - Feature selection theo tương quan
    - Chuẩn hóa dữ liệu
    - Train Linear & SVR, đánh giá valid
    - Chọn mô hình tốt nhất -> test
    - Phân tích feature importance, VIF
    - Phân tích residual (phân bố, top lỗi, tương quan với biến)
    - Tự động sinh nhận định thống kê residuals
"""

# =====================================================
# 1️⃣ Import & Config
# =====================================================
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import joblib
from scipy.stats import skew, kurtosis


# ---------- CONFIG ----------
DATA_DIR = '/Users/duoc.phamdinh/Documents/TERM 251/KTDL/devided_sets'
OUTPUT_DIR = '/Users/duoc.phamdinh/Documents/TERM 251/KTDL/ver2_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = 'LifeExpBirth'
TOP_K = 12
RANDOM_STATE = 42
PERM_REPEATS = 8

PALETTE = [
    "#115f9a", "#1984c5", "#22a7f0",
    "#48b5c4", "#76c68f", "#a6d75b",
    "#c9e52f", "#d0ee11", "#d0f400"
]

sns.set_palette(PALETTE)   # Set palette mặc định

# =====================================================
# Utility Functions
# =====================================================
def save_fig(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"✅ Saved figure: {path}")

def metrics(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

# =====================================================
# Load Data
# =====================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
valid_df = pd.read_csv(os.path.join(DATA_DIR, 'valid.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

for df in [train_df, valid_df, test_df]:
    if 'GNIAtlas' in df.columns:
        df.drop(columns=['GNIAtlas'], inplace=True)

print(f"Loaded datasets: Train={train_df.shape}, Valid={valid_df.shape}, Test={test_df.shape}")

# =====================================================
# Feature Selection
# =====================================================
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
for c in [TARGET, 'Year']:
    if c in numeric_cols:
        numeric_cols.remove(c)

corrs = train_df[numeric_cols + [TARGET]].corr()[TARGET].abs().drop(TARGET).sort_values(ascending=False)
top_features = corrs.head(TOP_K).index.tolist()
print(f"\nTop {TOP_K} correlated features with {TARGET}:")
print(corrs.head(TOP_K))

def prepare(df):
    df = df[top_features + [TARGET]].dropna()
    return df[top_features].values, df[TARGET].values

X_train, y_train = prepare(train_df)
X_valid, y_valid = prepare(valid_df)
X_test, y_test   = prepare(test_df)



# ---------- 12. MULTICOLLINEARITY (VIF) ----------
X_train_df = pd.DataFrame(X_train, columns=top_features)
X_vif = sm.add_constant(X_train_df)
vif_data = pd.DataFrame({
    "Feature": top_features,
    "VIF": [variance_inflation_factor(X_vif.values, i+1) for i in range(len(top_features))]
}).sort_values('VIF', ascending=False)
print("\nVIF (đa cộng tuyến):")
print(vif_data)


# =====================================================
# Scaling
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))



# =====================================================
# Linear Regression Training
# =====================================================
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_valid_lr = lr.predict(X_valid_scaled)
metrics_lr_valid = metrics(y_valid, y_pred_valid_lr)
print("\nLinear Regression (Validation):", metrics_lr_valid)

# =====================================================
# SVR Training
# =====================================================
svr = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
svr.fit(X_train_scaled, y_train)
y_pred_valid_svr = svr.predict(X_valid_scaled)
metrics_svr_valid = metrics(y_valid, y_pred_valid_svr)
print("\nSVR (Validation):", metrics_svr_valid)

# =====================================================
# THÊM PHẦN TẠO VÀ IN BẢNG METRICS
# =====================================================
metrics_df = pd.DataFrame({
    'Linear Regression (Valid)': metrics_lr_valid,
    'SVR (Valid)': metrics_svr_valid
}).T # .T để chuyển đổi cột thành hàng

print("\n\n------------------------------------------------------")
print("📊 BẢNG ĐÁNH GIÁ MÔ HÌNH (Tập Valid)")
print("------------------------------------------------------")
print(metrics_df.round(4)) # Làm tròn 4 chữ số thập phân cho bảng đẹp
print("------------------------------------------------------")



# =====================================================
# Model Selection by Valid set & Test Evaluation
# =====================================================
chosen_model = lr if metrics_lr_valid['R2'] > metrics_svr_valid['R2'] else svr
model_name = 'LinearRegression' if chosen_model == lr else 'SVR'
print(f"\n✅ Chọn mô hình: {model_name}")

y_pred_test = chosen_model.predict(X_test_scaled)
metrics_test = metrics(y_test, y_pred_test)
# print(f"\n📊 Test Performance ({model_name}): {metrics_test}")
print(f"\n📊 Test Performance ({model_name}):")
# Chuyển dictionary thành Pandas Series và in ra để mỗi giá trị xuống một dòng
print(pd.Series(metrics_test).round(4))


# =====================================================
# Feature Importance + Multicollinearity
# =====================================================
if model_name == 'LinearRegression':
    lr_coefs = pd.Series(lr.coef_, index=top_features).sort_values(key=lambda x: abs(x), ascending=False)
    plt.figure(figsize=(8,6))
    sns.barplot(x=lr_coefs.abs(), y=lr_coefs.index, color='steelblue')
    plt.title('Linear Regression - Feature Importance')
    save_fig(plt.gcf(), 'linear_top_features.png')
    plt.close()

perm = permutation_importance(chosen_model, X_test_scaled, y_test, n_repeats=PERM_REPEATS, random_state=RANDOM_STATE)
perm_series = pd.Series(perm.importances_mean, index=top_features).sort_values(ascending=False)
perm_series.to_csv(os.path.join(OUTPUT_DIR, 'perm_importance.csv'))

X_vif = sm.add_constant(pd.DataFrame(X_train, columns=top_features))
vif_data = pd.DataFrame({
    "Feature": top_features,
    "VIF": [variance_inflation_factor(X_vif.values, i+1) for i in range(len(top_features))]
}).sort_values('VIF', ascending=False)
vif_data.to_csv(os.path.join(OUTPUT_DIR, 'vif_summary.csv'))

# =====================================================
# Residual Analysis
# =====================================================
y_pred_lr = lr.predict(X_test_scaled)
y_pred_svr = svr.predict(X_test_scaled)

residuals_df = test_df[['Country', 'Year']].copy()

residuals_df = pd.DataFrame({
    'y_true': y_test,
    'y_pred_lr': y_pred_lr,
    'y_pred_svr': y_pred_svr
})
residuals_df['error_lr'] = residuals_df['y_true'] - residuals_df['y_pred_lr']
residuals_df['error_svr'] = residuals_df['y_true'] - residuals_df['y_pred_svr']
residuals_df['abs_error_lr'] = residuals_df['error_lr'].abs()
residuals_df['abs_error_svr'] = residuals_df['error_svr'].abs()

# =====================================================
# Residual Analysis (with Country & Year)
# =====================================================
y_pred_lr = lr.predict(X_test_scaled)
y_pred_svr = svr.predict(X_test_scaled)

# Giữ Country và Year từ test_df
residuals_df = test_df[['Country', 'Year']].copy()

# Thêm y_true và y_pred
residuals_df['y_true'] = y_test
residuals_df['y_pred_lr'] = y_pred_lr
residuals_df['y_pred_svr'] = y_pred_svr

# Residuals
residuals_df['error_lr'] = residuals_df['y_true'] - residuals_df['y_pred_lr']
residuals_df['error_svr'] = residuals_df['y_true'] - residuals_df['y_pred_svr']
residuals_df['abs_error_lr'] = residuals_df['error_lr'].abs()
residuals_df['abs_error_svr'] = residuals_df['error_svr'].abs()

# Top lỗi lớn nhất
top_err_lr = residuals_df.sort_values('abs_error_lr', ascending=False).head(10)
top_err_svr = residuals_df.sort_values('abs_error_svr', ascending=False).head(10)

print("\n🔍 Top 10 mẫu Linear Regression sai nhiều nhất:")
print(top_err_lr[['Country', 'Year', 'y_true', 'y_pred_lr', 'error_lr', 'abs_error_lr']])

print("\n🔍 Top 10 mẫu SVR sai nhiều nhất:")
print(top_err_svr[['Country', 'Year', 'y_true', 'y_pred_svr', 'error_svr', 'abs_error_svr']])

# Distribution
plt.figure(figsize=(10,6))
sns.kdeplot(residuals_df['error_lr'], fill=True, label='Linear Regression')
sns.kdeplot(residuals_df['error_svr'], fill=True, label='SVR')
plt.title('Residual Distribution Comparison')
plt.xlabel('Error (y_true - y_pred)')
plt.legend()
save_fig(plt.gcf(), 'residual_distribution_lr_vs_svr.png')
plt.close()

# Scatter
fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.scatterplot(x=residuals_df['y_true'], y=residuals_df['y_pred_lr'], ax=axes[0])
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_title('Linear Regression: y_pred vs y_true')
sns.scatterplot(x=residuals_df['y_true'], y=residuals_df['y_pred_svr'], ax=axes[1], color='orange')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_title('SVR: y_pred vs y_true')
save_fig(plt.gcf(), 'y_true_vs_pred_lr_svr.png')
plt.close()

# Top lỗi lớn nhất
top_err_lr = residuals_df.sort_values('abs_error_lr', ascending=False).head(10)
top_err_svr = residuals_df.sort_values('abs_error_svr', ascending=False).head(10)
print("\n🔍 Top 10 mẫu Linear Regression sai nhiều nhất:")
print(top_err_lr)
print("\n🔍 Top 10 mẫu SVR sai nhiều nhất:")
print(top_err_svr)

# Correlation giữa feature & residual
resid_corrs = pd.DataFrame({
    'corr_error_lr': [np.corrcoef(X_test_scaled[:, i], residuals_df['abs_error_lr'])[0,1] for i in range(len(top_features))],
    'corr_error_svr': [np.corrcoef(X_test_scaled[:, i], residuals_df['abs_error_svr'])[0,1] for i in range(len(top_features))]
}, index=top_features)
resid_corrs.to_csv(os.path.join(OUTPUT_DIR, 'residuals_feature_correlation.csv'))

# Phân tích thống kê residuals
stats_lr = {
    'mean': np.mean(residuals_df['error_lr']),
    'std': np.std(residuals_df['error_lr']),
    'skewness': skew(residuals_df['error_lr']),
    'kurtosis': kurtosis(residuals_df['error_lr'])
}
stats_svr = {
    'mean': np.mean(residuals_df['error_svr']),
    'std': np.std(residuals_df['error_svr']),
    'skewness': skew(residuals_df['error_svr']),
    'kurtosis': kurtosis(residuals_df['error_svr'])
}

print("\n📊 Residual Stats (LR):", stats_lr)
print("📊 Residual Stats (SVR):", stats_svr)

# Nhận định tự động
def interpret_residual(stats, model):
    desc = f"\n📈 {model} Residual Analysis:\n"
    if abs(stats['mean']) < 1:
        desc += "• Trung bình gần 0 → mô hình không bias đáng kể.\n"
    else:
        desc += "• Có bias → mô hình dự đoán lệch.\n"
    if stats['std'] > 5:
        desc += "• Phương sai cao → sai số không ổn định.\n"
    if stats['skewness'] > 0.5:
        desc += "• Phân phối lệch phải → mô hình thường **underpredict** (dự đoán thấp hơn thực tế).\n"
    elif stats['skewness'] < -0.5:
        desc += "• Phân phối lệch trái → mô hình thường **overpredict**.\n"
    if stats['kurtosis'] > 3:
        desc += "• Có đuôi dài → tồn tại outlier hoặc phi tuyến mạnh.\n"
    return desc

print(interpret_residual(stats_lr, "Linear Regression"))
print(interpret_residual(stats_svr, "SVR"))

# Save residuals
residuals_df.to_csv(os.path.join(OUTPUT_DIR, 'residuals_comparison_lr_svr.csv'), index=False)
print("\n✅ All residual analyses and visualizations saved successfully.")