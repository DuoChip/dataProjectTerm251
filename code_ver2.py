# -*- coding: utf-8 -*-
# Full Linear Regression + SVR Pipeline with Extended Residual Analysis
"""
Mục tiêu: Xác định yếu tố kinh tế – xã hội ảnh hưởng mạnh nhất đến tuổi thọ trung bình (LifeExpBirth)
Phương pháp: Linear Regression & SVR
Quy trình:
    - Load train/valid/test
    - Feature selection theo tương quan
    - Loại biến đa cộng tuyến cao (VIF > 10)
    - Chuẩn hóa dữ liệu (StandardScaler)
    - Train Linear & SVR, đánh giá valid
    - Chọn mô hình tốt nhất -> test
    - Phân tích feature importance
    - Phân tích residual (phân bố, top lỗi, tương quan với biến)
"""

# Import & Config
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


# ================= CONFIG =====================
DATA_DIR = 'devided_sets'
OUTPUT_DIR = 'output'
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

sns.set_palette(PALETTE)

# Utility Functions
def save_fig(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches='tight', dpi=150)

def metrics(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

# ================= Load Data =====================
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
valid_df = pd.read_csv(os.path.join(DATA_DIR, 'valid.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

for df in [train_df, valid_df, test_df]:
    if 'GNIAtlas' in df.columns:
        df.drop(columns=['GNIAtlas'], inplace=True)

# ================= Feature Selection =====================
# Correlation-based feature selection
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
for c in [TARGET, 'Year']:
    if c in numeric_cols:
        numeric_cols.remove(c)

corrs = train_df[numeric_cols + [TARGET]].corr()[TARGET].abs().drop(TARGET).sort_values(ascending=False)
top_features = corrs.head(TOP_K).index.tolist() # Top K features

def prepare(df):
    df = df[top_features + [TARGET]].dropna()
    return df[top_features].values, df[TARGET].values

# VIF Analysis - Check multicollinearity before training
X_train_temp, _ = prepare(train_df)     # Prepare data for VIF calculation
X_vif = sm.add_constant(pd.DataFrame(X_train_temp, columns=top_features))
vif_data = pd.DataFrame({
    "Feature": top_features,
    "VIF": [variance_inflation_factor(X_vif.values, i+1) for i in range(len(top_features))]
}).sort_values('VIF', ascending=False)
vif_data.to_csv(os.path.join(OUTPUT_DIR, 'vif_summary.csv'), index=False)

# Remove features with VIF > 10
VIF_THRESHOLD = 10
high_vif_features = vif_data[vif_data['VIF'] > VIF_THRESHOLD]['Feature'].tolist()

if high_vif_features:
    print(f"\n⚠️ Removing {len(high_vif_features)} features with VIF > {VIF_THRESHOLD}:")
    for feat in high_vif_features:
        vif_val = vif_data[vif_data['Feature'] == feat]['VIF'].values[0]
        print(f"   - {feat} (VIF={vif_val:.2f})")
    
    # Remove high VIF features
    top_features = [f for f in top_features if f not in high_vif_features]
    
    # Recalculate data and VIF after removal
    X_train, y_train = prepare(train_df)
    X_valid, y_valid = prepare(valid_df)
    X_test, y_test = prepare(test_df)
    
    X_vif = sm.add_constant(pd.DataFrame(X_train, columns=top_features))
    vif_data = pd.DataFrame({
        "Feature": top_features,
        "VIF": [variance_inflation_factor(X_vif.values, i+1) for i in range(len(top_features))]
    }).sort_values('VIF', ascending=False)
    vif_data.to_csv(os.path.join(OUTPUT_DIR, 'vif_summary_final.csv'), index=False)
    
    print(f"✅ Final features: {len(top_features)} (max VIF={vif_data['VIF'].max():.2f})")
else:
    print(f"✅ All features have VIF < {VIF_THRESHOLD} (max VIF={vif_data['VIF'].max():.2f})")
    # Prepare final data with all features
    X_train, y_train = prepare(train_df)
    X_valid, y_valid = prepare(valid_df)
    X_test, y_test = prepare(test_df)

# ================= Scaling =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))

# ==================== Model Training & Evaluation ======================
# Linear Regression Training
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_valid_lr = lr.predict(X_valid_scaled)
metrics_lr_valid = metrics(y_valid, y_pred_valid_lr)

# SVR Training
svr = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
svr.fit(X_train_scaled, y_train)
y_pred_valid_svr = svr.predict(X_valid_scaled)
metrics_svr_valid = metrics(y_valid, y_pred_valid_svr)

# Model Selection & Test Evaluation
chosen_model = lr if metrics_lr_valid['R2'] > metrics_svr_valid['R2'] else svr
model_name = 'LinearRegression' if chosen_model == lr else 'SVR'
y_pred_test = chosen_model.predict(X_test_scaled)
metrics_test = metrics(y_test, y_pred_test)

# ==================== Outputs ======================
# Predictions vs Actual
pred_vs_actual = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred_test,
})
pred_vs_actual.to_csv(os.path.join(OUTPUT_DIR, 'pred_vs_actual.csv'), index=False)
joblib.dump(lr, os.path.join(OUTPUT_DIR, 'linearregression_final_model.joblib'))
joblib.dump(svr, os.path.join(OUTPUT_DIR, 'svr_final_model.joblib'))

# Performance Summary
y_pred_test_lr = lr.predict(X_test_scaled)
y_pred_test_svr = svr.predict(X_test_scaled)
metrics_lr_test = metrics(y_test, y_pred_test_lr)
metrics_svr_test = metrics(y_test, y_pred_test_svr)
performance_summary = pd.DataFrame([
    {'Model': 'LinearRegression', **metrics_lr_valid, 'Phase': 'Validation'},
    {'Model': 'SVR', **metrics_svr_valid, 'Phase': 'Validation'},
    {'Model': 'LinearRegression', **metrics_lr_test, 'Phase': 'Test'},
    {'Model': 'SVR', **metrics_svr_test, 'Phase': 'Test'}
])
performance_summary.to_csv(os.path.join(OUTPUT_DIR, 'model_performance_summary.csv'), index=False)

# Feature Importance
if model_name == 'LinearRegression':
    lr_coefs = pd.Series(lr.coef_, index=top_features).sort_values(key=lambda x: abs(x), ascending=False)
    plt.figure(figsize=(8,6))
    sns.barplot(x=lr_coefs.abs(), y=lr_coefs.index, color='steelblue')
    plt.title('Linear Regression - Feature Importance')
    save_fig(plt.gcf(), 'linear_top_features.png')
    plt.close()

perm = permutation_importance(chosen_model, X_test_scaled, y_test, n_repeats=PERM_REPEATS, random_state=RANDOM_STATE)
perm_series = pd.Series(perm.importances_mean, index=top_features).sort_values(ascending=False)
perm_df = pd.DataFrame({'Feature': perm_series.index, 'Importance': perm_series.values})
perm_df.to_csv(os.path.join(OUTPUT_DIR, 'perm_importance.csv'), index=False)

# =========== Residual Analysis Comparison ===========
y_pred_lr = lr.predict(X_test_scaled)
y_pred_svr = svr.predict(X_test_scaled)

residuals_df = test_df[['Country', 'Year']].copy()
residuals_df['y_true'] = y_test
residuals_df['y_pred_lr'] = y_pred_lr
residuals_df['y_pred_svr'] = y_pred_svr
residuals_df['error_lr'] = residuals_df['y_true'] - residuals_df['y_pred_lr']
residuals_df['error_svr'] = residuals_df['y_true'] - residuals_df['y_pred_svr']
residuals_df['abs_error_lr'] = residuals_df['error_lr'].abs()
residuals_df['abs_error_svr'] = residuals_df['error_svr'].abs()

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

# Feature correlation with residuals
resid_corrs = pd.DataFrame({
    'corr_error_lr': [np.corrcoef(X_test_scaled[:, i], residuals_df['abs_error_lr'])[0,1] for i in range(len(top_features))],
    'corr_error_svr': [np.corrcoef(X_test_scaled[:, i], residuals_df['abs_error_svr'])[0,1] for i in range(len(top_features))]
}, index=top_features)
resid_corrs.to_csv(os.path.join(OUTPUT_DIR, 'residuals_feature_correlation.csv'))

# Residual statistics
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

residuals_df.to_csv(os.path.join(OUTPUT_DIR, 'residuals_comparison_lr_svr.csv'), index=False)

print(f"Pipeline completed. Model selected: {model_name}")
print(f"Test R²: {metrics_test['R2']:.4f}, RMSE: {metrics_test['RMSE']:.4f}, MAE: {metrics_test['MAE']:.4f}")
print(f"All outputs saved to '{OUTPUT_DIR}/' directory")