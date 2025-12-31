"""Validate RailRadar imputations.

Produces:
 - reports/validation_summary.md
 - reports/rr_distribution_comparison.csv
 - reports/high_uncertainty_imputations.csv
 - reports/plots_rr_mean.png
 - reports/plots_rr_std.png

Usage: python scripts/validate_imputations.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)

LABELS = DATA / 'railradar_labels.csv'
FEATURES_LABELED = DATA / 'railradar_features.csv'
FEATURES_FULL = DATA / 'features_full.csv'
MODEL_MEAN = ROOT / 'models' / 'rr_mean_model.joblib'
MODEL_STD = ROOT / 'models' / 'rr_std_model.joblib'
MODEL_MEAN_TUNED = ROOT / 'models' / 'rr_mean_model_tuned.joblib'
MODEL_STD_TUNED = ROOT / 'models' / 'rr_std_model_tuned.joblib'


def _choose_model(tuned_path, default_path):
    if tuned_path.exists():
        print(f'Using tuned model: {tuned_path.name}')
        return tuned_path
    print(f'Using default model: {default_path.name}')
    return default_path

# Load data
labels = pd.read_csv(LABELS)
features_labeled = pd.read_csv(FEATURES_LABELED)
features_full = pd.read_csv(FEATURES_FULL)

# Helper: select numeric feature columns used for modeling
def select_feature_cols(df):
    skip = {'train_id', 'rr_mean', 'rr_std'}
    cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    return cols

feat_cols_labeled = select_feature_cols(features_labeled)
feat_cols_full = select_feature_cols(features_full)
# Use intersection to be safe
feat_cols = [c for c in feat_cols_full if c in feat_cols_labeled]
if not feat_cols:
    # fallback to labeled numeric columns
    feat_cols = feat_cols_labeled

X_lab = features_labeled.set_index('train_id')[feat_cols]
X_all = features_full.set_index('train_id')[feat_cols]

# Load models (prefer tuned if available)
mean_model = joblib.load(_choose_model(MODEL_MEAN_TUNED, MODEL_MEAN))
std_model = joblib.load(_choose_model(MODEL_STD_TUNED, MODEL_STD))

# Predict with ensembles (per-estimator prediction -> mean & std)
def ensemble_preds_std(model, X):
    # Align features to model.feature_names_in_ when available (fill missing with zeros)
    X2 = X.copy()
    if hasattr(model, 'feature_names_in_'):
        expected = list(model.feature_names_in_)
        for c in expected:
            if c not in X2.columns:
                X2[c] = 0.0
        # Reorder to expected
        X2 = X2[expected]
    try:
        est_preds = np.vstack([est.predict(X2) for est in model.estimators_])
        mean = est_preds.mean(axis=0)
        std = est_preds.std(axis=0)
        return mean, std
    except Exception:
        # fallback to model.predict only (no uncertainty)
        mean = model.predict(X2)
        std = np.zeros_like(mean)
        return mean, std

# Predictions for labeled set
pred_mean_lab, pred_mean_std_lab = ensemble_preds_std(mean_model, X_lab)
pred_std_lab, pred_std_std_lab = ensemble_preds_std(std_model, X_lab)

labels = labels.set_index('train_id')
labels['pred_rr_mean'] = pred_mean_lab
labels['pred_rr_mean_unc'] = pred_mean_std_lab
labels['pred_rr_std'] = pred_std_lab
labels['pred_rr_std_unc'] = pred_std_std_lab

# Metrics for labeled set
metrics = []
for col_true, col_pred, col_unc, name in [
    ('rr_mean','pred_rr_mean','pred_rr_mean_unc','rr_mean'),
    ('rr_std','pred_rr_std','pred_rr_std_unc','rr_std')]:
    y_true = labels[col_true].values
    y_pred = labels[col_pred].values
    y_unc = labels[col_unc].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    # coverage: fraction of truths within k*unc
    cover_1 = np.mean(np.abs(y_true - y_pred) <= 1.0 * y_unc)
    cover_196 = np.mean(np.abs(y_true - y_pred) <= 1.96 * y_unc)
    cover_2 = np.mean(np.abs(y_true - y_pred) <= 2.0 * y_unc)
    z = (y_true - y_pred) / np.where(y_unc==0, 1e-8, y_unc)
    metrics.append({'stat': name, 'mae': mae, 'rmse': rmse, 'coverage_1std': cover_1, 'coverage_1.96std': cover_196, 'coverage_2std': cover_2, 'z_mean': np.mean(z), 'z_std': np.std(z)})

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(REPORTS / 'rr_model_validation_metrics.csv', index=False)

# Predictions for all trains (we'll use these for distribution comparison and high-uncertainty flags)
pred_mean_all, pred_mean_std_all = ensemble_preds_std(mean_model, X_all)
pred_std_all, pred_std_std_all = ensemble_preds_std(std_model, X_all)

all_preds = pd.DataFrame({
    'train_id': X_all.index.astype(int),
    'pred_rr_mean': pred_mean_all,
    'pred_rr_mean_unc': pred_mean_std_all,
    'pred_rr_std': pred_std_all,
    'pred_rr_std_unc': pred_std_std_all,
}).set_index('train_id')

# Distribution comparisons
summary_rows = []
for stat in ['rr_mean','rr_std']:
    true_vals = labels[stat].dropna().values
    pred_lab = labels[f'pred_{stat}'].dropna().values
    pred_all = all_preds[f'pred_{stat}'].dropna().values
    row = {
        'stat': stat,
        'label_mean': np.mean(true_vals),
        'label_median': np.median(true_vals),
        'label_std': np.std(true_vals),
        'pred_lab_mean': np.mean(pred_lab),
        'pred_lab_std': np.std(pred_lab),
        'pred_all_mean': np.mean(pred_all),
        'pred_all_std': np.std(pred_all),
    }
    # KS test between label true distribution and predicted imputed distribution
    try:
        ks = stats.ks_2samp(true_vals, pred_all)
        row['ks_stat'] = ks.statistic
        row['ks_pvalue'] = ks.pvalue
    except Exception:
        row['ks_stat'] = None
        row['ks_pvalue'] = None
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(REPORTS / 'rr_distribution_comparison.csv', index=False)

# Plots
plt.figure(figsize=(8,4))
plt.hist(labels['rr_mean'].dropna(), bins=25, alpha=0.6, label='label_rr_mean')
plt.hist(all_preds['pred_rr_mean'].dropna(), bins=25, alpha=0.6, label='pred_all_rr_mean')
plt.legend(); plt.title('rr_mean: labeled vs predicted(all)')
plt.savefig(REPORTS / 'plots_rr_mean.png', dpi=150)
plt.close()

plt.figure(figsize=(8,4))
plt.hist(labels['rr_std'].dropna(), bins=25, alpha=0.6, label='label_rr_std')
plt.hist(all_preds['pred_rr_std'].dropna(), bins=25, alpha=0.6, label='pred_all_rr_std')
plt.legend(); plt.title('rr_std: labeled vs predicted(all)')
plt.savefig(REPORTS / 'plots_rr_std.png', dpi=150)
plt.close()

# Flag high-uncertainty imputations
imputed_df = all_preds.reset_index()
# Heuristic: flag if uncertainty > 0.5 * predicted magnitude OR in top 5% absolute uncertainty
imputed_df['flag_mean'] = (imputed_df['pred_rr_mean_unc'] > 0.5 * np.abs(imputed_df['pred_rr_mean'])) | (imputed_df['pred_rr_mean_unc'] >= imputed_df['pred_rr_mean_unc'].quantile(0.95))
imputed_df['flag_std'] = (imputed_df['pred_rr_std_unc'] > 0.5 * np.abs(imputed_df['pred_rr_std'])) | (imputed_df['pred_rr_std_unc'] >= imputed_df['pred_rr_std_unc'].quantile(0.95))
imputed_df['flag_any'] = imputed_df['flag_mean'] | imputed_df['flag_std']
flags = imputed_df[imputed_df['flag_any']].copy()
flags.to_csv(REPORTS / 'high_uncertainty_imputations.csv', index=False)

# Write a short summary markdown
with open(REPORTS / 'validation_summary.md', 'w') as f:
    f.write('# RailRadar Imputation Validation Summary\n\n')
    f.write('## Metrics on labeled set (100 examples)\n')
    # to_markdown requires tabulate which may not be installed in minimal envs; use to_string for portability
    f.write(metrics_df.to_string(index=False))
    f.write('\n\n')
    f.write('## Distribution comparison\n')
    f.write(summary_df.to_string(index=False))
    f.write('\n\n')
    f.write(f'High-uncertainty imputations: {len(flags)} rows written to `reports/high_uncertainty_imputations.csv`\n')

print('Validation finished. Reports written to', REPORTS)
print('Metrics file:', REPORTS / 'rr_model_validation_metrics.csv')
print('Distribution comparison:', REPORTS / 'rr_distribution_comparison.csv')
print('High-uncertainty CSV:', REPORTS / 'high_uncertainty_imputations.csv')
print('Summary MD:', REPORTS / 'validation_summary.md')
