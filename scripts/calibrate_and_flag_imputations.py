"""Calibrate ensemble uncertainty using labeled residuals and flag high-uncertainty imputations.

Outputs:
 - reports/uncertainty_calibration_factors.csv
 - reports/high_uncertainty_imputations_calibrated.csv
 - data/ap_trains_master_clean_with_delays_v5.csv (master with calibrated flags)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

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


def _choose_model(tuned, default):
    if tuned.exists():
        print('Using tuned model:', tuned.name)
        return tuned
    print('Using default model:', default.name)
    return default
MASTER_V4 = DATA / 'ap_trains_master_clean_with_delays_v4.csv'
MASTER_V5 = DATA / 'ap_trains_master_clean_with_delays_v5.csv'

labels = pd.read_csv(LABELS).set_index('train_id')
features_labeled = pd.read_csv(FEATURES_LABELED).set_index('train_id')
features_full = pd.read_csv(FEATURES_FULL).set_index('train_id')

mean_model = joblib.load(_choose_model(MODEL_MEAN_TUNED, MODEL_MEAN))
std_model = joblib.load(_choose_model(MODEL_STD_TUNED, MODEL_STD))

# helper to align features to model
def align_features_for_model(X, model):
    X2 = X.copy()
    if hasattr(model, 'feature_names_in_'):
        expected = list(model.feature_names_in_)
        for c in expected:
            if c not in X2.columns:
                X2[c] = 0.0
        X2 = X2[expected]
    return X2

# ensemble predictions (mean & std)
def ensemble_preds_std(model, X):
    X2 = align_features_for_model(X, model)
    try:
        est_preds = np.vstack([est.predict(X2) for est in model.estimators_])
        mean = est_preds.mean(axis=0)
        std = est_preds.std(axis=0)
        return mean, std
    except Exception:
        mean = model.predict(X2)
        std = np.zeros_like(mean)
        return mean, std

# Predict on labeled set
# Choose numeric feature columns that appear in both labeled and full feature sets
skip = {'rr_mean','rr_std'}
def select_numeric(df):
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
cols_lab = set(select_numeric(features_labeled))
cols_all = set(select_numeric(features_full))
feat_cols = sorted(list(cols_lab.intersection(cols_all)))
if not feat_cols:
    # fall back to labeled numeric columns
    feat_cols = select_numeric(features_labeled)
X_lab = features_labeled[feat_cols]
X_all = features_full[feat_cols]

pred_mean_lab, pred_mean_std_lab = ensemble_preds_std(mean_model, X_lab)
pred_std_lab, pred_std_std_lab = ensemble_preds_std(std_model, X_lab)

labels['pred_rr_mean'] = pred_mean_lab
labels['pred_rr_mean_unc'] = pred_mean_std_lab
labels['pred_rr_std'] = pred_std_lab
labels['pred_rr_std_unc'] = pred_std_std_lab

# normalized residuals (avoid divide-by-zero)
eps = 1e-6
r_mean = np.abs(labels['rr_mean'] - labels['pred_rr_mean']) / np.maximum(labels['pred_rr_mean_unc'], eps)
r_std = np.abs(labels['rr_std'] - labels['pred_rr_std']) / np.maximum(labels['pred_rr_std_unc'], eps)

# compute calibration quantiles
quantiles = [0.80, 0.90, 0.95]
calib = {}
for q in quantiles:
    calib[f'mean_norm_q{int(q*100)}'] = np.quantile(r_mean.values, q)
    calib[f'std_norm_q{int(q*100)}'] = np.quantile(r_std.values, q)

calib_df = pd.DataFrame([calib])
calib_df.to_csv(REPORTS / 'uncertainty_calibration_factors.csv', index=False)

# Apply to all: get raw preds
pred_mean_all, pred_mean_std_all = ensemble_preds_std(mean_model, X_all)
pred_std_all, pred_std_std_all = ensemble_preds_std(std_model, X_all)
all_preds = pd.DataFrame({
    'train_id': X_all.index.astype(int),
    'pred_rr_mean': pred_mean_all,
    'pred_rr_mean_unc': pred_mean_std_all,
    'pred_rr_std': pred_std_all,
    'pred_rr_std_unc': pred_std_std_all,
}).set_index('train_id')

# Apply calibration (use 95% by default)
scale_mean = calib[f'mean_norm_q95']
scale_std = calib[f'std_norm_q95']
all_preds['pred_rr_mean_unc_calib'] = all_preds['pred_rr_mean_unc'] * scale_mean
all_preds['pred_rr_std_unc_calib'] = all_preds['pred_rr_std_unc'] * scale_std

# Flagging heuristic
all_preds['flag_mean'] = (all_preds['pred_rr_mean_unc_calib'] > 0.5 * np.abs(all_preds['pred_rr_mean'])) | (all_preds['pred_rr_mean_unc_calib'] >= all_preds['pred_rr_mean_unc_calib'].quantile(0.95))
all_preds['flag_std'] = (all_preds['pred_rr_std_unc_calib'] > 0.5 * np.abs(all_preds['pred_rr_std'])) | (all_preds['pred_rr_std_unc_calib'] >= all_preds['pred_rr_std_unc_calib'].quantile(0.95))
all_preds['flag_any'] = all_preds['flag_mean'] | all_preds['flag_std']

# Write calibrated high-uncertainty CSV
flags = all_preds[all_preds['flag_any']].reset_index()
flags.to_csv(REPORTS / 'high_uncertainty_imputations_calibrated.csv', index=False)

# Merge into master v4 -> v5
master = pd.read_csv(MASTER_V4)
master = master.set_index('train_id')
# add calibrated columns (only for trains we imputed)
for col in ['pred_rr_mean_unc_calib','pred_rr_std_unc_calib','flag_any']:
    master[col] = all_preds[col]

# create a boolean flag column rr_imputation_flag (True if flag_any true or rr_source_endpoint indicates model)
master['rr_imputation_flag'] = master['flag_any'].fillna(False)
# write master v5
master.reset_index().to_csv(MASTER_V5, index=False)

# Append a brief note to validation_summary.md
vs = REPORTS / 'validation_summary.md'
with open(vs, 'a') as f:
    f.write('\n\n## Calibration and Flagging\n')
    f.write(f"Applied normalized-residual calibration using q95 scales: mean_scale={scale_mean:.3f}, std_scale={scale_std:.3f}.\n")
    f.write(f"Flagged {len(flags)} trains as high-uncertainty; CSV: reports/high_uncertainty_imputations_calibrated.csv\n")

print('Calibration applied and master v5 written:', MASTER_V5)
print('Calibration factors:', REPORTS / 'uncertainty_calibration_factors.csv')
print('High-uncertainty (calibrated):', REPORTS / 'high_uncertainty_imputations_calibrated.csv')
