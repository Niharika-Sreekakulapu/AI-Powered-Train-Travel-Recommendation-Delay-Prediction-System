"""Compute split-conformal intervals for rr_mean and rr_std using labeled features, apply to full imputed set, and produce new master and reports.
- Uses `data/railradar_features.csv` (labeled) to compute OOF residuals via KFold.
- Computes quantiles (q80,q90,q95) of abs residuals for rr_mean and rr_std.
- Applies existing models in `models/` to predict on `data/features_full.csv` and constructs intervals.
- Writes:
  - `reports/conformal_calibration_factors.csv`
  - `data/ap_trains_master_clean_with_delays_v6.csv`
  - `reports/high_uncertainty_imputations_conformal.csv` (top 10% by interval width for rr_mean)
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
REPORTS_DIR = os.path.join(ROOT, 'reports')
MODELS_DIR = os.path.join(ROOT, 'models')

os.makedirs(REPORTS_DIR, exist_ok=True)

labeled_path = os.path.join(DATA_DIR, 'railradar_features.csv')
features_full_path = os.path.join(DATA_DIR, 'features_full.csv')
master_in = os.path.join(DATA_DIR, 'ap_trains_master_clean_with_delays_v5_conservative.csv')
master_out = os.path.join(DATA_DIR, 'ap_trains_master_clean_with_delays_v6.csv')

print('Loading labeled features:', labeled_path)
lab = pd.read_csv(labeled_path)
print('Loading full features:', features_full_path)
full = pd.read_csv(features_full_path)

# Prepare features and targets
TARGETS = {'rr_mean': 'rr_mean', 'rr_std': 'rr_std'}
ID_COL = 'train_id'

# Candidate feature columns: all columns except id and targets
exclude = [ID_COL] + list(TARGETS.values())
feature_cols = [c for c in lab.columns if c not in exclude]
print('Using feature cols:', feature_cols)

# Ensure features exist in full
for c in feature_cols:
    if c not in full.columns:
        full[c] = 0

# KFold OOF residuals
kf = KFold(n_splits=5, shuffle=True, random_state=0)
residuals = {t: [] for t in TARGETS}

a = lab.dropna(subset=TARGETS.values())
X = a[feature_cols].values

for target in TARGETS:
    y = a[TARGETS[target]].values
    oof_preds = np.zeros_like(y, dtype=float)
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        # simple RF; parameters can be tuned later
        rf = RandomForestRegressor(n_estimators=200, random_state=0)
        rf.fit(X_train, y_train)
        oof_preds[val_idx] = rf.predict(X_val)
    abs_resid = np.abs(y - oof_preds)
    residuals[target] = abs_resid

# Compute quantiles
qs = [0.80, 0.90, 0.95]
rows = []
for target in TARGETS:
    vals = residuals[target]
    row = {'target': target}
    for q in qs:
        row[f'q{int(q*100)}'] = np.quantile(vals, q)
    rows.append(row)

calib_df = pd.DataFrame(rows)
calib_out = os.path.join(REPORTS_DIR, 'conformal_calibration_factors.csv')
calib_df.to_csv(calib_out, index=False)
print('Wrote calibration factors:', calib_out)

# Load models (prefer tuned if available)
mean_model_path = os.path.join(MODELS_DIR, 'rr_mean_model_tuned.joblib') if os.path.exists(os.path.join(MODELS_DIR, 'rr_mean_model_tuned.joblib')) else os.path.join(MODELS_DIR, 'rr_mean_model.joblib')
std_model_path = os.path.join(MODELS_DIR, 'rr_std_model_tuned.joblib') if os.path.exists(os.path.join(MODELS_DIR, 'rr_std_model_tuned.joblib')) else os.path.join(MODELS_DIR, 'rr_std_model.joblib')
print('Loading models:', mean_model_path, std_model_path)
mean_model = joblib.load(mean_model_path)
std_model = joblib.load(std_model_path)

# Prepare full X
X_full = full[feature_cols].copy()
# Align columns to model.feature_names_in_ if available (handles saved feature names)
def align_features_for_model(df, model):
    cols = list(df.columns)
    if hasattr(model, 'feature_names_in_'):
        required = list(model.feature_names_in_)
        # add missing cols with zeros
        for c in required:
            if c not in df.columns:
                df[c] = 0
        df = df[required]
    return df

print('Predicting on full features...')
X_full_mean = align_features_for_model(X_full.copy(), mean_model)
X_full_std = align_features_for_model(X_full.copy(), std_model)
full['pred_rr_mean'] = mean_model.predict(X_full_mean)
full['pred_rr_std'] = std_model.predict(X_full_std)

# Apply q95 intervals (symmetric)
q95_mean = calib_df.loc[calib_df['target']=='rr_mean', 'q95'].iloc[0]
q95_std = calib_df.loc[calib_df['target']=='rr_std', 'q95'].iloc[0]
full['pred_rr_mean_conf_lower_95'] = full['pred_rr_mean'] - q95_mean
full['pred_rr_mean_conf_upper_95'] = full['pred_rr_mean'] + q95_mean
full['pred_rr_std_conf_lower_95'] = full['pred_rr_std'] - q95_std
full['pred_rr_std_conf_upper_95'] = full['pred_rr_std'] + q95_std

# Merge intervals back to master by train_id
print('Merging intervals into master...')
master = pd.read_csv(master_in, dtype=str)
master_ids = master[ID_COL].astype(str)
full_idxed = full.set_index(ID_COL)
# Add interval cols to master with numeric types
for col in ['pred_rr_mean','pred_rr_mean_conf_lower_95','pred_rr_mean_conf_upper_95','pred_rr_std','pred_rr_std_conf_lower_95','pred_rr_std_conf_upper_95']:
    master[col] = master_ids.map(lambda x: full_idxed[col].get(x) if x in full_idxed.index else np.nan)

# Flagging: mark top 10% by rr_mean interval width
master['pred_rr_mean_conf_width_95'] = pd.to_numeric(master['pred_rr_mean_conf_upper_95'], errors='coerce') - pd.to_numeric(master['pred_rr_mean_conf_lower_95'], errors='coerce')
mask_imputed = master['rr_imputed_at'].notnull() | master['rr_model_version'].notnull()
imputed_master = master[mask_imputed].copy()
n_imputed = len(imputed_master)
if n_imputed == 0:
    print('No imputed rows found. Exiting.')
    raise SystemExit(1)

imputed_master['__width'] = pd.to_numeric(imputed_master['pred_rr_mean_conf_width_95'], errors='coerce').fillna(-1)
n_flag = max(1, int(np.ceil(0.10 * n_imputed)))
flagged_idx = imputed_master.sort_values('__width', ascending=False).head(n_flag).index
master['rr_imputation_flag_conformal'] = False
master.loc[flagged_idx, 'rr_imputation_flag_conformal'] = True

# Save master v6
master.to_csv(master_out, index=False)
print('Wrote master v6:', master_out)

# Save flagged rows
flagged = master[master['rr_imputation_flag_conformal'] == True]
out_flagged = os.path.join(REPORTS_DIR, 'high_uncertainty_imputations_conformal.csv')
cols = ['train_id','train_name','distance_km','station_count','pred_rr_mean','pred_rr_mean_conf_lower_95','pred_rr_mean_conf_upper_95','pred_rr_mean_conf_width_95']
available = [c for c in cols if c in master.columns]
master[available].to_csv(out_flagged, index=False)
print('Wrote flagged conformal:', out_flagged)

print('Done.')