"""Conservative re-flagging script
- Reads `data/ap_trains_master_clean_with_delays_v5.csv`
- Marks top 10% of imputed trains by `pred_rr_mean_unc_calib` as `rr_imputation_flag_conservative=True`
- Writes:
  - `data/ap_trains_master_clean_with_delays_v5_conservative.csv`
  - `reports/reflag_conservative_summary.csv`
  - `reports/reflag_conservative_sample.csv` (30 flagged + 30 unflagged samples)

Run: python scripts/reflag_conservative.py
"""
import os
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
REPORTS_DIR = os.path.join(ROOT, 'reports')

os.makedirs(REPORTS_DIR, exist_ok=True)

MASTER_IN = os.path.join(DATA_DIR, 'ap_trains_master_clean_with_delays_v5.csv')
MASTER_OUT = os.path.join(DATA_DIR, 'ap_trains_master_clean_with_delays_v5_conservative.csv')
SUMMARY_OUT = os.path.join(REPORTS_DIR, 'reflag_conservative_summary.csv')
SAMPLE_OUT = os.path.join(REPORTS_DIR, 'reflag_conservative_sample.csv')

print('Loading master:', MASTER_IN)
df = pd.read_csv(MASTER_IN, dtype=str)
# Convert numeric cols where relevant
for col in ['pred_rr_mean_unc_calib','pred_rr_std_unc_calib','rr_imputed_unc_mean','rr_imputed_unc_std']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Identify imputed rows (where a model imputed value exists)
imputed_mask = df['rr_imputed_at'].notnull() | df['rr_model_version'].notnull()
imputed_df = df[imputed_mask].copy()
if imputed_df.empty:
    print('No imputed rows found. Exiting.')
    raise SystemExit(1)

# Choose a primary uncertainty metric (prefer calibrated mean uncertainty)
if 'pred_rr_mean_unc_calib' in imputed_df.columns and imputed_df['pred_rr_mean_unc_calib'].notnull().any():
    metric = 'pred_rr_mean_unc_calib'
elif 'rr_imputed_unc_mean' in imputed_df.columns and imputed_df['rr_imputed_unc_mean'].notnull().any():
    metric = 'rr_imputed_unc_mean'
else:
    # Fallback: use pred_rr_std_unc_calib or rr_imputed_unc_std
    for c in ['pred_rr_std_unc_calib','rr_imputed_unc_std']:
        if c in imputed_df.columns and imputed_df[c].notnull().any():
            metric = c
            break
    else:
        print('No reasonable uncertainty metric found in imputed rows. Exiting.')
        raise SystemExit(1)

print('Using metric for ranking:', metric)
# Rank by metric (descending)
imputed_df['__metric'] = imputed_df[metric].replace([np.inf, -np.inf], np.nan)
imputed_df['__metric'] = imputed_df['__metric'].fillna(-1)

n_imputed = len(imputed_df)
n_flag = max(1, int(np.ceil(0.10 * n_imputed)))

imputed_df = imputed_df.sort_values('__metric', ascending=False)
flagged_idx = imputed_df.head(n_flag).index

# Create conservative flag column
df['rr_imputation_flag_conservative'] = False
df.loc[flagged_idx, 'rr_imputation_flag_conservative'] = True

# Save updated master copy
print(f'Flagging {n_flag} of {n_imputed} imputed trains ({n_flag / n_imputed:.2%})')

df.to_csv(MASTER_OUT, index=False)

# Summary CSV
summary = pd.DataFrame({
    'metric_used':[metric],
    'n_imputed':[n_imputed],
    'n_flagged_conservative':[n_flag],
    'flag_fraction':[n_flag / n_imputed],
    'metric_median_all':[imputed_df['__metric'].median()],
    'metric_median_flagged':[imputed_df.head(n_flag)['__metric'].median()],
    'metric_min_flagged':[imputed_df.head(n_flag)['__metric'].min()],
    'metric_max_flagged':[imputed_df.head(n_flag)['__metric'].max()]
})
summary.to_csv(SUMMARY_OUT, index=False)

# Sample: top 30 flagged and 30 random unflagged imputed
n_sample = 30
flag_sample = imputed_df.head(n_flag).head(n_sample)
unflagged_pool = imputed_df.drop(flagged_idx)
unflag_sample = unflagged_pool.sample(n=min(n_sample, len(unflagged_pool)), random_state=42)

sample_df = pd.concat([flag_sample, unflag_sample])
sample_df = sample_df.drop(columns=['__metric'])
sample_df.to_csv(SAMPLE_OUT, index=False)

print('Wrote:', MASTER_OUT)
print('Wrote:', SUMMARY_OUT)
print('Wrote:', SAMPLE_OUT)
print('Done.')