"""Compute diagnostics comparing flagged vs unflagged imputed trains.
- Reads `data/ap_trains_master_clean_with_delays_v5_conservative.csv` (falls back to v5)
- Produces `reports/flagged_vs_unflagged_stats.csv` and `reports/flagged_vs_unflagged_plot.png`
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
REPORTS_DIR = os.path.join(ROOT, 'reports')

os.makedirs(REPORTS_DIR, exist_ok=True)

CANDIDATES = [
    os.path.join(DATA_DIR, 'ap_trains_master_clean_with_delays_v5_conservative.csv'),
    os.path.join(DATA_DIR, 'ap_trains_master_clean_with_delays_v5.csv')
]
for p in CANDIDATES:
    if os.path.exists(p):
        MASTER_IN = p
        break
else:
    raise SystemExit('No master file found (v5_conservative or v5)')

print('Loading master:', MASTER_IN)
df = pd.read_csv(MASTER_IN, dtype=str)
# Convert numeric
for col in ['distance_km','station_count','pred_rr_mean_unc_calib','pred_rr_std_unc_calib','rr_imputed_unc_mean','rr_imputed_unc_std','avg_delay_min']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Identify imputed rows
imputed_mask = df['rr_imputed_at'].notnull() | df['rr_model_version'].notnull()
df_imputed = df[imputed_mask].copy()
if df_imputed.empty:
    raise SystemExit('No imputed rows found')

flag_col = 'rr_imputation_flag_conservative'
if flag_col not in df_imputed.columns:
    # Fallback to rr_imputation_flag if conservative missing
    flag_col = 'rr_imputation_flag'

# Robust boolean coercion for flag column (handles 'True'/'true'/'1')
flag_vals = df_imputed[flag_col]
flag_bool = flag_vals.astype(str).str.lower().isin(['true','1','t','y','yes'])
flagged = df_imputed[flag_bool]
unflagged = df_imputed[~flag_bool]

print(f'Total imputed: {len(df_imputed)}, flagged: {len(flagged)}, unflagged: {len(unflagged)}')

# Metrics to compare
metrics = ['distance_km','station_count','pred_rr_mean_unc_calib','pred_rr_std_unc_calib','rr_imputed_unc_mean','rr_imputed_unc_std','avg_delay_min']

rows = []
for m in metrics:
    if m not in df_imputed.columns:
        continue
    f = flagged[m].dropna()
    u = unflagged[m].dropna()
    # summary stats
    row = {
        'metric': m,
        'n_flagged': len(f),
        'n_unflagged': len(u),
        'flagged_mean': np.nan if f.empty else f.mean(),
        'unflagged_mean': np.nan if u.empty else u.mean(),
        'flagged_median': np.nan if f.empty else np.median(f),
        'unflagged_median': np.nan if u.empty else np.median(u),
        'flagged_std': np.nan if f.empty else f.std(),
        'unflagged_std': np.nan if u.empty else u.std(),
    }
    # KS test if both non-empty
    if len(f) >= 2 and len(u) >= 2:
        try:
            ks = stats.ks_2samp(f, u)
            row['ks_stat'] = ks.statistic
            row['ks_pvalue'] = ks.pvalue
        except Exception:
            row['ks_stat'] = np.nan
            row['ks_pvalue'] = np.nan
    else:
        row['ks_stat'] = np.nan
        row['ks_pvalue'] = np.nan
    rows.append(row)

summary = pd.DataFrame(rows)
SUMMARY_OUT = os.path.join(REPORTS_DIR, 'flagged_vs_unflagged_stats.csv')
summary.to_csv(SUMMARY_OUT, index=False)
print('Wrote:', SUMMARY_OUT)

# Plot: compare distributions for key numeric metrics
PLOT_OUT = os.path.join(REPORTS_DIR, 'flagged_vs_unflagged_plot.png')
plt.figure(figsize=(12, 6))
plot_metrics = [m for m in ['pred_rr_mean_unc_calib','distance_km','station_count'] if m in df_imputed.columns]

n = len(plot_metrics)
for i, m in enumerate(plot_metrics, 1):
    plt.subplot(1, n, i)
    f = flagged[m].dropna()
    u = unflagged[m].dropna()
    bins = 30
    if m == 'station_count':
        bins = range(int(df_imputed[m].min() or 0), int((df_imputed[m].max() or 1) + 2))
    plt.hist(u, bins=bins, alpha=0.5, label='unflagged')
    plt.hist(f, bins=bins, alpha=0.5, label='flagged')
    plt.title(m)
    plt.legend()

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=150)
print('Wrote:', PLOT_OUT)
print('Done.')
