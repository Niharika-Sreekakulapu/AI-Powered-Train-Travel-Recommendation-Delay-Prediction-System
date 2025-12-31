"""Generate a short flag review CSV mapping flagged trains to human-readable reasons.
Rules (heuristics):
 - long_route: distance_km >= 900
 - many_stations: station_count >= 15
 - high_uncertainty: pred_rr_mean_unc_calib >= 300
 - historical_high_delay: avg_delay_min >= 50

Writes: reports/flag_review.csv
"""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
REPORTS_DIR = os.path.join(ROOT, 'reports')

os.makedirs(REPORTS_DIR, exist_ok=True)

MASTER_IN = os.path.join(DATA_DIR, 'ap_trains_master_clean_with_delays_v5_conservative.csv')
OUT = os.path.join(REPORTS_DIR, 'flag_review.csv')

print('Loading master:', MASTER_IN)
df = pd.read_csv(MASTER_IN, dtype=str)
for col in ['distance_km','station_count','pred_rr_mean_unc_calib','avg_delay_min']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Ensure flag column boolean
flag_col = 'rr_imputation_flag_conservative'
if flag_col not in df.columns:
    flag_col = 'rr_imputation_flag'

flag_vals = df[flag_col].astype(str).str.lower().isin(['true','1','t','y','yes'])
flagged = df[flag_vals].copy()

rules = []
for idx, row in flagged.iterrows():
    reasons = []
    if pd.notnull(row.get('distance_km')) and row['distance_km'] >= 900:
        reasons.append('long_route')
    if pd.notnull(row.get('station_count')) and row['station_count'] >= 15:
        reasons.append('many_stations')
    if pd.notnull(row.get('pred_rr_mean_unc_calib')) and row['pred_rr_mean_unc_calib'] >= 300:
        reasons.append('high_uncertainty')
    if pd.notnull(row.get('avg_delay_min')) and row['avg_delay_min'] >= 50:
        reasons.append('historical_high_delay')
    if not reasons:
        reasons.append('other')
    rules.append({
        'train_id': row['train_id'],
        'train_name': row.get('train_name', ''),
        'distance_km': row.get('distance_km', ''),
        'station_count': row.get('station_count', ''),
        'pred_rr_mean_unc_calib': row.get('pred_rr_mean_unc_calib', ''),
        'avg_delay_min': row.get('avg_delay_min', ''),
        'reasons': ';'.join(reasons)
    })

out_df = pd.DataFrame(rules)
out_df.to_csv(OUT, index=False)
print('Wrote:', OUT)
print('Done.')