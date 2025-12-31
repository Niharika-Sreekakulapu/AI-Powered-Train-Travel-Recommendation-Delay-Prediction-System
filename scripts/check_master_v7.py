"""Quick check script to validate master v7 and report counts."""
import os
import pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
MASTER_V7 = os.path.join(DATA_DIR, 'ap_trains_master_clean_with_delays_v7.csv')
if not os.path.exists(MASTER_V7):
    print('MASTER V7 not found')
    raise SystemExit(1)

m = pd.read_csv(MASTER_V7, dtype=str)
print('MASTER V7 loaded, rows:', len(m))
cols = list(m.columns)
print('Columns:', cols[:50])
print('Columns include final flag:', 'rr_imputation_flag_final' in cols)
for col in ['rr_imputation_flag','rr_imputation_flag_conservative','rr_imputation_flag_conformal','rr_imputation_flag_final']:
    if col in cols:
        count = int(m[col].astype(str).str.lower().isin(['true','1','t','y','yes']).sum())
        print(f'{col}: {count}')
    else:
        print(f'{col}: (missing)')
