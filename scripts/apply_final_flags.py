"""Apply final flagging policy and write master v7.
Policy: rr_imputation_flag_final = rr_imputation_flag_conformal OR rr_imputation_flag_conservative OR rr_imputation_flag
Writes:
 - data/ap_trains_master_clean_with_delays_v7.csv
 - reports/flagging_final_summary.csv
"""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
REPORTS_DIR = os.path.join(ROOT, 'reports')

os.makedirs(REPORTS_DIR, exist_ok=True)

v6 = os.path.join(DATA_DIR, 'ap_trains_master_clean_with_delays_v6.csv')
v5c = os.path.join(DATA_DIR, 'ap_trains_master_clean_with_delays_v5_conservative.csv')
out = os.path.join(DATA_DIR, 'ap_trains_master_clean_with_delays_v7.csv')
summary_out = os.path.join(REPORTS_DIR, 'flagging_final_summary.csv')

if not os.path.exists(v6):
    print('Master v6 not found. Exiting.')
    raise SystemExit(1)

m = pd.read_csv(v6, dtype=str)
# Ensure bool coercion
for col in ['rr_imputation_flag_conformal','rr_imputation_flag_conservative','rr_imputation_flag']:
    if col in m.columns:
        m[col] = m[col].astype(str).str.lower().isin(['true','1','t','y','yes'])
    else:
        m[col] = False

# Final policy: use conservative OR conformal (ignore legacy rr_imputation_flag which may be noisy)
m['rr_imputation_flag_final'] = m['rr_imputation_flag_conformal'] | m['rr_imputation_flag_conservative']

m.to_csv(out, index=False)

summary = pd.DataFrame([{
    'n_total': len(m),
    'n_conformal': int(m['rr_imputation_flag_conformal'].sum()),
    'n_conservative': int(m['rr_imputation_flag_conservative'].sum()),
    'n_final': int(m['rr_imputation_flag_final'].sum())
}])
summary.to_csv(summary_out, index=False)
print('Wrote master v7 and summary')