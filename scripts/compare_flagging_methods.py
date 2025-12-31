"""Compare conservative (v5_conservative) vs conformal (v6) flagging methods.
Generates:
 - reports/flagging_comparison.csv (one row per imputed train with both flags)
 - reports/flagging_comparison_summary.csv (counts & overlap metrics)
 - reports/flagging_only_conservative.csv
 - reports/flagging_only_conformal.csv
"""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
REPORTS_DIR = os.path.join(ROOT, 'reports')

os.makedirs(REPORTS_DIR, exist_ok=True)

cons_path = os.path.join(DATA_DIR, 'ap_trains_master_clean_with_delays_v5_conservative.csv')
conf_path = os.path.join(DATA_DIR, 'ap_trains_master_clean_with_delays_v6.csv')

print('Loading:', cons_path)
cons = pd.read_csv(cons_path, dtype=str)
print('Loading:', conf_path)
conf = pd.read_csv(conf_path, dtype=str)

# Standardize
for df in (cons, conf):
    for col in ['train_id','train_name','distance_km','station_count']:
        if col in df.columns:
            df[col] = df[col]

# Merge on train_id
cons = cons[['train_id','train_name','rr_imputation_flag_conservative']]
conf = conf[['train_id','train_name','rr_imputation_flag_conformal']]

merged = pd.merge(cons, conf, on=['train_id','train_name'], how='outer')
# Coerce bools
for col in ['rr_imputation_flag_conservative','rr_imputation_flag_conformal']:
    merged[col] = merged[col].astype(str).str.lower().isin(['true','1','t','y','yes'])

# Restrict to imputed trains (where any flag present or model version existed)
# We'll assume these masters already contain only relevant trains but be safe
merged['any_flag'] = merged[['rr_imputation_flag_conservative','rr_imputation_flag_conformal']].any(axis=1)

# Derive sets
only_cons = merged[(merged['rr_imputation_flag_conservative']==True) & (merged['rr_imputation_flag_conformal']==False)]
only_conf = merged[(merged['rr_imputation_flag_conservative']==False) & (merged['rr_imputation_flag_conformal']==True)]
both = merged[(merged['rr_imputation_flag_conservative']==True) & (merged['rr_imputation_flag_conformal']==True)]
neither = merged[(merged['rr_imputation_flag_conservative']==False) & (merged['rr_imputation_flag_conformal']==False)]

# Summary
summary = {
    'n_total_imputed': len(merged),
    'n_only_conservative': len(only_cons),
    'n_only_conformal': len(only_conf),
    'n_both': len(both),
    'n_neither': len(neither),
}
summary_df = pd.DataFrame([summary])
summary_out = os.path.join(REPORTS_DIR, 'flagging_comparison_summary.csv')
summary_df.to_csv(summary_out, index=False)
print('Wrote:', summary_out)

merged_out = os.path.join(REPORTS_DIR, 'flagging_comparison.csv')
merged.to_csv(merged_out, index=False)
print('Wrote:', merged_out)

if not only_cons.empty:
    only_cons_out = os.path.join(REPORTS_DIR, 'flagging_only_conservative.csv')
    only_cons.to_csv(only_cons_out, index=False)
    print('Wrote:', only_cons_out)

if not only_conf.empty:
    only_conf_out = os.path.join(REPORTS_DIR, 'flagging_only_conformal.csv')
    only_conf.to_csv(only_conf_out, index=False)
    print('Wrote:', only_conf_out)

print('Done.')