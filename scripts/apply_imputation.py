#!/usr/bin/env python3
"""Apply trained imputation models to trains missing RailRadar stats and merge into master CSV.
Writes:
 - data/ap_trains_master_clean_with_delays_v4.csv (master with imputations applied)
 - data/railradar_imputed_stats.csv (per-train imputed values and uncertainties)

Usage: python scripts/apply_imputation.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
MASTER_IN = ROOT / 'data' / 'ap_trains_master_clean_with_delays_v3.csv'
MASTER_OUT = ROOT / 'data' / 'ap_trains_master_clean_with_delays_v4.csv'
FEATURES = ROOT / 'data' / 'features_full.csv'
LABELS = ROOT / 'data' / 'railradar_labels.csv'
MODEL_MEAN = ROOT / 'models' / 'rr_mean_model.joblib'
MODEL_STD = ROOT / 'models' / 'rr_std_model.joblib'
MODEL_MEAN_TUNED = ROOT / 'models' / 'rr_mean_model_tuned.joblib'
MODEL_STD_TUNED = ROOT / 'models' / 'rr_std_model_tuned.joblib'
OUT_IMPUTED = ROOT / 'data' / 'railradar_imputed_stats.csv'

MODEL_VERSION = 'rr_imputer_v1'

# Feature columns for prediction (matches build_features outputs)
FEATURE_COLS = ['distance_km', 'station_count', 'route_density','avg_delay_min_prev','avg_delay_std_prev','is_exp','is_superfast','is_vande','is_shatabdi','is_mail','origin_count','dest_count','origin_avg_delay','dest_avg_delay','long_route','train_name_len','origin_dest_same']


def _choose_model(tuned_path, default_path):
    if tuned_path.exists():
        print(f'Using tuned model: {tuned_path.name}')
        return tuned_path
    print(f'Using default model: {default_path.name}')
    return default_path


def rf_predict_with_uncertainty(rf_model, X):
    # RandomForestRegressor: collect predictions from each estimator and compute mean and std
    all_preds = np.vstack([est.predict(X) for est in rf_model.estimators_])
    mean_pred = np.mean(all_preds, axis=0)
    std_pred = np.std(all_preds, axis=0, ddof=0)
    return mean_pred, std_pred


def main():
    master = pd.read_csv(MASTER_IN)
    features = pd.read_csv(FEATURES)
    labels = pd.read_csv(LABELS)

    labeled_ids = set(labels['train_id'].astype(int).tolist())

    # Identify trains without RailRadar source (i.e. rr_source_endpoint is NA)
    master['train_id'] = master['train_id'].astype(int)
    missing_mask = master['train_id'].apply(lambda t: t not in labeled_ids)

    # We'll impute for all trains not in labels
    X_all = features.copy()
    # For consistency with training, rename station_count_x -> station_count if desired
    X = X_all[FEATURE_COLS].fillna(0)

    mean_model_path = _choose_model(MODEL_MEAN_TUNED, MODEL_MEAN)
    std_model_path = _choose_model(MODEL_STD_TUNED, MODEL_STD)
    mean_model = joblib.load(mean_model_path)
    std_model = joblib.load(std_model_path)

    mean_preds, mean_unc = rf_predict_with_uncertainty(mean_model, X)
    std_preds, std_unc = rf_predict_with_uncertainty(std_model, X)

    X_all['pred_rr_mean'] = mean_preds
    X_all['pred_rr_mean_unc'] = mean_unc
    X_all['pred_rr_std'] = std_preds
    X_all['pred_rr_std_unc'] = std_unc

    # select only trains that are missing labels
    to_impute = X_all[~X_all['train_id'].isin(list(labeled_ids))]
    imputed_rows = []
    now = datetime.utcnow().isoformat() + 'Z'
    for _, row in to_impute.iterrows():
        tid = int(row['train_id'])
        imputed_rows.append({'train_id': tid, 'rr_mean_imputed': float(row['pred_rr_mean']), 'rr_mean_imputed_unc': float(row['pred_rr_mean_unc']), 'rr_std_imputed': float(row['pred_rr_std']), 'rr_std_imputed_unc': float(row['pred_rr_std_unc']), 'rr_imputed_at': now, 'rr_model_version': MODEL_VERSION})

    imputed_df = pd.DataFrame(imputed_rows)
    imputed_df.to_csv(OUT_IMPUTED, index=False)
    print(f'Wrote {len(imputed_df)} imputed rows to {OUT_IMPUTED}')

    # Merge imputations into master for trains that had no rr_source_endpoint
    imputed_map = imputed_df.set_index('train_id')

    master['rr_source_endpoint'] = master.get('rr_source_endpoint', pd.NA)
    for idx, row in master.iterrows():
        tid = int(row['train_id'])
        if pd.isna(row.get('rr_source_endpoint')) or row.get('rr_source_endpoint') == '':
            if tid in imputed_map.index:
                m = imputed_map.loc[tid]
                master.at[idx, 'avg_delay_min_prev'] = master.at[idx, 'avg_delay_min']
                master.at[idx, 'avg_delay_std_prev'] = master.at[idx, 'avg_delay_std']
                master.at[idx, 'avg_delay_min'] = m['rr_mean_imputed']
                master.at[idx, 'avg_delay_std'] = m['rr_std_imputed']
                master.at[idx, 'rr_merged_at'] = now
                master.at[idx, 'rr_station_count'] = master.at[idx, 'station_count']
                master.at[idx, 'rr_source_endpoint'] = f'model:{MODEL_VERSION}'
                master.at[idx, 'rr_imputed_at'] = m['rr_imputed_at']
                master.at[idx, 'rr_model_version'] = m['rr_model_version']
                master.at[idx, 'rr_imputed_unc_mean'] = m['rr_mean_imputed_unc']
                master.at[idx, 'rr_imputed_unc_std'] = m['rr_std_imputed_unc']

    master.to_csv(MASTER_OUT, index=False)
    print(f'Wrote updated master with imputations to {MASTER_OUT}')

    # For convenience, also save imputed_df
    imputed_df.to_csv(OUT_IMPUTED, index=False)


if __name__ == '__main__':
    main()
