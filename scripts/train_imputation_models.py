#!/usr/bin/env python3
"""Train imputation models for rr_mean and rr_std using labeled features.
Saves models to models/rr_mean_model.joblib and models/rr_std_model.joblib and metrics to data/rr_model_metrics.csv

Usage: python scripts/train_imputation_models.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

ROOT = Path(__file__).resolve().parents[1]
LABELED = ROOT / 'data' / 'railradar_features.csv'
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)
OUT_METRICS = ROOT / 'data' / 'rr_model_metrics.csv'

FEATURE_COLS = ['distance_km','station_count_x','route_density','avg_delay_min_prev','avg_delay_std_prev','is_exp','is_superfast','is_vande','is_shatabdi','is_mail','origin_count','dest_count','origin_avg_delay','dest_avg_delay','long_route','train_name_len','origin_dest_same']

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 8, 12, None],
    # Avoid 'auto' which can be invalid across sklearn versions; prefer explicit options
    'max_features': ['sqrt', 'log2', 0.5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

def tune_and_fit(X, y, name):
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(rf, param_dist, n_iter=10, scoring='neg_mean_absolute_error', cv=cv, random_state=42, n_jobs=-1, verbose=0)
    search.fit(X, y)
    print(f'{name} best params:', search.best_params_)
    best = search.best_estimator_
    best.fit(X, y)
    joblib.dump(best, MODELS_DIR / f'{name}_model_tuned.joblib')
    return best, search


def eval_model(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(RandomForestRegressor(n_estimators=100, random_state=42), X, y, cv=kf, n_jobs=-1)
    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds, squared=False)
    return {'mae': mae, 'rmse': rmse}, preds


def main():
    data = pd.read_csv(LABELED)
    X = data[FEATURE_COLS].fillna(0)

    # rr_mean model
    y_mean = data['rr_mean']
    metrics_mean, preds_mean = eval_model(X, y_mean)
    print('rr_mean baseline CV metrics:', metrics_mean)

    mean_model, mean_search = tune_and_fit(X, y_mean, 'rr_mean')

    # rr_std model
    y_std = data['rr_std']
    metrics_std, preds_std = eval_model(X, y_std)
    print('rr_std baseline CV metrics:', metrics_std)

    std_model, std_search = tune_and_fit(X, y_std, 'rr_std')

    # Save metrics and search results
    dfm = pd.DataFrame([{'target':'rr_mean','mae':metrics_mean['mae'],'rmse':metrics_mean['rmse']},{'target':'rr_std','mae':metrics_std['mae'],'rmse':metrics_std['rmse']}])
    dfm.to_csv(OUT_METRICS, index=False)
    print('Wrote metrics to', OUT_METRICS)

    # Save search results
    pd.DataFrame(mean_search.cv_results_).to_csv(MODELS_DIR / 'rr_mean_search_results.csv', index=False)
    pd.DataFrame(std_search.cv_results_).to_csv(MODELS_DIR / 'rr_std_search_results.csv', index=False)
    print('Saved RandomizedSearchCV results to models/')


if __name__ == '__main__':
    main()
