import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(ROOT, 'models')
DATA_DIR = os.path.join(ROOT, 'data')
REPORTS_DIR = os.path.join(ROOT, 'reports')


def test_tuned_models_exist():
    mean_tuned = os.path.join(MODELS_DIR, 'rr_mean_model_tuned.joblib')
    std_tuned = os.path.join(MODELS_DIR, 'rr_std_model_tuned.joblib')
    assert os.path.exists(mean_tuned) or os.path.exists(os.path.join(MODELS_DIR, 'rr_mean_model.joblib'))
    assert os.path.exists(std_tuned) or os.path.exists(os.path.join(MODELS_DIR, 'rr_std_model.joblib'))


def test_master_v7_and_columns():
    v7 = os.path.join(DATA_DIR, 'ap_trains_master_clean_with_delays_v7.csv')
    assert os.path.exists(v7)
    df = pd.read_csv(v7)
    assert 'pred_rr_mean_conf_lower_95' in df.columns
    assert 'pred_rr_mean_conf_upper_95' in df.columns
    assert 'rr_imputation_flag_final' in df.columns


def test_conformal_calib_exists():
    c = os.path.join(REPORTS_DIR, 'conformal_calibration_factors.csv')
    assert os.path.exists(c)
    df = pd.read_csv(c)
    assert not df.empty
