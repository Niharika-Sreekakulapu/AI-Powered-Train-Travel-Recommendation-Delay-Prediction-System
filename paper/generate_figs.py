import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

OUTDIR = os.path.join(os.path.dirname(__file__), 'figs')
os.makedirs(OUTDIR, exist_ok=True)

# Try loading model
MODEL_PATHS = [
    os.path.join('backend', 'model_v2.pkl'),
    os.path.join('backend', 'model.pkl'),
]
model = None
model_path = None
for p in MODEL_PATHS:
    if os.path.exists(p):
        model = joblib.load(p)
        model_path = p
        break
if model is None:
    raise FileNotFoundError('No model file found in backend/*.pkl')
print('Loaded model from', model_path)

# Load feature info if available
feature_info = None
info_path = os.path.join('backend', 'model_v2_info.pkl')
if os.path.exists(info_path):
    feature_info = joblib.load(info_path)
    features = feature_info.get('features')
else:
    # fallback features guess
    features = ['route_encoded', 'day_of_week', 'month', 'distance_km', 'weather_condition_encoded', 'season_encoded']

# Load dataset (sample to speed up)
DATA_CANDIDATES = [
    os.path.join('data', 'train_data.csv'),
    os.path.join('data', 'train_data_ap_only.csv'),
    os.path.join('data', 'ap_trains_master_clean_with_delays_v7.csv')
]
df = None
for p in DATA_CANDIDATES:
    if os.path.exists(p):
        print('Loading data from', p)
        # read with low_memory False for big csv
        try:
            df = pd.read_csv(p, nrows=20000)
        except Exception:
            df = pd.read_csv(p)
        break
if df is None:
    raise FileNotFoundError('No suitable data file found')

# Ensure target exists
if 'avg_delay_min' not in df.columns:
    raise ValueError('Dataset missing avg_delay_min column for ground truth')

# Attempt to build features expected by model
X = pd.DataFrame()
# route_encoded: try to use route column or compose
if 'route' in df.columns:
    X['route_encoded'] = pd.factorize(df['route'])[0]
else:
    if 'source' in df.columns and 'destination' in df.columns:
        X['route_encoded'] = pd.factorize(df['source'].astype(str) + '-' + df['destination'].astype(str))[0]
    else:
        X['route_encoded'] = 0

# day_of_week and month
if 'day_of_week' in df.columns:
    X['day_of_week'] = df['day_of_week'].fillna(1).astype(int)
else:
    X['day_of_week'] = 1
if 'month' in df.columns:
    X['month'] = df['month'].fillna(1).astype(int)
else:
    X['month'] = 1

# distance
if 'distance_km' in df.columns:
    X['distance_km'] = df['distance_km'].fillna(df['distance_km'].median())
else:
    X['distance_km'] = 0

# weather and season encoding (simple factorize)
if 'weather_condition' in df.columns:
    X['weather_condition_encoded'] = pd.factorize(df['weather_condition'].fillna('Clear'))[0]
else:
    X['weather_condition_encoded'] = 0
if 'season' in df.columns:
    X['season_encoded'] = pd.factorize(df['season'].fillna('Other'))[0]
else:
    X['season_encoded'] = 0

# Limit to rows with target
mask = df['avg_delay_min'].notnull()
df = df[mask]
X = X.loc[df.index]

y_true = df['avg_delay_min'].astype(float)

# Prepare prediction input matching model feature names to avoid unseen/mismatched features
X_pred = X.copy()
model_feature_names = list(getattr(model, 'feature_names_in_', []))
if model_feature_names:
    # build DataFrame with exactly the model's feature names (in order)
    X_model = pd.DataFrame(index=X.index, columns=model_feature_names)
    for col in model_feature_names:
        if col in X.columns:
            X_model[col] = X[col]
        else:
            # try to derive sensible defaults from df if possible
            if col in df.columns:
                X_model[col] = df[col]
            elif col.startswith('is_'):
                X_model[col] = 0
            else:
                X_model[col] = 0
    # ensure proper dtypes
    X_model = X_model.fillna(0)
    X_pred = X_model
else:
    # fall back to X; if a feature list was provided, try to use it
    if isinstance(features, list) and all([f in X.columns for f in features]):
        X_pred = X[features]

# Predict with model, catch and report errors
try:
    y_pred = model.predict(X_pred)
except Exception as e:
    print('Model predict failed on X_pred:', e)
    # as a last resort, try predicting on a dense numeric array (drop non-numeric)
    try:
        y_pred = model.predict(X_pred.select_dtypes(include=[np.number]).fillna(0))
    except Exception as e2:
        print('Final model predict attempt failed:', e2)
        raise

# Metrics
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f'MAE: {mae:.3f}, R2: {r2:.3f}')

# Figure 1: Feature importance (if available)
try:
    importances = model.feature_importances_
    feat_names = features if len(importances) == len(features) else getattr(model, 'feature_names_in_', features)
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(6,4))
    plt.barh([feat_names[i] for i in order], importances[order])
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'feature_importance.png'), dpi=300)
    plt.close()
    print('Saved feature importance')
except Exception as e:
    print('Failed to create feature importance:', e)

# Figure 2: Prediction vs Ground Truth scatter
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.5, s=10)
lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
plt.plot(lims, lims, 'r--')
plt.xlabel('Ground Truth avg_delay_min (min)')
plt.ylabel('Predicted delay (min)')
plt.title(f'Predicted vs Ground Truth (MAE={mae:.2f} min, R2={r2:.3f})')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'pred_vs_true_scatter.png'), dpi=300)
plt.close()
print('Saved pred vs true scatter')

# Figure 3: Calibration-like coverage vs threshold
errors = np.abs(y_true - y_pred)
thresholds = np.array([0.5,1,2,5,10,20,30,60,120])
coverage = [(errors <= t).mean() for t in thresholds]
plt.figure(figsize=(6,4))
plt.plot(thresholds, coverage, marker='o')
plt.xscale('log')
plt.xlabel('Absolute error threshold (min, log scale)')
plt.ylabel('Coverage (fraction within threshold)')
plt.title('Error coverage vs threshold')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'error_coverage.png'), dpi=300)
plt.close()
print('Saved error coverage plot')

# Also create a small dataset stats table image
stats = pd.DataFrame({
    'stat': ['n', 'mean_delay', 'std_delay', 'median_delay', 'max_delay'],
    'value': [len(y_true), float(y_true.mean()), float(y_true.std()), float(y_true.median()), float(y_true.max())]
})
fig, ax = plt.subplots(figsize=(4,2))
ax.axis('off')
tbl = ax.table(cellText=stats.values, colLabels=stats.columns, loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'dataset_stats.png'), dpi=300)
plt.close()
print('Saved dataset stats table')

print('All figures saved in', OUTDIR)
