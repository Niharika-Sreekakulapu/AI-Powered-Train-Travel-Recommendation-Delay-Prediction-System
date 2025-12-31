import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

import os

# Prefer daily-expanded dataset if present
p = 'data/ap_trains_final11_daily.csv' if os.path.exists('data/ap_trains_final11_daily.csv') else 'data/ap_trains_final11.csv'
print('Loading', p)
df = pd.read_csv(p)

# Features: distance_km, day_of_week, month, weather_condition, season, price
features = ['distance_km', 'day_of_week', 'month', 'weather_condition', 'season', 'price']
for f in features:
    if f not in df.columns:
        raise SystemExit(f'Missing feature: {f}')

X = df[features].copy()
y = df['avg_delay_min']

# Encode categorical features
le_weather = LabelEncoder()
le_season = LabelEncoder()
X['weather_encoded'] = le_weather.fit_transform(X['weather_condition'])
X['season_encoded'] = le_season.fit_transform(X['season'])
X = X.drop(['weather_condition', 'season'], axis=1)

# Fill any NaNs (shouldn't be many)
X = X.fillna(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train size: {len(X_train)}, Test size: {len(X_test)}')

# Model
model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Test MAE
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Test MAE: {mae:.3f} minutes')

# 5-fold CV (negative MAE -> convert). Some environments kill parallel workers,
# so try parallel first and fall back to single-process if it fails.
cv = KFold(n_splits=5, shuffle=True, random_state=42)
try:
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
except Exception:
    print('Parallel CV failed, retrying with n_jobs=1')
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=1)
cv_mae = -cv_scores
print(f'5-fold CV MAE scores: {cv_mae}')
print(f'CV MAE mean: {cv_mae.mean():.3f}, std: {cv_mae.std():.3f}')

# Feature importances
fi = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
print('\nFeature importances:')
print(fi.to_string(index=False))

# Save model and encoders for quick use
import joblib
joblib.dump(model, 'backend/baseline_model_no_route.pkl')
joblib.dump(le_weather, 'backend/le_weather_no_route.pkl')
joblib.dump(le_season, 'backend/le_season_no_route.pkl')
print('\nSaved baseline model and encoders to backend/')
