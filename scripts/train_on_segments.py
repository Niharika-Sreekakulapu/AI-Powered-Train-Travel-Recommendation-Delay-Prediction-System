import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib

P = 'data/segments_ap.csv'
print('Loading', P)
df = pd.read_csv(P)
print('Rows:', len(df))

# Features: distance_km, day_of_week, month, weather_condition, season, price
features = ['distance_km', 'day_of_week', 'month', 'weather_condition', 'season', 'price']
for f in features:
    if f not in df.columns:
        raise SystemExit(f'Missing feature: {f}')

X = df[features].copy()
y = df['avg_delay_min']

le_weather = LabelEncoder()
le_season = LabelEncoder()
X['weather_encoded'] = le_weather.fit_transform(X['weather_condition'])
X['season_encoded'] = le_season.fit_transform(X['season'])
X = X.drop(['weather_condition', 'season'], axis=1)

X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train size:', len(X_train), 'Test size:', len(X_test))

model = RandomForestRegressor(n_estimators=200, max_depth=16, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
print(f'Test MAE (segments): {mae:.3f} minutes')

# CV
cv = KFold(n_splits=5, shuffle=True, random_state=42)
try:
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
except Exception:
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=1)
cv_mae = -cv_scores
print('5-fold CV MAE:', cv_mae)
print('CV mean:', cv_mae.mean(), 'std:', cv_mae.std())

fi = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
print('\nFeature importances:')
print(fi.to_string(index=False))

joblib.dump(model, 'backend/segments_model.pkl')
joblib.dump(le_weather, 'backend/segments_le_weather.pkl')
joblib.dump(le_season, 'backend/segments_le_season.pkl')
print('\nSaved segment model and encoders')
