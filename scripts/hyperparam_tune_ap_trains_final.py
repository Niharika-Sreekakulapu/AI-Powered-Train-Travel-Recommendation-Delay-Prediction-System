import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, mean_absolute_error
import joblib
import os

P = 'data/ap_trains_final_v3.csv'
if not os.path.exists(P):
    raise SystemExit(f'Missing data file: {P}')

print('Loading', P)
df = pd.read_csv(P)

features = ['distance_km', 'day_of_week', 'month', 'weather_condition', 'season']
X = df[features].copy()
y = df['avg_delay_min']

le_weather = LabelEncoder()
le_season = LabelEncoder()
X['weather_encoded'] = le_weather.fit_transform(X['weather_condition'].astype(str))
X['season_encoded'] = le_season.fit_transform(X['season'].astype(str))
X = X.drop(['weather_condition', 'season'], axis=1)
X = X.fillna(0)

cv = KFold(n_splits=3, shuffle=True, random_state=42)

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [8, 12, 16, 20, None],
    'max_features': ['auto', 'sqrt', 'log2', 0.5, 0.8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

scorer = make_scorer(mean_absolute_error, greater_is_better=False)

search = RandomizedSearchCV(rf, param_dist, n_iter=20, scoring=scorer, cv=cv, random_state=42, n_jobs=-1, verbose=1)
search.fit(X, y)

print('Best params:', search.best_params_)
print('Best CV negative MAE:', search.best_score_)

best = search.best_estimator_
os.makedirs('backend', exist_ok=True)
joblib.dump(best, 'backend/model_from_ap_trains_final_tuned.pkl')
joblib.dump(le_weather, 'backend/le_weather_ap_trains_final_tuned.pkl')
joblib.dump(le_season, 'backend/le_season_ap_trains_final_tuned.pkl')

# Save results
results_df = pd.DataFrame(search.cv_results_)
results_df.to_csv('backend/hyperparam_search_results.csv', index=False)

print('Saved tuned model and results to backend/')
