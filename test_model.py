import os
import pandas as pd
import joblib
import numpy as np

# Load data - be robust to different dataset names in AP-only workspace
data_path = 'data/train_data.csv'
if not os.path.exists(data_path):
    data_path = 'data/train_data_ap_only.csv'
print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)
print('Data shape:', df.shape)

# Load model and encoders
model = joblib.load('backend/model.pkl')
route_encoder = joblib.load('backend/route_encoder.pkl')
weather_encoder = joblib.load('backend/weather_encoder.pkl')
season_encoder = joblib.load('backend/season_encoder.pkl')

print('Model loaded')

# Sample a few rows for testing
sample = df.sample(10, random_state=42)
print('Sample predictions:')
errors = []
for idx, row in sample.iterrows():
    try:
        route = f"{row['source']}-{row['destination']}"
        route_encoded = route_encoder.transform([route])[0]
        weather_encoded = weather_encoder.transform([row['weather_condition']])[0]
        season_encoded = season_encoder.transform([row['season']])[0]

        features = np.array([[route_encoded, row['day_of_week'], row['month'], row['distance_km'], weather_encoded, season_encoded]])
        pred = model.predict(features)[0]
        actual = row['avg_delay_min']
        error = abs(pred - actual)
        errors.append(error)
        print(f'Route: {route}, Pred: {pred:.1f}, Actual: {actual:.1f}, Error: {error:.1f}')
    except Exception as e:
        print(f'Error for route {route}: {e}')

if len(errors) > 0:
    print(f'Average error: {np.mean(errors):.2f} minutes')
    print(f'Median error: {np.median(errors):.2f} minutes')
    print(f'Max error: {np.max(errors):.2f} minutes')
else:
    print('No successful predictions to report (all sample predictions failed due to encoder/model mismatch)')

# Check overfitting: test on random unseen features
print('Testing on synthetic new data:')
for i in range(5):
    # Generate synthetic new data
    route_idx = np.random.randint(0, len(route_encoder.classes_))
    route_encoded = route_idx
    day = np.random.randint(1, 8)
    month = np.random.randint(1, 13)
    distance = np.random.uniform(50, 1500)
    weather_idx = np.random.randint(0, len(weather_encoder.classes_))
    season_idx = np.random.randint(0, len(season_encoder.classes_))
    weather_encoded = weather_idx
    season_encoded = season_idx

    features = np.array([[route_encoded, day, month, distance, weather_encoded, season_encoded]])
    try:
        pred = model.predict(features)[0]
        print(f'Random features: distance={distance:.0f}, pred={pred:.2f}')
    except Exception as e:
        print(f'Error for synthetic features: {e}')

# Analyze data statistics
print('\nData analysis:')
# Add route column for analysis
df['route'] = df['source'] + '-' + df['destination']

print(f'Unique routes: {df["route"].nunique()}')
print(f'Average delay: {df["avg_delay_min"].mean():.2f}')
print(f'Delay std: {df["avg_delay_min"].std():.2f}')
print(f'Max delay: {df["avg_delay_min"].max():.2f}')
print(f'Min delay: {df["avg_delay_min"].min():.2f}')
print(f'Median delay: {df["avg_delay_min"].median():.2f}')
