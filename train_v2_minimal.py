"""Train a compact v2 model that uses route encoding, distance, day_of_week, month, is_peak_day"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

# reuse realistic delay generator from simple_train
from simple_train import generate_realistic_delays

def main():
    print("Loading merged route datasets via backend merge logic...")
    # We'll follow similar merging as backend.load_model by reading schedules and price data
    repo_root = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(repo_root, 'datasets')
    schedules_file = os.path.join(datasets_dir, 'schedules.csv')
    price_file = os.path.join(datasets_dir, 'price_data.csv')

    schedules_df = pd.read_csv(schedules_file)
    price_df = pd.read_csv(price_file)

    # Build route entries similarly to backend but simplified: just source/destination/distance/day_of_week
    rows = []
    for idx, row in schedules_df.iterrows():
        try:
            train_number = str(row['trainNumber']).zfill(5)
            source = row['stationFrom']
            destination = row['stationTo']
            if pd.isna(row['stationList']):
                distance_km = 0
            else:
                try:
                    station_list = pd.io.json.loads(str(row['stationList']).replace("'", '"'))
                    if station_list and isinstance(station_list, list):
                        src_dist = None
                        dst_dist = None
                        for st in station_list:
                            if st.get('stationCode') == source:
                                src_dist = st.get('distance', None)
                            if st.get('stationCode') == destination:
                                dst_dist = st.get('distance', None)
                        if src_dist is not None and dst_dist is not None:
                            distance_km = abs(float(dst_dist) - float(src_dist))
                        else:
                            distance_km = 0
                    else:
                        distance_km = 0
                except Exception:
                    distance_km = 0

            days = []
            if row['trainRunsOnMon'] == 'Y': days.append(1)
            if row['trainRunsOnTue'] == 'Y': days.append(2)
            if row['trainRunsOnWed'] == 'Y': days.append(3)
            if row['trainRunsOnThu'] == 'Y': days.append(4)
            if row['trainRunsOnFri'] == 'Y': days.append(5)
            if row['trainRunsOnSat'] == 'Y': days.append(6)
            if row['trainRunsOnSun'] == 'Y': days.append(7)

            for d in days:
                rows.append({
                    'train_id': train_number,
                    'source': source,
                    'destination': destination,
                    'source_code': source,
                    'destination_code': destination,
                    'distance_km': distance_km,
                    'day_of_week': d,
                    'month': 6,  # placeholder; we'll randomize month for training distribution
                    'weather_condition': np.random.choice(['Clear','Cloudy','Rainy','Foggy','Windy']),
                })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    print('Built route dataset, rows:', len(df))

    # Randomize month for variety
    df['month'] = np.random.randint(1, 13, size=len(df))

    # Build route label
    df['route'] = df['source_code'].astype(str) + '-' + df['destination_code'].astype(str)

    # Ensure we have enough data
    df = df[df['distance_km'] > 0]
    print('Filtered to distance>0 rows:', len(df))

    # Generate synthetic target delays
    df['avg_delay_min'] = generate_realistic_delays(df)

    # Feature: is_peak_day (weekend)
    df['is_peak_day'] = df['day_of_week'].isin([6,7]).astype(int)

    # Encode route
    route_encoder = LabelEncoder()
    df['route_encoded'] = route_encoder.fit_transform(df['route'])

    # Prepare features
    features = ['route_encoded','distance_km','day_of_week','month','is_peak_day']
    X = df[features]
    y = df['avg_delay_min']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Training model v2...')
    model = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print('v2 MAE:', mae)

    # Save artifacts
    joblib.dump(model, 'backend/model_v2.pkl')
    joblib.dump(route_encoder, 'backend/route_encoder_v2.pkl')
    joblib.dump({'features':features,'mae':mae}, 'backend/model_v2_info.pkl')

    print('Saved model_v2 and route_encoder_v2')

if __name__ == '__main__':
    main()
