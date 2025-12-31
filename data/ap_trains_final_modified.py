import pandas as pd
import numpy as np

# Load your original file (ensure the name matches)
df = pd.read_csv('data/ap_trains_final.csv')

def create_varied_dataset(df):
    data = []
    # Logic for realistic variation
    weather_impact = {'Clear': 0, 'Cloudy': 10, 'Windy': 15, 'Humid': 5, 'Rainy': 40, 'Foggy': 60}
    season_impact = {'Summer': 5, 'Monsoon': 25, 'Autumn': 0, 'Winter': 15}
    
    for _, row in df.iterrows():
        # Create 5 historical versions of each trip to provide variety
        for i in range(5):
            new_row = row.copy()
            hour = np.random.choice([2, 5, 8, 11, 14, 17, 20, 23])
            
            # Formula: (Dist/20) + Weather + Season + Peak Hour (8,17,20) + Random Noise
            base = (row['distance_km'] / 100) * 5
            w_val = weather_impact.get(row['weather_condition'], 0)
            s_val = season_impact.get(row['season'], 0)
            peak_boost = 20 if hour in [8, 17, 20] else 0
            noise = np.random.normal(10, 5)
            
            total_delay = max(2, round(base + w_val + s_val + peak_boost + noise, 2))
            
            # Speed logic: Scheduled time @ 80km/h + delay
            sched_time_min = (row['distance_km'] / 80) * 60
            actual_time_min = sched_time_min + total_delay
            avg_speed = round(row['distance_km'] / (actual_time_min / 60), 2)
            
            new_row['dep_hour'] = hour
            new_row['avg_delay_min'] = total_delay
            new_row['avg_speed_kmh'] = avg_speed
            data.append(new_row)
            
    return pd.DataFrame(data)

# Generate and save
final_df = create_varied_dataset(df)
final_df.to_csv('data/ap_trains_final_v3.csv', index=False)
print("Success! 'data/ap_trains_final_v3.csv' has been created with varied entries.")