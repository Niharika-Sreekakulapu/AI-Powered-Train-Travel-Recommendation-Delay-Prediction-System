import os
import pandas as pd
import numpy as np
import os, sys
# Ensure repo root is on sys.path for imports when run from scripts/
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from simple_train import generate_realistic_delays

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
MASTER_CLEAN = os.path.join(DATA_DIR, 'ap_trains_master_clean.csv')
OUT = os.path.join(DATA_DIR, 'ap_trains_master_clean_with_delays.csv')

if __name__ == '__main__':
    if not os.path.exists(MASTER_CLEAN):
        raise FileNotFoundError(MASTER_CLEAN)
    df = pd.read_csv(MASTER_CLEAN)

    # Create minimal day_of_week/month/weather_condition/season columns expected by generator
    # Use random but reproducible values to approximate distribution
    rng = np.random.RandomState(42)
    df['day_of_week'] = rng.randint(1, 8, size=len(df))
    df['month'] = rng.randint(1, 13, size=len(df))

    # Heuristic weather and season for Andhra Pradesh
    def guess_weather(month):
        if month in [6,7,8,9]:
            return 'Rainy'
        if month in [12,1,2]:
            return 'Foggy'
        # hot months
        return 'Hot'

    df['weather_condition'] = df['month'].apply(guess_weather)

    print(f"Generating realistic delays for {len(df)} trains...")
    df['avg_delay_min'] = generate_realistic_delays(df)

    # Add per-train std estimate by sampling small noise
    df['avg_delay_std'] = (df['avg_delay_min'] * 0.25).round(1).clip(lower=1.0)

    # Save enriched master
    df.to_csv(OUT, index=False)

    # Report stats
    print('Delay stats:')
    print(f"  mean: {df['avg_delay_min'].mean():.2f} min")
    print(f"  median: {df['avg_delay_min'].median():.2f} min")
    print(f"  std: {df['avg_delay_min'].std():.2f} min")
    print(f"Wrote enriched master to {OUT}")
