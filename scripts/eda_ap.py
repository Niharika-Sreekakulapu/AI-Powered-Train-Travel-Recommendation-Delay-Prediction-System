import pandas as pd
import numpy as np

import os

# Prefer daily-expanded dataset if present
p = 'data/ap_trains_final11_daily.csv' if os.path.exists('data/ap_trains_final11_daily.csv') else 'data/ap_trains_final11.csv'
print('Loading', p)
df = pd.read_csv(p)
print('\nBasic shape:')
print('  rows:', len(df))

# route using codes (consistent)
df['route'] = df['source_code'].astype(str) + '-' + df['destination_code'].astype(str)

print('\nRoutes:')
print('  unique routes:', df['route'].nunique())
print('  unique sources:', df['source_code'].nunique())
print('  unique destinations:', df['destination_code'].nunique())

route_counts = df['route'].value_counts()
print('\nRoute samples (counts):')
print(route_counts.describe())
print('\nTop 10 routes by samples:')
print(route_counts.head(10))

print('\nNumber of routes with single sample:', (route_counts==1).sum())

# target stats
print('\nTarget `avg_delay_min` stats:')
print(df['avg_delay_min'].describe())
print('  median:', df['avg_delay_min'].median())
print('  pct <=15 min (on-time threshold):', (df['avg_delay_min'] <= 15).mean())

# distance stats
print('\n`distance_km` stats:')
print(df['distance_km'].describe())
print('  zeros or <=0 count:', (df['distance_km'] <= 0).sum())

# missing values
print('\nMissing values per column:')
print(df.isna().sum())

# categorical distributions
for col in ['weather_condition', 'season']:
    if col in df.columns:
        print(f"\nTop values for {col}:")
        print(df[col].value_counts().head(10))

# correlation distance vs delay
try:
    corr = df['distance_km'].corr(df['avg_delay_min'])
    print('\nPearson corr distance_km vs avg_delay_min:', corr)
except Exception as e:
    print('Corr error:', e)

# Per-route delay variability (for routes with >=2 samples)
grp = df.groupby('route')['avg_delay_min']
route_var = grp.agg(['count','mean','std'])
route_var_nonzero = route_var[route_var['count']>=2]
print('\nPer-route sample counts distribution:')
print(route_var['count'].describe())
print('\nFor routes with 2+ samples, mean std of delay:')
print(route_var_nonzero['std'].describe())

# show a few example rows
print('\nSample rows:')
print(df.head(5).to_string(index=False))

print('\nEDA finished')
