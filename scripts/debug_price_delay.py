import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend import app as backend_app

# Ensure model and data loaded
backend_app.load_model()

print('Model v2 loaded:', backend_app.model_v2 is not None)
print('Model features:', getattr(backend_app.model_v2, 'feature_names_in_', None))

# Test predict_delay across distances
routes = ['NDLS-LKO', 'HYB-VSKP', 'MYS-MAS', 'NDLS-MAS', 'CSMT-NED']
distances = [10, 50, 150, 500, 1200]
for r in routes:
    for d in distances:
        pred = backend_app.predict_delay(r, 3, 12, d, 'Clear', 'Winter')
        print(f'route={r:12s} dist={d:4d} -> pred={pred:.2f} min')

# Pick a sample train that likely has price_lookup entries
# Use price_lookup.csv to find a train with a short segment price
import pandas as pd
pl = pd.read_csv('datasets/price_lookup.csv')
pl['train_id'] = pl['train_id'].astype(str).str.zfill(5)
# pick first sample
sample = pl.iloc[0]
tid = sample['train_id']
src = sample['source_code']
dst = sample['destination_code']
print('\nSample price lookup row:', tid, src, dst, sample['avg_price'], sample['avg_distance'])

price = backend_app._get_price_for_train_segment(tid, src, dst)
print('Lookup price:', price)

# Now simulate segment where lookup doesn't exist - choose train from schedules where station_list contains both
from backend.app import find_intermediate_segment_trains
# Try an intermediate search for a random train route example
res = find_intermediate_segment_trains(src, dst, 3, 12, 'Clear', 'Winter')
print('find_intermediate_segment_trains result sample:', res)
