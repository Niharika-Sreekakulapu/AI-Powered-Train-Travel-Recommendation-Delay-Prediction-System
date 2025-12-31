import os
import pandas as pd
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'ap_trains_final_v3.csv')
OUT_MASTER = os.path.join(DATA_DIR, 'ap_trains_master.csv')
OUT_STATIONS = os.path.join(DATA_DIR, 'ap_unique_stations.txt')

print(f"Reading {INPUT_FILE}...")
chunks = []
for chunk in pd.read_csv(INPUT_FILE, chunksize=5000, dtype=str):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
print(f"Loaded {len(df)} rows")

# parse station lists
train_groups = {}
for _, row in df.iterrows():
    train_id = row.get('train_id')
    if pd.isna(train_id):
        continue
    train_id = str(train_id)
    st_list = []
    val = row.get('station_list')
    if pd.notna(val):
        try:
            arr = json.loads(val)
            st_list = [s.get('stationCode') for s in arr if 'stationCode' in s]
        except Exception:
            continue
    if train_id not in train_groups:
        train_groups[train_id] = {
            'train_id': train_id,
            'train_name': row.get('train_name'),
            'source': row.get('source'),
            'destination': row.get('destination'),
            'distance_km': row.get('distance_km'),
            'station_codes': st_list
        }
    else:
        existing = train_groups[train_id]['station_codes']
        # prefer longer station list
        if len(st_list) > len(existing):
            train_groups[train_id]['station_codes'] = st_list

# write master CSV
rows = []
unique_stations = set()
for t in train_groups.values():
    codes = t['station_codes']
    unique_stations.update([c for c in codes if c])
    rows.append({
        'train_id': t['train_id'],
        'train_name': t['train_name'],
        'source': t['source'],
        'destination': t['destination'],
        'distance_km': t['distance_km'],
        'station_count': len(codes),
        'station_codes': ','.join(codes)
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_MASTER, index=False)
print(f"Wrote master file with {len(out_df)} trains to {OUT_MASTER}")

with open(OUT_STATIONS, 'w') as fh:
    for c in sorted(unique_stations):
        fh.write(c + '\n')
print(f"Wrote {len(unique_stations)} unique station codes to {OUT_STATIONS}")
