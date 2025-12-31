import pandas as pd
import json
import os

IN = 'data/ap_trains_final_v3.csv'
OUT = 'data/segments_ap.csv'

if not os.path.exists(IN):
    raise SystemExit(f'Missing input file: {IN}')

print('Loading', IN)
df = pd.read_csv(IN)

rows = []
count_skipped = 0
for _, r in df.iterrows():
    stlist = r.get('station_list')
    try:
        stations = json.loads(str(stlist).replace("'","\""))
        for i in range(len(stations)-1):
            s_from = stations[i]
            s_to = stations[i+1]
            try:
                dist_from = float(s_from.get('distance', 0)) if s_from.get('distance') not in (None, '') else 0
                dist_to = float(s_to.get('distance', 0)) if s_to.get('distance') not in (None, '') else 0
            except Exception:
                continue
            seg_dist = abs(dist_to - dist_from)
            if seg_dist <= 0:
                continue
            rows.append({
                'parent_train_id': r['train_id'],
                'parent_train_name': r.get('train_name'),
                'source_code': s_from.get('stationCode',''),
                'destination_code': s_to.get('stationCode',''),
                'distance_km': seg_dist,
                'day_of_week': r.get('day_of_week'),
                'month': r.get('month'),
                'weather_condition': r.get('weather_condition'),
                'season': r.get('season'),
                'price': r.get('price', 0),
                'parent_distance': r.get('distance_km'),
                'parent_avg_delay': r.get('avg_delay_min')
            })
    except Exception:
        count_skipped += 1

seg_df = pd.DataFrame(rows)
print('Created segment rows:', len(seg_df))

seg_df['avg_delay_min'] = seg_df.apply(lambda x: (x['distance_km']/x['parent_distance'])*x['parent_avg_delay'] if x['parent_distance'] and x['parent_distance']>0 else x['parent_avg_delay'], axis=1)

seg_df = seg_df[seg_df['distance_km']>0]

seg_df.to_csv(OUT, index=False)
print('Saved segments to', OUT)
print('Skipped rows:', count_skipped)
