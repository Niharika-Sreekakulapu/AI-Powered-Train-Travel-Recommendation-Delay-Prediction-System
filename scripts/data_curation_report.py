import os
import pandas as pd
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

report = []

for f in files:
    path = os.path.join(DATA_DIR, f)
    print(f"Processing {f}...")
    try:
        station_codes = set()
        train_ids = set()
        rows = 0
        # read in chunks
        for chunk in pd.read_csv(path, chunksize=10000, dtype=str):
            rows += len(chunk)
            if 'train_id' in chunk.columns:
                train_ids.update(chunk['train_id'].dropna().unique().tolist())
            if 'station_list' in chunk.columns:
                for val in chunk['station_list'].dropna().unique():
                    try:
                        # station_list may be a JSON-like string or something
                        arr = json.loads(val)
                        for s in arr:
                            if 'stationCode' in s:
                                station_codes.add(s['stationCode'])
                    except Exception:
                        # try to parse as list of dicts using replace tricks
                        try:
                            arr = json.loads(val.replace("\"\"", '"'))
                            for s in arr:
                                if 'stationCode' in s:
                                    station_codes.add(s['stationCode'])
                        except Exception:
                            continue
        report.append({
            'file': f,
            'rows': rows,
            'unique_trains': len(train_ids),
            'unique_stations': len(station_codes),
            'sample_stations': sorted(list(station_codes))[:20]
        })
    except Exception as e:
        report.append({'file': f, 'error': str(e)})

print('\nData curation report:')
for r in report:
    print(r)

# write report to data folder
out = os.path.join(DATA_DIR, 'data_curation_report.json')
import json as _json
with open(out, 'w') as fh:
    _json.dump(report, fh, indent=2)
print(f"Report written to {out}")
