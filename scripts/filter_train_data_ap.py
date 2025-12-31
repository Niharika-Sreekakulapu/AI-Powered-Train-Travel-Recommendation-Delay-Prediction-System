import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
TRAIN_DATA = os.path.join(DATA_DIR, 'train_data.csv')
MASTER_STATIONS = os.path.join(DATA_DIR, 'ap_unique_stations.txt')
OUT_FILE = os.path.join(DATA_DIR, 'train_data_ap_only.csv')

with open(MASTER_STATIONS) as fh:
    master_codes = set(line.strip() for line in fh if line.strip())
print(f"Loaded {len(master_codes)} master station codes")

chunksize = 20000
keep_count = 0
rows_processed = 0
with pd.read_csv(TRAIN_DATA, chunksize=chunksize, dtype=str) as reader:
    for chunk in reader:
        rows_processed += len(chunk)
        mask = pd.Series(False, index=chunk.index)
        if 'source_code' in chunk.columns:
            mask = mask | chunk['source_code'].isin(master_codes)
        if 'destination_code' in chunk.columns:
            mask = mask | chunk['destination_code'].isin(master_codes)
        # fallback: check station_list
        if 'station_list' in chunk.columns:
            def contains_master(val):
                if pd.isna(val):
                    return False
                for code in master_codes:
                    if code in val:
                        return True
                return False
            sl_mask = chunk['station_list'].apply(contains_master)
            mask = mask | sl_mask
        filtered = chunk[mask]
        if not filtered.empty:
            header = not os.path.exists(OUT_FILE)
            filtered.to_csv(OUT_FILE, mode='a', index=False, header=header)
            keep_count += len(filtered)
        print(f"Processed {rows_processed}; kept so far {keep_count}")

print(f"Done. Total rows kept: {keep_count}")
