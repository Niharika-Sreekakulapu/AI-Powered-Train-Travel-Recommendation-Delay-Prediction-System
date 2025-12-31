"""Make all trains available on all days of the week.

This script reads data/ap_trains_final11.csv, expands each row into 7 rows
(day_of_week = 1..7), deduplicates by ['train_id','source_code','destination_code','day_of_week']
and writes the result back (keeping a .bak backup).
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / 'data' / 'ap_trains_final11.csv'
BACKUP_FILE = ROOT / 'data' / 'ap_trains_final11.csv.bak'

if __name__ == '__main__':
    print(f"Reading {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    original_count = len(df)
    print(f"Original rows: {original_count}")

    # Expand to 7 days. Drop any existing day_of_week so we replace it with all days
    if 'day_of_week' in df.columns:
        df = df.drop(columns=['day_of_week'])
    days = pd.DataFrame({'day_of_week': list(range(1, 8))})
    df['_tmp_key'] = 1
    days['_tmp_key'] = 1
    expanded = df.merge(days, on='_tmp_key').drop(columns=['_tmp_key'])

    # If original had day_of_week column, keep the expanded day_of_week value (we replaced it)
    # Deduplicate by desired subset
    before_dedup = len(expanded)
    deduped = expanded.drop_duplicates(subset=['train_id', 'source_code', 'destination_code', 'day_of_week'])
    after_dedup = len(deduped)

    print(f"Expanded rows: {before_dedup}, after dedupe: {after_dedup}")

    # Backup original
    BACKUP_FILE.write_bytes(DATA_FILE.read_bytes())
    print(f"Backup saved to {BACKUP_FILE}")

    # Save updated file
    NEW_FILE = ROOT / 'data' / 'ap_trains_final11_daily.csv'
    deduped.to_csv(NEW_FILE, index=False)
    print(f"Wrote {NEW_FILE} with {after_dedup} rows")
    print(f"Original file was not overwritten due to permission issues. If you want to use the daily file as the primary dataset, replace the original with the new file.")
