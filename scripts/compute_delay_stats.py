"""
Script: compute_delay_stats.py
Purpose: Compute per-train authoritative delay statistics (mean & std) from `data/ap_trains_final_v3.csv` and
merge/update them into `data/ap_trains_master_clean_with_delays.csv` (or create a new version).

Usage:
    python scripts/compute_delay_stats.py --src data/ap_trains_final_v3.csv --master data/ap_trains_master_clean.csv --out data/ap_trains_master_clean_with_delays_v2.csv

"""
import argparse
import os
import pandas as pd


def compute_delay_stats(src, master, out):
    if not os.path.exists(src):
        print(f"Source trains file not found: {src}")
        return 1
    df = pd.read_csv(src)
    # Expect column 'train_id' and 'avg_delay_min'
    if 'train_id' not in df.columns:
        if 'trainId' in df.columns:
            df['train_id'] = df['trainId'].astype(str).str.zfill(5)
    else:
        df['train_id'] = df['train_id'].astype(str).str.zfill(5)

    # Aggregate statistics per train
    agg = df.groupby('train_id')['avg_delay_min'].agg(['count', 'mean', 'std']).reset_index()
    agg.columns = ['train_id', 'delay_count', 'avg_delay_min', 'avg_delay_std']

    # Merge into master if provided
    if os.path.exists(master):
        master_df = pd.read_csv(master)
        master_df['train_id'] = master_df['train_id'].astype(str).str.zfill(5)
        merged = master_df.merge(agg[['train_id', 'avg_delay_min', 'avg_delay_std']], on='train_id', how='left')
    else:
        merged = agg.copy()

    print(f"Computed delay stats for {len(agg)} trains. Writing out to {out}")
    merged.to_csv(out, index=False)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='data/ap_trains_final_v3.csv')
    p.add_argument('--master', default='data/ap_trains_master_clean.csv')
    p.add_argument('--out', default='data/ap_trains_master_clean_with_delays_v2.csv')
    args = p.parse_args()
    exit(compute_delay_stats(args.src, args.master, args.out))