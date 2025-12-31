"""
Script: validate_price_mapping.py
Purpose: Validate that the compact `datasets/price_lookup.csv` covers trains from `data/ap_trains_master_clean_with_delays.csv`.
Generates a short report printed to stdout and writes `scripts/price_mapping_report.csv` with per-train coverage stats.

Usage:
    python scripts/validate_price_mapping.py --price datasets/price_lookup.csv --master data/ap_trains_master_clean_with_delays.csv --out scripts/price_mapping_report.csv
"""
import argparse
import os
import pandas as pd


def validate(price_file, master_file, out_file):
    if not os.path.exists(price_file):
        print(f"Price file not found: {price_file}")
        return 1
    if not os.path.exists(master_file):
        print(f"Master file not found: {master_file}")
        return 1

    price = pd.read_csv(price_file)
    master = pd.read_csv(master_file)

    price['train_id'] = price['train_id'].astype(str).str.zfill(5)
    master['train_id'] = master['train_id'].astype(str).str.zfill(5)

    trains_in_price = set(price['train_id'].unique())
    master_trains = master['train_id'].unique()

    coverage = [(t, t in trains_in_price) for t in master_trains]
    cov_df = pd.DataFrame(coverage, columns=['train_id', 'has_price'])

    covered = cov_df['has_price'].sum()
    total = len(cov_df)

    print(f"Price lookup covers {covered}/{total} trains ({covered/total*100:.1f}%)")

    # Frequency distribution of prices per km for sanity
    merged = price.merge(master[['train_id', 'distance_km']].drop_duplicates('train_id'), on='train_id', how='left')
    merged['price_per_km'] = merged['avg_price'] / merged['avg_distance']
    stats = merged['price_per_km'].describe()
    print("Price per km stats:")
    print(stats)

    cov_df.to_csv(out_file, index=False)
    print(f"Wrote coverage report to {out_file}")
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--price', default='datasets/price_lookup.csv')
    p.add_argument('--master', default='data/ap_trains_master_clean_with_delays.csv')
    p.add_argument('--out', default='scripts/price_mapping_report.csv')
    args = p.parse_args()
    exit(validate(args.price, args.master, args.out))