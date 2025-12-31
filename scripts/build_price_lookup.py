"""
Script: build_price_lookup.py
Purpose: Read the (potentially large) `datasets/price_data.csv` in chunks, aggregate average fares by
(trainNumber, fromStnCode, toStnCode) and write a compact `datasets/price_lookup.csv`.

This makes lookups fast and avoids loading the full large file in memory at runtime.

Usage:
    python scripts/build_price_lookup.py --src datasets/price_data.csv --out datasets/price_lookup.csv --chunksize 200000

"""
import argparse
import os
import pandas as pd


def build_price_lookup(src, out, chunksize=200000, class_code='3A'):
    if not os.path.exists(src):
        print(f"Source price file not found: {src}")
        return 1

    agg = {}
    total_rows = 0

    for i, chunk in enumerate(pd.read_csv(src, chunksize=chunksize)):
        total_rows += len(chunk)
        print(f"Processing chunk {i+1}, rows={len(chunk)}, cumulative={total_rows}")
        # Filter class if present
        # Optionally filter by class (if class_code provided and not 'ALL')
        if 'classCode' in chunk.columns and class_code and str(class_code).upper() != 'ALL':
            chunk = chunk[chunk['classCode'] == class_code]

        # Ensure cols exist
        expected = ['trainNumber', 'fromStnCode', 'toStnCode', 'totalFare', 'distance']
        for col in expected:
            if col not in chunk.columns:
                chunk[col] = None

        # Normalize trainNumber
        chunk['trainNumber'] = chunk['trainNumber'].astype(str).str.zfill(5)
        chunk['fromStnCode'] = chunk['fromStnCode'].astype(str).str.upper()
        chunk['toStnCode'] = chunk['toStnCode'].astype(str).str.upper()

        # Aggregate per-group sums and counts for mean computation later
        for row in chunk.itertuples(index=False):
            t = row.trainNumber
            src_code = row.fromStnCode
            dst_code = row.toStnCode
            try:
                fare = float(row.totalFare) if row.totalFare not in (None, '') else None
            except Exception:
                fare = None
            try:
                dist = float(row.distance) if row.distance not in (None, '') else None
            except Exception:
                dist = None

            key = (t, src_code, dst_code)
            if key not in agg:
                agg[key] = {'fare_sum': 0.0, 'fare_count': 0, 'dist_sum': 0.0, 'dist_count': 0}
            if fare is not None:
                agg[key]['fare_sum'] += fare
                agg[key]['fare_count'] += 1
            if dist is not None:
                agg[key]['dist_sum'] += dist
                agg[key]['dist_count'] += 1

    # Build DataFrame from agg
    rows = []
    for (train_id, src_code, dst_code), v in agg.items():
        avg_price = (v['fare_sum'] / v['fare_count']) if v['fare_count'] > 0 else None
        avg_distance = (v['dist_sum'] / v['dist_count']) if v['dist_count'] > 0 else None
        rows.append({'train_id': train_id, 'source_code': src_code, 'destination_code': dst_code, 'avg_price': avg_price, 'avg_distance': avg_distance})

    price_summary = pd.DataFrame(rows)
    print(f"Built price summary with {len(price_summary)} unique train-route combinations (from total rows {total_rows})")

    price_summary.to_csv(out, index=False)
    print(f"Wrote compact price lookup to {out}")
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='datasets/price_data.csv')
    p.add_argument('--out', default='datasets/price_lookup.csv')
    p.add_argument('--chunksize', type=int, default=200000)
    p.add_argument('--classCode', default='3A')
    args = p.parse_args()
    exit(build_price_lookup(args.src, args.out, chunksize=args.chunksize, class_code=args.classCode))