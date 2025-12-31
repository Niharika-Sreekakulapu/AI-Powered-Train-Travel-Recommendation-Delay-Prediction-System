#!/usr/bin/env python3
"""Build features for modeling RailRadar rr_mean/rr_std imputation.
Writes:
 - data/railradar_features.csv  (features + targets for trains with RailRadar labels)
 - data/features_full.csv       (features for all trains)

Usage: python scripts/build_features.py
"""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / 'data' / 'ap_trains_master_clean_with_delays_v3.csv'
LABELS = ROOT / 'data' / 'railradar_labels.csv'
OUT_LABELED = ROOT / 'data' / 'railradar_features.csv'
OUT_ALL = ROOT / 'data' / 'features_full.csv'


def name_flags(name: str):
    name = (name or '').upper()
    return {
        'is_exp': 1 if ('EXP' in name or 'EXPRESS' in name or 'EX' in name) else 0,
        'is_superfast': 1 if ('SUPERFAST' in name or 'SF' in name or 'SUF' in name) else 0,
        'is_vande': 1 if 'VANDE' in name else 0,
        'is_shatabdi': 1 if 'SHATABDI' in name else 0,
        'is_mail': 1 if 'MAIL' in name or 'MEX' in name else 0,
    }


def build_features(df):
    df = df.copy()
    # numeric features
    df['distance_km'] = pd.to_numeric(df['distance_km'], errors='coerce')
    df['station_count'] = pd.to_numeric(df['station_count'], errors='coerce')
    df['route_density'] = df['distance_km'] / df['station_count']
    df['avg_delay_min_prev'] = pd.to_numeric(df.get('avg_delay_min_prev', df['avg_delay_min']), errors='coerce')
    df['avg_delay_std_prev'] = pd.to_numeric(df.get('avg_delay_std_prev', df['avg_delay_std']), errors='coerce')

    # textual flags from train_name
    flags = df['train_name'].apply(name_flags).apply(pd.Series)
    df = pd.concat([df, flags], axis=1)

    # station_codes count (if station_count missing)
    df['station_codes_len'] = df['station_codes'].fillna('').apply(lambda s: len(s.split(',')) if s else 0)
    df['station_count'] = df['station_count'].fillna(df['station_codes_len'])

    # origin/destination aggregate features
    origin_stats = df.groupby('source').agg(origin_count=('train_id','count'), origin_avg_delay=('avg_delay_min_prev','mean')).reset_index()
    dest_stats = df.groupby('destination').agg(dest_count=('train_id','count'), dest_avg_delay=('avg_delay_min_prev','mean')).reset_index()
    df = df.merge(origin_stats, left_on='source', right_on='source', how='left')
    df = df.merge(dest_stats, left_on='destination', right_on='destination', how='left')

    # simple additional flags
    df['long_route'] = (df['distance_km'] > 800).astype(int)
    df['train_name_len'] = df['train_name'].fillna('').apply(len)
    df['origin_dest_same'] = (df['source'] == df['destination']).astype(int)

    # fill NaNs
    df['distance_km'] = df['distance_km'].fillna(df['distance_km'].median())
    df['route_density'] = df['route_density'].fillna(df['route_density'].median())
    df['avg_delay_min_prev'] = df['avg_delay_min_prev'].fillna(df['avg_delay_min_prev'].median())
    df['avg_delay_std_prev'] = df['avg_delay_std_prev'].fillna(df['avg_delay_std_prev'].median())
    df['origin_count'] = df['origin_count'].fillna(0)
    df['dest_count'] = df['dest_count'].fillna(0)
    df['origin_avg_delay'] = df['origin_avg_delay'].fillna(df['avg_delay_min_prev'].median())
    df['dest_avg_delay'] = df['dest_avg_delay'].fillna(df['avg_delay_min_prev'].median())

    # select features
    features = ['train_id','distance_km','station_count','route_density','avg_delay_min_prev','avg_delay_std_prev','is_exp','is_superfast','is_vande','is_shatabdi','is_mail','origin_count','dest_count','origin_avg_delay','dest_avg_delay','long_route','train_name_len','origin_dest_same']
    return df[features]


def main():
    master = pd.read_csv(MASTER)
    labels = pd.read_csv(LABELS)

    features_all = build_features(master)
    features_all.to_csv(OUT_ALL, index=False)
    print(f"Wrote full features to {OUT_ALL} ({features_all.shape[0]} rows)")

    # join with labels to produce training set
    labeled = features_all.merge(labels[['train_id','rr_mean','rr_std','station_count']], on='train_id', how='inner')
    labeled = labeled.drop_duplicates(subset=['train_id']).reset_index(drop=True)
    labeled.to_csv(OUT_LABELED, index=False)
    print(f"Wrote labeled features to {OUT_LABELED} ({labeled.shape[0]} rows)")
    print(labeled.head())


if __name__ == '__main__':
    main()
