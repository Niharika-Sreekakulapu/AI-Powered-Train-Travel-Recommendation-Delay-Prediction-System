#!/usr/bin/env python3
"""
Merge RailRadar-derived delay stats into master CSV for covered trains.
Outputs:
 - data/ap_trains_master_clean_with_delays_v3.csv (updated master)
 - data/railradar_merged_stats.csv (per-train computed rr_mean/rr_std/station_count/endpoint)
 - data/railradar_unmatched.csv (trains present in railradar file but not in master)

Usage: python scripts/merge_railradar_delays.py
"""
import json
from pathlib import Path
from datetime import datetime
import math

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Prefer freshly fetched full RailRadar responses when available
RR_RAW_FULL = ROOT / 'data' / 'railradar_raw_full.jsonl'
RR_FILE = RR_RAW_FULL if RR_RAW_FULL.exists() else ROOT / 'data' / 'railradar_positive_examples.jsonl'
MASTER_IN = ROOT / 'data' / 'ap_trains_master_clean_with_delays.csv'
MASTER_OUT = ROOT / 'data' / 'ap_trains_master_clean_with_delays_v3.csv'
MERGED_STATS = ROOT / 'data' / 'railradar_merged_stats.csv'
UNMATCHED = ROOT / 'data' / 'railradar_unmatched.csv'


def safe_parse_sample(sample_field):
    # sample_field is often a string containing JSON; sometimes NaN or already object
    if sample_field is None:
        return None
    if isinstance(sample_field, dict):
        return sample_field
    try:
        return json.loads(sample_field)
    except Exception:
        # sometimes the string contains trailing truncated content; try to fix common issues
        # fallback: find first '{"success' occurrence and try to json.loads from there
        try:
            start = sample_field.find('{')
            if start != -1:
                return json.loads(sample_field[start:])
        except Exception:
            return None
    return None


import re

def compute_train_stats(sample_json_or_text):
    """
    Accept either parsed JSON or raw sample text. We extract station-level arrival/departure delay numbers robustly.
    """
    station_means = []

    # If we have parsed JSON dict, try the normal route
    if isinstance(sample_json_or_text, dict):
        data = sample_json_or_text.get('data')
        if not data:
            return None
        station_delays = data.get('stationDelays') or []
        for s in station_delays:
            a = s.get('averageArrivalDelayMinutes')
            d = s.get('averageDepartureDelayMinutes')
            vals = [v for v in (a, d) if v is not None and not (isinstance(v, float) and math.isnan(v))]
            if len(vals) == 0:
                continue
            station_means.append(sum(vals) / len(vals))
    elif isinstance(sample_json_or_text, str):
        text = sample_json_or_text
        # Find station-like objects and extract numeric arrival/departure fields
        # This is resilient to truncated or invalid unicode escapes in the JSON capture
        station_pattern = re.compile(r"\{[^}]*?averageArrivalDelayMinutes\s*:\s*([-]?[0-9]+(?:\.[0-9]+)?)|averageDepartureDelayMinutes\s*:\s*([-]?[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
        # Simpler approach: find all occurrences of the two fields and group them by proximity
        arr_re = re.compile(r"averageArrivalDelayMinutes\s*:\s*([-]?[0-9]+(?:\.[0-9]+)?)")
        dep_re = re.compile(r"averageDepartureDelayMinutes\s*:\s*([-]?[0-9]+(?:\.[0-9]+)?)")
        arrs = [float(m.group(1)) for m in arr_re.finditer(text)]
        deps = [float(m.group(1)) for m in dep_re.finditer(text)]
        # Pair up by position: we'll use min length of arrs/deps sequences by index; if counts differ, pair as possible
        length = max(len(arrs), len(deps))
        for i in range(length):
            vals = []
            if i < len(arrs):
                vals.append(arrs[i])
            if i < len(deps):
                vals.append(deps[i])
            if vals:
                station_means.append(sum(vals)/len(vals))
    else:
        return None

    if len(station_means) == 0:
        return None
    mean = float(pd.Series(station_means).mean())
    std = float(pd.Series(station_means).std(ddof=0))
    return {'rr_mean': mean, 'rr_std': std, 'station_count': len(station_means)}


def main():
    rr_stats = {}
    unmatched = []

    if not RR_FILE.exists():
        print(f"RailRadar file not found: {RR_FILE}")
        return

    with RR_FILE.open('r', encoding='utf-8') as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                # some lines may use single quotes or NaN; try replace NaN
                line_fixed = line.replace('NaN', 'null')
                try:
                    entry = json.loads(line_fixed)
                except Exception as e:
                    print(f'Skipping unparsable line {line_num}:', e)
                    continue

            status = entry.get('status')
            if status != 200:
                continue

            # Helper to extract/compute stats for a given train id from this entry
            def stats_for_train(tid, src_entry):
                # prefer parsed JSON fields if present
                parsed = src_entry.get('parsed') or src_entry.get('response_json')
                if parsed:
                    stats = compute_train_stats(parsed)
                else:
                    sample_field = src_entry.get('sample') or src_entry.get('response_text') or src_entry.get('response')
                    sample_json = safe_parse_sample(sample_field)
                    if sample_json:
                        stats = compute_train_stats(sample_json)
                    elif isinstance(sample_field, str):
                        stats = compute_train_stats(sample_field)
                    else:
                        stats = None
                if not stats:
                    # diagnostic: show keys for this entry on first few failures
                    if line_num <= 10:
                        print(f"No stats for train {tid} at line {line_num}; keys: {list(src_entry.keys())}")
                    return None
                stats['endpoint'] = src_entry.get('endpoint') or src_entry.get('endpoints')
                return stats

            # If this entry is a grouped fetch with 'train_ids', expand it
            if isinstance(entry.get('train_ids'), list) and len(entry.get('train_ids')) > 0:
                for tid in entry.get('train_ids'):
                    try:
                        stats = stats_for_train(tid, entry)
                    except Exception as e:
                        print(f'Error computing stats for train {tid} at line {line_num}:', e)
                        stats = None
                    if stats:
                        rr_stats[int(tid)] = stats
                continue

            # Otherwise, look for single-train fields
            train_id = entry.get('train_id') or entry.get('matched') or entry.get('train') or entry.get('train_no')
            if train_id is None:
                # try keys that might contain list or string
                train_ids = entry.get('train_ids')
                if train_ids and isinstance(train_ids, list) and len(train_ids) == 1:
                    train_id = train_ids[0]

            if train_id is None:
                # nothing to do for this entry
                if line_num <= 5:
                    print(f"Skipping entry with no train id at line {line_num}; keys: {list(entry.keys())}")
                continue

            stats = stats_for_train(train_id, entry)
            if not stats:
                continue
            rr_stats[int(train_id)] = stats

    if len(rr_stats) == 0:
        print('No usable RailRadar stats parsed.')
        return

    master = pd.read_csv(MASTER_IN)
    master['train_id'] = master['train_id'].astype(int)

    # Prepare columns for backups/metadata
    if 'avg_delay_min_prev' not in master.columns:
        master['avg_delay_min_prev'] = master['avg_delay_min']
    if 'avg_delay_std_prev' not in master.columns:
        master['avg_delay_std_prev'] = master['avg_delay_std']
    master['rr_merged_at'] = master.get('rr_merged_at', pd.NA)
    master['rr_station_count'] = master.get('rr_station_count', pd.NA)
    master['rr_source_endpoint'] = master.get('rr_source_endpoint', pd.NA)

    matched_rows = []
    for tid, s in rr_stats.items():
        rows = master[master['train_id'] == int(tid)]
        if rows.empty:
            unmatched.append({'train_id': tid, **s})
            continue
        idx = rows.index[0]
        master.at[idx, 'avg_delay_min_prev'] = master.at[idx, 'avg_delay_min']
        master.at[idx, 'avg_delay_std_prev'] = master.at[idx, 'avg_delay_std']
        master.at[idx, 'avg_delay_min'] = s['rr_mean']
        master.at[idx, 'avg_delay_std'] = s['rr_std']
        master.at[idx, 'rr_merged_at'] = datetime.utcnow().isoformat() + 'Z'
        master.at[idx, 'rr_station_count'] = s['station_count']
        master.at[idx, 'rr_source_endpoint'] = s.get('endpoint')
        matched_rows.append({'train_id': tid, **s})

    # Write outputs
    master.to_csv(MASTER_OUT, index=False)

    pd.DataFrame(matched_rows).to_csv(MERGED_STATS, index=False)
    pd.DataFrame(unmatched).to_csv(UNMATCHED, index=False)

    print(f"Merged {len(matched_rows)} trains. Wrote updated master: {MASTER_OUT}")
    if len(unmatched) > 0:
        print(f"{len(unmatched)} railradar trains had no match in master; wrote {UNMATCHED}")


if __name__ == '__main__':
    main()
