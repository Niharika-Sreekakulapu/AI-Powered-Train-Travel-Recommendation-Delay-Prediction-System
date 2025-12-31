#!/usr/bin/env python3
"""Extract per-train RailRadar labels (rr_mean, rr_std) from positive examples.
Writes: data/railradar_labels.csv

Usage: python scripts/extract_railradar_labels.py
"""
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_FILE = ROOT / 'data' / 'railradar_positive_examples.jsonl'
OUT = ROOT / 'data' / 'railradar_labels.csv'

# Implement compute_train_stats and safe_parse_sample locally (same logic as merge script)
import json as _json
import math as _math
import re as _re

def safe_parse_sample(sample_field):
    if sample_field is None:
        return None
    if isinstance(sample_field, dict):
        return sample_field
    try:
        return _json.loads(sample_field)
    except Exception:
        try:
            start = sample_field.find('{')
            if start != -1:
                return _json.loads(sample_field[start:])
        except Exception:
            return None
    return None


def compute_train_stats(sample_json_or_text):
    station_means = []
    import pandas as _pd
    # parsed JSON
    if isinstance(sample_json_or_text, dict):
        data = sample_json_or_text.get('data')
        if not data:
            return None
        station_delays = data.get('stationDelays') or []
        for s in station_delays:
            a = s.get('averageArrivalDelayMinutes')
            d = s.get('averageDepartureDelayMinutes')
            vals = [v for v in (a, d) if v is not None and not (isinstance(v, float) and _math.isnan(v))]
            if len(vals) == 0:
                continue
            station_means.append(sum(vals) / len(vals))
    elif isinstance(sample_json_or_text, str):
        text = sample_json_or_text
        arr_re = _re.compile(r"averageArrivalDelayMinutes\s*:\s*([-]?[0-9]+(?:\.[0-9]+)?)", _re.IGNORECASE)
        dep_re = _re.compile(r"averageDepartureDelayMinutes\s*:\s*([-]?[0-9]+(?:\.[0-9]+)?)", _re.IGNORECASE)
        arrs = [float(m.group(1)) for m in arr_re.finditer(text)]
        deps = [float(m.group(1)) for m in dep_re.finditer(text)]
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
    mean = float(_pd.Series(station_means).mean())
    std = float(_pd.Series(station_means).std(ddof=0))
    return {'rr_mean': mean, 'rr_std': std, 'station_count': len(station_means)}


def main():
    rows = []
    if not IN_FILE.exists():
        print(f"Input file not found: {IN_FILE}")
        return

    with IN_FILE.open('r', encoding='utf-8') as fh:
        for ln, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                line_fixed = line.replace('NaN', 'null')
                try:
                    entry = json.loads(line_fixed)
                except Exception as e:
                    print(f"Skipping unparsable line {ln}: {e}")
                    continue

            status = entry.get('status')
            if status != 200:
                continue

            train_id = entry.get('train_id')
            if train_id is None:
                # try to expand grouped
                train_ids = entry.get('train_ids')
                if train_ids and isinstance(train_ids, list):
                    for tid in train_ids:
                        parsed = entry.get('parsed') or entry.get('response_json')
                        stats = None
                        if parsed:
                            stats = compute_train_stats(parsed)
                        else:
                            sample_field = entry.get('sample') or entry.get('response_text')
                            sample_json = safe_parse_sample(sample_field)
                            stats = compute_train_stats(sample_json) if sample_json else compute_train_stats(sample_field) if isinstance(sample_field, str) else None
                        if stats:
                            rows.append({'train_id': int(tid), 'rr_mean': stats['rr_mean'], 'rr_std': stats['rr_std'], 'station_count': stats['station_count'], 'endpoint': entry.get('endpoint') or entry.get('endpoints'), 'extracted_at': datetime.utcnow().isoformat() + 'Z'})
                    continue

            parsed = entry.get('parsed') or entry.get('response_json')
            stats = None
            if parsed:
                stats = compute_train_stats(parsed)
            else:
                sample_field = entry.get('sample') or entry.get('response_text')
                sample_json = safe_parse_sample(sample_field)
                if sample_json:
                    stats = compute_train_stats(sample_json)
                elif isinstance(sample_field, str):
                    stats = compute_train_stats(sample_field)

            if not stats:
                if ln <= 10:
                    print(f"No stats for line {ln}; keys: {list(entry.keys())}")
                continue

            rows.append({'train_id': int(train_id), 'rr_mean': stats['rr_mean'], 'rr_std': stats['rr_std'], 'station_count': stats['station_count'], 'endpoint': entry.get('endpoint'), 'extracted_at': datetime.utcnow().isoformat() + 'Z'})

    if len(rows) == 0:
        print('No labels extracted.')
        return

    df = pd.DataFrame(rows).drop_duplicates(subset=['train_id']).sort_values('train_id')
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} labels to {OUT}")
    print(df.head())


if __name__ == '__main__':
    main()
