#!/usr/bin/env python3
"""
Fetch full RailRadar payloads for trains that previously returned status 200
in `data/railradar_coverage_report.csv` and save them to
`data/railradar_raw_full.jsonl` (one JSON object per line).

Usage: python scripts/fetch_railradar_full.py
"""
import csv
import json
import time
from pathlib import Path
from datetime import datetime
import requests

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / 'data' / 'railradar_coverage_report.csv'
OUT = ROOT / 'data' / 'railradar_raw_full.jsonl'

RATE_LIMIT = 0.2  # seconds between requests
TIMEOUT = 8

session = requests.Session()

if not COVERAGE.exists():
    print(f"Coverage report not found: {COVERAGE}")
    raise SystemExit(1)

rows = []
with COVERAGE.open('r', encoding='utf-8') as fh:
    reader = csv.DictReader(fh)
    for r in reader:
        # choose rows where found is True and status == 200 and endpoint present
        try:
            found = str(r.get('found', '')).strip().lower() in ('true', '1', 'yes')
            status = int(r.get('status') or 0)
        except Exception:
            found = False
            status = 0
        endpoint = r.get('endpoint') or ''
        tid = r.get('train_id')
        if found and status == 200 and endpoint:
            rows.append({'train_id': tid, 'endpoint': endpoint})

print(f"Will fetch {len(rows)} endpoints")

# deduplicate by endpoint (but keep train_id mapping for metadata)
by_endpoint = {}
for r in rows:
    ep = r['endpoint']
    if ep not in by_endpoint:
        by_endpoint[ep] = []
    by_endpoint[ep].append(r['train_id'])

OUT.parent.mkdir(parents=True, exist_ok=True)
written = 0
errors = []
with OUT.open('w', encoding='utf-8') as outfh:
    for endpoint, tids in by_endpoint.items():
        try:
            print(f"Fetching {endpoint} (trains: {', '.join(tids)})")
            r = session.get(endpoint, timeout=TIMEOUT)
            fetched_at = datetime.utcnow().isoformat() + 'Z'
            entry = {
                'endpoints': endpoint,
                'train_ids': tids,
                'status': r.status_code,
                'fetched_at': fetched_at,
            }
            text = r.text
            entry['response_text'] = text
            try:
                entry['response_json'] = r.json()
            except Exception:
                entry['response_json'] = None
            outfh.write(json.dumps(entry, ensure_ascii=False) + '\n')
            written += 1
        except Exception as e:
            print(f"Error fetching {endpoint}: {e}")
            errors.append({'endpoint': endpoint, 'error': str(e)})
        time.sleep(RATE_LIMIT)

print(f"Wrote {written} entries to {OUT}")
if errors:
    print(f"Encountered {len(errors)} errors; sample: {errors[:3]}")

# Also write a summary JSON
summary = {
    'fetched_count': written,
    'errors': len(errors),
    'generated_at': datetime.utcnow().isoformat() + 'Z'
}
with (OUT.parent / 'railradar_raw_full_summary.json').open('w', encoding='utf-8') as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2)

print('Done.')
