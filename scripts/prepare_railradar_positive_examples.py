#!/usr/bin/env python3
"""
Convert `data/railradar_raw_full.jsonl` (endpoint responses grouped by endpoint) into
`data/railradar_positive_examples.jsonl` where each line corresponds to a single train_id
and includes fields compatible with the existing `merge_railradar_delays.py`.

Usage: python scripts/prepare_railradar_positive_examples.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / 'data' / 'railradar_raw_full.jsonl'
OUT = ROOT / 'data' / 'railradar_positive_examples.jsonl'

if not IN.exists():
    print(f"Input not found: {IN}")
    raise SystemExit(1)

OUT.parent.mkdir(parents=True, exist_ok=True)
count = 0
with IN.open('r', encoding='utf-8') as inf, OUT.open('w', encoding='utf-8') as outf:
    for line in inf:
        line=line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        endpoint = e.get('endpoints') or e.get('endpoint')
        status = e.get('status')
        text = e.get('response_text') or ''
        # If a parsed JSON exists, embed it as well
        parsed = e.get('response_json')
        for tid in e.get('train_ids', []) or []:
            out_entry = {
                'train_id': int(tid) if tid is not None and str(tid).isdigit() else tid,
                'status': status,
                'endpoint': endpoint,
                'sample': text,
            }
            # Also attach parsed if available to facilitate merging
            if parsed is not None:
                out_entry['parsed'] = parsed
            outf.write(json.dumps(out_entry, ensure_ascii=False) + '\n')
            count += 1

print(f"Wrote {count} entries to {OUT}")
