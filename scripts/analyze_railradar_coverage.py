#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import json

csv_in = Path('data/railradar_coverage_report.csv')
print(f"Reading {csv_in}")
df = pd.read_csv(csv_in)

total = len(df)
found = df['found'].sum()
percent = found / total * 100

print(f"Total trains: {total}")
print(f"Covered by RailRadar: {found} ({percent:.2f}%)")

# endpoints summary
endpoints = df[df['found']]['endpoint'].value_counts()
print('\nEndpoints used (found results):')
print(endpoints.to_string())

# top matched_train
if 'matched_train' in df.columns:
    top_matched = df[df['matched_train']!='']['matched_train'].value_counts().head(10)
    print('\nTop matched train numbers:')
    print(top_matched.to_string())

# examples
print('\nRepresentative examples (up to 5):')
examples = df[df['found']].head(5)
for idx, row in examples.iterrows():
    print('\n---')
    print(f"train_id: {row['train_id']}")
    print(f"matched: {row['matched_train']}")
    print(f"endpoint: {row['endpoint']}")
    print(f"status: {row['status']}")
    if isinstance(row['sample'], str) and row['sample'].strip():
        sample = row['sample']
        # try to pretty print JSON part if possible
        try:
            j = json.loads(sample)
            print('sample JSON keys:', list(j.keys())[:10])
        except Exception:
            print('sample (truncated):', sample[:400])
    else:
        print('sample: (empty)')

# Save a small JSON lines examples for quick inspection
out_examples = Path('data/railradar_positive_examples.jsonl')
with out_examples.open('w', encoding='utf-8') as f:
    for idx, row in df[df['found']].iterrows():
        rec = {
            'train_id': row['train_id'],
            'matched': row['matched_train'],
            'endpoint': row['endpoint'],
            'status': row['status'],
            'message': row['message'],
            'sample': row['sample']
        }
        f.write(json.dumps(rec) + '\n')

print(f"\nWrote positive examples to {out_examples} (one JSON per line)")
