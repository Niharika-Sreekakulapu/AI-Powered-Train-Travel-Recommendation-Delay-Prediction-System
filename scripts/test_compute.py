from scripts.merge_railradar_delays import compute_train_stats, safe_parse_sample
import json
from pathlib import Path
p=Path('data/railradar_positive_examples.jsonl')
lines=p.read_text(encoding='utf-8').splitlines()
for i,l in enumerate(lines[:10]):
    entry=json.loads(l.replace('NaN','null'))
    sample=entry.get('sample')
    print('\nLINE',i+1,'train',entry.get('train_id'))
    parsed=safe_parse_sample(sample)
    print('safe_parse_sample returned', type(parsed))
    stats1=None
    stats2=None
    if parsed:
        stats1=compute_train_stats(parsed)
    else:
        print('parsed is None')
    if isinstance(sample,str):
        stats2=compute_train_stats(sample)
    print('stats1',stats1)
    print('stats2',stats2)
