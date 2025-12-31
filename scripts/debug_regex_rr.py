import json, re
from pathlib import Path
p=Path('data/railradar_positive_examples.jsonl')
lines=p.read_text(encoding='utf-8').splitlines()
for i,l in enumerate(lines[:6]):
    entry=json.loads(l.replace('NaN','null'))
    sample=entry.get('sample')
    print('\nLINE',i+1,'train',entry.get('train_id'))
    if not isinstance(sample,str):
        print('sample not str')
        continue
    arr_re = re.compile(r"averageArrivalDelayMinutes\s*:\s*([-]?[0-9]+(?:\.[0-9]+)?)")
    dep_re = re.compile(r"averageDepartureDelayMinutes\s*:\s*([-]?[0-9]+(?:\.[0-9]+)?)")
    arrs = [float(m.group(1)) for m in arr_re.finditer(sample)]
    deps = [float(m.group(1)) for m in dep_re.finditer(sample)]
    print('arrs',arrs[:10])
    print('deps',deps[:10])
    print('sample head:', sample[:300])
    m=arr_re.search(sample)
    if m:
        start=max(0,m.start()-80)
        print('context',sample[start:m.start()+80])
print('done')
