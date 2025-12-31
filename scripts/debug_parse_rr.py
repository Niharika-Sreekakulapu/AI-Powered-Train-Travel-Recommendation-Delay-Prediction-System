import json
from pathlib import Path
p=Path('data/railradar_positive_examples.jsonl')
text=p.read_text(encoding='utf-8')
lines=text.splitlines()
for i,l in enumerate(lines[:30]):
    print('LINE',i+1)
    try:
        entry=json.loads(l)
    except Exception as e:
        print('json.loads entry failed', e)
        continue
    print('status',entry.get('status'),'train_id',entry.get('train_id'))
    s=entry.get('sample')
    if s is None:
        print('no sample')
        continue
    print('sample type',type(s))
    # try parse
    try:
        sj=json.loads(s)
        print('parsed sample data keys', list(sj.keys())[:5])
        if 'data' in sj and 'stationDelays' in sj['data']:
            print('has stationDelays count', len(sj['data']['stationDelays']))
    except Exception as e:
        print('failed parsing sample', e)

print('done')
