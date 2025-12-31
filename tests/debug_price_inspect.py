import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backend.app as app_module

assert app_module.load_model()
app_module.app.config['TESTING'] = True
client = app_module.app.test_client()

tr_resp = client.get('/api/trains')
tr = tr_resp.get_json().get('trains', [])
chosen = None
for t in tr:
    if int(t.get('distance_km', 0) or 0) > 0:
        chosen = t
        break
print('Chosen route:', chosen)
payload = {'source': chosen['source'], 'destination': chosen['destination'], 'travel_date': '2025-12-25'}
rec = client.post('/api/recommend', json={**payload, 'preference': 'cheapest'}).get_json()
pred = client.post('/api/predict', json=payload).get_json()
print('\nRECOMMENDATIONS:\n', json.dumps(rec, indent=2)[:4000])
print('\nPREDICTIONS:\n', json.dumps(pred, indent=2)[:4000])
# Dump mismatches
recs = rec.get('recommendations') or []
preds = pred.get('all_trains') or []
rec_map = {str(r['train_id']): r for r in recs}
pred_map = {str(p['train_id']): p for p in preds}
common = set(rec_map.keys()) & set(pred_map.keys())
print('\nCOMMON TRAIN IDS:', common)
for tid in common:
    print('---', tid)
    print('RECOMMEND:', rec_map[tid])
    print('PREDICT :', pred_map[tid])
