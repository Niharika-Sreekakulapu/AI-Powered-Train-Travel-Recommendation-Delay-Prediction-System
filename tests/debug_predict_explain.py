import backend.app as app_module

ok = app_module.load_model()
print('load_model', ok)
app_module.app.config['TESTING'] = True
with app_module.app.test_client() as client:
    # pick a train from /api/trains
    r = client.get('/api/trains')
    trains = r.get_json().get('trains') or []
    if not trains:
        print('No trains found')
    else:
        t = trains[0]
        src = t.get('source')
        dst = t.get('destination')
        print('Calling explain for', src, '->', dst)
        resp = client.post('/api/predict/explain', json={'source': src, 'destination': dst, 'travel_date': '2025-12-30'})
        print('/api/predict/explain status', resp.status_code)
        data = resp.get_json()
        print('Keys:', list(data.keys()))
        print('feature_contributions present at top-level?', 'feature_contributions' in data)
        if 'feature_contributions' in data:
            print(data['feature_contributions'])
        else:
            print('No top-level feature contributions; dumping full response...')
            import json
            print(json.dumps(data, indent=2))
