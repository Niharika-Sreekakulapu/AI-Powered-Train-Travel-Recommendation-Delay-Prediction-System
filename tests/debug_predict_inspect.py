import backend.app as app_module

print('Loading model...')
ok = app_module.load_model()
print('load_model returned', ok)
app_module.app.config['TESTING'] = True
with app_module.app.test_client() as client:
    r = client.get('/api/trains')
    print('/api/trains status', r.status_code)
    trains = r.get_json().get('trains') or []
    print('Found trains:', len(trains))
    if not trains:
        print('No trains returned; aborting')
    else:
        t = trains[0]
        s = t.get('source')
        d = t.get('destination')
        print('Testing predict for', s, '->', d)
        resp = client.post('/api/predict', json={'source': s, 'destination': d, 'travel_date': '2025-12-30'})
        print('/api/predict status', resp.status_code)
        data = resp.get_json()
        # check top-level fields and all_trains
        keys = list(data.keys())
        print('Top-level keys:', keys)
        print('Has all_trains?', 'all_trains' in data)
        if 'all_trains' in data:
            print('Num all_trains:', len(data['all_trains']))
            sample = data['all_trains'][0]
            print('Sample train keys:', list(sample.keys()))
            print('feature_contributions present?', 'feature_contributions' in sample)
            if 'feature_contributions' in sample:
                print('feature_contributions:', sample['feature_contributions'])
            else:
                print('No feature_contributions in sample; checking best_route')
                if 'best_route' in data and data['best_route']:
                    print('best_route keys:', list(data['best_route'].keys()))
                    print('best_route.feature_contributions?', 'feature_contributions' in data['best_route'])
