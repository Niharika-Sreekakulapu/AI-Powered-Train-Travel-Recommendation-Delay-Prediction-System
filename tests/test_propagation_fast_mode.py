import json
import os, sys
# Ensure project root is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import importlib
app_mod = importlib.import_module('backend.app')
app = app_mod.app


def test_propagation_historical_fast_mode():
    import pandas as pd
    from datetime import datetime

    # prepare a small synthetic train_data so the endpoint doesn't rely on full dataset
    date_str = '2025-12-15'
    dow = datetime.strptime(date_str, '%Y-%m-%d').weekday() + 1
    rows = []
    for i in range(1, 501):
        rows.append({
            'train_id': str(10000 + i),
            'source': 'AAA',
            'destination': 'BBB',
            'distance_km': 50,
            'avg_delay_min': 0.0,
            'departure_time': '08:00',
            'arrival_time': '09:00',
            'day_of_week': dow,
            'station_list': 'AAA,BBB'
        })
    df = pd.DataFrame(rows)
    # inject lightweight dataset and prediction function
    app_mod.train_data = df
    app_mod.predict_delay_cached = lambda *args, **kwargs: 5.0

    client = app.test_client()
    payload = {'date': date_str, 'mode': 'fast', 'sample_n': 200}
    r = client.post('/api/predict/propagate/historical', json=payload)
    assert r.status_code == 200, r.data
    j = r.get_json()
    assert j.get('mode') == 'fast'
    assert 'mode_params' in j
    assert j.get('n_trains_before') >= j.get('n_trains_after')
    assert j.get('n_trains_after') <= 200
    assert 'metrics' in j
    assert 'viz_base64_png' in j
    assert len(j.get('viz_base64_png') or '') > 0
