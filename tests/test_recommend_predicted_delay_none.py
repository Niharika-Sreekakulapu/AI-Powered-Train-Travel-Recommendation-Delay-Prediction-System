import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backend.app as app_module


@pytest.fixture(scope='module')
def client():
    assert app_module.load_model(), "load_model failed"
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as client:
        yield client


def test_recommend_handles_none_predicted_delay(client, monkeypatch):
    # Pick a multi-train route from the enriched master
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    clean_master = os.path.join(repo_root, 'data', 'ap_trains_master_clean_with_delays.csv')
    assert os.path.exists(clean_master), 'Enriched master missing'
    import pandas as pd
    df = pd.read_csv(clean_master)

    route_df = df.groupby(['source', 'destination']).size().reset_index(name='n')
    multi = route_df[route_df['n'] > 1]
    if multi.empty:
        pytest.skip('No multi-train routes found in enriched master')

    row = multi.iloc[0]
    source = row['source']
    destination = row['destination']

    # Monkeypatch predict_delay to simulate a None return (edge case)
    def fake_predict_delay(route, day_of_week, month, distance_km, weather_condition, season):
        return None

    monkeypatch.setattr(app_module, 'predict_delay', fake_predict_delay)

    payload = {
        'source': source,
        'destination': destination,
        'travel_date': '2025-02-01'
    }

    resp = client.post('/api/recommend', json=payload)
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} - {resp.get_data(as_text=True)}"
    data = resp.get_json()
    recs = data.get('recommendations') or []
    assert recs, 'No recommendations returned'

    # Ensure all predicted delays are numeric and non-negative
    for r in recs:
        assert isinstance(r.get('predicted_delay_min'), (int, float)), f"Non-numeric predicted_delay_min: {r.get('predicted_delay_min')}"
        assert r.get('predicted_delay_min') >= 0
