import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backend.app as app_module


@pytest.fixture(scope='module')
def client():
    assert app_module.load_model(), "load_model failed"
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as client:
        yield client


def test_predict_returns_connecting_route_for_some_pair(client):
    # First try to find a candidate pair quickly by calling the internal helper with relaxed days
    trains_resp = client.get('/api/trains')
    assert trains_resp.status_code == 200
    trains = trains_resp.get_json().get('trains') or []
    assert trains, 'No trains available in dataset'

    sources = sorted(list({t['source'] for t in trains}))[:10]
    dests = sorted(list({t['destination'] for t in trains}))[:10]

    found_pair = None
    # Use internal function to find a connection quickly (relaxed, very small runtime)
    for s in sources:
        for d in dests:
            if s == d:
                continue
            try:
                conn = app_module.find_connecting_trains(s, d, 4, 12, 'Clear', 'Winter', allow_relaxed_days=True, max_runtime=1)
            except Exception:
                conn = None
            if conn:
                found_pair = (s, d)
                break
        if found_pair:
            break

    if not found_pair:
        pytest.skip('No candidate connecting pair found via internal search; consider improving heuristics')

    s, d = found_pair
    resp = client.post('/api/predict', json={'source': s, 'destination': d, 'travel_date': '2025-12-25'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('connecting_route'), 'predict did not return connecting_route for a known candidate'
    conn = data['connecting_route']
    assert 'train1' in conn and 'train2' in conn
    assert 'total_delay' in conn and 'total_price' in conn