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


def test_predict_returns_feature_contributions(client):
    # pick a route from /api/trains
    r = client.get('/api/trains')
    assert r.status_code == 200
    trains = r.get_json().get('trains') or []
    assert trains
    t = trains[0]
    src = t.get('source')
    dst = t.get('destination')

    resp = client.post('/api/predict', json={'source': src, 'destination': dst, 'travel_date': '2025-12-30'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'all_trains' in data
    assert isinstance(data['all_trains'], list)
    # At least one train in all_trains should have feature_contributions
    has = any('feature_contributions' in tr and tr['feature_contributions'] for tr in data['all_trains'])
    assert has, 'No feature_contributions found in any returned train' 
