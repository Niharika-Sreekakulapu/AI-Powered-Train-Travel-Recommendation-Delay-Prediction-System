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


def test_available_days_for_existing_route(client):
    # Pick a real route by querying /api/trains
    trains_resp = client.get('/api/trains')
    assert trains_resp.status_code == 200
    trains = trains_resp.get_json().get('trains') or []
    assert trains, 'No trains available in dataset'

    chosen = None
    for t in trains:
        if int(t.get('distance_km', 0) or 0) > 0:
            chosen = t
            break
    assert chosen, 'No suitable train found for testing'

    src = chosen['source']
    dst = chosen['destination']

    resp = client.get(f'/api/available_days?source={src}&destination={dst}')
    assert resp.status_code == 200, f"Available days endpoint failed: {resp.get_data(as_text=True)}"
    data = resp.get_json()
    assert 'available_days' in data, 'available_days missing from response'
    # We expect at least one available day for an existing route
    assert len(data['available_days']) >= 1, 'Expected at least one available day for an existing route'