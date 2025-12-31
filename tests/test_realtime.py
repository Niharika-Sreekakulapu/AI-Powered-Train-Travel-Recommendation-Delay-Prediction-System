import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backend.app as app_module


@pytest.fixture(scope='module')
def client():
    # Ensure model and train_data loaded
    assert app_module.load_model(), "load_model failed"
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as client:
        yield client


def test_realtime_endpoint_basic(client):
    # Pick a train from cleaned master if available
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    clean_master = os.path.join(repo_root, 'data', 'ap_trains_master_clean.csv')
    if os.path.exists(clean_master):
        import pandas as pd
        df = pd.read_csv(clean_master)
        row = df.iloc[0]
        source = row['source'] if 'source' in row else row['source_code']
        destination = row['destination'] if 'destination' in row else row['destination_code']
    else:
        # fallback to simple codes
        source = 'BZA'
        destination = 'VSKP'

    ev = {
        'event_time': '2025-01-01T00:00:00Z',
        'train_id': str( row.get('train_id') ) if os.path.exists(clean_master) else '99999',
        'source': source,
        'destination': destination,
        'scheduled_time': '2025-01-01T10:00:00Z',
        'distance_km': 100.0,
        'simulated_delay_min': 5.0,
        'status': 'on_time'
    }

    resp = client.post('/api/realtime', json=ev)
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'predicted_delay_min' in data
    assert isinstance(data['predicted_delay_min'], (int, float))
    # Ensure route and weather info are present
    assert 'route' in data
    assert 'weather' in data


def test_realtime_invalid_input(client):
    resp = client.post('/api/realtime', json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'predicted_delay_min' in data
