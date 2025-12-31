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


def test_recommend_variability(client):
    # Find a route with multiple trains
    # Use cleaned master to pick source/destination
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    clean_master = os.path.join(repo_root, 'data', 'ap_trains_master_clean_with_delays.csv')
    assert os.path.exists(clean_master), 'Enriched master missing'
    import pandas as pd
    df = pd.read_csv(clean_master)

    # Pick a route where there are multiple trains with same source/destination
    route_df = df.groupby(['source', 'destination']).size().reset_index(name='n')
    multi = route_df[route_df['n'] > 1]
    if multi.empty:
        pytest.skip('No multi-train routes found in enriched master')

    row = multi.iloc[0]
    source = row['source']
    destination = row['destination']

    payload = {
        'source': source,
        'destination': destination,
        'travel_date': '2025-02-01'
    }

    resp = client.post('/api/recommend', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    trains = data.get('recommendations') or data.get('trains') or []
    if not trains or len(trains) < 2:
        pytest.skip('Not enough trains returned for this route')

    delays = [t['predicted_delay_min'] for t in trains]
    # Assert that not all delays are identical
    assert len(set(delays)) > 1, f"All predicted delays are identical: {set(delays)}"

    # Also assert speed varies across trains (not all the same)
    speeds = [t['speed_kmph'] for t in trains]
    assert len(set(speeds)) > 1, f"All speeds are identical: {set(speeds)}"

    # Assert prices are reasonable and at least match a train_data price for returned train_ids
    train_ids = [t['train_id'] for t in trains]
    prices = [t['price'] for t in trains]
    df_subset = df[df['train_id'].isin(train_ids)]
    for tid, price in zip(train_ids, prices):
        # If train exists in master, price should match one of its reported prices
        m = df_subset[df_subset['train_id'] == tid]
        if not m.empty and m['price'].notna().any():
            assert int(price) in set(m['price'].dropna().astype(int).unique()), f"Price for {tid} ({price}) not found in master"