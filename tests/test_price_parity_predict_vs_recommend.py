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


def test_price_parity_between_predict_and_recommend(client):
    # Dynamically pick a route that exists by querying /api/trains
    trains_resp = client.get('/api/trains')
    assert trains_resp.status_code == 200, f"Trains endpoint failed: {trains_resp.get_data(as_text=True)}"
    trains = trains_resp.get_json().get('trains') or []
    assert trains, 'No trains available in dataset'

    # Pick first train with non-zero distance
    chosen = None
    for t in trains:
        if int(t.get('distance_km', 0) or 0) > 0:
            chosen = t
            break
    assert chosen, 'No suitable train found for testing'

    payload = {
        'source': chosen['source'],
        'destination': chosen['destination'],
        'travel_date': '2025-12-25'
    }

    # Call recommend
    rec_resp = client.post('/api/recommend', json={**payload, 'preference': 'cheapest'})
    assert rec_resp.status_code == 200, f"Recommend failed: {rec_resp.get_data(as_text=True)}"
    rec_data = rec_resp.get_json()
    recs = rec_data.get('recommendations') or []
    assert recs, 'No recommendations returned'

    # Call predict (no specific train_id; should return all_trains)
    pred_resp = client.post('/api/predict', json=payload)
    assert pred_resp.status_code == 200, f"Predict failed: {pred_resp.get_data(as_text=True)}"
    pred_data = pred_resp.get_json()
    preds = pred_data.get('all_trains') or []
    assert preds, 'No predictions returned'

    # Build dicts by train_id for quick lookup
    rec_map = {str(r['train_id']): r for r in recs}
    pred_map = {str(p['train_id']): p for p in preds}

    # For each train present in both, assert price parity
    common = set(rec_map.keys()) & set(pred_map.keys())
    assert common, 'No common trains between predict and recommend outputs to compare'

    price_mismatches = []
    delay_mismatches = []

    for tid in common:
        r_price = int(rec_map[tid].get('price', 0) or 0)
        p_price = int(pred_map[tid].get('price', 0) or 0)
        if r_price != p_price:
            price_mismatches.append({'train_id': tid, 'recommend_price': r_price, 'predict_price': p_price})

        # Compare delays (rounded to 1 decimal) to match the API output
        r_delay = round(float(rec_map[tid].get('predicted_delay_min', 0) or 0), 1)
        p_delay = round(float(pred_map[tid].get('predicted_delay_min', 0) or 0), 1)
        if r_delay != p_delay:
            delay_mismatches.append({'train_id': tid, 'recommend_delay': r_delay, 'predict_delay': p_delay})

    assert not price_mismatches, f"Price mismatches found for trains: {json.dumps(price_mismatches, indent=2)}"
    assert not delay_mismatches, f"Delay mismatches found for trains: {json.dumps(delay_mismatches, indent=2)}"
