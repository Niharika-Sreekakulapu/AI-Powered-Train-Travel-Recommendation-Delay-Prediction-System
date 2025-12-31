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


def test_best_route_not_duplicated_in_recommendations(client):
    # Pick a sample route from /api/trains that has at least one train
    trains_resp = client.get('/api/trains')
    assert trains_resp.status_code == 200
    trains = trains_resp.get_json().get('trains') or []
    assert trains, 'No trains available in dataset'

    # Choose a route with non-zero distance
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

    rec_resp = client.post('/api/recommend', json={**payload, 'preference': 'cheapest'})
    assert rec_resp.status_code == 200, f"Recommend failed: {rec_resp.get_data(as_text=True)}"
    rec_data = rec_resp.get_json()

    best = rec_data.get('best_route')
    recs = rec_data.get('recommendations') or []

    # If a best route exists, ensure it is NOT present in the recommendations list (no duplication)
    if best:
        rec_train_ids = {str(r['train_id']) for r in recs}
        assert str(best['train_id']) not in rec_train_ids, "Best route should not be duplicated in recommendations"