import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.app import app as flask_app

flask_app.testing = True
client = flask_app.test_client()

payload = {
    'source': 'BZA',
    'destination': 'MTM',
    'travel_date': '2025-12-15',
    'preference': 'cheapest'
}

try:
    resp = client.post('/api/recommend', json=payload)
    print('status', resp.status_code)
    print(json.dumps(resp.get_json(), indent=2))
except Exception as e:
    import traceback
    print('Exception calling /api/recommend:', e)
    traceback.print_exc()
