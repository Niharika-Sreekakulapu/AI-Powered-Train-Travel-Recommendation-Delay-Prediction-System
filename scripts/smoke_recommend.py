import json, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.app import app as flask_app

client = flask_app.test_client()

payload = {
    "source": "NDLS",
    "destination": "LKO",
    "travel_date": "2024-12-01",
    "preference": "cheapest"
}

resp = client.post('/api/recommend', json=payload)
print('status', resp.status_code)
try:
    print(json.dumps(resp.get_json(), indent=2))
except Exception:
    print('Non-JSON response', resp.data)
