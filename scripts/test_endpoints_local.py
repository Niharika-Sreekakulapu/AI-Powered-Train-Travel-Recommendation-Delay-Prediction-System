import sys, os
sys.path.append(os.getcwd())
from backend import app as backend_app
import json

client = backend_app.app.test_client()
# Ensure model loaded
backend_app.load_model()

# Test trains endpoint
resp = client.get('/api/trains?source=HYB&destination=VSKP')
print('trains status', resp.status_code)
print(json.dumps(resp.get_json(), indent=2)[:2000])

# Test predict
resp2 = client.post('/api/predict', json={'source':'HYB','destination':'VSKP','travel_date':'2025-12-15'})
print('predict status', resp2.status_code)
print('predict keys:', list(resp2.get_json().keys()))

# Test recommend
resp3 = client.post('/api/recommend', json={'source':'HYB','destination':'VSKP','travel_date':'2025-12-15','preference':'fastest'})
print('recommend status', resp3.status_code)
print('recommend response:')
print(json.dumps(resp3.get_json(), indent=2))
