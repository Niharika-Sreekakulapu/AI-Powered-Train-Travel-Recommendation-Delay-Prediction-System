import json
import sys
import os
sys.path.append(os.getcwd())
from backend import app as backend_app


def test_trains_pass_through():
    backend_app.load_model()
    client = backend_app.app.test_client()

    resp = client.get('/api/trains?source=HYB&destination=VSKP')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'trains' in data
    assert data['total'] >= 2, f"Expected multiple trains, got {data['total']}" 


if __name__ == '__main__':
    test_trains_pass_through()
    print('PASS')
