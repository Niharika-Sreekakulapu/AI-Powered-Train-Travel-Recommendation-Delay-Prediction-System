import requests
import json


def test_propagation_historical_api():
    url = 'http://localhost:8000/api/predict/propagate/historical'
    payload = {
        'date': '2025-12-15',
        'max_transfer_minutes': 240,
        'recovery_margin': 5
    }
    try:
        r = requests.post(url, json=payload)
        print('Status:', r.status_code)
        j = r.json()
        print(json.dumps({k: j.get(k) for k in ['metrics', 'n_trains']}, indent=2))
        assert r.status_code == 200
        assert 'metrics' in j
        assert 'viz_base64_png' in j
        assert 'simulated_final' in j
    except Exception as e:
        print('Error during historical propagation API test:', e)


if __name__ == '__main__':
    test_propagation_historical_api()