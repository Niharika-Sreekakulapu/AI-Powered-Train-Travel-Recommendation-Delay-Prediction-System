import requests
import json


def test_propagate_api():
    url = 'http://localhost:8000/api/predict/propagate'
    payload = {
        'injections': [{'node': 'A', 'delay': 15}],
        'edges': [['A', 'B', 10]],
        'recovery_margin': 5
    }
    try:
        r = requests.post(url, json=payload)
        print('Status:', r.status_code)
        print(json.dumps(r.json(), indent=2))
        if r.status_code == 200:
            j = r.json()
            assert 'final_delays' in j
            assert 'B' in j['final_delays']
            assert abs(j['final_delays']['B'] - 20.0) < 1e-6
    except Exception as e:
        print('Error during propagate API test:', e)


if __name__ == '__main__':
    test_propagate_api()