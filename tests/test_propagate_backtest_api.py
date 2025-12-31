import requests
import json


def test_propagate_backtest_api():
    url = 'http://localhost:8000/api/predict/propagate/backtest'
    payload = {
        'injections': [{'node': 'A', 'delay': 15}],
        'edges': [['A', 'B', 10], ['B', 'C', 5]],
        'observed_final': {'A':15, 'B':20, 'C':20},
        'recovery_margin': 5
    }
    try:
        r = requests.post(url, json=payload)
        print('Status:', r.status_code)
        j = r.json()
        print(json.dumps(j, indent=2))
        assert r.status_code == 200
        assert 'metrics' in j
        assert abs(j['metrics']['mae']) < 1e-6
        assert 'viz_base64_png' in j
    except Exception as e:
        print('Backtest API test failed:', e)


if __name__ == '__main__':
    test_propagate_backtest_api()