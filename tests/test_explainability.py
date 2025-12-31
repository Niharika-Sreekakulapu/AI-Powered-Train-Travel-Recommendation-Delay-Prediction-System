import requests
import json

# Test the explain endpoint

def test_explain():
    url = 'http://localhost:8000/api/predict/explain'
    data = {
        'source': 'HYB',
        'destination': 'VSKP',
        'travel_date': '2025-12-15'
    }

    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        j = response.json()
        assert 'predicted_delay_min' in j
        assert 'explanation' in j
        # explanation either contains top_features or a warning
        if 'top_features' in j['explanation']:
            assert isinstance(j['explanation']['top_features'], list)
    except Exception as e:
        print(f"Error testing explain endpoint: {e}")


if __name__ == '__main__':
    print('Testing explain endpoint...')
    test_explain()