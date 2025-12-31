import requests
import json

# Test the predict endpoint
def test_predict():
    url = 'http://localhost:8000/api/predict'
    data = {
        'source': 'HYB',
        'destination': 'VSKP',
        'travel_date': '2025-12-15',
        'preference': 'fastest'
    }

    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print("Response:")
        j = response.json()
        print(json.dumps(j, indent=2))
        # Basic assertions: response should contain 'all_trains' and first train should contain conformal fields
        if 'all_trains' in j and len(j['all_trains']) > 0:
            first = j['all_trains'][0]
            assert 'predicted_delay_min' in first
            # Either pred_rr_mean_conf_lower_95 or pred_rr_std_conf_lower_95 should be present when master v6 exists
            if any(k in first for k in ['pred_rr_mean_conf_lower_95','pred_rr_std_conf_lower_95']):
                print('Conformal interval fields present in API response (good)')
            else:
                print('Warning: Conformal interval fields not present in this response')
    except Exception as e:
        print(f"Error: {e}")

# Test the recommend endpoint
def test_recommend():
    url = 'http://localhost:8000/api/recommend'
    data = {
        'source': 'HYB',
        'destination': 'VSKP',
        'travel_date': '2025-12-15',
        'preference': 'fastest'
    }

    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    print("Testing Predict API...")
    test_predict()
    print("\n" + "="*50 + "\n")
    print("Testing Recommend API...")
    test_recommend()
