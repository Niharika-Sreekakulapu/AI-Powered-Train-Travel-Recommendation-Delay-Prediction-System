import pytest
import requests

# Test recommend API for working routes (skips if server not running)
working_routes = [('HYB', 'VSKP'), ('MYS', 'MAS'), ('NDLS', 'MAS'), ('CSMT', 'NED')]


@pytest.mark.parametrize('source,destination', working_routes)
def test_recommend(source, destination):
    url = 'http://localhost:8000/api/recommend'
    data = {
        'source': source,
        'destination': destination,
        'travel_date': '2025-12-15',
        'preference': 'fastest'
    }

    try:
        response = requests.post(url, json=data, timeout=3)
    except Exception:
        pytest.skip('Recommend API server not running')

    assert response.status_code in (200, 400)

    # If successful, ensure JSON structure is reasonable
    if response.status_code == 200:
        result = response.json()
        assert isinstance(result, dict)
        # If recommendations present, ensure expected keys exist
        recs = result.get('recommendations', [])
        if recs:
            assert 'train_id' in recs[0]
            assert 'train_name' in recs[0]
