import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backend.app as app_module


def test_analytics_endpoint_returns_data():
    assert app_module.load_model(), "load_model failed"
    client = app_module.app.test_client()
    resp = client.get('/api/analytics/route_trends?source=AGTL&destination=ARCL')
    assert resp.status_code == 200
    json = resp.get_json()
    assert 'delayTrendData' in json
    assert len(json['delayTrendData']) == 12
    assert 'reliabilityData' in json
    assert 'seasonData' in json
    assert 'keyInsights' in json


def test_analytics_endpoint_fallback_to_model_when_no_history():
    assert app_module.load_model(), "load_model failed"
    client = app_module.app.test_client()
    # Use unlikely route codes to trigger no historical data
    resp = client.get('/api/analytics/route_trends?source=XXXX&destination=YYYY')
    assert resp.status_code == 200
    json = resp.get_json()
    assert 'delayTrendData' in json
    assert len(json['delayTrendData']) == 12
    assert 'reliabilityData' in json
    assert 'seasonData' in json
    assert 'keyInsights' in json
    # Ensure keyInsights are present and numeric
    assert isinstance(json['keyInsights'].get('average_delay_min'), (int, float))
    assert 'on_time_percentage' in json['keyInsights']
    # Confirm fallback is model-based and shows month-to-month variation
    assert json.get('model_based', False) is True
    delays = [d['delay'] for d in json['delayTrendData']]
    # There should be some variation after modulation
    assert len(set(delays)) > 1, f"Expected variation in monthly delays, got {delays}"
