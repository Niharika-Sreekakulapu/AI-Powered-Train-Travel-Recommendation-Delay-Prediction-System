import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backend.app as backend_app


def test_delay_floor_and_long_route_magnitude():
    assert backend_app.load_model() is True

    # For a short segment (e.g., 80 km) enforce baseline
    short_dist = 80
    pred_short = backend_app.predict_delay('BZA-MTM', 3, 12, short_dist, 'Clear', 'Winter')
    assert pred_short >= short_dist * backend_app.DELAY_MIN_PER_KM

    # For a long route, expect at least tens of minutes in many cases
    long_dist = 1200
    pred_long = backend_app.predict_delay('NDLS-MAS', 3, 12, long_dist, 'Clear', 'Monsoon')
    assert pred_long >= 20
