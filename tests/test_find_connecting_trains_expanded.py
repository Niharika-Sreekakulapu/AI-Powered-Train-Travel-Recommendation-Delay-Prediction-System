import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backend.app as app_module


def test_internal_expanded_search_returns_connection():
    # Ensure model and data are loaded (fast) before invoking internal helper
    assert app_module.load_model(), "load_model failed"
    assert app_module.train_data is not None

    # Use a pair that debug logs showed earlier as having a connection (AGTL -> ARCL)
    # Give the expanded search sufficient runtime for the dataset on slower dev machines
    conn = app_module.find_connecting_trains('AGTL', 'ARCL', 4, 12, 'Clear', 'Winter', allow_relaxed_days=True, max_runtime=8)
    assert conn is not None, 'Expanded internal search did not return a connection for AGTL -> ARCL'
    assert 'train1' in conn and 'train2' in conn
    assert 'total_delay' in conn and 'total_distance' in conn