import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backend.app as app_module


def test_find_connecting_trains_is_fast():
    assert app_module.load_model(), "load_model failed"
    start = time.time()
    conn = app_module.find_connecting_trains('AGTL', 'ARCL', 4, 12, 'Clear', 'Winter', allow_relaxed_days=True, max_runtime=6)
    elapsed = time.time() - start
    assert conn is not None, 'Expected to find a connecting route'
    assert elapsed < 10, f'find_connecting_trains too slow: {elapsed:.1f}s'