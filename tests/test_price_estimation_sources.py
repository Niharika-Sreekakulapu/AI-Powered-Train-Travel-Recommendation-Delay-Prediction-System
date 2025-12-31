import pandas as pd
import numpy as np
import os
import pytest

from backend import app as backend_app

PRICE_LOOKUP_NAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', 'price_lookup.csv')


def make_df(rows):
    return pd.DataFrame(rows, columns=['train_id', 'source_code', 'destination_code', 'avg_price', 'avg_distance'])


def fake_read_csv_factory(df_map):
    """Return a fake pd.read_csv that returns a DataFrame when path endswith 'price_lookup.csv' and otherwise delegates to real read_csv."""
    real_read = pd.read_csv

    def fake_read(path, *args, **kwargs):
        if str(path).endswith('price_lookup.csv'):
            # Return copy to allow in-test mutation
            return df_map.copy()
        return real_read(path, *args, **kwargs)

    return fake_read


def test_estimated_by_rate_with_sufficient_candidates(monkeypatch, tmp_path):
    # Create a synthetic price_lookup with multiple candidate rates for train 12345
    rows = [
        ['12345', 'BZA', 'XXX', 100.0, 50.0],
        ['12345', 'YYY', 'ZZZ', 130.0, 65.0],
        ['12345', 'AAA', 'BBB', 90.0, 45.0],
        ['99999', 'BZA', 'MTM', 500.0, 300.0],  # other train row
    ]
    df = make_df(rows)

    monkeypatch.setattr(pd, 'read_csv', fake_read_csv_factory(df))
    monkeypatch.setattr(os.path, 'exists', lambda p: True)

    price, source = backend_app._estimate_segment_price('12345', 'BZA', 'MTM', 80, train_price=1000)
    assert source.startswith('estimated_by_rate')
    assert price <= 1000


def test_fallback_to_train_price_with_insufficient_candidates(monkeypatch):
    # Only one candidate row for train 22222; should prefer train_price
    rows = [
        ['22222', 'BZA', 'CCC', 120.0, 40.0],
    ]
    df = make_df(rows)

    monkeypatch.setattr(pd, 'read_csv', fake_read_csv_factory(df))
    monkeypatch.setattr(os.path, 'exists', lambda p: True)

    price, source = backend_app._estimate_segment_price('22222', 'BZA', 'MTM', 80, train_price=800)
    # With insufficient per-train candidates we prefer a conservative fraction of the full train price
    assert source == 'train_price_fraction_fallback'
    assert price == 200


def test_global_estimate_when_no_train_rows(monkeypatch):
    # No rows for train 33333, but station-based matches exist
    rows = [
        ['44444', 'BZA', 'XYZ', 60.0, 30.0],
        ['44444', 'MTM', 'ABC', 100.0, 50.0],
        ['55555', 'BZA', 'DEF', 80.0, 40.0],
        ['66666', 'GHI', 'JKL', 70.0, 35.0],
    ]
    df = make_df(rows)

    monkeypatch.setattr(pd, 'read_csv', fake_read_csv_factory(df))
    monkeypatch.setattr(os.path, 'exists', lambda p: True)

    price, source = backend_app._estimate_segment_price('33333', 'BZA', 'MTM', 45, train_price=None)
    # Could be estimated_by_rate_global or distance-based depending on implementation; accept either estimated_by_rate_global* or distance-based
    assert source.startswith('estimated_by_rate') or source == 'distance_fallback'
    assert price > 0


def test_reversed_lookup_scaling(monkeypatch):
    # Price exists only in the opposite direction (MTM->BZA). The estimator should use the reversed entry and scale by distance.
    rows = [
        ['12345', 'MTM', 'BZA', 100.0, 50.0],  # rate = 2.0
    ]
    df = make_df(rows)
    monkeypatch.setattr(pd, 'read_csv', fake_read_csv_factory(df))
    monkeypatch.setattr(os.path, 'exists', lambda p: True)

    # Load model to build caches
    assert backend_app.load_model() is True

    price, source = backend_app._estimate_segment_price('12345', 'BZA', 'MTM', 25, train_price=None)
    assert source == 'lookup_reversed'
    assert abs(price - 50.0) < 1e-6


def test_global_rate_fallback(monkeypatch):
    # No rows for the target train; global median rate should be used
    rows = [
        ['44444', 'AAA', 'CCC', 120.0, 40.0],  # rate = 3.0
        ['55555', 'DDD', 'EEE', 80.0, 40.0],   # rate = 2.0
    ]
    df = make_df(rows)
    monkeypatch.setattr(pd, 'read_csv', fake_read_csv_factory(df))
    monkeypatch.setattr(os.path, 'exists', lambda p: True)

    # Load model to compute PRICE_GLOBAL_RATE
    assert backend_app.load_model() is True

    # global median rate = median([3.0,2.0]) = 2.5
    price, source = backend_app._estimate_segment_price('22222', 'XXX', 'YYY', 40, train_price=None)
    assert source == 'estimated_by_global_rate'
    assert abs(price - (2.5 * 40)) < 1e-6

