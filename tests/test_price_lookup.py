import os
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backend.app as backend_app


def test_price_lookup_available_and_matches():
    # Ensure model and price lookup loaded
    backend_app.load_model()

    # Find a candidate mapping in the price lookup that also exists in master
    pl = pd.read_csv('datasets/price_lookup_allclasses.csv')
    pl['train_id'] = pl['train_id'].astype(str).str.zfill(5)

    # Pick a sample row where avg_price is not null
    sample = pl[pl['avg_price'].notna()].iloc[0]
    tid = str(sample['train_id']).zfill(5)
    src = str(sample['source_code']).strip().upper()
    dst = str(sample['destination_code']).strip().upper()
    # Expected price should match backend's loaded lookup (may be built from class-filtered file)
    expected_price = backend_app.price_lookup_dict.get((tid, src, dst))

    price_val = backend_app._get_price_for_train_segment(tid, src, dst)
    assert price_val is not None
    assert expected_price is not None
    assert abs(price_val - float(expected_price)) < 0.01
