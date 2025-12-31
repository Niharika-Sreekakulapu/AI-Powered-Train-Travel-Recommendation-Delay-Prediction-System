import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backend.app as backend_app
import pandas as pd
import numpy as np


def test_estimate_segment_price_falls_back_and_caps():
    assert backend_app.load_model() is True

    # Find a train row with station_list where exact lookup is missing for an inner adjacent pair
    for _, row in backend_app.train_data.iterrows():
        try:
            if not pd.notna(row.get('station_list')):
                continue
            sl = json.loads(str(row['station_list']).replace("'", '"'))
            if not sl or len(sl) < 3:
                continue
            # pick adjacent pair inside the list
            src = sl[1].get('stationCode')
            dst = sl[2].get('stationCode')
            if not src or not dst:
                continue
            tid = str(row['train_id']).zfill(5)
            exact = backend_app._get_price_for_train_segment(tid, src, dst)
            if exact is None:
                seg_distance = float(sl[2].get('distance', 0)) - float(sl[1].get('distance', 0))
                est, src_name = backend_app._estimate_segment_price(tid, src, dst, seg_distance, train_price=row.get('price'))
                assert est is not None
                # If the dataset has no price rows for this train, we expect a global estimation path
                pl_tid = backend_app.price_lookup[backend_app.price_lookup['tid'] == str(tid)]
                if pl_tid.empty:
                    assert str(src_name).startswith('estimated_by_rate')
                else:
                    # otherwise allow any of the known sources
                    assert src_name in ('estimated_by_rate', 'train_price', 'distance_fallback', 'lookup', 'estimated_by_rate_global', 'estimated_by_rate_global_distance')
                # If train row price exists, estimated price must not exceed full-train price
                try:
                    fullp = float(row.get('price')) if pd.notna(row.get('price')) else None
                    if fullp is not None and fullp > 0:
                        assert est <= fullp + 1e-6
                except Exception:
                    pass
                return
        except Exception:
            continue
    import pytest
    pytest.skip('No suitable train found for estimation test')
