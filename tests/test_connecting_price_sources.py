import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backend.app as app_module


def test_connecting_route_per_leg_prices_and_sources():
    # Ensure model and data are loaded
    assert app_module.load_model(), "load_model failed"
    assert app_module.train_data is not None

    # Known pair (from prior debug logs) that yields a connection
    conn = app_module.find_connecting_trains('AGTL', 'ARCL', 4, 12, 'Clear', 'Winter', allow_relaxed_days=True, max_runtime=4)
    assert conn is not None, 'No connecting route found for AGTL -> ARCL'

    # Get train ids and legs
    t1 = conn['train1']
    t2 = conn['train2']

    tid1 = str(t1['train_id'])
    tid2 = str(t2['train_id'])

    # Lookup train rows to provide station_list and train-level price
    row1 = app_module.train_data[app_module.train_data['train_id'] == tid1]
    row2 = app_module.train_data[app_module.train_data['train_id'] == tid2]

    station_list1 = row1.iloc[0].get('station_list') if not row1.empty else None
    station_list2 = row2.iloc[0].get('station_list') if not row2.empty else None

    train_price1 = row1.iloc[0].get('price') if not row1.empty else None
    train_price2 = row2.iloc[0].get('price') if not row2.empty else None

    # Compute expected price and source for leg1
    p1 = app_module._get_price_for_train_segment(tid1, t1['source'], t1['destination'], station_list1)
    if p1 is not None:
        expected_p1 = int(round(float(p1)))
        expected_src1 = 'lookup'
    else:
        est1, expected_src1 = app_module._estimate_segment_price(tid1, t1['source'], t1['destination'], t1['distance_km'], train_price=train_price1)
        expected_p1 = int(round(float(est1 or 0)))

    # Compute expected price and source for leg2
    p2 = app_module._get_price_for_train_segment(tid2, t2['source'], t2['destination'], station_list2)
    if p2 is not None:
        expected_p2 = int(round(float(p2)))
        expected_src2 = 'lookup'
    else:
        est2, expected_src2 = app_module._estimate_segment_price(tid2, t2['source'], t2['destination'], t2['distance_km'], train_price=train_price2)
        expected_p2 = int(round(float(est2 or 0)))

    # Assert per-leg prices match expected
    assert int(t1['price']) == expected_p1, f"Train1 price mismatch: got {t1['price']} expected {expected_p1}"
    assert int(t2['price']) == expected_p2, f"Train2 price mismatch: got {t2['price']} expected {expected_p2}"

    # Assert price_source fields exist and match expected source labels
    assert 'price_source' in t1 and t1['price_source'] == expected_src1
    assert 'price_source' in t2 and t2['price_source'] == expected_src2
