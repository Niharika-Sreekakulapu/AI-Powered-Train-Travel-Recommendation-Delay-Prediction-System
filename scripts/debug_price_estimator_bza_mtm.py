"""Smoke script to print price estimate and source for a sample train/segment (BZA->MTM example).
Run: python scripts/debug_price_estimator_bza_mtm.py
"""
from backend import app as backend_app

if __name__ == '__main__':
    # Example variables - adjust as needed
    train_id = '11464'
    origin = 'BZA'
    dest = 'MTM'
    seg_distance = 80

    price, source = backend_app._estimate_segment_price(train_id, origin, dest, seg_distance, train_price=None)
    print(f"Estimated price for train {train_id} {origin}->{dest} ({seg_distance} km): {price} (source={source})")
