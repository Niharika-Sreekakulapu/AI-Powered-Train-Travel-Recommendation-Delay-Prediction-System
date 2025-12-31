import json
import sys
from backend import app as backend_app


def run_test():
    # Load model and datasets
    loaded = backend_app.load_model()
    if not loaded:
        print('FAIL: Model failed to load')
        sys.exit(2)

    if backend_app.train_data is None or len(backend_app.train_data) == 0:
        print('FAIL: train_data not loaded')
        sys.exit(2)

    # Sample train IDs
    sample_ids = ['19955', '28317', '14065']

    client = backend_app.app.test_client()

    for tid in sample_ids:
        tidz = str(tid).zfill(5)
        # Direct check in DataFrame
        matches = backend_app.train_data[backend_app.train_data['train_id'] == tidz]
        if len(matches) == 0:
            print(f'FAIL: Train {tidz} not found in train_data')
            sys.exit(2)

        # Check API endpoint
        resp = client.get(f"/api/train/{tidz}")
        if resp.status_code != 200:
            print(f'FAIL: API returned {resp.status_code} for train {tidz}')
            sys.exit(2)
        data = json.loads(resp.data)
        if data.get('train_id') != tidz:
            print(f"FAIL: API returned train_id {data.get('train_id')} for train {tidz}")
            sys.exit(2)

    print('PASS: All sample train IDs found and API responded correctly')


if __name__ == '__main__':
    run_test()
