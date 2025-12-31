import json
import pytest
from backend import app as backend_app


def test_ap_train_ids_loaded_and_searchable():
    # Load model and datasets
    loaded = backend_app.load_model()
    assert loaded is True, "Model failed to load"

    assert backend_app.train_data is not None
    assert len(backend_app.train_data) > 0

    # Sample train IDs from ap_trains_final11.csv (present in repo data)
    sample_ids = ['19955', '28317', '14065']

    client = backend_app.app.test_client()

    for tid in sample_ids:
        tidz = str(tid).zfill(5)
        # Direct check in DataFrame
        matches = backend_app.train_data[backend_app.train_data['train_id'] == tidz]
        assert len(matches) > 0, f"Train {tidz} not found in train_data"

        # Check API endpoint
        resp = client.get(f"/api/train/{tidz}")
        assert resp.status_code == 200, f"API returned {resp.status_code} for train {tidz}"
        data = json.loads(resp.data)
        assert data['train_id'] == tidz
