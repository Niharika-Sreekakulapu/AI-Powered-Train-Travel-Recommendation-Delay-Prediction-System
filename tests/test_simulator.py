import os
import sys
import json
import pytest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts import simulate_realtime as sim


def test_run_simulation_writes_file(tmp_path, monkeypatch):
    # Disable sleep to speed up test
    monkeypatch.setattr(sim.time, 'sleep', lambda s: None)

    out_file = tmp_path / "events.jsonl"

    # Patch requests.post to capture calls
    posted = []

    def fake_post(url, json=None, timeout=5):
        posted.append((url, json))
        class R: pass
        return R()

    monkeypatch.setattr(sim.requests, 'post', fake_post)

    # Run simulation for a tiny fraction of a day
    sim.run_simulation(days=0.001, speed=600, output=str(out_file), endpoint='http://localhost:8000/api/realtime', seed=123, anomaly_rate=0.0)

    # Output file exists and has at least one line
    assert out_file.exists()
    with open(out_file, 'r') as fh:
        lines = [l.strip() for l in fh if l.strip()]
    assert len(lines) > 0

    # Ensure we posted at least once
    assert len(posted) > 0


def test_inject_anomalies_changes_delay():
    ev = {'simulated_delay_min': 5.0, 'status': 'on_time'}
    ev2 = sim.inject_anomalies(ev.copy(), severity='high')
    assert ev2['simulated_delay_min'] >= 5.0
    assert ev2['status'] == 'anomaly'
