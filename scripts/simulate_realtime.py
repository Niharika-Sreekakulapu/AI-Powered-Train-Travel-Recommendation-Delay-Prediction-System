"""
Simple synthetic real-time simulator for Train Delay Prediction project.
Produces JSON Lines (one event per line) or posts events to a local endpoint (e.g., http://localhost:8000/api/realtime).

Usage examples:
  python scripts/simulate_realtime.py --days 1 --speed 60 --output events.jsonl
  python scripts/simulate_realtime.py --days 1 --speed 200 --endpoint http://localhost:8000/api/realtime

The script imports backend.app.load_model() to get `train_data` (the merged dataset).
It generates events with fields: event_time, train_id, source, destination, scheduled_time, delay_minutes (simulated), status.
It can inject anomalies based on `--anomaly_rate` and `--anomaly_severity`.
"""

import argparse
import json
import random
import time
from datetime import datetime, timedelta
import requests

# Import backend.app dynamically so we can observe its runtime-updated globals
try:
    import importlib
    backend_app = importlib.import_module('backend.app')
    load_model = backend_app.load_model
except Exception:
    # Try import when running from scripts/ or other working dirs: add repo root to sys.path
    import os
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        import importlib
        backend_app = importlib.import_module('backend.app')
        load_model = backend_app.load_model
    except Exception as e:
        raise ModuleNotFoundError(f"Could not import 'backend.app' even after adding repo root ({repo_root}) to sys.path: {e}")

# Note: do NOT import `train_data` by name at module import time — we access backend_app.train_data dynamically after loading.


def ensure_data_loaded():
    # Ensure backend model and train_data are loaded and accessible via backend_app
    ok = True
    try:
        td = getattr(backend_app, 'train_data', None)
        if td is None or (hasattr(td, 'empty') and td.empty):
            ok = load_model()
    except Exception:
        ok = load_model()

    if not ok:
        raise RuntimeError('Could not load model/train_data via backend.app.load_model()')

    # After load, retrieve train_data from module
    td = getattr(backend_app, 'train_data', None)
    if td is None:
        raise RuntimeError('backend.app.train_data is still None after load_model()')
    return td


def simulate_event_from_row(row, base_datetime):
    """Create a simulated event dict for a given train row and base datetime"""
    train_id = str(row.get('train_id'))
    source = str(row.get('source'))
    destination = str(row.get('destination'))

    # Scheduled time: try departure_time, else random time in the day
    scheduled_time = row.get('departure_time') or row.get('arrival_time')
    if scheduled_time and isinstance(scheduled_time, str) and scheduled_time.strip() and scheduled_time != '--':
        # scheduled_time may be HH:MM
        try:
            hh_mm = scheduled_time.strip().split(':')
            hh = int(hh_mm[0])
            mm = int(hh_mm[1]) if len(hh_mm) > 1 else 0
            sched_dt = base_datetime.replace(hour=hh, minute=mm, second=0, microsecond=0)
        except Exception:
            sched_dt = base_datetime.replace(hour=random.randint(0, 23), minute=random.randint(0, 59), second=0, microsecond=0)
    else:
        sched_dt = base_datetime.replace(hour=random.randint(0, 23), minute=random.randint(0, 59), second=0, microsecond=0)

    # Base delay distribution: depends on distance
    try:
        distance = float(row.get('distance_km') or 100)
    except Exception:
        distance = 100.0

    # mean delay minutes increases with distance and with small random noise
    mean_delay = max(0.0, distance * 0.02)
    # Sample actual delay from a skewed distribution: exponential + small normal noise
    delay = max(0.0, random.expovariate(1 / max(1.0, mean_delay + 1)) + random.gauss(0, mean_delay * 0.1))
    delay = round(delay, 1)

    status = 'on_time' if delay < 5 else ('minor_delay' if delay < 30 else 'major_delay')

    event = {
        'event_time': datetime.utcnow().isoformat(),
        'train_id': train_id,
        'source': source,
        'destination': destination,
        'scheduled_time': sched_dt.isoformat(),
        'distance_km': distance,
        'simulated_delay_min': delay,
        'status': status
    }
    return event


def inject_anomalies(event, severity='high'):
    """Mutate event to represent an anomaly: spike delay"""
    if severity == 'low':
        event['simulated_delay_min'] = round(event['simulated_delay_min'] + random.uniform(10, 30), 1)
    elif severity == 'medium':
        event['simulated_delay_min'] = round(event['simulated_delay_min'] + random.uniform(30, 90), 1)
    else:
        event['simulated_delay_min'] = round(event['simulated_delay_min'] + random.uniform(90, 300), 1)

    event['status'] = 'anomaly'
    return event


def run_simulation(days=1, speed=60, output=None, endpoint=None, anomaly_rate=0.01, anomaly_severity='medium', seed=None):
    """Run simulation for `days` of events. `speed` factor accelerates time (minutes per second).
    If output is a filepath, events are written as JSON Lines. If endpoint is provided, events are POSTed to it.
    """
    if seed is not None:
        random.seed(seed)

    td = ensure_data_loaded()

    # Choose a subset of trains to simulate per day (e.g., 1000 events/day by default)
    all_trains = td.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if all_trains is None or all_trains.empty:
        raise RuntimeError('train_data is empty')

    start_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + timedelta(days=days)

    out_f = open(output, 'w') if output else None

    current_sim_time = start_date
    real_start = time.time()

    try:
        while current_sim_time < end_date:
            # pick a random sample of trains for this simulated minute
            sample_rows = all_trains.sample(n=min(50, len(all_trains)), replace=False)
            for _, row in sample_rows.iterrows():
                ev = simulate_event_from_row(row, current_sim_time)
                # Inject anomalies occasionally
                if random.random() < anomaly_rate:
                    ev = inject_anomalies(ev, severity=anomaly_severity)

                # Output or POST
                if out_f:
                    out_f.write(json.dumps(ev) + '\n')
                if endpoint:
                    try:
                        requests.post(endpoint, json=ev, timeout=5)
                    except Exception as e:
                        print(f"Warning: POST to {endpoint} failed: {e}")
                else:
                    print(json.dumps(ev))

            # advance simulated time by 1 minute
            current_sim_time += timedelta(minutes=1)
            # Sleep scaled by speed: real_sleep_seconds = 60 / speed
            time.sleep(max(0.01, 60.0 / max(1, speed)))
    finally:
        if out_f:
            out_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthetic real-time simulator for train events')
    parser.add_argument('--days', type=float, default=1.0, help='Number of simulated days (can be fractional, e.g., 0.01)')
    parser.add_argument('--speed', type=int, default=60, help='Simulation speed (simulated minutes per real second)')
    parser.add_argument('--output', type=str, default=None, help='Write events to JSONL file')
    parser.add_argument('--endpoint', type=str, default=None, help='POST events to this endpoint (e.g., http://localhost:8000/api/realtime)')
    parser.add_argument('--anomaly_rate', type=float, default=0.01, help='Probability per event to inject an anomaly')
    parser.add_argument('--anomaly_severity', choices=['low', 'medium', 'high'], default='medium')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    run_simulation(days=args.days, speed=args.speed, output=args.output, endpoint=args.endpoint, anomaly_rate=args.anomaly_rate, anomaly_severity=args.anomaly_severity, seed=args.seed)
