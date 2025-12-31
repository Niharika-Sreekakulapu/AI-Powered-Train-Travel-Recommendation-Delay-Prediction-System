import sys, os, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backend.app as backend_app
from datetime import datetime

backend_app.load_model()

source='BZA'
destination='MTM'
travel_date='2025-12-15'
try:
    date_obj = datetime.strptime(travel_date, '%Y-%m-%d')
    day_of_week = date_obj.weekday() + 1
    month = date_obj.month
except Exception as e:
    print('bad date', e)

available_trains = backend_app.get_trains_passing_through(source, destination, day_of_week)

for idx, train in available_trains.iterrows():
    try:
        print('\n--- Row', idx, 'train_id', train.get('train_id'))
        train_distance = float(train.get('distance_km') or 0)
        train_price = train.get('price')

        departure_time = train.get('departure_time')
        arrival_time = train.get('arrival_time')

        # predict
        pred = backend_app.predict_delay(f"{source}-{destination}", day_of_week, month, train_distance, 'Clear', 'Winter')
        print('predicted_delay', pred)

        # price estimate using helper
        # compute seg_distance from station_list if available
        station_list_str = train.get('station_list')
        seg_distance = None
        if station_list_str:
            try:
                sl = json.loads(str(station_list_str).replace("'", '"'))
                # find origin/dest distances
                origin_dist = None
                dest_dist = None
                for s in sl:
                    code = s.get('stationCode','').strip().upper()
                    if code == source:
                        origin_dist = float(s.get('distance', 0) or 0)
                    if code == destination:
                        dest_dist = float(s.get('distance', 0) or 0)
                if origin_dist is not None and dest_dist is not None:
                    seg_distance = dest_dist - origin_dist
            except Exception:
                pass

        est, src_name = backend_app._estimate_segment_price(str(train.get('train_id')).zfill(5), source, destination, seg_distance if seg_distance is not None else train_distance, train_price=train_price)
        print('price_est', est, 'source', src_name)

    except Exception as e:
        print('Exception for row', idx, e)
        traceback.print_exc()
        raise
print('\ndone')
