import sys, os, json, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backend.app as backend_app
from datetime import datetime

backend_app.load_model()

payload = {'source': 'BZA', 'destination': 'MTM', 'travel_date': '2025-12-15', 'preference': 'cheapest'}

source = payload.get('source','').strip().upper()
destination = payload.get('destination','').strip().upper()
travel_date = payload.get('travel_date','')
try:
    date_obj = datetime.strptime(travel_date, '%Y-%m-%d')
    day_of_week = date_obj.weekday() + 1
    month = date_obj.month
except Exception as e:
    print('Bad date', e)
    raise

print('inputs:', source, destination, day_of_week, month)

try:
    available_trains = backend_app.get_trains_passing_through(source, destination, day_of_week)
    print('available_trains shape:', available_trains.shape)
except Exception as e:
    print('get_trains_passing_through raised:', e)
    traceback.print_exc()
    raise

# If no direct trains found, check intermediate logic similar to recommend
if available_trains.empty:
    print('No direct trains found, checking intermediate_trains...')
    intermediate_trains = backend_app.train_data[
        (backend_app.train_data['source'] == source) &
        (backend_app.train_data['destination'] == destination) &
        (backend_app.train_data['day_of_week'] == day_of_week) &
        (backend_app.train_data['is_intermediate_segment'].fillna(False) == True)
    ].copy()
    print('intermediate_trains shape:', intermediate_trains.shape)

    if not intermediate_trains.empty:
        available_trains = intermediate_trains
    else:
        try:
            intermediate_train = backend_app.find_intermediate_segment_trains(source, destination, day_of_week, month, 'Clear', 'Winter')
            print('find_intermediate_segment_trains result:', intermediate_train)
        except Exception as e:
            print('find_intermediate_segment_trains raised:', e)
            traceback.print_exc()
            raise

print('done')
