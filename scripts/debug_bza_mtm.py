import sys, os, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backend.app as backend_app
from datetime import datetime

backend_app.load_model()

date = '2025-12-15'
day_of_week = datetime.strptime(date, '%Y-%m-%d').weekday() + 1
try:
    res = backend_app.get_trains_passing_through('BZA', 'MTM', day_of_week=day_of_week)
    print('Success shape:', res.shape)
    print(res.head().to_dict(orient='records')[:5])
except Exception as e:
    print('Exception:', e)
    traceback.print_exc()
