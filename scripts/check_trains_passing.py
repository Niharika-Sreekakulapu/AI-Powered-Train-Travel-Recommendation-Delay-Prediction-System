import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend import app as backend_app
backend_app.load_model()
print('Loaded trains:', len(backend_app.train_data))
try:
    res = backend_app.get_trains_passing_through('NDLS', 'LKO', day_of_week=1)
    print('Result DF shape:', res.shape)
    print(res.head().to_dict(orient='records')[:3])
except Exception as e:
    print('Error calling get_trains_passing_through:', e)
    import traceback
    traceback.print_exc()
