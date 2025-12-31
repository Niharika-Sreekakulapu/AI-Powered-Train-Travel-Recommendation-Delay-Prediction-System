import sys, os
sys.path.append(os.getcwd())
from backend import app as backend_app
from flask import json

data = {'source':'HYB','destination':'VSKP','travel_date':'2025-12-15','preference':'fastest'}
with backend_app.app.test_request_context('/api/recommend', method='POST', json=data):
    try:
        backend_app.load_model()
        resp = backend_app.recommend()
        print('resp:', resp)
    except Exception as e:
        import traceback
        traceback.print_exc()
