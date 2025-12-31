import sys, os
sys.path.append(os.getcwd())
from backend import app as backend_app

client = backend_app.app.test_client()
backend_app.load_model()

test_routes = [
    ('HYB','VSKP'),
    ('DEL','LKO'),
    ('BNC','SC'),
    ('HYB','CHE')
]

for s,d in test_routes:
    resp = client.get(f'/api/trains?source={s}&destination={d}')
    data = resp.get_json()
    print(f'{s}->{d}:', resp.status_code, 'total:', data.get('total',0))
