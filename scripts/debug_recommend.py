import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import backend.app as app_module
from pprint import pprint

app_module.load_model()
app_module.app.config['PROPAGATE_EXCEPTIONS'] = True
client = app_module.app.test_client()

# Pick a multi-train route from enriched master
import pandas as pd
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
df = pd.read_csv(os.path.join(root, 'data', 'ap_trains_master_clean_with_delays.csv'))
route_df = df.groupby(['source','destination']).size().reset_index(name='n')
multi = route_df[route_df['n']>1]
if multi.empty:
    print('No multi routes')
    sys.exit(0)
row = multi.iloc[0]
print('Testing route', row['source'], row['destination'], 'count', row['n'])

payload = {'source': row['source'], 'destination': row['destination'], 'travel_date': '2025-02-01'}
resp = client.post('/api/recommend', json=payload)
print('status', resp.status_code)
try:
    pprint(resp.get_json())
except Exception as e:
    print('Failed to get json:', e)
    print(resp.data)
