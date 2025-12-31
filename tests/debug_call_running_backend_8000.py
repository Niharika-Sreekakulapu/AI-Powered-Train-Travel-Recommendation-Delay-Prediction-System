import requests
url='http://127.0.0.1:8000/api/predict'
payload={'source':'MJ','destination':'MJMG','travel_date':'2025-12-30'}
print('Calling',url)
try:
    r=requests.post(url,json=payload,timeout=10)
    print('status',r.status_code)
    data=r.json()
    print('keys',list(data.keys()))
    print('all_trains present?', 'all_trains' in data)
    if 'all_trains' in data and data['all_trains']:
        t=data['all_trains'][0]
        print('sample keys', list(t.keys()))
        print('feature_contributions?', 'feature_contributions' in t)
        if 'feature_contributions' in t:
            print(t['feature_contributions'])
except Exception as e:
    print('Error calling backend:', e)
