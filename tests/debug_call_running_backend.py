import requests
url='http://127.0.0.1:5000/api/predict'
payload={'source':'NDLS','destination':'BCT','travel_date':'2025-12-30'}
print('Calling',url)
try:
    r=requests.post(url,json=payload,timeout=5)
    print('status',r.status_code)
    print('keys', list(r.json().keys()))
    all_trains = r.json().get('all_trains')
    print('all_trains present?', bool(all_trains))
    if all_trains:
        t=all_trains[0]
        print('sample keys', list(t.keys()))
        print('feature_contributions in sample?', 'feature_contributions' in t)
        if 'feature_contributions' in t:
            print('feature_contributions:', t['feature_contributions'])
except Exception as e:
    print('Error calling local backend:', e)
