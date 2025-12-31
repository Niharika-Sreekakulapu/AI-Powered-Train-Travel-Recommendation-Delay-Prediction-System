#!/usr/bin/env python3
import requests, pandas as pd, time, json, csv
from pathlib import Path

csv_in = Path('data/ap_trains_master_clean.csv')
out_csv = Path('data/railradar_coverage_report.csv')

print(f"Reading train list from {csv_in}")
df = pd.read_csv(csv_in)
train_ids = df['train_id'].astype(str).tolist()

base = 'https://api.railradar.in'
rows = []
session = requests.Session()

print(f"Checking {len(train_ids)} trains against RailRadar...")
for i, tid in enumerate(train_ids, 1):
    tid = tid.strip()
    tried = []
    found = False
    matched = None
    status = None
    endpoint = None
    sample = ''
    message = ''

    candidates = [tid]
    z5 = tid.zfill(5)
    if z5 != tid:
        candidates.append(z5)

    for cand in candidates:
        url = f"{base}/api/v1/trains/{cand}/average-delay"
        tried.append(url)
        try:
            r = session.get(url, timeout=6)
            status = r.status_code
            if r.status_code == 200:
                found = True
                endpoint = url
                try:
                    j = r.json()
                    sample = json.dumps(j)[:1000]
                    message = j.get('message','') if isinstance(j, dict) else ''
                    if isinstance(j, dict) and j.get('success') is False:
                        found = False
                    else:
                        matched = cand
                        break
                except Exception:
                    sample = r.text[:1000]
                    matched = cand
                    break
            else:
                try:
                    j = r.json()
                    message = j.get('error', {}).get('message', '') or j.get('message','') or str(j)
                except:
                    message = r.text[:200]
        except Exception as e:
            message = str(e)

        url2 = f"{base}/api/v1/trains/{cand}?dataType=static"
        tried.append(url2)
        try:
            r2 = session.get(url2, timeout=6)
            if r2.status_code == 200:
                found = True
                endpoint = url2
                try:
                    j = r2.json()
                    sample = json.dumps(j)[:1000]
                    matched = cand
                except:
                    sample = r2.text[:1000]
                break
            else:
                try:
                    j = r2.json()
                    message = j.get('error', {}).get('message', '') or j.get('message','') or str(j)
                except:
                    message = r2.text[:200]
        except Exception as e:
            message = str(e)

    if not found:
        url3 = f"{base}/api/v1/trains/list?q={tid}"
        tried.append(url3)
        try:
            r3 = session.get(url3, timeout=6)
            if r3.status_code == 200:
                try:
                    j = r3.json()
                    trains = j.get('data', {}).get('trains', []) if isinstance(j, dict) else []
                    exact = None
                    for t in trains:
                        if t.get('train_number') and (t.get('train_number') == tid or t.get('train_number') == tid.zfill(5)):
                            exact = t.get('train_number')
                            break
                    if exact:
                        found = True
                        matched = exact
                        endpoint = url3
                        sample = json.dumps(trains[:4])[:1000]
                    else:
                        if trains:
                            sample = json.dumps(trains[:4])[:1000]
                            message = f"search returned {len(trains)} candidates"
                except Exception as e:
                    message = str(e)
            else:
                try:
                    message = r3.json()
                except:
                    message = r3.text[:200]
        except Exception as e:
            message = str(e)

    rows.append({
        'train_id': tid,
        'found': found,
        'matched_train': matched or '',
        'status': status or '',
        'endpoint': endpoint or '',
        'sample': sample,
        'message': message,
        'tried': '|'.join(tried)
    })

    if i % 10 == 0:
        print(f"Processed {i}/{len(train_ids)} (found so far: {sum(1 for r in rows if r['found'])})")
    time.sleep(0.15)

# write CSV
keys = ['train_id','found','matched_train','status','endpoint','message','sample','tried']
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, keys)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

found_count = sum(1 for r in rows if r['found'])
print(f"Done: {found_count}/{len(rows)} trains covered by RailRadar ({found_count/len(rows):.2%})")
print('\nFirst 10 rows of report:')
for r in rows[:10]:
    print(r['train_id'], r['found'], r['matched_train'], r['status'])
print(f"Wrote report to {out_csv}")
