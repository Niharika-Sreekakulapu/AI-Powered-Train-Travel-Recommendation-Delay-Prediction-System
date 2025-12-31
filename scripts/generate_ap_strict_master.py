import os
import json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
STATION_MAP = os.path.join(ROOT, 'frontend', 'src', 'utils', 'stationMapping.json')
AP_UNIQUE = os.path.join(DATA_DIR, 'ap_unique_stations.txt')
MASTER = os.path.join(DATA_DIR, 'ap_trains_master.csv')
OUT_STRICT = os.path.join(DATA_DIR, 'ap_strict_463_codes.txt')
OUT_MASTER_CLEAN = os.path.join(DATA_DIR, 'ap_trains_master_clean.csv')
OUT_REPORT = os.path.join(DATA_DIR, 'ap_strict_report.json')

# Load station mapping
with open(STATION_MAP, 'r', encoding='utf-8') as fh:
    station_mapping = json.load(fh)
all_stations = station_mapping.get('stations', [])

# Reproduce frontend heuristics for Telangana and AP filters
def is_telangana(station):
    name = station.get('name', '')
    tel_kw = ['Hyderabad','Secunderabad','Warangal','Karimnagar','Nizamabad','Khammam','Ramagundam','Mahbubnagar','Nalgonda','Adilabad','Siddipet','Medak','Kamareddy','Jangaon','Kazipet','Peddapalli','Kaghaznagar','Dornakal','Mahbubabad','Suryapet','Vikarabad','Tandur','Secundrabad','Kacheguda','Begumpet','Lingampalli','Medchal','Shamshabad','Hi-Tech','Hitech','Jubilee Hills','Khairatabad','Nampally']
    return any(k in name for k in tel_kw)

def is_ap_candidate(station):
    name = station.get('name','')
    ap_keywords = [
      'Vijayawada','Visakhapatnam','Tirupati','Nellore','Guntur','Rajahmundry','Kakinada','Anantapur','Kadapa','Chittoor','Eluru','Tenali','Srikakulam','Kadiri','Hindupur','Dharmavaram','Bhimavaram','Machilipatnam','Tadepalligudem','Tanuku','Palakollu','Gudur','Duvvada','Annavaram','Samalkot','Gudivada','Chirala','Amalapuram','Narsipatnam','Sompeta','Palasa','Berhampur','Khallikot','Kotabommali','Mandasa','Bobbili','Parvatipuram','Rayagada','Naupada','Pedana','Repalle','Vuyyuru','Vetapalem','Narsapur','Wamtum','Goligeni','Navabpalem','Marampalli','Nujvid','Kolena','Chandanagar','Diviti Pally','Gangavathi','Crianakalapalli','Peddana','Tarigoppula','Vijayawada Jn','Visakhapatnam Jn','Tirupati Jn','Gudur Jn','Ongole','Guntur Jn','Anantapur Jn','Dharmavaram Jn','Bhimavaram Town','Machilipatnam Jn','Samalkot Jn','Gudivada Jn','Parvatipuram Tn','Salur','Srikakulam Road','Tekkali','Gudlavalleru','Kaikaluru','Akividu','Bhimadolu','Manchalipuram','Undi','Kavutaram','Vissannapeta','Zamindar','Indupalli','Moturu','Rayanapadu','Krishna Canal','Mustabada','Gannavaram','Pedana','Manipal','Bantumilli','Mantripalem','Ruthiyai'
    ]
    return any(k in name for k in ap_keywords)

# Build AP candidate set (codes)
ap_candidates = [s for s in all_stations if (not is_telangana(s) and is_ap_candidate(s))]
ap_candidate_codes = set(s['code'] for s in ap_candidates)
print(f"Frontend heuristic found {len(ap_candidate_codes)} AP candidate station codes")

# Intersect with our ap_unique_stations (stations actually present in datasets)
our_ap_codes = set()
if os.path.exists(AP_UNIQUE):
    with open(AP_UNIQUE, 'r') as fh:
        for line in fh:
            c = line.strip()
            if c:
                our_ap_codes.add(c)
print(f"Our dataset has {len(our_ap_codes)} unique station codes")

# Intersection
candidates_in_data = sorted(list(ap_candidate_codes & our_ap_codes))
print(f"Candidates in our data count: {len(candidates_in_data)}")

# If resulting set is not 463, we will pick top 463 by frequency from master
master_df = pd.read_csv(MASTER)

# Compute station frequency in master (count of trains that include station)
def station_freq_from_master(station_code):
    return master_df['station_codes'].str.contains(station_code).sum()

freqs = {code: station_freq_from_master(code) for code in candidates_in_data}

# If less than 463, try to expand by including other codes from our dataset ordered by frequency
strict_codes = set(candidates_in_data)
if len(strict_codes) < 463:
    print(f"Need to expand: have {len(strict_codes)}, want 463. Expanding by frequency...")
    # compute all candidate codes from our data and order by frequency
    all_codes = sorted(list(our_ap_codes), key=lambda c: master_df['station_codes'].str.contains(c).sum(), reverse=True)
    for c in all_codes:
        if len(strict_codes) >= 463:
            break
        strict_codes.add(c)

elif len(strict_codes) > 463:
    print(f"Too many AP candidates ({len(strict_codes)}). Reducing to top 463 by frequency...")
    ordered = sorted(list(strict_codes), key=lambda c: station_freq_from_master(c), reverse=True)
    strict_codes = set(ordered[:463])

# Final strict set length
strict_list = sorted(list(strict_codes))
print(f"Final strict list length: {len(strict_list)}")

# Write out codes and names
with open(OUT_STRICT, 'w') as fh:
    for code in strict_list:
        fh.write(code + '\n')

with open(OUT_REPORT, 'w') as fh:
    report = {
        'initial_frontend_candidates': len(ap_candidate_codes),
        'our_unique_codes': len(our_ap_codes),
        'intersection_count': len(candidates_in_data),
        'final_strict_count': len(strict_list)
    }
    json.dump(report, fh, indent=2)

print(f"Wrote strict station codes to {OUT_STRICT}")
print(f"Wrote report to {OUT_REPORT}")

# Build cleaned master: keep trains that touch at least one strict code
strict_set = set(strict_list)

keep_mask = master_df['station_codes'].apply(lambda s: any(code in strict_set for code in (s.split(',') if isinstance(s,str) else [])))
clean_df = master_df[keep_mask].copy()

# Write cleaned master
clean_df.to_csv(OUT_MASTER_CLEAN, index=False)
print(f"Wrote cleaned master with {len(clean_df)} trains to {OUT_MASTER_CLEAN}")

# Report missing stations (strict list stations not covered by cleaned master)
covered_codes = set()
for s in clean_df['station_codes'].fillna(''):
    covered_codes.update([c for c in s.split(',') if c])
missing = sorted(list(strict_set - covered_codes))
print(f"Stations missing from cleaned master: {len(missing)}")

# Append coverage info to report
report['cleaned_master_trains'] = len(clean_df)
report['strict_covered'] = len(strict_set) - len(missing)
report['strict_missing'] = missing
with open(OUT_REPORT, 'w') as fh:
    json.dump(report, fh, indent=2)

print('Done')
