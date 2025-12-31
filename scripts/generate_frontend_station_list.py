import os
import json

ROOT = os.path.dirname(os.path.dirname(__file__))
FRONTEND_UTILS = os.path.join(ROOT, 'frontend', 'src', 'utils')
STATION_MAP = os.path.join(FRONTEND_UTILS, 'stationMapping.json')
STRICT_CODES = os.path.join(ROOT, 'data', 'ap_strict_463_codes.txt')
OUT_FILE = os.path.join(FRONTEND_UTILS, 'stationMapping_ap_strict.json')

if not os.path.exists(STATION_MAP):
    raise FileNotFoundError(STATION_MAP)
if not os.path.exists(STRICT_CODES):
    raise FileNotFoundError(STRICT_CODES)

with open(STATION_MAP, 'r', encoding='utf-8') as fh:
    mapping = json.load(fh)

with open(STRICT_CODES, 'r', encoding='utf-8') as fh:
    codes = set(line.strip() for line in fh if line.strip())

filtered_stations = [s for s in mapping.get('stations', []) if s.get('code') in codes]
found_codes = set(s.get('code') for s in filtered_stations)
missing_codes = sorted(list(codes - found_codes))
# Add placeholders for missing codes so the frontend still has entries for all 463 codes
for c in missing_codes:
    filtered_stations.append({'code': c, 'name': c})

filtered = {
    'stations': sorted(filtered_stations, key=lambda s: s.get('code'))
}

with open(OUT_FILE, 'w', encoding='utf-8') as fh:
    json.dump(filtered, fh, indent=2, ensure_ascii=False)

print(f"Wrote {len(filtered['stations'])} stations to {OUT_FILE} (including {len(missing_codes)} placeholders)")
