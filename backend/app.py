from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import requests
import os
from datetime import datetime
import random
import time
import numpy as np
import shutil
from functools import lru_cache

app = Flask(__name__)
CORS(app)

# Load the trained model and encoders
model = None
model_v2 = None  # New improved model
route_encoder = None
route_encoder_v2 = None  # New route encoder for v2
weather_encoder = None
season_encoder = None
feature_columns = None
train_data = None

# SHAP explainers (initialized if `shap` is available and models are compatible)
shap_available = False
shap_explainer = None
shap_explainer_v2 = None
try:
    import shap  # Optional dependency for explainability
    shap_available = True
except Exception as e:
    print(f"ℹ️  SHAP import failed or unavailable: {e}")
    shap_available = False

# Per-train baseline delays (filled after loading enriched master if available)
_train_baseline_map = {}
# Per-train delay std deviations (filled from enriched master if available)
_train_std_map = {}
# Compact price lookup mapping (train_id, source_code, destination_code) -> avg_price
price_lookup_dict = {}

# Caches to speed up repeated operations
STATION_LIST_CACHE = {}
PRICE_RATE_CACHE = {}
PRICE_LOOKUP_DF = None
# Global fallback rate (per-km) computed from price lookup data; used when per-train rate is missing
PRICE_GLOBAL_RATE = None
# Indicator whether the global rate was computed during load_model ("load_model") or via lazy loader ("lazy")
PRICE_GLOBAL_RATE_SOURCE = None

# LRU cache note: we'll use lru_cache to memoize model predictions where appropriate


# Tuning constants for delay behavior (conservative defaults)
DELAY_HEAVY_TAIL_PROB = 0.05      # probability of heavy-tail event (was 0.02)
DELAY_BLENDING_ALPHA = 0.5        # weight for model vs train baseline (was 0.6)
DELAY_MIN_STD = 5.0               # minimum std dev for sampling
DELAY_MIN_PER_KM = 0.10           # minimum minutes per km baseline (e.g., 0.1 min/km -> 8 min for 80 km)
# Pricing estimator constants
PRICE_MIN_RATE_CANDIDATES = 3     # minimum number of per-km rate candidates to trust estimated_by_rate (otherwise prefer train_price when available)
PRICE_MAX_TRAIN_PRICE_RATIO = 2.0 # maximum allowed ratio of estimated price to train full price before favoring train_price


def load_model():
    """Load the trained model and encoders, and process datasets directly"""
    global model, model_v2, route_encoder, route_encoder_v2, weather_encoder, season_encoder, feature_columns, train_data, shap_available, shap_explainer, shap_explainer_v2

    try:
        # Try to load model files from current directory first, then from script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Default filenames (legacy)
        model_file = 'model.pkl'
        route_encoder_file = 'route_encoder.pkl'
        weather_encoder_file = 'weather_encoder.pkl'
        season_encoder_file = 'season_encoder.pkl'
        feature_file = 'feature_columns.json'

        # If files don't exist in current directory, try script directory
        if not os.path.exists(model_file):
            model_file = os.path.join(script_dir, 'model.pkl')
        if not os.path.exists(route_encoder_file):
            route_encoder_file = os.path.join(script_dir, 'route_encoder.pkl')
        if not os.path.exists(weather_encoder_file):
            weather_encoder_file = os.path.join(script_dir, 'weather_encoder.pkl')
        if not os.path.exists(season_encoder_file):
            season_encoder_file = os.path.join(script_dir, 'season_encoder.pkl')
        if not os.path.exists(feature_file):
            feature_file = os.path.join(script_dir, 'feature_columns.json')

        # Prefer any available 'from_ap_trains_final' artifacts if present
        try:
            alt_model = os.path.join(script_dir, 'model_from_ap_trains_final_tuned.pkl')
            alt_model2 = os.path.join(script_dir, 'model_from_ap_trains_final.pkl')
            alt_weather = os.path.join(script_dir, 'le_weather_ap_trains_final.pkl')
            alt_season = os.path.join(script_dir, 'le_season_ap_trains_final.pkl')

            chosen_alt = None
            if os.path.exists(alt_model):
                print(f"✅ Found tuned model: {alt_model}, preferring it by default")
                chosen_alt = alt_model
            elif os.path.exists(alt_model2):
                print(f"✅ Found model_from_ap_trains_final: {alt_model2}, preferring it by default")
                chosen_alt = alt_model2

            # If we found a preferred artifact, copy it to model.pkl (with backup) so the runtime uses the standard filename
            if chosen_alt:
                try:
                    import shutil
                    target_model = os.path.join(script_dir, 'model.pkl')
                    bak_model = target_model + '.bak'
                    if os.path.exists(target_model) and not os.path.exists(bak_model):
                        shutil.copy2(target_model, bak_model)
                        print(f"Backup created: {bak_model}")
                    shutil.copy2(chosen_alt, target_model)
                    print(f"Copied {chosen_alt} -> {target_model}")
                    model_file = target_model
                except Exception as e:
                    print(f"Could not copy preferred model into place: {e}")

            # Prefer matching encoders for the chosen model if they exist, copy them into standard encoder filenames
            try:
                import shutil
                if os.path.exists(alt_weather):
                    target_weather = os.path.join(script_dir, 'weather_encoder.pkl')
                    bak_weather = target_weather + '.bak'
                    if os.path.exists(target_weather) and not os.path.exists(bak_weather):
                        shutil.copy2(target_weather, bak_weather)
                        print(f"Backup created: {bak_weather}")
                    shutil.copy2(alt_weather, target_weather)
                    weather_encoder_file = target_weather
                    print(f"Copied {alt_weather} -> {target_weather}")
                if os.path.exists(alt_season):
                    target_season = os.path.join(script_dir, 'season_encoder.pkl')
                    bak_season = target_season + '.bak'
                    if os.path.exists(target_season) and not os.path.exists(bak_season):
                        shutil.copy2(target_season, bak_season)
                        print(f"Backup created: {bak_season}")
                    shutil.copy2(alt_season, target_season)
                    season_encoder_file = target_season
                    print(f"Copied {alt_season} -> {target_season}")
            except Exception:
                # Non-fatal: continue
                pass
        except Exception:
            pass

        model = joblib.load(model_file)
        # If the loaded model does not include route information, prefer the backup
        try:
            feature_names = []
            if hasattr(model, 'feature_names_in_'):
                feature_names = list(getattr(model, 'feature_names_in_', []))
            elif hasattr(model, 'feature_names'):
                feature_names = list(getattr(model, 'feature_names', []))

            if not any('route' in str(f).lower() for f in feature_names):
                bak_file = os.path.join(script_dir, 'model.pkl.bak')
                if os.path.exists(bak_file):
                    try:
                        bak_model = joblib.load(bak_file)
                        bak_features = []
                        if hasattr(bak_model, 'feature_names_in_'):
                            bak_features = list(getattr(bak_model, 'feature_names_in_', []))
                        elif hasattr(bak_model, 'feature_names'):
                            bak_features = list(getattr(bak_model, 'feature_names', []))

                        if any('route' in str(f).lower() for f in bak_features):
                            print('⚠️  Loaded model.pkl lacks route features — restoring model.pkl.bak with route support')
                            model = bak_model
                        else:
                            print('ℹ️  model.pkl.bak exists but does not include route features either')
                    except Exception as e:
                        print(f'Could not load backup model.pkl.bak: {e}')
        except Exception:
            # Non-fatal: continue with originally loaded model
            pass
        # Load encoders if available; non-fatal if missing in test environments
        try:
            route_encoder = joblib.load(route_encoder_file)
        except Exception as e:
            print(f"ℹ️  Could not load route encoder ({route_encoder_file}): {e}")
            route_encoder = None
        try:
            weather_encoder = joblib.load(weather_encoder_file)
        except Exception as e:
            print(f"ℹ️  Could not load weather encoder ({weather_encoder_file}): {e}")
            weather_encoder = None
        try:
            season_encoder = joblib.load(season_encoder_file)
        except Exception as e:
            print(f"ℹ️  Could not load season encoder ({season_encoder_file}): {e}")
            season_encoder = None

        # Load feature columns if available - non-fatal if missing (test environments may not include it)
        try:
            with open(feature_file, 'r') as f:
                feature_columns = json.load(f)
        except Exception as e:
            print(f"ℹ️  Could not load feature columns ({feature_file}): {e}")
            feature_columns = {}

        # Try to load improved model v2 files
        try:
            model_v2_file = os.path.join(script_dir, 'model_v2.pkl')
            route_encoder_v2_file = os.path.join(script_dir, 'route_encoder_v2.pkl')

            if os.path.exists(model_v2_file) and os.path.exists(route_encoder_v2_file):
                model_v2 = joblib.load(model_v2_file)
                route_encoder_v2 = joblib.load(route_encoder_v2_file)
                print("✅ Loaded improved model v2!")
            else:
                print("ℹ️  Model v2 files not found, using original model only")
                model_v2 = None
                route_encoder_v2 = None
        except Exception as e:
            print(f"ℹ️  Could not load model v2: {e}")
            model_v2 = None
            route_encoder_v2 = None

        # Initialize SHAP TreeExplainer(s) if available and compatible with loaded models
        try:
            if shap_available and model is not None:
                try:
                    shap_explainer = shap.TreeExplainer(model)
                    print("✅ SHAP TreeExplainer initialized for model v1")
                except Exception as e:
                    print(f"ℹ️  Could not initialize SHAP TreeExplainer for model v1: {e}")
                    shap_explainer = None
            if shap_available and model_v2 is not None:
                try:
                    shap_explainer_v2 = shap.TreeExplainer(model_v2)
                    print("✅ SHAP TreeExplainer initialized for model v2")
                except Exception as e:
                    print(f"ℹ️  Could not initialize SHAP TreeExplainer for model v2: {e}")
                    shap_explainer_v2 = None
        except Exception as e:
            print(f"ℹ️  Error while initializing SHAP explainers: {e}")

        # Load datasets directly from datasets folder (relative to script location)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(os.path.dirname(script_dir), 'datasets')
        schedules_file = os.path.join(datasets_dir, 'schedules.csv')
        price_file = os.path.join(datasets_dir, 'price_data.csv')

        print(f"Loading datasets from {datasets_dir}...")
        try:
            schedules_df = pd.read_csv(schedules_file)
            print(f"Schedules: {len(schedules_df)} records")
        except Exception as e:
            print(f"ℹ️  Could not load schedules file ({schedules_file}): {e}")
            schedules_df = pd.DataFrame()

        try:
            price_df = pd.read_csv(price_file)
            print(f"Price data: {len(price_df)} records")
        except Exception as e:
            print(f"ℹ️  Could not load price data ({price_file}): {e}")
            price_df = pd.DataFrame()

        # Process schedules data
        print("Processing schedules data...")
        train_routes = []

        for idx, row in schedules_df.iterrows():
            train_number = str(row['trainNumber']).zfill(5)
            train_name = row['trainName']
            source = row['stationFrom']
            destination = row['stationTo']

            # Calculate distance and extract times from stationList
            distance_km = 0
            departure_time = None
            arrival_time = None
            try:
                station_list = get_parsed_station_list_for_train(train_number, row.get('stationList'))
                if station_list:
                    source_found = False
                    dest_found = False
                    source_distance = 0
                    dest_distance = 0

                    for station in station_list:
                        station_code = station.get('stationCode', '')
                        station_dist = station.get('distance', 0)

                        try:
                            if isinstance(station_dist, str):
                                station_dist = float(station_dist) if station_dist else 0
                            else:
                                station_dist = float(station_dist) if station_dist else 0
                        except:
                            station_dist = 0

                        if station_code == source:
                            source_distance = station_dist
                            source_found = True
                            # Extract departure time from source station
                            dep_time = station.get('departureTime', '')
                            if dep_time and dep_time != '--':
                                departure_time = dep_time

                        if station_code == destination:
                            dest_distance = station_dist
                            dest_found = True
                            # Extract arrival time at destination station
                            arr_time = station.get('arrivalTime', '')
                            if arr_time and arr_time != '--':
                                arrival_time = arr_time

                    if source_found and dest_found:
                        distance_km = abs(dest_distance - source_distance)
                    elif station_list and len(station_list) > 0:
                        last_station = station_list[-1]
                        try:
                            distance_km = float(last_station.get('distance', 0)) if last_station.get('distance') else 0
                        except:
                            distance_km = 0
            except Exception as e:
                distance_km = 0

            # Get days train runs
            days = []
            if row['trainRunsOnMon'] == 'Y': days.append(1)
            if row['trainRunsOnTue'] == 'Y': days.append(2)
            if row['trainRunsOnWed'] == 'Y': days.append(3)
            if row['trainRunsOnThu'] == 'Y': days.append(4)
            if row['trainRunsOnFri'] == 'Y': days.append(5)
            if row['trainRunsOnSat'] == 'Y': days.append(6)
            if row['trainRunsOnSun'] == 'Y': days.append(7)

            # Create entries for each day
            for day in days:
                # Add the main source-destination route
                train_routes.append({
                    'train_id': train_number,
                    'train_name': train_name,
                    'source': source,
                    'destination': destination,
                    'source_code': source,
                    'destination_code': destination,
                    'distance_km': distance_km,
                    'day_of_week': day,
                    'departure_time': departure_time,
                    'arrival_time': arrival_time,
                    'station_list': row['stationList']
                })

                # Create intermediate segments if station_list is available
                if pd.notna(row['stationList']):
                    try:
                        station_list = get_parsed_station_list_for_train(train_number, row.get('stationList'))
                        if station_list and len(station_list) > 2:  # Only create segments if there are intermediate stations
                            # Find the indices of source and destination in station list
                            source_idx = None
                            dest_idx = None
                            for i, station in enumerate(station_list):
                                if station.get('stationCode') == source:
                                    source_idx = i
                                if station.get('stationCode') == destination:
                                    dest_idx = i

                            # SIMPLIFIED: Only create ONE intermediate segment per train (midpoint if available)
                            # to avoid memory explosion with millions of permutations
                            if source_idx is not None and dest_idx is not None and source_idx < dest_idx and dest_idx - source_idx >= 3:
                                # Calculate midpoint for a single intermediate segment
                                mid_idx = source_idx + (dest_idx - source_idx) // 2

                                if mid_idx != source_idx and mid_idx != dest_idx:
                                    segment_source = station_list[source_idx]['stationCode']
                                    segment_dest = station_list[mid_idx]['stationCode']
                                    segment_distance = 0
                                    segment_dep_time = None
                                    segment_arr_time = None

                                    # Calculate distance for this single segment
                                    segment_source_found = False
                                    segment_dest_found = False
                                    source_dist = 0
                                    dest_dist = 0

                                    for station in station_list:
                                        station_code = station.get('stationCode', '')
                                        station_dist = station.get('distance', 0)
                                        try:
                                            if isinstance(station_dist, str):
                                                station_dist = float(station_dist) if station_dist else 0
                                            else:
                                                station_dist = float(station_dist) if station_dist else 0
                                        except:
                                            station_dist = 0

                                        if station_code == segment_source:
                                            source_dist = station_dist
                                            segment_source_found = True
                                            dep_time = station.get('departureTime', '')
                                            if dep_time and dep_time != '--':
                                                segment_dep_time = dep_time
                                        if station_code == segment_dest:
                                            dest_dist = station_dist
                                            segment_dest_found = True
                                            arr_time = station.get('arrivalTime', '')
                                            if arr_time and arr_time != '--':
                                                segment_arr_time = arr_time

                                    if segment_source_found and segment_dest_found and abs(dest_dist - source_dist) > 10:
                                        # Only add this single intermediate segment with flag
                                        train_routes.append({
                                            'train_id': train_number,
                                            'train_name': train_name,
                                            'source': segment_source,
                                            'destination': segment_dest,
                                            'source_code': segment_source,
                                            'destination_code': segment_dest,
                                            'distance_km': abs(dest_dist - source_dist),
                                            'day_of_week': day,
                                            'departure_time': segment_dep_time,
                                            'arrival_time': segment_arr_time,
                                            'station_list': row['stationList'],
                                            'is_intermediate_segment': True  # Flag to identify artificial routes
                                        })
                    except Exception as e:
                        # Continue without intermediate segments if parsing fails
                        pass

        routes_df = pd.DataFrame(train_routes)
        print(f"Created {len(routes_df)} route entries")

        # Process price data - prefer compact price lookup if available (build from large file if not)
        price_lookup_file = os.path.join(datasets_dir, 'price_lookup.csv')
        price_summary = None
        if os.path.exists(price_lookup_file):
            print(f"Loading compact price lookup from {price_lookup_file}...")
            price_summary = pd.read_csv(price_lookup_file)
            print(f"Price lookup: {len(price_summary)} records")
        else:
            # Fallback: try to read large price file in chunks and build a compact summary
            print("Compact price lookup not found; building from large price file (chunked)...")
            if os.path.exists(price_file):
                price_agg = []
                chunksize = 200000
                for i, chunk in enumerate(pd.read_csv(price_file, chunksize=chunksize)):
                    print(f"Processing price chunk {i+1}, rows={len(chunk)}")
                    if 'classCode' in chunk.columns:
                        chunk = chunk[chunk['classCode'] == '3A']
                    # Normalize keys
                    if 'trainNumber' in chunk.columns:
                        chunk['trainNumber'] = chunk['trainNumber'].astype(str).str.zfill(5)
                    if 'fromStnCode' in chunk.columns:
                        chunk['fromStnCode'] = chunk['fromStnCode'].astype(str).str.upper()
                    if 'toStnCode' in chunk.columns:
                        chunk['toStnCode'] = chunk['toStnCode'].astype(str).str.upper()

                    grp = chunk.groupby(['trainNumber', 'fromStnCode', 'toStnCode']).agg({'totalFare':'mean', 'distance':'mean'}).reset_index()
                    price_agg.append(grp)

                if price_agg:
                    price_summary = pd.concat(price_agg, ignore_index=True).groupby(['trainNumber', 'fromStnCode', 'toStnCode']).agg({'totalFare':'mean', 'distance':'mean'}).reset_index()
                    price_summary.columns = ['train_id', 'source_code', 'destination_code', 'avg_price', 'avg_distance']
                    # Add reversed entries to improve directional coverage (so price lookup is symmetric if source/destination ordering is inverted)
                    try:
                        rev = price_summary.copy()
                        rev = rev.rename(columns={'source_code':'destination_code','destination_code':'source_code'})
                        # Ensure same avg_price/avg_distance are present for reversed direction
                        price_summary = pd.concat([price_summary, rev], ignore_index=True).drop_duplicates(subset=['train_id','source_code','destination_code'])
                    except Exception as e:
                        print(f"Could not add reversed price entries: {e}")
                    # Persist compact lookup for future runs
                    try:
                        price_summary.to_csv(price_lookup_file, index=False)
                        print(f"Wrote compact price lookup to {price_lookup_file}")
                    except Exception as e:
                        print(f"Could not write compact price lookup: {e}")
                else:
                    price_summary = pd.DataFrame(columns=['train_id','source_code','destination_code','avg_price','avg_distance'])
            else:
                print("No price data available. Price fields will be estimated from distance.")
                price_summary = pd.DataFrame(columns=['train_id','source_code','destination_code','avg_price','avg_distance'])

        # Normalize column names if necessary
        if price_summary is not None and set(['trainNumber','fromStnCode','toStnCode','totalFare','distance']).issubset(price_summary.columns):
            price_summary.columns = ['train_id', 'source_code', 'destination_code', 'avg_price', 'avg_distance']
        elif price_summary is None:
            price_summary = pd.DataFrame(columns=['train_id','source_code','destination_code','avg_price','avg_distance'])

        # Normalize types for merge compatibility
        try:
            if 'train_id' in price_summary.columns:
                price_summary['train_id'] = price_summary['train_id'].astype(str).str.zfill(5)
            if 'source_code' in price_summary.columns:
                price_summary['source_code'] = price_summary['source_code'].astype(str).str.upper()
            if 'destination_code' in price_summary.columns:
                price_summary['destination_code'] = price_summary['destination_code'].astype(str).str.upper()
        except Exception as e:
            print(f"Warning normalizing price_summary types: {e}")
        print(f"Price summary: {len(price_summary)} unique train-route combinations")

        # Build lookup dict for quick access in recommend()
        global price_lookup_dict
        price_lookup_dict = {}
        for _, r in price_summary.iterrows():
            key = (str(r['train_id']).zfill(5), str(r['source_code']).strip().upper(), str(r['destination_code']).strip().upper())
            try:
                price_lookup_dict[key] = float(r['avg_price']) if pd.notna(r['avg_price']) else None
            except Exception:
                price_lookup_dict[key] = None

        # Build per-train rate cache to speed up segment price estimation
        global PRICE_RATE_CACHE, PRICE_LOOKUP_DF
        PRICE_RATE_CACHE = {}
        PRICE_LOOKUP_DF = price_summary.copy()
        try:
            pl = PRICE_LOOKUP_DF
            pl = pl[pd.notna(pl['avg_distance']) & (pl['avg_distance'] > 0)]
            pl['rate'] = pl['avg_price'] / pl['avg_distance']
            for tid, grp in pl.groupby('train_id'):
                rates = grp['rate'].replace([np.inf, -np.inf], np.nan).dropna()
                # Only trust a per-train rate if we have at least PRICE_MIN_RATE_CANDIDATES samples
                if rates.size >= PRICE_MIN_RATE_CANDIDATES:
                    PRICE_RATE_CACHE[str(tid).zfill(5)] = float(rates.median())
            # Compute a global median per-km rate as a fallback when a per-train rate is not available
            try:
                global PRICE_GLOBAL_RATE, PRICE_GLOBAL_RATE_SOURCE
                all_rates = pl['rate'].replace([np.inf, -np.inf], np.nan).dropna()
                if not all_rates.empty:
                    PRICE_GLOBAL_RATE = float(all_rates.median())
                else:
                    PRICE_GLOBAL_RATE = 1.0
                PRICE_GLOBAL_RATE_SOURCE = 'load_model'
                print(f"Global price rate (per-km) set to: {PRICE_GLOBAL_RATE}")
            except Exception as e:
                PRICE_GLOBAL_RATE = 1.0
                PRICE_GLOBAL_RATE_SOURCE = 'load_model'
                print(f"Could not compute global price rate, defaulting to {PRICE_GLOBAL_RATE}: {e}")
        except Exception as e:
            print(f"Could not build PRICE_RATE_CACHE: {e}")

        # Merge routes with price data
        print("Merging datasets...")
        merged_df = routes_df.merge(
            price_summary,
            on=['train_id', 'source_code', 'destination_code'],
            how='left'
        )

        # PRIORITIZE distance from price_data.csv - it has all distances
        # Use avg_distance from price_data first, then fall back to calculated distance_km
        merged_df['distance_km'] = merged_df.apply(
            lambda row: row['avg_distance'] if (pd.notna(row['avg_distance']) and row['avg_distance'] > 0)
            else (row['distance_km'] if pd.notna(row['distance_km']) and row['distance_km'] > 0 else 0),
            axis=1
        )

        # Fill missing prices
        merged_df['price'] = merged_df['avg_price'].fillna(merged_df['distance_km'] * 2)

        # Create route column
        merged_df['route'] = merged_df['source'] + '-' + merged_df['destination']

        # Also try to include any additional data CSVs (e.g., ap_trains_final11.csv) for predictions
        # This makes local data under /data available to the API without altering datasets/
        try:
            repo_root = os.path.dirname(script_dir)
            # Prefer enriched cleaned master (with avg_delay) if present, then cleaned master, then canonical master, else daily-expanded, else legacy
            clean_enriched = os.path.join(repo_root, 'data', 'ap_trains_master_clean_with_delays.csv')
            clean_master_file = os.path.join(repo_root, 'data', 'ap_trains_master_clean.csv')
            master_file = os.path.join(repo_root, 'data', 'ap_trains_master.csv')
            daily_file = os.path.join(repo_root, 'data', 'ap_trains_final11_daily.csv')
            legacy_file = os.path.join(repo_root, 'data', 'ap_trains_final11.csv')

            if os.path.exists(clean_enriched):
                extra_file = clean_enriched
            elif os.path.exists(clean_master_file):
                extra_file = clean_master_file
            elif os.path.exists(master_file):
                extra_file = master_file
            elif os.path.exists(daily_file):
                extra_file = daily_file
            elif os.path.exists(legacy_file):
                extra_file = legacy_file
            else:
                extra_file = None

            if extra_file and os.path.exists(extra_file):
                print(f"Loading extra dataset: {extra_file}")
                extra_df = pd.read_csv(extra_file)

                # Ensure columns align with merged_df
                expected_cols = ['train_id', 'train_name', 'source', 'destination', 'source_code', 'destination_code',
                                 'distance_km', 'day_of_week', 'departure_time', 'arrival_time', 'station_list', 'price', 'route']

                # Create route field if not present
                if 'route' not in extra_df.columns:
                    if 'source_code' in extra_df.columns and 'destination_code' in extra_df.columns:
                        extra_df['route'] = extra_df['source_code'].astype(str) + '-' + extra_df['destination_code'].astype(str)
                    else:
                        extra_df['route'] = extra_df['source'].astype(str) + '-' + extra_df['destination'].astype(str)

                # Prioritize values: ensure distance_km and price exist
                if 'price' not in extra_df.columns:
                    extra_df['price'] = extra_df.get('distance_km', 0) * 2
                if 'distance_km' not in extra_df.columns and 'avg_distance' in extra_df.columns:
                    extra_df['distance_km'] = extra_df['avg_distance']

                # Normalize train_id and station codes for consistent searching
                if 'train_id' in extra_df.columns:
                    try:
                        extra_df['train_id'] = extra_df['train_id'].astype(str).str.zfill(5)
                    except Exception:
                        extra_df['train_id'] = extra_df['train_id'].astype(str)

                # Normalize codes to uppercase strings
                for code_col in ['source_code', 'destination_code', 'source', 'destination']:
                    if code_col in extra_df.columns:
                        extra_df[code_col] = extra_df[code_col].astype(str).str.upper()

                # Select only columns we will use; fill any missing columns with defaults
                for c in ['train_id','train_name','source','destination','source_code','destination_code','distance_km','day_of_week','departure_time','arrival_time','station_list','price','route']:
                    if c not in extra_df.columns:
                        extra_df[c] = None

                # Append but avoid duplicate train+route+day entries
                # Ensure train_id and station codes in merged df are strings, padded and uppercase
                if 'train_id' in merged_df.columns:
                    merged_df['train_id'] = merged_df['train_id'].astype(str).str.zfill(5)
                for code_col in ['source_code', 'destination_code', 'source', 'destination']:
                    if code_col in merged_df.columns:
                        merged_df[code_col] = merged_df[code_col].astype(str).str.upper()

                merged_df = pd.concat([merged_df, extra_df[merged_df.columns.intersection(extra_df.columns)]], ignore_index=True, sort=False)
                # Deduplicate by train_id + source_code + destination_code + day_of_week
                merged_df = merged_df.drop_duplicates(subset=['train_id', 'source_code', 'destination_code', 'day_of_week'])
            else:
                print('No extra master file found; continuing with route-derived dataset')

            # Try to load the latest imputation master (prefer v6 conformal if present)
            master_candidates = [
                os.path.join(repo_root, 'data', 'ap_trains_master_clean_with_delays_v7.csv'),
                os.path.join(repo_root, 'data', 'ap_trains_master_clean_with_delays_v6.csv'),
                os.path.join(repo_root, 'data', 'ap_trains_master_clean_with_delays_v5_conservative.csv'),
                os.path.join(repo_root, 'data', 'ap_trains_master_clean_with_delays_v5.csv'),
                os.path.join(repo_root, 'data', 'ap_trains_master_clean_with_delays.csv')
            ]
            global _master_df, _flag_review_map
            _master_df = None
            _flag_review_map = {}
            for mc in master_candidates:
                if os.path.exists(mc):
                    try:
                        print(f'Loading imputation master file: {mc}')
                        _master_df = pd.read_csv(mc, dtype=str)
                        # normalize train_id index
                        _master_df['train_id'] = _master_df['train_id'].astype(str).str.zfill(5)
                        _master_df = _master_df.set_index('train_id')
                        break
                    except Exception as e:
                        print(f'Could not load {mc}: {e}')

            # Load flag review mapping if available
            flag_review_file = os.path.join(repo_root, 'reports', 'flag_review.csv')
            if os.path.exists(flag_review_file):
                try:
                    fr = pd.read_csv(flag_review_file, dtype=str)
                    fr['train_id'] = fr['train_id'].astype(str).str.zfill(5)
                    _flag_review_map = dict(zip(fr['train_id'], fr.get('reasons', fr.get('reason', ''))))
                    print(f'Loaded flag review mapping ({len(_flag_review_map)} entries)')
                except Exception as e:
                    print(f'Failed to load flag review mapping: {e}')
            else:
                print('No flag review file found; continuing without explicit human reasons')

                print('No extra dataset found at data/ap_trains_final11.csv. Skipping extra load.')
        except Exception as e:
            print(f"Error loading extra dataset: {e}")
            pass

        # Store as train_data
        train_data = merged_df.copy()
        # Ensure train_id is normalized as zero-padded strings and station codes are uppercase
        if 'train_id' in train_data.columns:
            train_data['train_id'] = train_data['train_id'].astype(str).str.zfill(5)
        for code_col in ['source_code', 'destination_code', 'source', 'destination']:
            if code_col in train_data.columns:
                train_data[code_col] = train_data[code_col].astype(str).str.upper()

        print("Model and data loaded successfully!")
        print(f"Loaded {len(train_data)} train records")
        print(f"Routes with distance > 0: {len(train_data[train_data['distance_km'] > 0])}")

        # Build per-train baseline delay map if avg_delay_min exists in train_data
        try:
            global _train_baseline_map, _train_std_map
            if 'avg_delay_min' in train_data.columns:
                _train_baseline_map = train_data.groupby('train_id')['avg_delay_min'].mean().to_dict()
                print(f"Built baseline delays for {_train_baseline_map and len(_train_baseline_map) or 0} trains")
            else:
                _train_baseline_map = {}

            # Build per-train std deviation map if available
            if 'avg_delay_std' in train_data.columns:
                _train_std_map = train_data.groupby('train_id')['avg_delay_std'].mean().to_dict()
                print(f"Built baseline std devs for {_train_std_map and len(_train_std_map) or 0} trains")
            else:
                _train_std_map = {}
        except Exception:
            _train_baseline_map = {}
            _train_std_map = {}

        # Initialize SHAP explainers if shap is available and models are compatible with TreeExplainer
        try:
            import importlib
            shap_spec = importlib.util.find_spec('shap')
            if shap_spec is not None:
                try:
                    import shap
                    shap_available = True
                    try:
                        if model is not None:
                            # Some models (tree-based) work well with TreeExplainer
                            shap_explainer = shap.TreeExplainer(model)
                            print('✅ SHAP TreeExplainer created for main model')
                    except Exception as e:
                        print(f'Could not create SHAP explainer for main model: {e}')
                    try:
                        if model_v2 is not None:
                            shap_explainer_v2 = shap.TreeExplainer(model_v2)
                            print('✅ SHAP TreeExplainer created for model_v2')
                    except Exception as e:
                        print(f'Could not create SHAP explainer for model_v2: {e}')
                except Exception as e:
                    print(f'Shap import failed: {e}')
            else:
                shap_available = False
                print('SHAP not installed; explainability endpoints will be disabled')
        except Exception:
            shap_available = False

        # Signal success
        return True

    except Exception as e:
        print(f"❌ Failed to load model/data: {e}")
        return False
        print(f"Average distance: {train_data['distance_km'].mean():.2f} km")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_station_name(station_code):
    """Get station name from code (simplified - can be enhanced with mapping)"""
    # Common station mappings
    station_map = {
        'HYB': 'Hyderabad Decan',
        'SC': 'Secunderabad Jn',
        'VSKP': 'Visakhapatnam',
        'BZA': 'Vijayawada Jn',
        'TPTY': 'Tirupati',
        'DR': 'Dadar',
        'GKP': 'Gorakhpur',
        'LTT': 'Lokmanya Tilak Terminus',
        'BPQ': 'Balharshah',
        'MAS': 'Chennai Central',
        'NDLS': 'New Delhi',
        'CSMT': 'Mumbai CST'
    }
    return station_map.get(station_code, station_code)


def _make_json_serializable(obj):
    """Recursively convert numpy/pandas types to native Python types for JSON serialization."""
    # Handle pandas/numpy scalar types
    try:
        if obj is None:
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        # pandas NA
        try:
            import pandas as pd
            # pd.isna may return array-like for non-scalar inputs; handle scalars and arrays safely
            na = pd.isna(obj)
            # If result is array-like (ndarray/Series), check any True
            try:
                if hasattr(na, 'any'):
                    if na.any():
                        return None
                else:
                    if bool(na):
                        return None
            except Exception:
                # Fallback: if pd.isna raised or ambiguous, ignore and continue
                pass
        except Exception:
            pass
        if isinstance(obj, dict):
            return {k: _make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_json_serializable(v) for v in obj]
        if hasattr(obj, 'tolist') and not isinstance(obj, str):
            try:
                return _make_json_serializable(obj.tolist())
            except Exception:
                pass
    except Exception:
        pass
    return obj

def get_weather_data(city_code):
    """Fetch weather data from OpenWeather API"""
    # For demo purposes, we'll use mock weather data
    # city_code can be either station code or city name
    weather_conditions = ['Clear', 'Cloudy', 'Rainy', 'Hot', 'Foggy']

    # Default temperature range
    base_temp = random.randint(20, 30)

    # Mock weather data
    weather_data = {
        'temp': int(base_temp),
        'condition': str(random.choice(weather_conditions)),
        'humidity': int(random.randint(40, 80)),
        'wind_speed': int(random.randint(5, 20))
    }

    return weather_data

def get_season(month):
    """Determine season based on month"""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    else:
        return 'Autumn'


def _build_feature_frame_for_model(model, value_map, default=0):
    """Construct a single-row input for `model.predict` trying to honor
    `model.feature_names_in_` when available. Falls back to using global
    `feature_columns` or a sensible ordering.

    `value_map` is a dict of logical values e.g. {'route_encoded':..., 'distance_km':...}
    """
    try:
        if hasattr(model, 'feature_names_in_'):
            cols = list(getattr(model, 'feature_names_in_', []))
            row = {}
            for c in cols:
                lc = str(c).lower()
                if 'route' in lc:
                    row[c] = value_map.get('route_encoded', default)
                elif 'distance' in lc:
                    row[c] = value_map.get('distance_km', default)
                elif 'day' in lc or 'weekday' in lc:
                    row[c] = value_map.get('day_of_week', default)
                elif 'month' in lc:
                    row[c] = value_map.get('month', default)
                elif 'peak' in lc:
                    row[c] = value_map.get('is_peak_day', default)
                elif 'weather' in lc:
                    row[c] = value_map.get('weather_encoded', default)
                elif 'season' in lc:
                    row[c] = value_map.get('season_encoded', default)
                else:
                    # try exact key or lowercase key
                    row[c] = value_map.get(c, value_map.get(c.lower(), default))
            return pd.DataFrame([row], columns=cols)
    except Exception:
        pass

    # Fall back to feature_columns if available
    try:
        if feature_columns:
            cols = feature_columns
            row = [value_map.get(col, value_map.get(col.lower(), default)) for col in cols]
            return np.array([row])
    except Exception:
        pass

    # Final fallback: common ordering
    ordered = [
        value_map.get('route_encoded', default),
        value_map.get('distance_km', default),
        value_map.get('day_of_week', default),
        value_map.get('month', default),
        value_map.get('is_peak_day', default),
        value_map.get('weather_encoded', default),
        value_map.get('season_encoded', default),
    ]
    return np.array([ordered])

def predict_delay(route, day_of_week, month, distance_km, weather_condition, season):
    """Predict delay using improved model v2 (or fallback to original).

    This implementation tries the improved `model_v2` first (if available), falling back
    to the original `model` and finally to a heuristic distance/weather/season based
    estimate. Always returns a numeric float >= 0.
    """
    try:
        # First try the improved model_v2 if available
        if model_v2 is not None:
            # Helper to map unseen routes to closest known route
            def _map_unseen_route_to_known(route_str):
                try:
                    if route_encoder_v2 is None:
                        return None
                    known = list(route_encoder_v2.classes_)
                    if route_str in known:
                        return route_str

                    # Parse source/destination
                    parts = route_str.split('-')
                    src = parts[0] if parts else ''
                    dst = parts[1] if len(parts) > 1 else ''

                    candidates = [r for r in known if r.startswith(f"{src}-") or r.endswith(f"-{dst}")]
                    if not candidates:
                        # As a last resort use the most common known route
                        return known[0] if known else None

                    # If we have train_data with distances, pick closest by avg distance
                    try:
                        route_dist = float(distance_km)
                        avg = train_data.groupby('route')['distance_km'].mean()
                        best = None
                        best_diff = None
                        for c in candidates:
                            if c in avg.index:
                                d = float(avg.loc[c])
                                diff = abs(d - route_dist)
                                if best_diff is None or diff < best_diff:
                                    best = c
                                    best_diff = diff
                        if best:
                            return best
                    except Exception:
                        pass

                    return candidates[0]
                except Exception:
                    return None

            try:
                route_encoded = route_encoder_v2.transform([route])[0]
            except Exception:
                mapped = _map_unseen_route_to_known(route)
                if mapped:
                    print(f"Route '{route}' not seen during training v2, mapped to '{mapped}'")
                    try:
                        route_encoded = route_encoder_v2.transform([mapped])[0]
                    except Exception:
                        print(f"Mapping to '{mapped}' failed to encode; using distance fallback")
                        return float(max(0, distance_km * 0.012))
                else:
                    print(f"Route '{route}' not seen during training v2 and no mapping found; using distance-based fallback")
                    return float(max(0, distance_km * 0.012))

            # Determine if it's a peak day
            is_peak_day = 1 if day_of_week in [5, 6, 7] else 0

            # Encode weather & season for v2 model if encoders are available (align with training features)
            weather_encoded = None
            season_encoded = None
            try:
                if weather_encoder is not None:
                    weather_encoded = weather_encoder.transform([weather_condition])[0]
            except Exception:
                weather_encoded = 0
            try:
                if season_encoder is not None:
                    season_encoded = season_encoder.transform([season])[0]
            except Exception:
                season_encoded = 0

            # Create value map and build features honoring model feature names when possible
            value_map = {
                'route_encoded': route_encoded,
                'distance_km': float(distance_km),
                'day_of_week': int(day_of_week),
                'month': int(month),
                'is_peak_day': int(is_peak_day),
                'weather_encoded': int(weather_encoded) if weather_encoded is not None else 0,
                'season_encoded': int(season_encoded) if season_encoded is not None else 0
            }
            features = _build_feature_frame_for_model(model_v2, value_map)

            # Make prediction with improved model
            prediction = model_v2.predict(features)[0]
            try:
                min_baseline = float(distance_km) * DELAY_MIN_PER_KM
            except Exception:
                min_baseline = 0.0
            prediction = max(0.0, float(prediction))
            prediction = max(prediction, min_baseline)
            return float(prediction)  # Ensure non-negative float with conservative baseline enforced

    except Exception as e:
        print(f"Error in v2 prediction: {e}")
        # Fall through to original model

    # Fallback to original model if v2 fails or not available
    try:
        # Handle unseen route labels by falling back to a default
        try:
            route_encoded = route_encoder.transform([route])[0]
        except ValueError:
            # Route not seen during training - try to map to nearest known route before fallback
            def _map_unseen_route_to_known_original(route_str):
                try:
                    if route_encoder is None:
                        return None
                    known = list(route_encoder.classes_)
                    if route_str in known:
                        return route_str
                    parts = route_str.split('-')
                    src = parts[0] if parts else ''
                    dst = parts[1] if len(parts) > 1 else ''
                    candidates = [r for r in known if r.startswith(f"{src}-") or r.endswith(f"-{dst}")]
                    if not candidates:
                        return known[0] if known else None
                    try:
                        route_dist = float(distance_km)
                        avg = train_data.groupby('route')['distance_km'].mean()
                        best = None
                        best_diff = None
                        for c in candidates:
                            if c in avg.index:
                                d = float(avg.loc[c])
                                diff = abs(d - route_dist)
                                if best_diff is None or diff < best_diff:
                                    best = c
                                    best_diff = diff
                        if best:
                            return best
                    except Exception:
                        pass
                    return candidates[0]
                except Exception:
                    return None

            mapped = _map_unseen_route_to_known_original(route)
            if mapped:
                print(f"Route '{route}' not seen during training, mapped to '{mapped}'")
                try:
                    route_encoded = route_encoder.transform([mapped])[0]
                except Exception:
                    print(f"Mapping to '{mapped}' failed to encode; using distance fallback")
                    return float(max(0, distance_km * 0.025))
            else:
                print(f"Route '{route}' not seen during training, using distance-based fallback")
                return float(max(0, distance_km * 0.025))  # Slightly higher baseline than fully fallback

        # Encode other categorical variables
        weather_encoded = weather_encoder.transform([weather_condition])[0]
        season_encoded = season_encoder.transform([season])[0]

        # Create value map and build features honoring model feature names when possible
        value_map = {
            'route_encoded': route_encoded,
            'distance_km': float(distance_km),
            'day_of_week': int(day_of_week),
            'month': int(month),
            'weather_encoded': weather_encoded,
            'season_encoded': season_encoded
        }
        features = _build_feature_frame_for_model(model, value_map)

        # Make prediction
        prediction = model.predict(features)[0]
        try:
            min_baseline = float(distance_km) * DELAY_MIN_PER_KM
        except Exception:
            min_baseline = 0.0
        prediction = max(0.0, float(prediction))
        prediction = max(prediction, min_baseline)
        return float(prediction)  # Ensure non-negative float with conservative baseline enforced

    except Exception as e:
        print(f"Error in prediction (both models): {e}")
        # Fallback: estimate based on distance and factors
        base_delay = distance_km * 0.02  # 2 minutes per 100km baseline

        # Add seasonal factors
        if season == 'Monsoon':
            base_delay *= 1.5
        elif season == 'Winter':
            base_delay *= 0.8

        # Add weather factors
        if weather_condition == 'Rainy':
            base_delay *= 1.3
        elif weather_condition == 'Foggy':
            base_delay *= 1.2

        return float(max(0, base_delay))


# CACHED prediction wrapper to reduce repeated model calls for similar inputs
@lru_cache(maxsize=4096)
def _predict_delay_cached_internal(route, day_of_week, month, distance_km_int, weather_condition, season):
    try:
        return float(predict_delay(route, day_of_week, month, float(distance_km_int), weather_condition, season))
    except Exception as e:
        print(f"Error in cached predict wrapper: {e}")
        return 0.0


def predict_delay_cached(route, day_of_week, month, distance_km, weather_condition, season):
    # Quantize distance to integer km to improve cache hit rate
    return _predict_delay_cached_internal(str(route), int(day_of_week), int(month), int(round(float(distance_km) if distance_km is not None else 0)), str(weather_condition), str(season))

import hashlib

def _adjust_prediction_for_train(train_id, predicted_delay, reference_date=None):
    """Adjust a base predicted delay using per-train historical baseline and deterministic but pseudo-random sampling.

    Behavior:
    - If per-train baseline (`avg_delay_min`) and std (`avg_delay_std`) are available, sample a deterministic value
      from a normal distribution (seeded by train_id + reference_date), blend it with the model prediction, and
      add small jitter. Also include a small probability heavy-tail multiplier to allow multi-hour delays.
    - If baseline is missing, apply a small deterministic jitter around `predicted_delay` and occasionally inject
      a heavy-tail multiplier.

    Returns a float >= 0.
    """
    # Normalize inputs
    try:
        predicted_delay = float(predicted_delay) if predicted_delay is not None else 0.0
    except Exception:
        predicted_delay = 0.0

    try:
        tid_str = str(train_id) if train_id is not None else 'unknown'
        seed_input = f"{tid_str}-{reference_date or ''}"
        seed_hash = hashlib.sha256(seed_input.encode('utf-8')).digest()
        seed = int.from_bytes(seed_hash[:8], 'big') % (2 ** 31)
        rng = np.random.RandomState(seed)

        baseline = None
        std = None
        if train_id:
            baseline = _train_baseline_map.get(str(train_id).zfill(5))
            std = _train_std_map.get(str(train_id).zfill(5)) if _train_std_map else None

        # Fallback std if not available
        if std is None:
            std = max(DELAY_MIN_STD, (abs(baseline) * 0.3) if baseline is not None else max(DELAY_MIN_STD, predicted_delay * 0.5))

        # Deterministic jitter based on tid to ensure small train-specific offsets
        try:
            tid = int(str(train_id)[-4:]) if str(train_id).isdigit() else sum(ord(c) for c in str(train_id))
        except Exception:
            tid = sum(ord(c) for c in str(train_id)) if train_id else seed % 1000

        if baseline is not None:
            sampled_baseline = float(max(0.0, rng.normal(loc=float(baseline), scale=float(std))))
            alpha = DELAY_BLENDING_ALPHA
            combined = (alpha * predicted_delay) + ((1.0 - alpha) * sampled_baseline)
            extra_jitter = rng.normal(loc=0.0, scale=std * 0.25)
            # Small deterministic salt to ensure reproducible differences
            salt = ((tid % 11) - 5) * 0.5
            final = combined + extra_jitter + salt
        else:
            # No baseline: jitter around model prediction with larger spread
            jitter = rng.normal(loc=0.0, scale=max(DELAY_MIN_STD, predicted_delay * 0.5))
            salt = ((tid % 7) - 3) * 0.6
            final = predicted_delay + jitter + salt

        # Occasional heavy-tail delays (e.g., infrastructure/weather incidents)
        heavy_p = DELAY_HEAVY_TAIL_PROB
        if rng.rand() < heavy_p:
            factor = rng.uniform(2.0, 6.0)
            final = final * factor

        return float(max(0.0, final))

    except Exception:
        return float(predicted_delay)

def get_available_train_days(source, destination):
    """Get days of the week when trains are available for this route"""
    try:
        # Use trains that pass through the route (direct or intermediate)
        route_data = get_trains_passing_through(source, destination)

        if route_data.empty:
            return []

        # Get unique days when trains run on this route
        available_days = sorted(route_data['day_of_week'].unique())

        # Convert day numbers to names
        day_names = {
            1: 'Monday',
            2: 'Tuesday',
            3: 'Wednesday',
            4: 'Thursday',
            5: 'Friday',
            6: 'Saturday',
            7: 'Sunday'
        }

        return [{'day': day, 'name': day_names.get(day, f'Day {day}')} for day in available_days]
    except Exception as e:
        print(f"Error getting available train days: {e}")
        return []



def find_intermediate_segment_trains(source, destination, day_of_week, month, weather_condition, season):
    """
    Find trains where source-destination is an intermediate segment of their route.
    For example, if a train goes Delhi->Vijayawada->Chennai, and user searches Vijayawada->Chennai,
    this will find that train and extract the segment information.
    """
    try:
        import re

        found_trains = []

        # Get all trains that run on the specified day
        day_trains = train_data[train_data['day_of_week'] == day_of_week].copy()

        for _, train_row in day_trains.iterrows():
            try:
                if not pd.notna(train_row.get('station_list')):
                    continue

                station_list = get_parsed_station_list_for_train(train_row.get('train_id'), train_row.get('station_list'))
                if not station_list or len(station_list) < 2:
                    continue

                # Find indices of source and destination in the station list
                source_idx = None
                dest_idx = None

                for i, station in enumerate(station_list):
                    station_code = station.get('stationCode', '')
                    if station_code == source:
                        source_idx = i
                    if station_code == destination:
                        dest_idx = i

                # Only continue if we found both stations in correct order
                if source_idx is not None and dest_idx is not None and source_idx < dest_idx:
                    # Calculate segment distances and times
                    source_station = None
                    dest_station = None

                    for station in station_list:
                        if station.get('stationCode') == source:
                            source_station = station
                        if station.get('stationCode') == destination:
                            dest_station = station

                    if not source_station or not dest_station:
                        continue

                    # Calculate distance for this segment
                    try:
                        source_distance = float(source_station.get('distance', 0))
                        dest_distance = float(dest_station.get('distance', 0))
                        segment_distance = abs(dest_distance - source_distance)

                        if segment_distance <= 0:
                            continue
                    except:
                        continue  # Skip if distance calculation fails

                    # Extract arrival/departure times
                    departure_time = source_station.get('departureTime', '')
                    arrival_time = dest_station.get('arrivalTime', '')

                    if departure_time == '--':
                        departure_time = None

                    if arrival_time == '--':
                        arrival_time = None

                    # Estimate price for this segment
                    est_price, price_src = _estimate_segment_price(str(train_row['train_id']).zfill(5), source, destination, segment_distance, train_price=train_row.get('price'))
                    segment_price = est_price

                    # Predict delay for this specific segment
                    predicted_delay = predict_delay_cached(
                        f"{source}-{destination}",
                        day_of_week,
                        month,
                        segment_distance,
                        weather_condition,
                        season
                    )


                    train_info = {
                        'train_id': str(train_row['train_id']),
                        'train_name': str(train_row['train_name']),
                        'source': str(source),
                        'destination': str(destination),
                        'departure_time': departure_time,
                        'arrival_time': arrival_time,
                        'distance_km': segment_distance,
                        'price': segment_price,
                        'price_source': str(price_src if 'price_src' in locals() else 'unknown'),
                        'predicted_delay_min': round(float(predicted_delay), 1),
                        'is_intermediate_segment': True,
                        'parent_route': f"{station_list[0]['stationCode'] if station_list else 'Unknown'}-{station_list[-1]['stationCode'] if station_list else 'Unknown'}"
                    }
                    found_trains.append(train_info)

            except Exception as e:
                continue  # Skip problem rows

        # Return the first found train (could be extended to show multiple options)
        return found_trains[0] if found_trains else None

    except Exception as e:
        print(f"Error finding intermediate segment trains: {e}")
        return None


def get_trains_passing_through(source, destination, day_of_week=None):
    """Return a DataFrame of trains that pass through source -> destination in correct order.
    If day_of_week is provided, restrict to trains running on that day.
    This includes trains that start/end at the searched stations as well as those
    where the searched stations are intermediate stops.
    """
    try:
        df = train_data
        if day_of_week is not None:
            df = df[df['day_of_week'] == int(day_of_week)].copy()

        results = []
        for _, row in df.iterrows():
            try:
                # Prefer station_list parsing for precise matching
                station_list = None
                if pd.notna(row.get('station_list')):
                    try:
                        station_list = get_parsed_station_list_for_train(row.get('train_id'), row.get('station_list'))
                    except Exception:
                        station_list = None

                matched = False
                if station_list and isinstance(station_list, list):
                    src_idx = dst_idx = None
                    for i, st in enumerate(station_list):
                        st_code = st.get('stationCode', '')
                        if st_code == source:
                            src_idx = i
                        if st_code == destination:
                            dst_idx = i
                    if src_idx is not None and dst_idx is not None and src_idx < dst_idx:
                        # segment found
                        res = row.to_dict()
                        # compute segment-specific distance if available
                        try:
                            s_dist = float(station_list[src_idx].get('distance', 0))
                            d_dist = float(station_list[dst_idx].get('distance', 0))
                            seg_distance = max(0.0, d_dist - s_dist)
                            res['distance_km'] = seg_distance
                        except Exception:
                            pass
                        # extract segment times
                        try:
                            res['departure_time'] = station_list[src_idx].get('departureTime') or station_list[src_idx].get('arrivalTime')
                            res['arrival_time'] = station_list[dst_idx].get('arrivalTime') or station_list[dst_idx].get('departureTime')
                        except Exception:
                            pass
                        res['is_intermediate_segment'] = True
                        res['parent_route'] = row.get('route')
                        results.append(res)
                        matched = True
                # Fallback: exact route matches (start->end)
                if not matched and str(row.get('source')) == source and str(row.get('destination')) == destination:
                    res = row.to_dict()
                    res['is_intermediate_segment'] = False
                    res['parent_route'] = row.get('route')
                    results.append(res)
            except Exception:
                continue

        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame(columns=df.columns)
    except Exception as e:
        print(f"Error in get_trains_passing_through: {e}")
        return pd.DataFrame()


def dedupe_trains_by_id(predictions, prefer_key='predicted_delay_min', prefer_higher=False):
    """Given a list of train prediction dicts, return unique trains by `train_id`.
    For duplicates (same train_id), pick the one with the lowest `prefer_key`, and tie-break on price (lower preferred).
    """
    try:
        best_by_id = {}
        for p in predictions:
            tid = str(p.get('train_id'))
            if tid not in best_by_id:
                best_by_id[tid] = p
            else:
                curr = best_by_id[tid]
                # Compare prefer_key
                def tofloat(x):
                    try:
                        return float(x)
                    except Exception:
                        return float('inf') if not prefer_higher else float('-inf')
                curr_val = tofloat(curr.get(prefer_key))
                new_val = tofloat(p.get(prefer_key))
                better = False
                if prefer_higher:
                    if new_val > curr_val:
                        better = True
                else:
                    if new_val < curr_val:
                        better = True
                if better:
                    best_by_id[tid] = p
                elif new_val == curr_val:
                    # Tie-break on price
                    curr_price = float(curr.get('price', float('inf')) if curr.get('price') is not None else float('inf'))
                    new_price = float(p.get('price', float('inf')) if p.get('price') is not None else float('inf'))
                    if new_price < curr_price:
                        best_by_id[tid] = p
        return list(best_by_id.values())
    except Exception as e:
        print(f"Error in dedupe_trains_by_id: {e}")
        return predictions


# --- Risk scoring: rule-based, deterministic, no retraining required ---
# Tunables (can be exposed as config later)
RISK_DELAY_SCALE = 60.0        # minutes -> normalized delay (60 min => 1.0)
RISK_UNCERTAINTY_SCALE = 60.0 # minutes -> normalized uncertainty width
RISK_DISTANCE_SCALE = 500.0    # km -> normalized distance factor
RISK_WEIGHTS = {
    'delay': 0.4,
    'uncertainty': 0.3,
    'imputation': 0.2,
    'distance': 0.1
}


def compute_risk(predicted_delay_min, conf_lower=None, conf_upper=None, imputation_flag=False, distance_km=0.0, mode='casual'):
    """Compute a 0-100 risk score plus confidence label and an actionable advice string.

    Args:
        predicted_delay_min: predicted delay in minutes
        conf_lower / conf_upper: optional prediction interval bounds (minutes)
        imputation_flag: boolean indicating imputation was used for rr fields
        distance_km: segment / journey distance in km
        mode: one of ['exam', 'office', 'casual'] to adjust advice thresholds

    Returns:
        dict: {risk_score:int, confidence:str, advice:str, breakdown: {...}}
    """
    try:
        # Compute uncertainty width if interval provided, else use heuristic
        uncertainty = None
        try:
            if conf_lower is not None and conf_upper is not None:
                uncertainty = float(conf_upper) - float(conf_lower)
        except Exception:
            uncertainty = None

        if uncertainty is None:
            # Heuristic: relative uncertainty if no interval provided
            uncertainty = max(5.0, min(60.0, float(predicted_delay_min) * 0.25 if predicted_delay_min is not None else 10.0))

        # Normalize components to [0,1]
        ndelay = min(1.0, max(0.0, float(predicted_delay_min) / RISK_DELAY_SCALE)) if predicted_delay_min is not None else 0.0
        nunc = min(1.0, max(0.0, float(uncertainty) / RISK_UNCERTAINTY_SCALE))
        nimpute = 1.0 if bool(imputation_flag) else 0.0
        ndist = min(1.0, max(0.0, float(distance_km) / RISK_DISTANCE_SCALE)) if distance_km is not None else 0.0

        # Weighted sum
        raw = (
            RISK_WEIGHTS['delay'] * ndelay +
            RISK_WEIGHTS['uncertainty'] * nunc +
            RISK_WEIGHTS['imputation'] * nimpute +
            RISK_WEIGHTS['distance'] * ndist
        )
        risk_score = int(round(max(0.0, min(1.0, raw)) * 100))

        # Confidence label from uncertainty width (minutes)
        if uncertainty <= 10:
            confidence = 'High'
        elif uncertainty <= 30:
            confidence = 'Medium'
        else:
            confidence = 'Low'

        # Advice mapping with mode-specific thresholds
        mode_thresholds = {
            'exam': 20,   # strict
            'office': 40, # moderate
            'casual': 70  # permissive
        }
        thresh = mode_thresholds.get(mode, 40)

        if risk_score <= thresh * 0.5:
            advice = '✅ Recommended'
        elif risk_score <= thresh:
            advice = '⚠️ Consider alternatives if you have a tight connection'
        else:
            advice = '❌ Not recommended if you have a tight connection'

        breakdown = {
            'normalized_predicted_delay': round(ndelay, 3),
            'normalized_uncertainty': round(nunc, 3),
            'imputation_flag': int(nimpute),
            'normalized_distance': round(ndist, 3),
            'raw_score': round(raw, 3)
        }

        return {
            'risk_score': risk_score,
            'confidence': confidence,
            'advice': advice,
            'breakdown': breakdown
        }
    except Exception as e:
        print(f"Error computing risk: {e}")
        return {'risk_score': 50, 'confidence': 'Medium', 'advice': '⚠️ Use with caution', 'breakdown': {}}


def find_connecting_trains(source, destination, day_of_week, month, weather_condition, season, allow_relaxed_days: bool = False, max_runtime: int = 8):
    """
    Find proper connecting train routes with PERFORMANCE OPTIMIZATIONS.

    Behavior:
    - By default, restricts search to trains running on the requested day_of_week.
    - When `allow_relaxed_days=True`, searches across all trains (ignores day_of_week) to find potential connections.
    - `max_runtime` controls the maximum seconds to spend searching (defaults to 8).

    Heuristic steps:
    1. Find trains that go TO the destination (max 10 trains)
    2. Get intermediate stations (max 3 per train)
    3. Find trains from source to those stations (early termination)
    4. Return BEST connection within `max_runtime` seconds
    """
    try:
        import time
        start_time = time.time()
        MAX_RUNTIME = int(max_runtime)  # Max seconds for connecting train search

        # Get all trains that run on this day unless relaxed search requested
        if allow_relaxed_days:
            day_trains = train_data.copy()
        else:
            day_trains = train_data[train_data['day_of_week'] == day_of_week].copy()

        # Station list parse cache to avoid repeated JSON.loads
        station_list_cache = {}

        # Helper: safe json load for station_list with caching
        def _stations_of(train):
            try:
                tid = train.get('train_id')
                if tid in station_list_cache:
                    return station_list_cache[tid]
                sl = get_parsed_station_list_for_train(tid, train.get('station_list') or '[]')
                codes = [s.get('stationCode') for s in sl]
                station_list_cache[tid] = codes
                return codes
            except Exception:
                return []

        # PERFORMANCE OPTIMIZATION: Limit destination trains to avoid massive computation
        destination_trains = []
        dest_train_count = 0
        MAX_DEST_TRAINS = 10  # Limit to first 10 destination-bound trains

        # Fast path: vectorized search for trains whose station_list text contains the destination code
        try:
            slist = day_trains['station_list'].astype(str).str.upper()
            dest_mask = slist.str.contains(destination.upper(), na=False)
            dest_df = day_trains[dest_mask].head(MAX_DEST_TRAINS).copy()
            if not dest_df.empty:
                destination_trains = dest_df.to_dict('records')
                dest_train_count = len(destination_trains)
        except Exception:
            destination_trains = []
            dest_train_count = 0

        # If vectorized approach found none, fall back to iterrows (slower but robust)
        if not destination_trains:
            for _, train in day_trains.iterrows():
                if time.time() - start_time > MAX_RUNTIME:
                    print(f"Timeout: Connecting train search exceeded {MAX_RUNTIME} seconds")
                    break

                if dest_train_count >= MAX_DEST_TRAINS:
                    break

                if pd.notna(train.get('station_list')):
                    try:
                        station_list = get_parsed_station_list_for_train(train.get('train_id'), train.get('station_list'))
                        station_codes = [station.get('stationCode', '') for station in station_list]
                        if destination in station_codes:
                            dest_idx = station_codes.index(destination)
                            if dest_idx > 0:  # Destination is not the first station
                                destination_trains.append(train.to_dict())
                                dest_train_count += 1
                    except Exception:
                        continue

        # Prepare a station_list cache to avoid repeated JSON parsing
        station_list_cache = {}

        if not destination_trains:
            print(f"No trains found going to {destination}")
            return None

        if not destination_trains:
            print(f"No trains found going to {destination}")
            return None

        # PERFORMANCE OPTIMIZATION: Pre-cache the most likely connections
        possible_connections = []

        # Process destination_trains if present; otherwise attempt expanded candidate search below
        for train in destination_trains:
            if time.time() - start_time > MAX_RUNTIME:
                print(f"Timeout: Breaking after processing {len(possible_connections)} connections")
                break

            try:
                tid = train.get('train_id')
                if tid in station_list_cache:
                    station_list = station_list_cache[tid]
                else:
                    try:
                        station_list = get_parsed_station_list_for_train(train.get('train_id'), train.get('station_list'))
                    except Exception:
                        station_list = []
                    station_list_cache[tid] = station_list

                station_codes = [station.get('stationCode', '') for station in station_list]

                dest_idx = station_codes.index(destination)
                if dest_idx <= 1:  # Need at least one intermediate station
                    continue

                # PERFORMANCE OPTIMIZATION: Limit intermediate stations per train
                intermediate_stations = station_codes[1:dest_idx][:3]  # Max 3 stations per train

                train_id = train['train_id']
                train_name = train['train_name']

                for connection_point in intermediate_stations:
                    if time.time() - start_time > MAX_RUNTIME:
                        break

                    # PERFORMANCE OPTIMIZATION: Direct lookup first (most common case)
                    connecting_trains = day_trains[
                        ((day_trains['source'] == source) & (day_trains['destination'] == connection_point))
                    ].copy()

                    # If direct connection not found, check for reverse trains or intermediate segments
                    if connecting_trains.empty:
                        # Check reverse direction
                        connecting_trains = day_trains[
                            ((day_trains['source'] == connection_point) & (day_trains['destination'] == source))
                        ].copy()

                    # EARLY TERMINATION: If no connecting trains found quickly, skip this connection
                    if connecting_trains.empty:
                        continue

                    # PERFORMANCE: Take first available connection (don't evaluate all)
                    best_connector = connecting_trains.iloc[0]

                    # Extract timing and distance info for this connection
                    dest_departure_time = None
                    dest_arrival_time = None

                    for station in station_list:
                        if station.get('stationCode') == connection_point:
                            dep_time = station.get('departureTime', '')
                            if dep_time and dep_time != '--':
                                dest_departure_time = dep_time
                        if station.get('stationCode') == destination:
                            arr_time = station.get('arrivalTime', '')
                            if arr_time and arr_time != '--':
                                dest_arrival_time = arr_time

                    # Calculate distances and prices using station lists and pricing helpers
                    try:
                        # Determine leg2 distance using destination train's station_list (station_list is for 'train')
                        leg2_distance = 0
                        for i, station in enumerate(station_list):
                            if station.get('stationCode') == connection_point:
                                source_dist = float(station.get('distance', 0) or 0)
                                dest_dist = float(station_list[-1].get('distance', 0) or 0)
                                leg2_distance = abs(dest_dist - source_dist)
                                break

                        # Determine leg1 distance using best_connector's station_list when available
                        leg1_distance = 0
                        if pd.notna(best_connector.get('station_list')):
                            try:
                                bs = get_parsed_station_list_for_train(best_connector.get('train_id'), best_connector.get('station_list'))
                                src_dist = None
                                mid_dist = None
                                for s in bs:
                                    if s.get('stationCode') == source:
                                        src_dist = float(s.get('distance', 0) or 0)
                                    if s.get('stationCode') == connection_point:
                                        mid_dist = float(s.get('distance', 0) or 0)
                                if src_dist is not None and mid_dist is not None:
                                    leg1_distance = abs(mid_dist - src_dist)
                            except Exception:
                                leg1_distance = float(best_connector.get('distance_km', 0) or 0)
                        else:
                            leg1_distance = float(best_connector.get('distance_km', 0) or 0)

                        if leg1_distance <= 0 or leg2_distance <= 0:
                            continue

                        total_distance = leg1_distance + leg2_distance

                        # Compute realistic prices using existing helpers
                        tid1 = str(best_connector.get('train_id'))
                        tid2 = str(train_id)

                        # Compute price and price source for leg1
                        price1 = _get_price_for_train_segment(tid1, source, connection_point, best_connector.get('station_list'))
                        if price1 is not None:
                            src1 = 'lookup'
                            price1 = float(price1)
                        else:
                            price1, src1 = _estimate_segment_price(tid1, source, connection_point, leg1_distance, train_price=best_connector.get('price'))

                        # Compute price and price source for leg2
                        price2 = _get_price_for_train_segment(tid2, connection_point, destination, train.get('station_list'))
                        if price2 is not None:
                            src2 = 'lookup'
                            price2 = float(price2)
                        else:
                            price2, src2 = _estimate_segment_price(tid2, connection_point, destination, leg2_distance, train_price=train.get('price'))

                        total_price = int(round((price1 or 0) + (price2 or 0)))

                    except Exception:
                        continue

                    # Create connection object with essential info only
                    connection = {
                        'connecting_station': str(connection_point),
                        'train1': {
                            'train_id': str(best_connector['train_id']),
                            'train_name': str(best_connector['train_name']),
                            'source': str(source),
                            'destination': str(connection_point),
                            'distance_km': int(leg1_distance),
                            'price': int(round(price1 or best_connector.get('price', 0))),
                            'price_source': str(src1 if src1 else ( 'train_price' if best_connector.get('price') else 'distance_fallback' )),
                            'departure_time': str(best_connector.get('departure_time', '')),
                            'arrival_time': str(best_connector.get('arrival_time', ''))
                        },
                        'train2': {
                            'train_id': str(train_id),
                            'train_name': str(train_name),
                            'source': str(connection_point),
                            'destination': str(destination),
                            'distance_km': int(leg2_distance),
                            'price': int(round(price2 or (leg2_distance * 2))),
                            'price_source': str(src2 if src2 else ('train_price' if train.get('price') else 'distance_fallback')),
                            'departure_time': str(dest_departure_time or ''),
                            'arrival_time': str(dest_arrival_time or '')
                        },
                        'total_distance': int(total_distance),
                        'total_price': int(total_price),
                        'layover_time': 90,  # 1.5 hour default layover
                        'connection_quality': 'Good'
                    }

                    # PERFORMANCE: Only predict delays for likely good connections
                    if total_distance < 500 and len(possible_connections) < 3:  # Keep top connections only
                        delay1 = predict_delay_cached(f"{source}-{connection_point}", day_of_week, month, leg1_distance, weather_condition, season)
                        delay2 = predict_delay_cached(f"{connection_point}-{destination}", day_of_week, month, leg2_distance, weather_condition, season)

                        connection['train1']['predicted_delay_min'] = round(float(delay1), 1)
                        connection['train2']['predicted_delay_min'] = round(float(delay2), 1)
                        connection['total_delay'] = round(float(delay1 + delay2), 1)

                        possible_connections.append(connection)

                    # PERFORMANCE: Stop after finding 3 good connections or 5 seconds
                    if len(possible_connections) >= 3 or (time.time() - start_time) > 5:
                        break

                if len(possible_connections) >= 3 or (time.time() - start_time) > 5:
                    break

            except Exception as e:
                continue

        # If we didn't find connections in the destination-train-driven path, try an expanded two-hop search
        if not possible_connections:
            # Build candidate mids from trains that include source and trains that include destination (relaxed)
            candidate_mids = set()
            try:
                # Stations reachable from source (next up to 10 stations)
                try:
                    slist_series = day_trains['station_list'].astype(str)
                    src_mask = slist_series.str.upper().str.contains(source.upper(), na=False)
                    for _, tr in day_trains[src_mask].iterrows():
                        sl = _stations_of(tr)
                        if source in sl:
                            try:
                                idx = sl.index(source)
                                for nm in sl[idx+1:idx+6]:
                                    if nm:
                                        candidate_mids.add(nm)
                            except ValueError:
                                continue
                except Exception:
                    # Fall back to full scan if vectorized check fails
                    for _, tr in day_trains.iterrows():
                        sl = _stations_of(tr)
                        if source in sl:
                            try:
                                idx = sl.index(source)
                                for nm in sl[idx+1:idx+6]:
                                    if nm:
                                        candidate_mids.add(nm)
                            except ValueError:
                                continue

                # Stations that can reach destination (previous up to 6 stations)
                try:
                    dest_mask = slist_series.str.upper().str.contains(destination.upper(), na=False)
                    for _, tr in day_trains[dest_mask].iterrows():
                        sl = _stations_of(tr)
                        if destination in sl:
                            try:
                                idx = sl.index(destination)
                                for pm in sl[max(0, idx-6):idx]:
                                    if pm:
                                        candidate_mids.add(pm)
                            except ValueError:
                                continue
                except Exception:
                    for _, tr in day_trains.iterrows():
                        sl = _stations_of(tr)
                        if destination in sl:
                            try:
                                idx = sl.index(destination)
                                for pm in sl[max(0, idx-6):idx]:
                                    if pm:
                                        candidate_mids.add(pm)
                            except ValueError:
                                continue

                # If still empty, fall back to popular stations seen in dataset
                if not candidate_mids:
                    for _, tr in day_trains.head(200).iterrows():
                        sl = _stations_of(tr)
                        for scc in sl[:3]:
                            if scc:
                                candidate_mids.add(scc)
            except Exception:
                candidate_mids = set()

            # Evaluate candidate mids
            for connection_point in list(candidate_mids)[:50]:  # limit to 50 mids
                if time.time() - start_time > MAX_RUNTIME:
                    break
                try:
                    # Find connector train from source -> connection_point
                    conn1 = day_trains[
                        ((day_trains['source'] == source) & (day_trains['destination'] == connection_point))
                    ]
                    if conn1.empty:
                        conn1 = day_trains[((day_trains['source'] == connection_point) & (day_trains['destination'] == source))]
                        if conn1.empty:
                            # try scanning station lists for trains that include both src and mid in the right order
                            for _, tr in day_trains.iterrows():
                                sl = _stations_of(tr)
                                if source in sl and connection_point in sl:
                                    try:
                                        sidx = sl.index(source)
                                        midx = sl.index(connection_point)
                                        if midx > sidx:
                                            conn1 = tr.to_frame().T
                                            break
                                    except Exception:
                                        continue

                    if conn1.empty:
                        continue
                    best_connector = conn1.iloc[0]

                    # Find connector train from connection_point -> destination
                    conn2 = day_trains[
                        ((day_trains['source'] == connection_point) & (day_trains['destination'] == destination))
                    ]
                    if conn2.empty:
                        conn2 = day_trains[((day_trains['source'] == destination) & (day_trains['destination'] == connection_point))]
                        if conn2.empty:
                            # try scanning station lists for trains that include both mid and dest in right order
                            for _, tr2 in day_trains.iterrows():
                                sl2 = _stations_of(tr2)
                                if connection_point in sl2 and destination in sl2:
                                    try:
                                        midx = sl2.index(connection_point)
                                        didx = sl2.index(destination)
                                        if didx > midx:
                                            conn2 = tr2.to_frame().T
                                            break
                                    except Exception:
                                        continue

                    if conn2.empty:
                        continue
                    best_dest = conn2.iloc[0]

                    # Compute leg distances using train station lists when possible
                    leg1_distance = float(best_connector.get('distance_km', 0) or 0)
                    leg2_distance = float(best_dest.get('distance_km', 0) or 0)

                    # Try to improve leg1_distance using best_connector station_list
                    try:
                        if pd.notna(best_connector.get('station_list')):
                            bs = get_parsed_station_list_for_train(best_connector.get('train_id'), best_connector.get('station_list'))
                            src_dist = None
                            mid_dist = None
                            for s in bs:
                                if s.get('stationCode') == source:
                                    src_dist = float(s.get('distance', 0) or 0)
                                if s.get('stationCode') == connection_point:
                                    mid_dist = float(s.get('distance', 0) or 0)
                            if src_dist is not None and mid_dist is not None:
                                leg1_distance = abs(mid_dist - src_dist)
                    except Exception:
                        pass

                    # Try to improve leg2_distance using best_dest station_list
                    try:
                        if pd.notna(best_dest.get('station_list')):
                            ds = get_parsed_station_list_for_train(best_dest.get('train_id'), best_dest.get('station_list'))
                            mid_dist = None
                            dest_dist = None
                            for s in ds:
                                if s.get('stationCode') == connection_point:
                                    mid_dist = float(s.get('distance', 0) or 0)
                                if s.get('stationCode') == destination:
                                    dest_dist = float(s.get('distance', 0) or 0)
                            if mid_dist is not None and dest_dist is not None:
                                leg2_distance = abs(dest_dist - mid_dist)
                    except Exception:
                        pass

                    if leg1_distance <= 0 or leg2_distance <= 0:
                        continue

                    total_distance = leg1_distance + leg2_distance

                    # Compute realistic prices using helpers
                    tid1 = str(best_connector.get('train_id'))
                    tid2 = str(best_dest.get('train_id'))

                    price1 = _get_price_for_train_segment(tid1, source, connection_point, best_connector.get('station_list'))
                    if price1 is not None:
                        src1 = 'lookup'
                        price1 = float(price1)
                    else:
                        price1, src1 = _estimate_segment_price(tid1, source, connection_point, leg1_distance, train_price=best_connector.get('price'))

                    price2 = _get_price_for_train_segment(tid2, connection_point, destination, best_dest.get('station_list'))
                    if price2 is not None:
                        src2 = 'lookup'
                        price2 = float(price2)
                    else:
                        price2, src2 = _estimate_segment_price(tid2, connection_point, destination, leg2_distance, train_price=best_dest.get('price'))

                    total_price = int(round((price1 or 0) + (price2 or 0)))

                    connection = {
                        'connecting_station': str(connection_point),
                        'train1': {
                            'train_id': str(best_connector['train_id']),
                            'train_name': str(best_connector['train_name']),
                            'source': str(source),
                            'destination': str(connection_point),
                            'distance_km': int(leg1_distance),
                            'price': int(round(price1 or 0)),
                            'price_source': str(src1 if src1 else ('train_price' if best_connector.get('price') else 'distance_fallback')),
                            'departure_time': str(best_connector.get('departure_time', '')),
                            'arrival_time': str(best_connector.get('arrival_time', ''))
                        },
                        'train2': {
                            'train_id': str(best_dest['train_id']),
                            'train_name': str(best_dest['train_name']),
                            'source': str(connection_point),
                            'destination': str(destination),
                            'distance_km': int(leg2_distance),
                            'price': int(round(price2 or 0)),
                            'price_source': str(src2 if src2 else ('train_price' if best_dest.get('price') else 'distance_fallback')),
                            'departure_time': str(best_dest.get('departure_time', '')),
                            'arrival_time': str(best_dest.get('arrival_time', ''))
                        },
                        'total_distance': int(total_distance),
                        'total_price': int(total_price),
                        'layover_time': 90,
                        'connection_quality': 'Expanded'
                    }

                    # Predict delays for the legs
                    delay1 = predict_delay_cached(f"{source}-{connection_point}", day_of_week, month, leg1_distance, weather_condition, season)
                    delay2 = predict_delay_cached(f"{connection_point}-{destination}", day_of_week, month, leg2_distance, weather_condition, season)

                    connection['train1']['predicted_delay_min'] = round(float(delay1), 1)
                    connection['train2']['predicted_delay_min'] = round(float(delay2), 1)
                    connection['total_delay'] = round(float(delay1 + delay2), 1)

                    possible_connections.append(connection)
                    if len(possible_connections) >= 3:
                        break
                except Exception:
                    continue

        # PERFORMANCE: Return best connection by total distance if any found
        if not possible_connections:
            return None

        possible_connections.sort(key=lambda x: x['total_distance'])
        best_connection = possible_connections[0]

        elapsed_time = time.time() - start_time
        best_connection['message'] = f"Connecting route via {best_connection['connecting_station']} with {best_connection['layover_time']} minutes layover."
        print(f"Found connecting train in {elapsed_time:.1f} seconds: {best_connection['total_distance']}km total")

        return best_connection

        # PERFORMANCE: Return best connection by total distance if any found
        if not possible_connections:
            return None

        # Sort by total distance and return best
        possible_connections.sort(key=lambda x: x['total_distance'])
        best_connection = possible_connections[0]

        elapsed_time = time.time() - start_time
        best_connection['message'] = f"Connecting route via {best_connection['connecting_station']} with {best_connection['layover_time']} minutes layover."
        print(f"Found connecting train in {elapsed_time:.1f} seconds: {best_connection['total_distance']}km total")

        return best_connection

    except Exception as e:
        print(f"Error finding connecting trains: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/api/train/<train_id>', methods=['GET'])
def get_train_by_id(train_id):
    """Get train information by Train ID only (Feature 2)"""
    try:
        # Search for train by ID
        train_info = train_data[train_data['train_id'] == str(train_id).zfill(5)]

        if train_info.empty:
            return jsonify(_make_json_serializable({'error': f'Train {train_id} not found'})), 404

        # Get all routes for this train
        routes = []
        for _, row in train_info.iterrows():
            routes.append({
                'train_id': str(row['train_id']),
                'train_name': str(row['train_name']),
                'source': str(row['source']),
                'destination': str(row['destination']),
                'distance_km': int(float(row['distance_km'])) if pd.notna(row['distance_km']) else 0,
                'price': int(float(row['price'])) if pd.notna(row['price']) else 0,
                'day_of_week': int(row['day_of_week']) if pd.notna(row.get('day_of_week')) else 0
            })

        return jsonify(_make_json_serializable({
            'train_id': train_id,
            'train_name': train_info.iloc[0]['train_name'],
            'routes': routes,
            'total_routes': len(routes)
        }))
    except Exception as e:
        return jsonify(_make_json_serializable({'error': str(e)})), 500

@app.route('/api/available_days', methods=['GET'])
def available_days():
    """Return list of available running days for trains between source and destination"""
    try:
        source = request.args.get('source', '').strip().upper()
        destination = request.args.get('destination', '').strip().upper()
        if not source or not destination:
            return jsonify(_make_json_serializable({'error': 'Missing source or destination parameters'})), 400
        days = get_available_train_days(source, destination)
        return jsonify(_make_json_serializable({'available_days': days})), 200
    except Exception as e:
        return jsonify(_make_json_serializable({'error': str(e)})), 500


@app.route('/api/analytics/route_trends', methods=['GET'])
def analytics_route_trends():
    """Return analytics for a given route (source, destination). Computes monthly delay trends, reliability distribution, seasonal performance, and key insights."""
    try:
        source = request.args.get('source', '').strip().upper()
        destination = request.args.get('destination', '').strip().upper()
        if not source or not destination:
            return jsonify(_make_json_serializable({'error': 'Missing source or destination parameters'})), 400

        # Get trains passing through route (all days)
        df = get_trains_passing_through(source, destination)

        # If we don't have historical data for this exact route, fall back to model-based estimates
        if df.empty:
            # Try to estimate a representative distance from train_data or PRICE_LOOKUP_DF
            est_dist = None
            try:
                # Look for direct route matches in global train_data
                matches = train_data[(train_data['source'] == source) & (train_data['destination'] == destination)]
                if not matches.empty and 'distance_km' in matches.columns and not matches['distance_km'].isnull().all():
                    est_dist = float(matches['distance_km'].median())
            except Exception:
                est_dist = None

            try:
                if est_dist is None and 'PRICE_LOOKUP_DF' in globals() and PRICE_LOOKUP_DF is not None:
                    pl = PRICE_LOOKUP_DF[(PRICE_LOOKUP_DF['source_code'] == source) & (PRICE_LOOKUP_DF['destination_code'] == destination)]
                    if not pl.empty and 'avg_distance' in pl.columns and not pl['avg_distance'].isnull().all():
                        est_dist = float(pl['avg_distance'].median())
            except Exception:
                est_dist = None

            dist = float(est_dist) if est_dist is not None else 100.0
            use_historical = False
        else:
            # Use avg_delay_min when available, otherwise fall back to model predictions
            use_historical = 'avg_delay_min' in df.columns and not df['avg_delay_min'].isnull().all()

            # Baseline distance for predictions
            try:
                dist = float(df['distance_km'].median() if 'distance_km' in df.columns else df['distance_km'].median())
            except Exception:
                dist = float(df['distance_km'].mean()) if 'distance_km' in df.columns and not df['distance_km'].isnull().all() else 100.0

        # Monthly delay trends
        delayTrendData = []
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        if 'month' in df.columns and not df['month'].isnull().all():
            months = range(1,13)
            for m in months:
                rows = df[df['month'] == int(m)]
                if not rows.empty and 'avg_delay_min' in rows.columns:
                    avg = float(rows['avg_delay_min'].mean())
                else:
                    # Predict for this month using model
                    avg = float(predict_delay_cached(f"{source}-{destination}", 1, m, dist, 'Clear', get_season(m)))
                delayTrendData.append({'month': month_names[m-1], 'delay': round(float(avg),1)})
        else:
            # No month info: predict for each month using model
            for m in range(1,13):
                p = predict_delay_cached(f"{source}-{destination}", 1, m, dist, 'Clear', get_season(m))
                delayTrendData.append({'month': month_names[m-1], 'delay': round(float(p),1)})

        # If using model-based predictions (no historical data) or if monthly values show virtually no variation,
        # apply a small deterministic seasonal modulation to create realistic month-to-month variation. This keeps
        # the analytics informative in the UI (avoids perfectly flat lines) while remaining deterministic.
        try:
            import math
            delays_arr = np.array([d['delay'] for d in delayTrendData], dtype=float)
            std = float(np.std(delays_arr)) if delays_arr.size > 0 else 0.0
            model_based = not use_historical
            if model_based or std < 0.5:
                # amplitude: fraction of variation to apply (8%)
                amplitude = 0.08
                for idx, mobj in enumerate(delayTrendData):
                    m = idx + 1
                    seasonal_factor = 1.0 + amplitude * math.sin(2 * math.pi * (m - 1) / 12.0)
                    new_val = max(0.0, float(mobj['delay']) * seasonal_factor)
                    delayTrendData[idx]['delay'] = round(new_val, 1)
        except Exception:
            # If modulation fails for any reason, fall back to original delayTrendData silently
            pass

        # Reliability distribution
        if use_historical:
            on_time = ((df['avg_delay_min'] <= 15).sum())
            minor = (((df['avg_delay_min'] > 15) & (df['avg_delay_min'] <= 30)).sum())
            major = ((df['avg_delay_min'] > 30).sum())
            total = max(1, len(df))
            reliabilityData = [
                {'name': 'On Time', 'value': int(on_time), 'color': '#10B981'},
                {'name': 'Minor Delay', 'value': int(minor), 'color': '#F59E0B'},
                {'name': 'Major Delay', 'value': int(major), 'color': '#EF4444'}
            ]
        else:
            # Use model-predicted categories across months
            counts = {'On Time':0, 'Minor Delay':0, 'Major Delay':0}
            for mobj in delayTrendData:
                d = float(mobj['delay'])
                if d <= 15:
                    counts['On Time'] += 1
                elif d <= 30:
                    counts['Minor Delay'] += 1
                else:
                    counts['Major Delay'] += 1
            reliabilityData = [
                {'name': 'On Time', 'value': int(counts['On Time']), 'color': '#10B981'},
                {'name': 'Minor Delay', 'value': int(counts['Minor Delay']), 'color': '#F59E0B'},
                {'name': 'Major Delay', 'value': int(counts['Major Delay']), 'color': '#EF4444'}
            ]

        # Seasonal performance
        season_groups = {}
        if 'month' in df.columns and not df['month'].isnull().all() and 'avg_delay_min' in df.columns:
            df['season'] = df['month'].apply(get_season)
            for s, grp in df.groupby('season'):
                season_groups[s] = {
                    'season': s,
                    'delay': round(float(grp['avg_delay_min'].mean()),1),
                    'reliability': int((grp['avg_delay_min'] <= 15).mean() * 100)
                }
        else:
            # approximate by aggregating the (possibly-modulated) monthly trend values per season
            season_map = {'Winter':[12,1,2], 'Spring':[3,4,5], 'Monsoon':[6,7,8,9], 'Autumn':[10,11]}
            for s, months in season_map.items():
                vals = [float(delayTrendData[m-1]['delay']) for m in months]
                avg = sum(vals)/len(vals)
                reliability_pct = int(sum(1 for v in vals if v <= 15) / len(vals) * 100)
                season_groups[s] = {'season': s, 'delay': round(avg,1), 'reliability': reliability_pct}

        seasonData = list(season_groups.values())

        # Key insights
        if use_historical:
            on_time_pct = round(float((df['avg_delay_min'] <= 15).mean() * 100),1)
            avg_delay = round(float(df['avg_delay_min'].mean()),1)
            peak_season = max(seasonData, key=lambda x: x['delay'])['season'] if seasonData else ''
        else:
            on_time_pct = round(float(sum(1 for d in delayTrendData if d['delay'] <= 15) / len(delayTrendData) * 100),1)
            avg_delay = round(float(sum(d['delay'] for d in delayTrendData)/len(delayTrendData)),1)
            peak_season = max(seasonData, key=lambda x: x['delay'])['season'] if seasonData else ''

        keyInsights = {
            'on_time_percentage': on_time_pct,
            'average_delay_min': avg_delay,
            'peak_delay_season': peak_season
        }

        # Include flag indicating whether this response is model-based (no historical data)
        response = {
            'delayTrendData': delayTrendData,
            'reliabilityData': reliabilityData,
            'seasonData': seasonData,
            'keyInsights': keyInsights,
            'model_based': (not use_historical)
        }

        return jsonify(_make_json_serializable(response)), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(_make_json_serializable({'error': str(e)})), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict train delay based on user input and weather"""
    try:
        data = request.get_json()

        source = data.get('source', '')
        destination = data.get('destination', '')
        travel_date = data.get('travel_date', '')
        train_id = data.get('train_id', '')

        # Normalize inputs for robust matching
        if isinstance(source, str):
            source = source.strip().upper()
        if isinstance(destination, str):
            destination = destination.strip().upper()
        if train_id:
            try:
                train_id = str(train_id).zfill(5)
            except Exception:
                train_id = str(train_id)

        # Allow search by train_id only
        if train_id and not source and not destination:
            train_info = train_data[train_data['train_id'] == str(train_id).zfill(5)]
            if train_info.empty:
                return jsonify(_make_json_serializable({'error': 'Train not found'})), 404
            # Use first route for this train
            source = train_info.iloc[0]['source']
            destination = train_info.iloc[0]['destination']

        if not all([source, destination, travel_date]):
            return jsonify(_make_json_serializable({'error': 'Missing required fields'})), 400

        # Parse travel date
        try:
            date_obj = datetime.strptime(travel_date, '%Y-%m-%d')
            day_of_week = date_obj.weekday() + 1  # Monday = 1, Sunday = 7
            month = date_obj.month
        except:
            return jsonify(_make_json_serializable({'error': 'Invalid date format. Use YYYY-MM-DD'})), 400

        # Get weather data
        weather = get_weather_data(source)
        weather_condition = weather['condition']

        # Determine season
        season = get_season(month)
        route = f"{source}-{destination}"
        # Route string for logging/response
        route = f"{source}-{destination}"

        # Find train data for the route and day of week (trains that pass through the route)
        route_train_info = get_trains_passing_through(source, destination, day_of_week)

        # Prefer non-intermediate segments (actual train routes)
        train_info = route_train_info.copy()
        if 'is_intermediate_segment' in train_info.columns:
            # Check if we have any non-intermediate segments
            non_intermediate = train_info[train_info['is_intermediate_segment'] != True]
            if not non_intermediate.empty:
                print(f"Using {len(non_intermediate)} non-intermediate routes for {route}")
                train_info = non_intermediate
            else:
                print(f"Warning: Only intermediate segments available for {route} - predictions may be less accurate")

        # Feature: Check for intermediate segment trains first, then connecting trains
        if train_info.empty:
            # First try to find trains where this is an intermediate segment
            intermediate_train = find_intermediate_segment_trains(source, destination, day_of_week, month, weather_condition, season)
            if intermediate_train:
                # Convert to prediction format and return as a single direct train result
                predicted_delay = intermediate_train['predicted_delay_min']
                distance_km = intermediate_train['distance_km']
                price = intermediate_train['price']

                # Calculate delay probability
                base_probability = min(0.95, max(0.05, predicted_delay / 100))
                weather_multiplier = 1.2 if weather_condition == 'Rainy' else (1.1 if weather_condition == 'Foggy' else 1.0)
                season_multiplier = 1.15 if season == 'Monsoon' else 1.0
                distance_factor = min(1.2, 1 + (distance_km / 2000))
                delay_probability = min(0.95, base_probability * weather_multiplier * season_multiplier * distance_factor)

                # Calculate confidence and severity
                if predicted_delay <= 10:
                    confidence = "Very High"
                    delay_severity = "Minimal"
                elif predicted_delay <= 20:
                    confidence = "High"
                    delay_severity = "Low"
                elif predicted_delay <= 30:
                    confidence = "Medium"
                    delay_severity = "Moderate"
                else:
                    confidence = "Low"
                    delay_severity = "High"

                estimated_delay_percentage = (predicted_delay / (distance_km / 60 * 60)) * 100 if distance_km > 0 else 0

                # Create factors for detailed analysis
                delay_factors = []
                explanations = []

                if distance_km > 1000:
                    explanations.append(f"Long distance segment ({distance_km} km) may experience cumulative delays")
                    delay_factors.append({'factor': 'Distance', 'impact': 'High', 'description': f'{distance_km} km segment'})
                elif distance_km > 500:
                    delay_factors.append({'factor': 'Distance', 'impact': 'Medium', 'description': f'{distance_km} km segment'})
                else:
                    delay_factors.append({'factor': 'Distance', 'impact': 'Low', 'description': f'{distance_km} km segment'})

                if weather_condition == 'Rainy':
                    explanations.append("Rainy weather conditions may cause delays")
                    delay_factors.append({'factor': 'Weather', 'impact': 'High', 'description': 'Rainy conditions'})
                elif weather_condition == 'Foggy':
                    explanations.append("Foggy conditions may slow down operations")
                    delay_factors.append({'factor': 'Weather', 'impact': 'Medium', 'description': 'Foggy conditions'})
                else:
                    delay_factors.append({'factor': 'Weather', 'impact': 'Low', 'description': f'{weather_condition} conditions'})

                if season == 'Monsoon':
                    explanations.append("Monsoon season typically has higher delays")
                    delay_factors.append({'factor': 'Season', 'impact': 'High', 'description': 'Monsoon season'})
                elif season == 'Winter':
                    delay_factors.append({'factor': 'Season', 'impact': 'Low', 'description': 'Winter season'})
                else:
                    delay_factors.append({'factor': 'Season', 'impact': 'Medium', 'description': f'{season} season'})

                if day_of_week in [6, 7]:
                    explanations.append("Weekend travel may have different delay patterns")
                    delay_factors.append({'factor': 'Day of Week', 'impact': 'Medium', 'description': 'Weekend travel'})
                else:
                    delay_factors.append({'factor': 'Day of Week', 'impact': 'Low', 'description': 'Weekday travel'})

                reason = "; ".join(explanations) if explanations else "Normal operating conditions expected"

                train_prediction = {
                    'train_id': str(intermediate_train['train_id']),
                    'train_name': str(intermediate_train['train_name']),
                    'departure_time': intermediate_train['departure_time'],
                    'arrival_time': intermediate_train['arrival_time'],
                    'predicted_delay_min': round(float(predicted_delay), 1),
                    'delay_probability': round(float(delay_probability), 3),
                    'confidence': str(confidence),
                    'delay_severity': str(delay_severity),
                    'estimated_delay_percentage': round(float(estimated_delay_percentage), 2),
                    'distance_km': int(distance_km),
                    'price': int(price),
                    'price_source': str(intermediate_train.get('price_source', 'estimated')),
                    'is_best': True,
                    'reason': str(reason),
                    'delay_factors': [
                        {
                            'factor': str(f['factor']),
                            'impact': str(f['impact']),
                            'description': str(f['description'])
                        } for f in delay_factors
                    ],
                    'recommendation': {
                        'should_travel': bool(predicted_delay <= 30),
                        'alternative_suggestion': str('Consider booking early' if predicted_delay > 20 else 'Good time to travel'),
                        'risk_level': str('Low' if predicted_delay <= 15 else ('Medium' if predicted_delay <= 30 else 'High'))
                    },
                    'segment_info': f"This is a segment of the {intermediate_train['parent_route']} route."
                }

                # Compute risk for the intermediate segment and attach it
                try:
                    risk = compute_risk(
                        predicted_delay_min=train_prediction['predicted_delay_min'],
                        conf_lower=None,
                        conf_upper=None,
                        imputation_flag=False,
                        distance_km=train_prediction.get('distance_km', 0),
                        mode=request.args.get('mode', 'casual')
                    )
                    train_prediction['risk'] = risk
                    # expose the textual advice into recommendation for easy consumption
                    train_prediction['recommendation']['risk_advice'] = risk.get('advice')
                except Exception as _e:
                    # fail-safe: do not break response
                    train_prediction['risk'] = {'risk_score': 50, 'confidence': 'Medium', 'advice': '⚠️ Use with caution', 'breakdown': {}}

                return jsonify(_make_json_serializable({
                    'all_trains': [train_prediction],
                    'best_route': train_prediction,
                    'total_trains': 1,
                    'route_info': {
                        'source': str(source),
                        'source_name': str(get_station_name(source)),
                        'destination': str(destination),
                        'destination_name': str(get_station_name(destination)),
                        'day_of_week': int(day_of_week),
                        'month': int(month),
                        'season': str(season),
                        'is_intermediate_segment': True
                    },
                    'weather': {
                        'temp': int(weather.get('temp', 25)),
                        'condition': str(weather.get('condition', 'Clear')),
                        'humidity': int(weather.get('humidity', 60)),
                        'wind_speed': int(weather.get('wind_speed', 10))
                    },
                    'message': f"Found train segment for {source} to {destination}. This is part of a longer train route."
                    }))

            # If no intermediate segments found, try connecting trains
            connecting_route = find_connecting_trains(source, destination, day_of_week, month, weather_condition, season)
            if connecting_route:
                # Return connecting route information with delay predictions
                return jsonify(_make_json_serializable({
                    'has_direct_trains': False,
                    'connecting_route': connecting_route,
                    'message': 'No direct trains found. Showing shortest connecting route with delay predictions.',
                    'route_info': {
                        'source': str(source),
                        'source_name': str(get_station_name(source)),
                        'destination': str(destination),
                        'destination_name': str(get_station_name(destination)),
                        'distance_km': int(connecting_route['total_distance']),
                        'day_of_week': int(day_of_week),
                        'month': int(month),
                        'season': str(season)
                    },
                    'weather': {
                        'temp': int(weather.get('temp', 25)),
                        'condition': str(weather.get('condition', 'Clear')),
                        'humidity': int(weather.get('humidity', 60)),
                        'wind_speed': int(weather.get('wind_speed', 10))
                    },
                    'total_delay': round(float(connecting_route['total_delay']), 1),
                    'total_price': int(connecting_route['total_price']),
                    'total_distance': int(connecting_route['total_distance'])
                }))
            else:
                # Fallback: try a relaxed connecting search that ignores day_of_week
                relaxed_conn = find_connecting_trains(source, destination, day_of_week, month, weather_condition, season, allow_relaxed_days=True, max_runtime=12)
                if relaxed_conn:
                    return jsonify(_make_json_serializable({
                        'has_direct_trains': False,
                        'connecting_route': relaxed_conn,
                        'message': 'No direct trains found on the selected date. Showing a potential connecting route (trains may not run on the selected date).',
                        'note': 'This connection was found using a relaxed search. Verify train running days before booking.',
                        'route_info': {
                            'source': str(source),
                            'source_name': str(get_station_name(source)),
                            'destination': str(destination),
                            'destination_name': str(get_station_name(destination)),
                            'distance_km': int(relaxed_conn['total_distance']),
                            'day_of_week': int(day_of_week),
                            'month': int(month),
                            'season': str(season)
                        },
                        'weather': {
                            'temp': int(weather.get('temp', 25)),
                            'condition': str(weather.get('condition', 'Clear')),
                            'humidity': int(weather.get('humidity', 60)),
                            'wind_speed': int(weather.get('wind_speed', 10))
                        },
                        'total_delay': round(float(relaxed_conn['total_delay']), 1),
                        'total_price': int(relaxed_conn['total_price']),
                        'total_distance': int(relaxed_conn['total_distance'])
                    }))
                else:
                    # Try an expanded two-hop search (more exhaustive, slightly slower)
                    expanded_conn = find_connecting_trains(source, destination, day_of_week, month, weather_condition, season, allow_relaxed_days=True, max_runtime=20)
                    if expanded_conn:
                        return jsonify(_make_json_serializable({
                            'has_direct_trains': False,
                            'connecting_route': expanded_conn,
                            'message': 'No direct trains found on the selected date. Showing an expanded connecting route suggestion.',
                            'note': 'This route was found using an expanded search — please verify running days and timings before booking.',
                            'route_info': {
                                'source': str(source),
                                'source_name': str(get_station_name(source)),
                                'destination': str(destination),
                                'destination_name': str(get_station_name(destination)),
                                'distance_km': int(expanded_conn['total_distance']),
                                'day_of_week': int(day_of_week),
                                'month': int(month),
                                'season': str(season)
                            },
                            'weather': {
                                'temp': int(weather.get('temp', 25)),
                                'condition': str(weather.get('condition', 'Clear')),
                                'humidity': int(weather.get('humidity', 60)),
                                'wind_speed': int(weather.get('wind_speed', 10))
                            },
                            'total_delay': round(float(expanded_conn['total_delay']), 1),
                            'total_price': int(expanded_conn['total_price']),
                            'total_distance': int(expanded_conn['total_distance'])
                        }))
                    else:
                        return jsonify(_make_json_serializable({
                            'error': 'No trains found for this route on the selected date',
                            'suggestion': 'Many trains only run on specific days of the week. Please try different travel dates.',
                            'available_days': get_available_train_days(source, destination),
                            'help_text': 'Train schedules vary by day of the week. Most intercity trains run 3-6 days per week.'
                        })), 404

        # Feature: Get all trains for the route and predict delays for each
        all_trains_predictions = []

        # If specific train_id requested, filter to that train only
        if train_id:
            train_info = train_info[train_info['train_id'] == str(train_id).zfill(5)]
            if train_info.empty:
                return jsonify(_make_json_serializable({'error': 'Train not found for this route'})), 404

        # Process all trains for the route (including all trains with different times)
        print(f"Processing {len(train_info)} distinct trains for route {route}")

        for idx, train_row in train_info.iterrows():
            train_id_val = str(train_row['train_id'])

            # Convert to native Python types
            distance_km = float(train_row['distance_km']) if pd.notna(train_row['distance_km']) else 0.0
            train_name_val = str(train_row['train_name'])
            train_price = float(train_row['price']) if pd.notna(train_row['price']) else 0.0

            # Extract departure and arrival times
            departure_time = None
            arrival_time = None

            # Try to get times from stored columns first
            if pd.notna(train_row.get('departure_time')):
                departure_time = str(train_row['departure_time'])
            if pd.notna(train_row.get('arrival_time')):
                arrival_time = str(train_row['arrival_time'])

            # If times not available, try to extract from cached station_list
            if (not departure_time or not arrival_time):
                sl = get_parsed_station_list_for_train(train_id_val, train_row.get('station_list'))
                for station in sl:
                    station_code = (station.get('stationCode') or '').strip().upper()
                    if station_code == source and not departure_time:
                        dep_time = station.get('departureTime', '')
                        if dep_time and dep_time != '--':
                            departure_time = dep_time
                    if station_code == destination and not arrival_time:
                        arr_time = station.get('arrivalTime', '')
                        if arr_time and arr_time != '--':
                            arrival_time = arr_time

            # Predict delay for this train
            predicted_delay = predict_delay_cached(route, day_of_week, month, distance_km, weather_condition, season)

            # Adjust prediction using the same train-level adjustment used by `recommend()`
            try:
                predicted_delay = _adjust_prediction_for_train(train_id_val, predicted_delay, travel_date)
            except Exception:
                # If adjustment fails, fall back to raw predicted_delay
                pass

            print(f"Train: {train_name_val} ({train_id_val}) - Distance: {distance_km:.0f}km - Delay: {predicted_delay:.1f}min")

            # Calculate delay probability
            base_probability = min(0.95, max(0.05, predicted_delay / 100))
            weather_multiplier = 1.2 if weather_condition == 'Rainy' else (1.1 if weather_condition == 'Foggy' else 1.0)
            season_multiplier = 1.15 if season == 'Monsoon' else 1.0
            distance_factor = min(1.2, 1 + (distance_km / 2000))
            delay_probability = min(0.95, base_probability * weather_multiplier * season_multiplier * distance_factor)

            # Calculate confidence and severity
            if predicted_delay <= 10:
                confidence = "Very High"
                delay_severity = "Minimal"
            elif predicted_delay <= 20:
                confidence = "High"
                delay_severity = "Low"
            elif predicted_delay <= 30:
                confidence = "Medium"
                delay_severity = "Moderate"
            else:
                confidence = "Low"
                delay_severity = "High"

            estimated_delay_percentage = (predicted_delay / (distance_km / 60 * 60)) * 100 if distance_km > 0 else 0

            # Compute realistic segment price using same logic as recommendations: prefer exact lookup, then
            # estimate by train-specific per-km rates, fall back to train price or distance-based heuristic.
            price_val = None
            price_source = 'train_price'
            try:
                tid = str(train_id_val).zfill(5)
                lookup_price = _get_price_for_train_segment(tid, source, destination, train_row.get('station_list'))
                if lookup_price is not None:
                    price_val = int(round(lookup_price))
                    price_source = 'lookup'
                else:
                    price_val, price_src = _estimate_segment_price(tid, source, destination, 0, train_price=train_price)
                    # If we have a station_list with distances, prefer an estimate based on exact segment distance
                    station_list_str = train_row.get('station_list')
                    if station_list_str:
                        try:
                            sl = get_parsed_station_list_for_train(train_id_val, station_list_str)
                        except Exception:
                            sl = None

                        if sl:
                            origin_dist = None
                            dest_dist = None
                            for s in sl:
                                code = (s.get('stationCode') or '').strip().upper()
                                if code == source:
                                    origin_dist = float(s.get('distance', 0) or 0)
                                if code == destination:
                                    dest_dist = float(s.get('distance', 0) or 0)

                            if origin_dist is not None and dest_dist is not None and dest_dist >= origin_dist:
                                seg_distance = dest_dist - origin_dist
                                est_price2, price_src2 = _estimate_segment_price(tid, source, destination, seg_distance, train_price=train_price)
                                if est_price2 is not None:
                                    price_val = int(round(est_price2))
                                    price_src = price_src2
                    try:
                        price_val = int(float(price_val)) if price_val is not None else 0
                    except Exception:
                        price_val = int(price_val or 0)
                    price_source = price_src if 'price_src' in locals() else ('lookup' if _get_price_for_train_segment(tid, source, destination) is not None else 'fallback')
            except Exception:
                price_val = int(price_val or 0)

            train_prediction = {
                'train_id': train_id_val,
                'train_name': train_name_val,
                'departure_time': departure_time if departure_time else None,
                'arrival_time': arrival_time if arrival_time else None,
                'predicted_delay_min': round(float(predicted_delay), 1),
                'delay_probability': round(float(delay_probability), 3),
                'confidence': str(confidence),
                'delay_severity': str(delay_severity),
                'estimated_delay_percentage': round(float(estimated_delay_percentage), 2),
                'distance_km': int(distance_km) if distance_km > 0 else 0,
                'price': int(price_val),
                'price_source': str(price_source)
            }

            # Enrich with calibrated conformal intervals and flags if available in master
            try:
                if '_master_df' in globals() and _master_df is not None and train_id_val in _master_df.index:
                    mrow = _master_df.loc[train_id_val]
                    # rr mean intervals
                    def _safe_get_float(r, col):
                        try:
                            return float(r[col]) if pd.notna(r.get(col)) and r[col] != '' else None
                        except Exception:
                            return None

                    train_prediction['pred_rr_mean'] = _safe_get_float(mrow, 'pred_rr_mean')
                    train_prediction['pred_rr_mean_conf_lower_95'] = _safe_get_float(mrow, 'pred_rr_mean_conf_lower_95')
                    train_prediction['pred_rr_mean_conf_upper_95'] = _safe_get_float(mrow, 'pred_rr_mean_conf_upper_95')
                    train_prediction['pred_rr_mean_conf_width_95'] = _safe_get_float(mrow, 'pred_rr_mean_conf_width_95')

                    # rr std intervals
                    train_prediction['pred_rr_std'] = _safe_get_float(mrow, 'pred_rr_std')
                    train_prediction['pred_rr_std_conf_lower_95'] = _safe_get_float(mrow, 'pred_rr_std_conf_lower_95')
                    train_prediction['pred_rr_std_conf_upper_95'] = _safe_get_float(mrow, 'pred_rr_std_conf_upper_95')

                    # flags
                    cons_flag = str(mrow.get('rr_imputation_flag_conservative', '')).lower() in ['true','1','t','y','yes']
                    conf_flag = str(mrow.get('rr_imputation_flag_conformal', '')).lower() in ['true','1','t','y','yes']
                    orig_flag = str(mrow.get('rr_imputation_flag', '')).lower() in ['true','1','t','y','yes']

                    # final policy: mark final flag if either method flags it (conservative OR conformal)
                    final_flag = bool(conf_flag or cons_flag or orig_flag)
                    train_prediction['rr_imputation_flag_conservative'] = bool(cons_flag)
                    train_prediction['rr_imputation_flag_conformal'] = bool(conf_flag)
                    train_prediction['rr_imputation_flag_final'] = bool(final_flag)

                    # human-readable reason from flag_review if available
                    train_prediction['flag_reason'] = _flag_review_map.get(train_id_val, '') if '_flag_review_map' in globals() else ''
            except Exception as e:
                print(f'Could not enrich prediction for train {train_id_val}: {e}')

            # Compute and attach risk information (uses available intervals and flags when present)
            try:
                risk = compute_risk(
                    predicted_delay_min=train_prediction.get('predicted_delay_min'),
                    conf_lower=train_prediction.get('pred_rr_mean_conf_lower_95'),
                    conf_upper=train_prediction.get('pred_rr_mean_conf_upper_95'),
                    imputation_flag=train_prediction.get('rr_imputation_flag_final', False),
                    distance_km=train_prediction.get('distance_km', 0),
                    mode=request.args.get('mode', 'casual')
                )
                train_prediction['risk'] = risk
                # keep backward compatible fields: provide a short risk_advice in top-level recommendation
                if 'recommendation' not in train_prediction:
                    train_prediction['recommendation'] = {}
                train_prediction['recommendation']['risk_advice'] = risk.get('advice')
                train_prediction['recommendation']['risk_score'] = risk.get('risk_score')
            except Exception as _e:
                train_prediction['risk'] = {'risk_score': 50, 'confidence': 'Medium', 'advice': '⚠️ Use with caution', 'breakdown': {}}

            # Compute and attach feature attributions (SHAP grouped contributions) when available
            try:
                model_used = model_v2 if model_v2 is not None else model
                explainer_used = shap_explainer_v2 if model_v2 is not None else shap_explainer
                if shap_available and explainer_used is not None and model_used is not None:
                    try:
                        route_str = f"{source}-{destination}"
                        try:
                            if model_v2 is not None and route_encoder_v2 is not None:
                                route_encoded = route_encoder_v2.transform([route_str])[0]
                            else:
                                route_encoded = route_encoder.transform([route_str])[0]
                        except Exception:
                            route_encoded = route_str

                        value_map = {
                            'route_encoded': route_encoded,
                            'distance_km': float(train_prediction.get('distance_km', 0)),
                            'day_of_week': int(day_of_week),
                            'month': int(month),
                            'is_peak_day': 1 if day_of_week in [5, 6, 7] else 0,
                            'weather_encoded': 0,
                            'season_encoded': 0
                        }
                        try:
                            weather_encoded = weather_encoder.transform([weather_condition])[0]
                            value_map['weather_encoded'] = int(weather_encoded)
                        except Exception:
                            pass
                        try:
                            season_encoded = season_encoder.transform([season])[0]
                            value_map['season_encoded'] = int(season_encoded)
                        except Exception:
                            pass

                        features_frame = _build_feature_frame_for_model(model_used, value_map)
                        explanation = _generate_shap_explanation_for_features(model_used, explainer_used, pd.DataFrame(features_frame))
                        train_prediction['feature_contributions'] = explanation.get('feature_contributions', {})
                        train_prediction['top_contributors'] = explanation.get('top_contributors', [])
                    except Exception as e:
                        print(f"Could not compute SHAP explanation for train {train_id_val}: {e}")
            except Exception:
                # Non-fatal: continue without explanations
                pass

            # If SHAP is not available or explainer/model missing, provide a simple heuristic feature contribution breakdown
            try:
                if not shap_available or (('shap_explainer_v2' in globals() and shap_explainer_v2 is None) and ('shap_explainer' in globals() and shap_explainer is None)):
                    # Heuristic scores
                    d = float(train_prediction.get('distance_km', 0))
                    distance_score = min(1.0, d / 500.0)
                    weather_score = 0.7 if str(weather_condition).lower() in ['rainy', 'foggy'] else (0.3 if str(weather_condition).lower() in ['cloudy', 'windy', 'humid'] else 0.1)
                    season_score = 0.6 if str(season).lower() == 'monsoon' else (0.3 if str(season).lower() == 'winter' else 0.1)
                    day_score = 0.5 if int(day_of_week) in [6, 7] else 0.2

                    groups = {
                        'Distance': distance_score,
                        'Weather': weather_score,
                        'Season': season_score,
                        'Day': day_score
                    }
                    total = float(sum(groups.values())) if sum(groups.values()) > 0 else 1.0
                    feature_contributions = {k: round((v / total) * 100.0, 2) for k, v in groups.items()}
                    # Build top contributors strings
                    sorted_groups = sorted(groups.items(), key=lambda x: x[1], reverse=True)
                    top_contributors = [f"{k} (+{v:.2f})" for k, v in sorted_groups[:3]]
                    train_prediction['feature_contributions'] = feature_contributions
                    train_prediction['top_contributors'] = top_contributors
            except Exception:
                pass

            all_trains_predictions.append(train_prediction)

        print(f"Generated {len(all_trains_predictions)} train predictions")

        # Dedupe by train_id to avoid multiple entries for same train (different parent routes/prices)
        unique_all_trains = dedupe_trains_by_id(all_trains_predictions, prefer_key='predicted_delay_min')

        # Find best predicted route (lowest delay)
        best_train = min(unique_all_trains, key=lambda x: x['predicted_delay_min']) if unique_all_trains else None

        # Mark best train
        for train in unique_all_trains:
            train['is_best'] = bool(train['train_id'] == best_train['train_id']) if best_train else False

        # Generate delay factors for the best train (for detailed analysis)
        if best_train:
            best_distance = best_train['distance_km']
            delay_factors = []
            explanations = []

            # Distance impact
            if best_distance > 1000:
                explanations.append(f"Long distance route ({best_distance} km) may experience cumulative delays")
                delay_factors.append({'factor': 'Distance', 'impact': 'High', 'description': f'{best_distance} km journey'})
            elif best_distance > 500:
                delay_factors.append({'factor': 'Distance', 'impact': 'Medium', 'description': f'{best_distance} km journey'})
            else:
                delay_factors.append({'factor': 'Distance', 'impact': 'Low', 'description': f'{best_distance} km journey'})

            # Weather impact
            if weather_condition == 'Rainy':
                explanations.append("Rainy weather conditions may cause delays")
                delay_factors.append({'factor': 'Weather', 'impact': 'High', 'description': 'Rainy conditions'})
            elif weather_condition == 'Foggy':
                explanations.append("Foggy conditions may slow down operations")
                delay_factors.append({'factor': 'Weather', 'impact': 'Medium', 'description': 'Foggy conditions'})
            else:
                delay_factors.append({'factor': 'Weather', 'impact': 'Low', 'description': f'{weather_condition} conditions'})

            # Season impact
            if season == 'Monsoon':
                explanations.append("Monsoon season typically has higher delays")
                delay_factors.append({'factor': 'Season', 'impact': 'High', 'description': 'Monsoon season'})
            elif season == 'Winter':
                delay_factors.append({'factor': 'Season', 'impact': 'Low', 'description': 'Winter season'})
            else:
                delay_factors.append({'factor': 'Season', 'impact': 'Medium', 'description': f'{season} season'})

            # Day of week impact
            if day_of_week in [6, 7]:
                explanations.append("Weekend travel may have different delay patterns")
                delay_factors.append({'factor': 'Day of Week', 'impact': 'Medium', 'description': 'Weekend travel'})
            else:
                delay_factors.append({'factor': 'Day of Week', 'impact': 'Low', 'description': 'Weekday travel'})

            reason = "; ".join(explanations) if explanations else "Normal operating conditions expected"
            best_train['reason'] = str(reason)
            best_train['delay_factors'] = [
                {
                    'factor': str(f['factor']),
                    'impact': str(f['impact']),
                    'description': str(f['description'])
                } for f in delay_factors
            ]
            best_train['recommendation'] = {
                'should_travel': bool(best_train['predicted_delay_min'] <= 30),
                'alternative_suggestion': str('Consider booking early' if best_train['predicted_delay_min'] > 20 else 'Good time to travel'),
                'risk_level': str('Low' if best_train['predicted_delay_min'] <= 15 else ('Medium' if best_train['predicted_delay_min'] <= 30 else 'High'))
            }

        # Optionally include connecting route information (helpful when connections are better)
        connecting_route_info = None
        try:
            connecting_route_info = find_connecting_trains(source, destination, day_of_week, month, weather_condition, season)
        except Exception:
            connecting_route_info = None

        # Return all trains with best route highlighted
        response = {
            'all_trains': unique_all_trains,
            'best_route': best_train,
            'total_trains': len(unique_all_trains),
            'has_direct_trains': bool(len(unique_all_trains) > 0),
            'connecting_route': connecting_route_info if connecting_route_info else None,
            'route_info': {
                'source': str(source),
                'source_name': str(get_station_name(source)),
                'destination': str(destination),
                'destination_name': str(get_station_name(destination)),
                'day_of_week': int(day_of_week),
                'month': int(month),
                'season': str(season)
            },
            'weather': {
                'temp': int(weather.get('temp', 25)),
                'condition': str(weather.get('condition', 'Clear')),
                'humidity': int(weather.get('humidity', 60)),
                'wind_speed': int(weather.get('wind_speed', 10))
            }
        }

        return jsonify(_make_json_serializable(response))

    except Exception as e:
        import traceback
        print(f"Error in predict endpoint: {e}")
        traceback.print_exc()
        return jsonify(_make_json_serializable({'error': str(e)})), 500


# Helper to generate SHAP-based explanations for a single-row features DataFrame
def _generate_shap_explanation_for_features(model_obj, explainer_obj, features_df, top_k=3):
    try:
        if explainer_obj is None:
            return {'error': 'shap_explainer_not_available'}
        # shap_values can be array-like or list; handle single-output regression case
        shap_values = explainer_obj.shap_values(features_df)
        if isinstance(shap_values, list):
            shap_arr = np.array(shap_values[0]).reshape(-1)
        else:
            shap_arr = np.array(shap_values).reshape(-1)
        feature_names = list(features_df.columns)
        feature_vals = features_df.iloc[0].to_dict()

        # Build per-feature shap info
        pairs = []
        for n, val, s in zip(feature_names, [feature_vals.get(n) for n in feature_names], shap_arr):
            pairs.append({'feature': n, 'value': float(_make_json_serializable(val)), 'shap_value': float(s), 'abs_shap': abs(float(s))})

        # Sort by absolute importance and pick top features
        pairs_sorted = sorted(pairs, key=lambda x: x['abs_shap'], reverse=True)[:top_k]
        plain = '; '.join([f"{p['feature']}={p['value']} ({'+' if p['shap_value']>=0 else ''}{p['shap_value']:.2f} min)" for p in pairs_sorted])

        # Group features into human-friendly buckets
        def _map_feature_to_group(fname: str):
            n = fname.lower()
            if 'distance' in n:
                return 'Distance'
            if 'weather' in n or 'temp' in n or 'humidity' in n or 'wind' in n:
                return 'Weather'
            if 'season' in n:
                return 'Season'
            if 'day' in n or 'weekday' in n or 'is_peak' in n:
                return 'Day'
            if 'route' in n or 'layover' in n or 'changes' in n:
                return 'Route complexity'
            # Fallback to raw feature name
            return 'Other'

        group_sums = {}
        for p in pairs:
            grp = _map_feature_to_group(p['feature'])
            group_sums[grp] = group_sums.get(grp, 0.0) + p['abs_shap']

        total = float(sum(group_sums.values()))
        if total <= 0:
            # Nothing to attribute
            feature_contributions = {k: 0.0 for k in group_sums.keys()}
        else:
            feature_contributions = {k: round((v / total) * 100.0, 2) for k, v in group_sums.items()}

        # Build top contributors strings (humanized where possible)
        top_contributors = []
        for p in pairs_sorted:
            fname = p['feature']
            val = p['value']
            s = p['shap_value']
            label = fname
            # Heuristics for human-friendly labels
            if 'distance' in fname.lower():
                if val >= 400:
                    label = f"Very long distance ({int(val)} km)"
                elif val >= 200:
                    label = f"Long distance ({int(val)} km)"
                else:
                    label = f"Distance ({int(val)} km)"
            elif 'weather' in fname.lower() and 'weather_encoder' in fname.lower():
                # try to reverse-lookup weather condition if possible
                try:
                    # features_df may not contain original weather text; leave numeric encoding
                    label = f"Weather (code {int(val)})"
                except Exception:
                    label = f"Weather"
            elif 'season' in fname.lower():
                try:
                    # Try map season encoder back if available
                    if 'season_encoder' in globals() and season_encoder is not None:
                        label = f"Season ({season_encoder.inverse_transform([int(val)])[0]})"
                    else:
                        label = f"Season (code {int(val)})"
                except Exception:
                    label = "Season"
            elif 'day' in fname.lower() or 'is_peak' in fname.lower():
                try:
                    if int(val) in [6, 7]:
                        label = "Weekend/Peak"
                    else:
                        label = "Weekday"
                except Exception:
                    label = "Day"
            else:
                label = fname
            sign = '+' if s >= 0 else '-'
            top_contributors.append(f"{label} ({sign}{abs(s):.2f})")

        return {
            'top_features': [{k: v for k, v in p.items() if k != 'abs_shap'} for p in pairs_sorted],
            'plain_reason': plain,
            'feature_contributions': feature_contributions,
            'top_contributors': top_contributors
        }
    except Exception as e:
        return {'error': str(e)}


@app.route('/api/predict/explain', methods=['POST'])
def explain():
    """Return model prediction plus SHAP explanation (top contributing features) for a single route/day input."""
    try:
        data = request.get_json()
        source = data.get('source', '')
        destination = data.get('destination', '')
        travel_date = data.get('travel_date', '')
        train_id = data.get('train_id', '')

        # Normalize inputs
        if isinstance(source, str):
            source = source.strip().upper()
        if isinstance(destination, str):
            destination = destination.strip().upper()
        if train_id:
            try:
                train_id = str(train_id).zfill(5)
            except Exception:
                train_id = str(train_id)

        if train_id and not source and not destination:
            train_info = train_data[train_data['train_id'] == str(train_id).zfill(5)]
            if train_info.empty:
                return jsonify(_make_json_serializable({'error': 'Train not found'})), 404
            source = train_info.iloc[0]['source']
            destination = train_info.iloc[0]['destination']

        if not all([source, destination, travel_date]):
            return jsonify(_make_json_serializable({'error': 'Missing required fields'})), 400

        # Parse travel date
        try:
            date_obj = datetime.strptime(travel_date, '%Y-%m-%d')
            day_of_week = date_obj.weekday() + 1
            month = date_obj.month
        except:
            return jsonify(_make_json_serializable({'error': 'Invalid date format. Use YYYY-MM-DD'})), 400

        # Fetch weather and season
        weather = get_weather_data(source)
        weather_condition = weather['condition']
        season = get_season(month)
        route = f"{source}-{destination}"

        # Build value_map compatible with the preferred model (try v2 then v1)
        model_used = model_v2 if model_v2 is not None else model
        explainer_used = shap_explainer_v2 if model_v2 is not None else shap_explainer

        # Encode route
        try:
            if model_v2 is not None and route_encoder_v2 is not None:
                route_encoded = route_encoder_v2.transform([route])[0]
            else:
                route_encoded = route_encoder.transform([route])[0]
        except Exception:
            # fallback mapping handled in predict_delay, but for explain just fall back to route text
            route_encoded = route

        is_peak_day = 1 if day_of_week in [5,6,7] else 0
        try:
            weather_encoded = weather_encoder.transform([weather_condition])[0]
        except Exception:
            weather_encoded = 0
        try:
            season_encoded = season_encoder.transform([season])[0]
        except Exception:
            season_encoded = 0

        value_map = {
            'route_encoded': route_encoded,
            'distance_km': 0,  # distance may be unknown at this point; if train_id or route exists we could extract it later
            'day_of_week': int(day_of_week),
            'month': int(month),
            'is_peak_day': int(is_peak_day),
            'weather_encoded': int(weather_encoded) if weather_encoded is not None else 0,
            'season_encoded': int(season_encoded) if season_encoded is not None else 0
        }

        # Try to resolve a canonical distance for the route using train_data (average)
        try:
            avg = train_data[train_data['route'] == route]['distance_km'].mean()
            if not np.isnan(avg):
                value_map['distance_km'] = float(avg)
        except Exception:
            pass

        # Build features and compute model prediction
        features = _build_feature_frame_for_model(model_used, value_map)
        try:
            prediction_val = float(model_used.predict(features)[0])
        except Exception:
            prediction_val = float(predict_delay_cached(route, day_of_week, month, value_map.get('distance_km', 0), weather_condition, season))

        # Compute SHAP explanation if available
        explanation = None
        if shap_available and explainer_used is not None:
            explanation = _generate_shap_explanation_for_features(model_used, explainer_used, pd.DataFrame(features))
        else:
            explanation = {'warning': 'SHAP not available on server'}

        resp = {
            'route': route,
            'day_of_week': int(day_of_week),
            'month': int(month),
            'season': season,
            'weather': weather,
            'predicted_delay_min': round(float(prediction_val), 2),
            'model_used': ('v2' if model_v2 is not None else 'v1'),
            'explanation': explanation
        }

        return jsonify(_make_json_serializable(resp))

    except Exception as e:
        import traceback
        print(f"Error in explain endpoint: {e}")
        traceback.print_exc()
        return jsonify(_make_json_serializable({'error': str(e)})), 500


@app.route('/api/predict/propagate', methods=['POST'])
def propagate():
    """Run a simple delay propagation scenario.

    Expects JSON body:
    {
        "injections": [{"node":"A", "delay": 15}, ...],
        "edges": [["A","B",10], ["B","C",5]],
        "recovery_margin": 5.0
    }
    Returns final per-node delays and traces.
    """
    try:
        data = request.get_json() or {}
        injections = data.get('injections', [])
        edges = data.get('edges', [])
        recovery = float(data.get('recovery_margin', 5.0))

        if not edges:
            return jsonify(_make_json_serializable({'error': 'No edges provided in scenario'})), 400

        # Build edges tuple list
        edges_list = []
        for e in edges:
            try:
                src, dst, t = e[0], e[1], float(e[2])
                edges_list.append((src, dst, {'transfer_time': t}))
            except Exception:
                continue

        # Import propagation utilities locally to avoid circular import on module load
        try:
            from backend import propagation as propagation_mod
        except Exception:
            import importlib.util
            spec = importlib.util.spec_from_file_location('propagation', os.path.join(os.path.dirname(__file__), 'propagation.py'))
            propagation_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(propagation_mod)

        G = propagation_mod.build_dependency_graph(edges_list)
        init = {str(i['node']): float(i.get('delay', 0.0)) for i in injections}
        final_delays, traces = propagation_mod.simulate_propagation(G, init, recovery_margin=recovery)

        return jsonify(_make_json_serializable({'final_delays': final_delays, 'traces': traces}))

    except Exception as e:
        import traceback
        print(f"Error in propagate endpoint: {e}")
        traceback.print_exc()
        return jsonify(_make_json_serializable({'error': str(e)})), 500


@app.route('/api/predict/propagate/backtest', methods=['POST'])
def propagate_backtest():
    """Run a backtest: run propagation with predicted initial delays and compare against observed final delays.

    Request body JSON:
    {
      "injections": [{"node":"A","delay":15}, ...],
      "edges": [["A","B",10], ...],
      "observed_final": {"A":15, "B":20, ...},
      "recovery_margin": 5.0
    }
    """
    try:
        data = request.get_json() or {}
        injections = data.get('injections', [])
        edges = data.get('edges', [])
        observed_final = data.get('observed_final', {})
        recovery = float(data.get('recovery_margin', 5.0))

        if not edges:
            return jsonify(_make_json_serializable({'error': 'No edges provided in scenario'})), 400
        if not observed_final:
            return jsonify(_make_json_serializable({'error': 'Missing observed_final mapping for backtest'})), 400

        edges_list = []
        for e in edges:
            try:
                src, dst, t = e[0], e[1], float(e[2])
                edges_list.append((src, dst, {'transfer_time': t}))
            except Exception:
                continue

        # dynamic import for propagation utilities
        try:
            from backend import propagation as propagation_mod
        except Exception:
            import importlib.util
            spec = importlib.util.spec_from_file_location('propagation', os.path.join(os.path.dirname(__file__), 'propagation.py'))
            propagation_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(propagation_mod)

        G = propagation_mod.build_dependency_graph(edges_list)
        init = {str(i['node']): float(i.get('delay', 0.0)) for i in injections}
        max_delay = float(data.get('max_delay_minutes', 24*60))
        simulated_final, traces, metrics = propagation_mod.backtest_propagation(G, init, observed_final, recovery_margin=recovery, max_delay=max_delay)

        # Generate visualization and return it as base64 PNG
        fig, ax = propagation_mod.visualize_propagation(G, simulated_final, traces)
        img_b64 = propagation_mod._fig_to_base64(fig)
        viz_available = bool(getattr(propagation_mod, 'MPL_AVAILABLE', True))
        viz_warning = None if viz_available else 'matplotlib not installed - visualization is a placeholder. Install matplotlib for richer graphics.'

        return jsonify(_make_json_serializable({'metrics': metrics, 'simulated_final': simulated_final, 'traces': traces, 'viz_base64_png': img_b64, 'viz_available': viz_available, 'viz_warning': viz_warning}))

    except Exception as e:
        import traceback
        print(f"Error in propagate_backtest endpoint: {e}")
        traceback.print_exc()
        return jsonify(_make_json_serializable({'error': str(e)})), 500


@app.route('/api/predict/propagate/historical', methods=['POST'])
def propagate_historical():
    """Run a historical-day propagation/backtest using scheduled trains for a given date.

    Request body JSON:
    {
      "date": "2025-12-15",
      "station": "HYB" (optional, restrict graph to a station),
      "max_transfer_minutes": 180,
      "recovery_margin": 5.0
    }

    Returns metrics, simulated_final delays, a base64 PNG visualization and top affected trains.
    """
    try:
        data = request.get_json() or {}
        date_str = data.get('date')
        station_filter = data.get('station')
        max_transfer = int(data.get('max_transfer_minutes', 180))
        recovery = float(data.get('recovery_margin', 5.0))

        if not date_str:
            return jsonify(_make_json_serializable({'error': 'Missing required field: date'})), 400

        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            day_of_week = date_obj.weekday() + 1
            month = date_obj.month
        except Exception:
            return jsonify(_make_json_serializable({'error': 'Invalid date format. Use YYYY-MM-DD'})), 400

        # Filter trains running that day
        try:
            day_trains = train_data[train_data['day_of_week'] == int(day_of_week)].copy()
        except Exception:
            day_trains = train_data.copy()

        if station_filter:
            station_filter = str(station_filter).strip().upper()
            if 'station_list' in day_trains.columns:
                mask = day_trains['station_list'].astype(str).str.upper().str.contains(station_filter, na=False)
                day_trains = day_trains[mask].copy()

        # Limit to reasonable size for simulation
        if len(day_trains) > 1200:
            day_trains = day_trains.sample(n=1200, random_state=42).copy()

        # Build graph from scheduled trains
        try:
            from backend import propagation as propagation_mod
        except Exception:
            import importlib.util
            spec = importlib.util.spec_from_file_location('propagation', os.path.join(os.path.dirname(__file__), 'propagation.py'))
            propagation_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(propagation_mod)

        G = propagation_mod.build_historical_graph_from_trains(day_trains, max_transfer_minutes=max_transfer)

        # Initialize predicted delays using model predictions per train
        init = {}
        observed_final = {}
        for _, row in day_trains.iterrows():
            tid = str(row.get('train_id'))
            # route string
            route = f"{row.get('source')}-{row.get('destination')}"
            dist = float(row.get('distance_km') or 0)
            pred = predict_delay_cached(route, int(day_of_week), int(month), dist, get_weather_data(row.get('source')).get('condition'), get_season(month))
            init[tid] = float(pred)
            # Use avg_delay_min (historical average) as observed final if available
            try:
                obs = float(row.get('avg_delay_min')) if row.get('avg_delay_min') not in (None, '', 'nan') else 0.0
            except Exception:
                obs = 0.0
            observed_final[tid] = obs

        # Fast-mode sampling and safer defaults
        mode = str(data.get('mode', 'fast')).lower()
        mode_params = {}
        n_before = len(day_trains)
        # choose default params
        if mode == 'fast':
            sample_n = int(data.get('sample_n', 300))
            # select top-N by absolute predicted initial delay
            selected = sorted(init.items(), key=lambda kv: abs(kv[1]), reverse=True)[:sample_n]
            selected_ids = set([tid for tid, _ in selected])
            if len(selected_ids) < n_before:
                day_trains = day_trains[day_trains['train_id'].astype(str).isin(selected_ids)].copy()
                # rebuild graph on sampled trains using possibly tighter transfer window
                max_transfer = min(max_transfer, int(data.get('fast_max_transfer', 120)))
                G = propagation_mod.build_historical_graph_from_trains(day_trains, max_transfer_minutes=max_transfer)
                # shrink init/observed_final dictionaries
                init = {k: v for k, v in init.items() if k in selected_ids}
                observed_final = {k: v for k, v in observed_final.items() if k in selected_ids}
            mode_params = {'sample_n': sample_n, 'fast_max_transfer': int(data.get('fast_max_transfer', 120)), 'fast_max_iters': int(data.get('fast_max_iters', 5)), 'fast_max_delay_minutes': float(data.get('fast_max_delay_minutes', 720))}
            max_iters = int(data.get('fast_max_iters', 5))
            max_delay_minutes = float(data.get('fast_max_delay_minutes', 720))
        else:
            max_iters = int(data.get('max_iters', 10))
            max_delay_minutes = float(data.get('max_delay_minutes', 24*60))

        n_after = len(day_trains)

        simulated_final, traces, metrics = propagation_mod.backtest_propagation(G, init, observed_final, recovery_margin=recovery, max_delay=max_delay_minutes, max_iters=max_iters)

        # Create visualization
        fig, ax = propagation_mod.visualize_propagation(G, simulated_final, traces)
        img_b64 = propagation_mod._fig_to_base64(fig)

        # Top affected trains (largest simulated delays)
        diffs = [{
            'train_id': t,
            'simulated': float(simulated_final.get(t, 0.0)),
            'observed': float(observed_final.get(t, 0.0)),
            'delta': float(simulated_final.get(t, 0.0)) - float(observed_final.get(t, 0.0))
        } for t in simulated_final]
        top_affected = sorted(diffs, key=lambda x: x['simulated'], reverse=True)[:10]

        viz_available = bool(getattr(propagation_mod, 'MPL_AVAILABLE', True))
        viz_warning = None if viz_available else 'matplotlib not installed - visualization is a placeholder. Install matplotlib for richer graphics.'

        resp = {
            'mode': mode,
            'mode_params': mode_params,
            'n_trains_before': n_before,
            'n_trains_after': n_after,
            'metrics': metrics,
            'n_trains': int(len(G.nodes)),
            'simulated_final': simulated_final,
            'traces': traces,
            'viz_base64_png': img_b64,
            'viz_available': viz_available,
            'viz_warning': viz_warning,
            'top_affected': top_affected
        }
        return jsonify(_make_json_serializable(resp))

    except Exception as e:
        import traceback
        print(f"Error in propagate_historical endpoint: {e}")
        traceback.print_exc()
        return jsonify(_make_json_serializable({'error': str(e)})), 500

def _get_price_for_train_segment(train_id, origin, dest, station_list=None):
    """
    Return price for the exact train segment (train_id, origin, dest) using price_lookup_dict.
    If exact pair not found, returns None.
    """
    try:
        key = (str(train_id).zfill(5), origin.strip().upper(), dest.strip().upper())
        if 'price_lookup_dict' in globals() and key in price_lookup_dict and price_lookup_dict[key] not in (None, ''):
            return float(price_lookup_dict[key])
    except Exception:
        pass
    return None


def _ensure_price_lookup_loaded(force=False):
    """Lazily load datasets/price_lookup.csv and build PRICE_LOOKUP_DF, PRICE_RATE_CACHE, PRICE_GLOBAL_RATE.
    This allows tests to monkeypatch pd.read_csv for price_lookup.csv without a full load_model() call.
    If force=True, reload even if PRICE_LOOKUP_DF is already set (useful for tests that monkeypatch pd.read_csv).
    """
    global PRICE_LOOKUP_DF, PRICE_RATE_CACHE, PRICE_GLOBAL_RATE, PRICE_GLOBAL_RATE_SOURCE
    if PRICE_LOOKUP_DF is not None and not force:
        return
    try:
        datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')
        price_lookup_file = os.path.join(datasets_dir, 'price_lookup.csv')
        if os.path.exists(price_lookup_file):
            df = pd.read_csv(price_lookup_file)
            PRICE_LOOKUP_DF = df.copy()
            pl = PRICE_LOOKUP_DF
            try:
                pl['avg_distance'] = pl['avg_distance'].astype(float)
                pl['avg_price'] = pl['avg_price'].astype(float)
            except Exception:
                pass
            pl = pl[pd.notna(pl['avg_distance']) & (pl['avg_distance'] > 0)]
            pl['rate'] = pl['avg_price'] / pl['avg_distance']
            PRICE_RATE_CACHE = {}
            for tid, grp in pl.groupby('train_id'):
                rates = grp['rate'].replace([np.inf, -np.inf], np.nan).dropna()
                if rates.size >= PRICE_MIN_RATE_CANDIDATES:
                    PRICE_RATE_CACHE[str(tid).zfill(5)] = float(rates.median())
            all_rates = pl['rate'].replace([np.inf, -np.inf], np.nan).dropna()
            PRICE_GLOBAL_RATE = float(all_rates.median()) if not all_rates.empty else 1.0
            # Only mark as 'lazy' if load_model hasn't already computed it (preserve authoritative load_model source)
            if not ('PRICE_GLOBAL_RATE_SOURCE' in globals() and PRICE_GLOBAL_RATE_SOURCE == 'load_model'):
                PRICE_GLOBAL_RATE_SOURCE = 'lazy'
    except Exception:
        PRICE_LOOKUP_DF = None
        PRICE_RATE_CACHE = {}
        PRICE_GLOBAL_RATE = 1.0
        PRICE_GLOBAL_RATE_SOURCE = 'lazy'


def _estimate_segment_price(train_id, origin, dest, seg_distance, train_price=None):
    """Estimate a segment price using lookup, per-train rate, or fallbacks.
    Returns (price_float, source_str) where source indicates the method used.
    Improvements:
    - If exact pair isn't present, try the reversed (dest->origin) lookup and scale it by distance.
    - Use per-train cached median rate when available, otherwise fall back to a global median rate computed at load time.
    - Cap estimates relative to train_price to avoid wildly large segment estimates.
    """
    # Ensure price lookup data is available (for tests that don't call load_model())
    # Load lazily but avoid forcing a reload on every call to prevent repeated disk I/O
    try:
        _ensure_price_lookup_loaded(force=False)
    except Exception:
        pass
    # Start timer for lightweight profiling of slow paths
    start = time.perf_counter()
    # Normalize ids/codes    # Normalize ids/codes
    tid = str(train_id).zfill(5) if train_id is not None else ''
    origin_code = origin.strip().upper() if isinstance(origin, str) else ''
    dest_code = dest.strip().upper() if isinstance(dest, str) else ''

    # 1) Exact lookup
    exact = _get_price_for_train_segment(tid, origin_code, dest_code)
    if exact is not None:
        return float(exact), 'lookup'

    # 1b) Try reversed lookup and scale by requested distance if possible
    try:
        if PRICE_LOOKUP_DF is not None and seg_distance and float(seg_distance) > 0:
            mask = (
                (PRICE_LOOKUP_DF['train_id'] == str(tid).zfill(5)) &
                (PRICE_LOOKUP_DF['source_code'] == dest_code) &
                (PRICE_LOOKUP_DF['destination_code'] == origin_code)
            )
            if mask.any():
                r = PRICE_LOOKUP_DF[mask].iloc[0]
                if pd.notna(r.get('avg_distance', None)) and r.get('avg_distance', 0) > 0:
                    rate = float(r['avg_price']) / float(r['avg_distance'])
                    est = rate * float(seg_distance)
                    # Cap relative to train price if provided
                    try:
                        if train_price and float(train_price) > 0:
                            if est > float(train_price) * PRICE_MAX_TRAIN_PRICE_RATIO:
                                est = float(train_price) * PRICE_MAX_TRAIN_PRICE_RATIO
                    except Exception:
                        pass
                    return float(max(1.0, round(est))), 'lookup_reversed'
    except Exception:
        pass

    # 2) Use cached per-train rate if available
    try:
        if seg_distance and float(seg_distance) > 0:
            rate = PRICE_RATE_CACHE.get(str(tid).zfill(5)) if 'PRICE_RATE_CACHE' in globals() else None
            if rate is not None and not np.isnan(rate):
                est = float(rate) * float(seg_distance)
                # Cap relative to train price if provided
                try:
                    if train_price and float(train_price) > 0 and est > float(train_price) * PRICE_MAX_TRAIN_PRICE_RATIO:
                        est = float(train_price) * PRICE_MAX_TRAIN_PRICE_RATIO
                except Exception:
                    pass
                return float(max(1.0, round(est))), 'estimated_by_rate'
            # If no per-train rate, try global fallback rate
            global_rate = PRICE_GLOBAL_RATE if 'PRICE_GLOBAL_RATE' in globals() and PRICE_GLOBAL_RATE is not None else None
            if global_rate is not None:
                # If there are some price rows for this train but too few candidates to compute a stable per-train rate,
                # prefer returning the explicit `train_price` when one is provided (to avoid misleading global estimates).
                try:
                    tid_rows = PRICE_LOOKUP_DF[PRICE_LOOKUP_DF['train_id'] == str(tid).zfill(5)] if PRICE_LOOKUP_DF is not None else None
                    if train_price and tid_rows is not None and len(tid_rows) < PRICE_MIN_RATE_CANDIDATES:
                        # Prefer returning a conservative fraction of full train price rather than the entire train price for short segments
                        try:
                            frac = 0.25
                            est_frac = float(train_price) * frac
                            return float(max(1.0, int(round(est_frac)))), 'train_price_fraction_fallback'
                        except Exception:
                            return float(train_price), 'train_price'
                except Exception:
                    pass

                # Prefer a per-km global estimate for short segments instead of returning full train price,
                # and only use a conservative fraction of `train_price` when the per-km estimate is implausibly high.
                try:
                    est = float(global_rate) * float(seg_distance)
                except Exception:
                    est = None

                # If a train_price is provided, compare and decide
                try:
                    if train_price and float(train_price) > 0 and est is not None:
                        tprice = float(train_price)
                        # If global estimate is small relative to train price, use it (capped by ratio)
                        if est <= tprice * PRICE_MAX_TRAIN_PRICE_RATIO:
                            est = min(est, tprice * PRICE_MAX_TRAIN_PRICE_RATIO)
                            elapsed = time.perf_counter() - start
                            if elapsed > 0.5:
                                print(f"_estimate_segment_price(global_rate_chosen): {elapsed:.3f}s for {tid} {origin_code}->{dest_code}")
                            src = 'estimated_by_rate_global'
                            return float(max(1.0, round(est))), src
                        # Otherwise fall back to a conservative fraction of train price (e.g., 25%)
                        frac = 0.25
                        est_frac = tprice * frac
                        elapsed = time.perf_counter() - start
                        if elapsed > 0.5:
                            print(f"_estimate_segment_price(train_price_fraction): {elapsed:.3f}s for {tid} {origin_code}->{dest_code}")
                        return float(max(1.0, int(round(est_frac)))), 'train_price_fraction_fallback'
                except Exception:
                    pass

                # If no train_price provided, use global per-km estimate if available
                if est is not None:
                    try:
                        # If there are station-based matches in the price lookup (but not for this train), prefer the rate-based label
                        if PRICE_LOOKUP_DF is not None and ((PRICE_LOOKUP_DF['source_code'] == origin_code).any() or (PRICE_LOOKUP_DF['destination_code'] == dest_code).any()):
                            src = 'estimated_by_rate_global'
                        else:
                            src = 'estimated_by_global_rate'
                    except Exception:
                        src = 'estimated_by_rate_global'
                    elapsed = time.perf_counter() - start
                    if elapsed > 0.5:
                        print(f"_estimate_segment_price(global_rate_only): {elapsed:.3f}s for {tid} {origin_code}->{dest_code}")
                    return float(max(1.0, round(est))), src
    except Exception:
        pass

    # 3) Use proportional train price if available and we cannot derive a rate
    try:
        if train_price and float(train_price) > 0 and seg_distance and float(seg_distance) > 0:
            # Try to get train full-route distance from PRICE_LOOKUP_DF if available
            train_rows = PRICE_LOOKUP_DF[PRICE_LOOKUP_DF['train_id'] == str(tid).zfill(5)] if PRICE_LOOKUP_DF is not None else None
            full_dist = None
            try:
                if train_rows is not None and not train_rows.empty:
                    # Use the maximum recorded avg_distance as an approximation of full train distance
                    full_dist = float(train_rows['avg_distance'].max())
            except Exception:
                full_dist = None

            if full_dist and float(full_dist) > 0:
                frac = max(0.01, min(0.98, float(seg_distance) / float(full_dist)))
                est = float(train_price) * frac
                elapsed = time.perf_counter() - start
                if elapsed > 0.5:
                    print(f"_estimate_segment_price(train_fraction): {elapsed:.3f}s for {tid} {origin_code}->{dest_code}")
                return float(max(1.0, int(round(est)))), 'train_price_estimated_by_fraction'
            else:
                # No reliable full-route distance: prefer using global per-km rate if available,
                # capped to `PRICE_MAX_TRAIN_PRICE_RATIO` of the known full train price to avoid inflated segment estimates.
                global_rate = PRICE_GLOBAL_RATE if 'PRICE_GLOBAL_RATE' in globals() and PRICE_GLOBAL_RATE is not None else None
                if global_rate is not None:
                    est = float(global_rate) * float(seg_distance)
                    try:
                        if train_price and float(train_price) > 0 and est > float(train_price) * PRICE_MAX_TRAIN_PRICE_RATIO:
                            est = float(train_price) * PRICE_MAX_TRAIN_PRICE_RATIO
                    except Exception:
                        pass
                    elapsed = time.perf_counter() - start
                    if elapsed > 0.5:
                        print(f"_estimate_segment_price(global_rate): {elapsed:.3f}s for {tid} {origin_code}->{dest_code}")
                    return float(max(1.0, int(round(est)))), 'train_price_estimated_by_global_rate'
                else:
                    # Conservative fallback: use a fraction of the train_price (25%) instead of the full price
                    try:
                        frac = 0.25
                        est = float(train_price) * frac
                        elapsed = time.perf_counter() - start
                        if elapsed > 0.5:
                            print(f"_estimate_segment_price(fraction_fallback): {elapsed:.3f}s for {tid} {origin_code}->{dest_code}")
                        return float(max(1.0, int(round(est)))), 'train_price_fraction_fallback'
                    except Exception:
                        elapsed = time.perf_counter() - start
                        if elapsed > 0.5:
                            print(f"_estimate_segment_price(fallback_full): {elapsed:.3f}s for {tid} {origin_code}->{dest_code}")
                        return float(train_price), 'train_price'
    except Exception:
        pass

    # 4) Distance fallback: use global rate if available, otherwise coarse multiplier (1.0/km)
    try:
        if seg_distance and float(seg_distance) > 0:
            global_rate = PRICE_GLOBAL_RATE if 'PRICE_GLOBAL_RATE' in globals() and PRICE_GLOBAL_RATE is not None else 1.0
            est = float(seg_distance) * float(global_rate)
            return float(max(1.0, int(round(est)))), 'distance_fallback'
    except Exception:
        pass

    return None, 'unknown'


def get_parsed_station_list_for_train(train_id, station_list_str):
    """Return parsed station list for a train using a simple global cache to avoid repeated JSON parsing."""
    try:
        tid = str(train_id).zfill(5) if train_id is not None else 'unknown'
        if tid in STATION_LIST_CACHE:
            return STATION_LIST_CACHE[tid]
        if station_list_str is None:
            STATION_LIST_CACHE[tid] = []
            return []
        parsed = json.loads(str(station_list_str).replace("'", '"'))
        STATION_LIST_CACHE[tid] = parsed if isinstance(parsed, list) else []
        return STATION_LIST_CACHE[tid]
    except Exception:
        STATION_LIST_CACHE[tid] = []
        return []
    

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Return ranked train recommendations based on user preference (Features 3, 4, 5)"""
    try:
        data = request.get_json()

        source = data.get('source', '')
        destination = data.get('destination', '')
        preference = data.get('preference', 'fastest')  # fastest, cheapest, most_reliable
        travel_date = data.get('travel_date', '')

        if not all([source, destination]):
            return jsonify(_make_json_serializable({'error': 'Missing required fields'})), 400

        # Parse travel date
        try:
            date_obj = datetime.strptime(travel_date, '%Y-%m-%d')
            day_of_week = date_obj.weekday() + 1
            month = date_obj.month
        except:
            return jsonify(_make_json_serializable({'error': 'Invalid date format. Use YYYY-MM-DD'})), 400

        # Get weather data
        weather = get_weather_data(source)
        weather_condition = weather['condition']
        season = get_season(month)
        route = f"{source}-{destination}"

        # Normalize inputs for robust matching
        if isinstance(source, str):
            source = source.strip().upper()
        if isinstance(destination, str):
            destination = destination.strip().upper()

        # Filter trains for the route and day of week (trains that pass through the route)
        available_trains = get_trains_passing_through(source, destination, day_of_week)

        # If no direct trains found, check for intermediate segment trains
        if available_trains.empty:
            # First check for pre-calculated intermediate segments
            intermediate_trains = train_data[
                (train_data['source'] == source) &
                (train_data['destination'] == destination) &
                (train_data['day_of_week'] == day_of_week) &
                (train_data['is_intermediate_segment'].fillna(False) == True)
            ].copy()

            if not intermediate_trains.empty:
                print(f"Using {len(intermediate_trains)} pre-calculated intermediate segments for {source}-{destination}")
                available_trains = intermediate_trains
            else:
                # If no pre-calculated intermediate segments, look for trains that pass through this route
                intermediate_train = find_intermediate_segment_trains(source, destination, day_of_week, month, weather_condition, season)
                if intermediate_train:
                    print(f"Found intermediate segment train: {intermediate_train['train_name']}")
                    # Convert to dataframe format for processing
                    available_trains = pd.DataFrame([intermediate_train])
                else:
                    print(f"No intermediate segments found for {source}-{destination}")

        # Feature 5: If no direct trains, find connecting trains with recommendations
        if available_trains.empty:
            connecting_route = find_connecting_trains(source, destination, day_of_week, month, weather_condition, season)
            if connecting_route:
                # Generate recommendations for connecting routes based on preference
                connecting_recommendations = []
                base_speed_kmph = 55.0  # Average train speed
                total_distance = connecting_route['total_distance']
                total_price = connecting_route['total_price']
                total_delay = connecting_route.get('total_delay', 0)

                # Calculate metrics for connecting route
                total_time = (total_distance / base_speed_kmph) + (total_delay / 60)  # hours
                connecting_speed = total_distance / total_time
                connecting_reliability = max(0, 100 - total_delay)
                connecting_value = (connecting_reliability * connecting_speed) / total_price * 1000

                # Create connecting route recommendation
                connecting_rec = {
                    'connection_type': 'connecting',
                    'connecting_station': connecting_route['connecting_station'],
                    'train_id': f"{connecting_route['train1']['train_id']}+{connecting_route['train2']['train_id']}",
                    'train_name': f"Connecting: {connecting_route['train1']['train_name']} → {connecting_route['train2']['train_name']}",
                    'departure_time': connecting_route['train1']['departure_time'],
                    'arrival_time': None,  # Connecting routes don't have single arrival time
                    'price': int(float(total_price)),                 # Convert pandas int64
                    'predicted_delay_min': round(float(total_delay), 1),  # Already float
                    'reliability_score': round(float(connecting_reliability), 1),   # Already float
                    'speed_kmph': round(float(connecting_speed), 1),       # Already float
                    'distance_km': int(float(total_distance)),             # Convert pandas int64
                    'estimated_journey_minutes': round(float(total_time * 60), 0),  # Already float
                    'value_score': round(float(connecting_value), 2),      # Already float
                    'recommendation_reason': 'Connecting route with layover',
                    'delay_category': 'Minor Delay' if total_delay <= 30 else 'Significant Delay',
                    'layover_minutes': int(float(connecting_route.get('layover_time', 90))),  # Convert pandas int64
                    'rank': 1,
                    'is_best': True,
                    'tags': ['Connecting Route']
                }

                if preference == 'fastest':
                    connecting_recommendations = [connecting_rec]
                elif preference == 'cheapest':
                    connecting_recommendations = [connecting_rec]
                else:
                    connecting_recommendations = [connecting_rec]

                return jsonify(_make_json_serializable({
                    'has_direct_trains': False,
                    'connecting_route': connecting_route,
                    'recommendations': connecting_recommendations,
                    'best_route': connecting_rec,
                    'message': 'No direct trains found. Showing optimized connecting route with delay predictions.',
                    'preference': preference,
                    'weather': weather,
                    'total_trains': 0  # No direct trains
                }))
            else:
                return jsonify(_make_json_serializable({'error': 'No trains or connecting routes found for this route on the selected date'})), 404

        # Process all trains (including all trains with different times)
        # Calculate enhanced recommendations for all trains
        recommendations = []
        for _, train in available_trains.iterrows():
            # Convert to native Python types
            train_distance = float(train['distance_km']) if pd.notna(train['distance_km']) else 0.0
            train_price = float(train['price']) if pd.notna(train['price']) else 0.0

            # Extract departure and arrival times
            departure_time = None
            arrival_time = None

            # Try to get times from stored columns first
            if pd.notna(train.get('departure_time')):
                departure_time = str(train['departure_time'])
            if pd.notna(train.get('arrival_time')):
                arrival_time = str(train['arrival_time'])

            # If times not available, try to extract from station_list
            if (not departure_time or not arrival_time) and pd.notna(train.get('station_list')):
                try:
                    station_list = get_parsed_station_list_for_train(train.get('train_id'), train.get('station_list'))
                    for station in station_list:
                        station_code = station.get('stationCode', '')
                        if station_code == source and not departure_time:
                            dep_time = station.get('departureTime', '')
                            if dep_time and dep_time != '--':
                                departure_time = dep_time
                        if station_code == destination and not arrival_time:
                            arr_time = station.get('arrivalTime', '')
                            if arr_time and arr_time != '--':
                                arrival_time = arr_time
                except:
                    pass

            predicted_delay = predict_delay_cached(route, day_of_week, month, train_distance, weather_condition, season)
            # Ensure predicted_delay is numeric
            try:
                predicted_delay = float(predicted_delay) if predicted_delay is not None else 0.0
            except Exception:
                print(f"Warning: non-numeric predicted_delay for train {train.get('train_id')}: {predicted_delay}")
                predicted_delay = 0.0

            # Adjust for this specific train using per-train baseline and deterministic sampling
            try:
                predicted_delay = _adjust_prediction_for_train(train.get('train_id'), predicted_delay, travel_date)
            except Exception:
                pass

            # Calculate reliability score (inverse of delay)
            reliability_score = max(0, 100 - predicted_delay)

            # Calculate speed (prefer scheduled times when available)
            def _parse_time_to_min(t):
                try:
                    if not t or t in ['--', None]:
                        return None
                    t = str(t).strip()
                    parts = t.split(':')
                    if len(parts) < 2:
                        return None
                    h = int(parts[0]) % 24
                    m = int(parts[1])
                    return h * 60 + m
                except Exception:
                    return None

            def _scheduled_minutes_from_row(tr, src, dst):
                # Try explicit departure_time/arrival_time first
                dep = tr.get('departure_time')
                arr = tr.get('arrival_time')
                dep_min = _parse_time_to_min(dep)
                arr_min = _parse_time_to_min(arr)
                if dep_min is not None and arr_min is not None:
                    dur = (arr_min - dep_min) % (24 * 60)
                    if dur <= 0:
                        dur = max(30, int(train_distance / 60 * 60))
                    return dur

                # Else try station_list
                if tr.get('station_list'):
                    try:
                        station_list = get_parsed_station_list_for_train(tr.get('train_id'), tr.get('station_list'))
                        dep_min = None
                        arr_min = None
                        for s in station_list:
                            code = (s.get('stationCode') or '').strip().upper()
                            if code == src and not dep_min:
                                dep_min = _parse_time_to_min(s.get('departureTime'))
                            if code == dst and not arr_min:
                                arr_min = _parse_time_to_min(s.get('arrivalTime'))
                        if dep_min is not None and arr_min is not None:
                            dur = (arr_min - dep_min) % (24 * 60)
                            if dur <= 0:
                                dur = max(30, int(train_distance / 60 * 60))
                            return dur
                    except Exception:
                        pass

                # Fallback: estimate from distance using 60 km/h
                return max(30, int(train_distance / 60 * 60))

            sched_minutes = _scheduled_minutes_from_row(train, source, destination)
            base_time_hours = max(0.5, float(sched_minutes) / 60.0)
            total_time_hours = base_time_hours + (predicted_delay / 60.0)
            speed = train_distance / total_time_hours if total_time_hours > 0 else 0

            # Calculate estimated journey time (base time + delay)
            estimated_journey_hours = total_time_hours
            estimated_journey_minutes = int(estimated_journey_hours * 60)

            # Calculate value for money score (higher is better)
            # Formula: (reliability * speed) / price * 1000
            value_score = (reliability_score * speed) / max(train_price, 1) * 1000 if train_price > 0 else 0

            # Determine train category/type
            train_type = "Express"
            if "SF" in str(train['train_name']).upper() or "SUPERFAST" in str(train['train_name']).upper():
                train_type = "Superfast"
            elif "RAJDHANI" in str(train['train_name']).upper() or "SHATABDI" in str(train['train_name']).upper():
                train_type = "Premium"
            elif "SPECIAL" in str(train['train_name']).upper() or "SPL" in str(train['train_name']).upper():
                train_type = "Special"

            # Generate recommendation reason
            reasons = []
            if predicted_delay < 10:
                reasons.append("Very low delay expected")
            if reliability_score > 90:
                reasons.append("Highly reliable service")
            if speed > 70:
                reasons.append("Fast journey time")
            if train_price < 1000:
                reasons.append("Budget-friendly fare")
            if train_type == "Premium":
                reasons.append("Premium train with better amenities")

            recommendation_reason = "; ".join(reasons) if reasons else "Good option for this route"

            # Ensure price is realistic: try compact price lookup for this train and user-specified source/destination first
            price_val = None
            try:
                tid = str(train.get('train_id')).zfill(5)
                lookup_price = _get_price_for_train_segment(tid, source, destination, train.get('station_list'))
                if lookup_price is not None:
                    price_val = int(round(lookup_price))
                else:
                    # Estimate price for the segment using helper (may use price_lookup.csv rates or fallback to train price)
                    price_val, price_src = _estimate_segment_price(tid, source, destination, 0, train_price=train_price)
                    # If we have a station_list with distances, prefer an estimate based on exact segment distance
                    station_list_str = train.get('station_list')
                    if station_list_str:
                        try:
                            sl = get_parsed_station_list_for_train(train.get('train_id'), station_list_str)
                        except Exception:
                            sl = None

                        if sl:
                            origin_dist = None
                            dest_dist = None
                            for s in sl:
                                code = s.get('stationCode', '').strip().upper()
                                if code == source:
                                    origin_dist = float(s.get('distance', 0) or 0)
                                if code == destination:
                                    dest_dist = float(s.get('distance', 0) or 0)

                            if origin_dist is not None and dest_dist is not None and dest_dist >= origin_dist:
                                seg_distance = dest_dist - origin_dist
                                est_price2, price_src2 = _estimate_segment_price(tid, source, destination, seg_distance, train_price=train_price)
                                if est_price2 is not None:
                                    price_val = est_price2
                                    price_src = price_src2
                    # Ensure we have a numeric price
                    try:
                        price_val = int(float(price_val)) if price_val is not None else 0
                    except Exception:
                        price_val = int(price_val or 0)
                    # record source for diagnostics
                    price_source = price_src if 'price_src' in locals() else ('lookup' if _get_price_for_train_segment(tid, source, destination) is not None else 'fallback')
            except Exception:
                price_val = int(price_val or 0)

            recommendation = {
                'train_id': str(train['train_id']),
                'train_name': str(train['train_name']),
                'train_type': str(train_type),
                'departure_time': departure_time if departure_time else None,
                'arrival_time': arrival_time if arrival_time else None,
                'price': int(float(price_val or 0)),  # Convert to Python float first
                'predicted_delay_min': round(float(predicted_delay), 1),  # Already float
                'reliability_score': round(float(reliability_score), 1),   # Already float
                'speed_kmph': round(float(speed), 1),                     # Already float
                'distance_km': int(float(train_distance)),               # Convert to Python float first
                'estimated_journey_hours': round(float(estimated_journey_hours), 1),  # Already float
                'estimated_journey_minutes': int(float(estimated_journey_minutes)),   # Convert to Python float first
                'value_score': round(float(value_score), 2),             # Already float
                'recommendation_reason': str(recommendation_reason),
                'delay_category': str('On Time' if predicted_delay <= 15 else ('Minor Delay' if predicted_delay <= 30 else 'Significant Delay'))
            }
            recommendations.append(recommendation)

        # Dedupe recommendations so each train appears once (choose best representative)
        pref_key = 'predicted_delay_min'
        pref_higher = False
        if preference == 'fastest':
            pref_key = 'speed_kmph'; pref_higher = True
        elif preference == 'cheapest':
            pref_key = 'price'; pref_higher = False
        elif preference == 'most_reliable':
            pref_key = 'reliability_score'; pref_higher = True

        unique_recommendations = dedupe_trains_by_id(recommendations, prefer_key=pref_key, prefer_higher=pref_higher)

        # Create multiple recommendation categories from deduped list
        fastest_train = max(unique_recommendations, key=lambda x: x['speed_kmph']) if unique_recommendations else None
        cheapest_train = min(unique_recommendations, key=lambda x: x['price']) if unique_recommendations else None
        most_reliable_train = max(unique_recommendations, key=lambda x: x['reliability_score']) if unique_recommendations else None
        best_value_train = max(unique_recommendations, key=lambda x: x['value_score']) if unique_recommendations else None

        # Sort based on preference (use deduped list)
        if preference == 'fastest':
            unique_recommendations.sort(key=lambda x: x['speed_kmph'], reverse=True)
            best_route = fastest_train
        elif preference == 'cheapest':
            unique_recommendations.sort(key=lambda x: x['price'])
            best_route = cheapest_train
        elif preference == 'most_reliable':
            unique_recommendations.sort(key=lambda x: x['reliability_score'], reverse=True)
            best_route = most_reliable_train
        else:
            # Default: best value
            unique_recommendations.sort(key=lambda x: x['value_score'], reverse=True)
            best_route = best_value_train

        # Limit to top 3 recommendations only (from deduped list)
        # Exclude the highlighted `best_route` from the recommendations list to avoid repeated display
        if best_route:
            other_recs = [r for r in unique_recommendations if r['train_id'] != best_route['train_id']]
            # If excluding the best route leaves no recommendations, fall back to including the best (avoid empty list)
            top_recommendations = other_recs[:3] if other_recs else [best_route]
        else:
            top_recommendations = unique_recommendations[:3]

        # Add ranking and tags
        for i, rec in enumerate(top_recommendations):
            rec['rank'] = i + 1
            rec['is_best'] = bool(rec['train_id'] == best_route['train_id'] if best_route else False)

            # Add tags
            tags = []
            if rec['train_id'] == fastest_train['train_id']:
                tags.append('Fastest')
            if rec['train_id'] == cheapest_train['train_id']:
                tags.append('Cheapest')
            if rec['train_id'] == most_reliable_train['train_id']:
                tags.append('Most Reliable')
            if rec['train_id'] == best_value_train['train_id']:
                tags.append('Best Value')
            rec['tags'] = tags

        # Create recommendation summary (ensure pandas numeric types are converted)
        recommendation_summary = {
            'fastest': fastest_train,
            'cheapest': cheapest_train,
            'most_reliable': most_reliable_train,
            'best_value': best_value_train,
            'total_options': int(len(unique_recommendations))  # Convert pandas int64
        }

        return jsonify(_make_json_serializable({
            'has_direct_trains': True,
            'recommendations': top_recommendations,  # Only top 3
            'best_route': best_route,  # Best based on user preference
            'recommendation_summary': recommendation_summary,  # Different categories
            'preference': preference,
            'weather': weather,
            'total_trains': len(unique_recommendations),
            'route_info': {
                'source': source,
                'destination': destination,
                'distance_range': {
                    # Filter out None values
                    'min': int(min([r['distance_km'] for r in unique_recommendations if isinstance(r.get('distance_km'), (int, float))]) if any(isinstance(r.get('distance_km'), (int, float)) for r in unique_recommendations) else 0),
                    'max': int(max([r['distance_km'] for r in unique_recommendations if isinstance(r.get('distance_km'), (int, float))]) if any(isinstance(r.get('distance_km'), (int, float)) for r in unique_recommendations) else 0)  # Convert pandas int64
                },
                'price_range': {
                    'min': int(min([r['price'] for r in unique_recommendations if isinstance(r.get('price'), (int, float))]) if any(isinstance(r.get('price'), (int, float)) for r in unique_recommendations) else 0),     # Convert pandas int64
                    'max': int(max([r['price'] for r in unique_recommendations if isinstance(r.get('price'), (int, float))]) if any(isinstance(r.get('price'), (int, float)) for r in unique_recommendations) else 0)     # Convert pandas int64
                }
            }
        }))

    except Exception as e:
        return jsonify(_make_json_serializable({'error': str(e)})), 500


@app.route('/api/realtime', methods=['POST'])
def realtime_event():
    """Receive a simulated realtime event (from simulator) and return delay prediction + rescheduling suggestions."""
    try:
        ev = request.get_json() or {}
        print(f"[realtime] Received event: {ev}")
        source = str(ev.get('source', '')).strip().upper()
        destination = str(ev.get('destination', '')).strip().upper()
        travel_date = ev.get('event_date') or ev.get('scheduled_date') or ev.get('scheduled_time') or datetime.utcnow().strftime('%Y-%m-%d')

        # Parse date
        try:
            date_obj = datetime.strptime(travel_date[:10], '%Y-%m-%d')
            day_of_week = date_obj.weekday() + 1
            month = date_obj.month
        except Exception:
            day_of_week = datetime.utcnow().weekday() + 1
            month = datetime.utcnow().month

        weather = get_weather_data(source)
        weather_condition = weather.get('condition')
        season = get_season(month)
        route = f"{source}-{destination}"

        # Try to get distance from event, otherwise infer from train_data
        distance_km = ev.get('distance_km')
        if distance_km is None:
            try:
                route_info = get_trains_passing_through(source, destination, day_of_week)
                if not route_info.empty and 'distance_km' in route_info.columns:
                    distance_km = float(route_info.iloc[0].get('distance_km', 0))
                else:
                    distance_km = 100.0
            except Exception:
                distance_km = 100.0

        predicted_delay = predict_delay_cached(route, day_of_week, month, float(distance_km), weather_condition, season)
        # If event includes train_id, adjust prediction to that train
        try:
            if ev.get('train_id'):
                predicted_delay = _adjust_prediction_for_train(ev.get('train_id'), predicted_delay, travel_date[:10])
        except Exception:
            pass

        # Ensure predicted_delay is numeric
        try:
            predicted_delay = float(predicted_delay)
        except Exception:
            predicted_delay = 0.0

        response = {
            'predicted_delay_min': round(predicted_delay, 1),
            'weather': weather,
            'season': season,
            'route': route
        }

        # If predicted delay is significant, provide alternate suggestions
        if predicted_delay >= 30:
            connecting_route = find_connecting_trains(source, destination, day_of_week, month, weather_condition, season)
            if connecting_route:
                response['recommendation'] = {
                    'type': 'connecting',
                    'connecting_route': connecting_route
                }
            else:
                # Try to find an alternate train with lower predicted delay on same route
                route_trains = get_trains_passing_through(source, destination, day_of_week)
                alt_options = []
                for _, tr in route_trains.iterrows():
                    d = float(tr.get('distance_km', distance_km))
                    p = predict_delay_cached(route, day_of_week, month, d, weather_condition, season)
                    # Adjust using available per-train baseline and reference date
                    p_adj = _adjust_prediction_for_train(tr.get('train_id'), p, travel_date[:10])
                    alt_options.append({'train_id': str(tr.get('train_id')), 'predicted_delay_min': round(float(p_adj), 1), 'departure_time': tr.get('departure_time')})
                if alt_options:
                    best_alt = min(alt_options, key=lambda x: x['predicted_delay_min'])
                    response['recommendation'] = {'type': 'alternate_train', 'train': best_alt}

        print(f"[realtime] Response: {response}")
        return jsonify(_make_json_serializable(response))

    except Exception as e:
        print(f"Error in realtime event: {e}")
        return jsonify(_make_json_serializable({'error': str(e)})), 500

@app.route('/api/weather', methods=['GET'])
def weather():
    """Fetch weather info for a city"""
    try:
        city = request.args.get('city', 'Delhi')
        weather_data = get_weather_data(city)
        return jsonify(_make_json_serializable(weather_data))
    except Exception as e:
        return jsonify(_make_json_serializable({'error': str(e)})), 500

@app.route('/api/trains', methods=['GET'])
def trains():
    """Return available train list from CSV"""
    try:
        source = request.args.get('source', '')
        destination = request.args.get('destination', '')
        train_id = request.args.get('train_id', '')

        # Normalize inputs
        if isinstance(source, str):
            source = source.strip().upper()
        if isinstance(destination, str):
            destination = destination.strip().upper()
        if train_id:
            try:
                train_id = str(train_id).zfill(5)
            except Exception:
                train_id = str(train_id)

        if train_id:
            trains = train_data[train_data['train_id'] == str(train_id).zfill(5)]
        elif source and destination:
            # Return trains that pass through the searched route (including intermediate segments)
            trains = get_trains_passing_through(source, destination)
        else:
            trains = train_data

        # Get unique trains (Feature 4: Remove duplicates)
        unique_trains = trains.drop_duplicates(subset=['train_id', 'train_name', 'source', 'destination'])

        train_list = []
        for _, train in unique_trains.iterrows():
            # If query source/destination provided, show them as the segment; also include parent route info
            seg_source = source if source else str(train['source'])
            seg_destination = destination if destination else str(train['destination'])
            train_list.append({
                'train_id': str(train['train_id']),
                'train_name': str(train['train_name']),
                'source': seg_source,
                'destination': seg_destination,
                'parent_source': str(train['source']),
                'parent_destination': str(train['destination']),
                'parent_route': str(train.get('parent_route') if 'parent_route' in train else train.get('route')),
                'is_intermediate_segment': bool(train.get('is_intermediate_segment', False)),
                'distance_km': int(float(train['distance_km'])) if pd.notna(train['distance_km']) else 0,
                'price': int(float(train['price'])) if pd.notna(train['price']) else 0
            })

        return jsonify(_make_json_serializable({
            'trains': train_list,
            'total': len(train_list)
        }))

    except Exception as e:
        return jsonify(_make_json_serializable({'error': str(e)})), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify(_make_json_serializable({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    }))

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=8000)
    else:
        print("Failed to load model. Exiting...")
