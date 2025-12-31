# TrainDelay AI — Comprehensive Project Documentation

## Abstract
- Predicts train delays using a RandomForestRegressor trained on historical route, season, weather, day-of-week, and distance features.
- Serves predictions and recommendations via a Python Flask API, consumed by a React TypeScript frontend.
- Demonstrates end‑to‑end ML integration: data prep, model training, API endpoints, and interactive UI with analytics.

## Objectives
- Build an accurate delay prediction model with practical inference speed.
- Provide user‑friendly web UI for predictions, recommendations, and analytics.
- Integrate real‑time contextual factors (mocked weather in demo) to refine outputs.
- Offer clear Mac and Windows setup with minimal friction.

## Related Work
- ML for transportation reliability often uses ensemble methods on multi‑factor features; this project applies a similar approach tailored to Indian train routes.
- Real‑time weather integration is common; here it is mocked for demo simplicity but structured to swap in OpenWeather.

## Methodology
- Feature engineering: encode categorical route, weather, season; use numeric day, month, distance.
- Model: `RandomForestRegressor` with performance around 98.26% R² and ~3.47 minutes MAE on the demo dataset.
- Inference: Flask API encodes inputs and predicts delay, derives simple probability and explanations.
- Recommendations: Rank trains by user preference (fastest, cheapest, most reliable) using predicted delay, price, and computed speed.

## Implementation
- Training script builds encoders and the model, saves artifacts used by the API.
- Backend loads artifacts, exposes endpoints for prediction, recommendation, weather, trains, and health.
- The imputation pipeline produces master releases (v3–v7); **v6** adds conformal 95% prediction intervals and **v7** applies the final flagging policy. The API surfaces intervals and flags so the frontend can display uncertainty and invite manual review where needed.
- Frontend calls API via Axios, renders results, charts, and UI interactions.

### Key Code References
- Training prediction function: `train_model.py:99`
- Backend prediction endpoint: `backend/app.py:100`
- Backend recommendation endpoint: `backend/app.py:191`
- Weather mock provider: `backend/app.py:46`
- Frontend API base URL: `frontend/src/services/api.ts:4`

## Code Explanation
- `train_model.py`: Trains the ML model, fits `LabelEncoder`s for route, weather, season, and exposes `predict_delay` used for testing. Saves `model.pkl`, encoder artifacts, and `feature_columns.json`.
- `backend/app.py`:
  - Initializes Flask with CORS and loads artifacts on startup.
  - `get_weather_data` returns mocked weather for demo; can be replaced with OpenWeather.
  - `predict` (`/api/predict`): Parses inputs, encodes features, computes delay, returns metadata, reason, and weather. The API now additionally returns **calibrated conformal 95% prediction intervals** for imputed rr_mean and rr_std when available, and exposes flag fields: `rr_imputation_flag_conservative`, `rr_imputation_flag_conformal`, and `rr_imputation_flag_final` so the frontend can surface high‑uncertainty imputations to users or ops.
  - `recommend` (`/api/recommend`): Predicts delays across trains on a route, ranks by preference.
  - `trains` (`/api/trains`): Serves available trains from CSV.
  - `health` (`/api/health`): Reports server status and model load.
  - AP-only canonical dataset: The repo now contains scripts to create and validate a strict Andhra Pradesh (AP) master dataset (463 station codes). Key scripts:
    - `scripts/create_ap_master.py` — generate canonical master from AP raw files.
    - `scripts/generate_ap_strict_master.py` — derive strict 463-station list (`data/ap_strict_463_codes.txt`) and produce cleaned master (`data/ap_trains_master_clean.csv`).
    - `scripts/test_load_model_prefers_clean_master.py` — verifies `backend.app.load_model()` loads and includes cleaned master trains.
- `frontend/src/services/api.ts`: Axios client configured to `http://localhost:8000/api` with helpers for predict, recommend, weather, trains, and health.
- UI Components (examples): `PredictionCard.tsx` renders delay result styling; `RecommendationsList.tsx` displays ranked trains with badges.

## Project Structure
```
TrainDelay AI/
├── backend/                 # Python Flask API
│   ├── app.py              # Main Flask application
│   ├── model.pkl           # Trained ML model
│   ├── route_encoder.pkl   # Route encoder
│   ├── weather_encoder.pkl # Weather encoder
│   ├── season_encoder.pkl  # Season encoder
│   └── feature_columns.json# Feature configuration
├── frontend/               # React TypeScript app
│   ├── src/                # Components, services, types, utils
│   └── package.json        # Dependencies and scripts
├── data/                   # Dataset
│   └── train_data.csv      # Historical train data
├── train_model.py          # ML model training script
├── requirements.txt        # Python dependencies
└── PROJECT_DOCUMENTATION.md# This document
```

## Setup (macOS)
- Prerequisites: `Python 3.9+`, `pip`, optional `Node.js 16+` for frontend.
- Install Python deps:
  - `cd "Train Delay Prediction"`
  - `pip3 install -r requirements.txt`
- Train the model (first run):
  - `python3 train_model.py`
- Start backend:
  - `cd backend`
  - `python3 app.py`
  - Backend runs at `http://localhost:8000`
- Start frontend (optional):
  - `cd frontend`
  - `npm install`
  - `npm start`
  - Frontend runs at `http://localhost:3000`
- One‑command startup (macOS/Linux with Bash):
  - `./run_project.sh` (manages ports, installs deps, starts both services)

## Setup (Windows)
- Prerequisites: Install `Python 3.9+` and `pip`. Optional `Node.js 16+` for the React app.
- Use Command Prompt or PowerShell.
- Install Python deps:
  - `cd "Train Delay Prediction"`
  - `pip install -r requirements.txt`
- Train the model (first run):
  - `python train_model.py`
- Start backend:
  - `cd backend`
  - `python app.py`
  - Backend runs at `http://localhost:8000`
- Start frontend (optional):
  - `cd frontend`
  - `npm install`
  - `npm start`
  - Frontend runs at `http://localhost:3000`
- Note: `.sh` scripts (like `run_project.sh`) require WSL/Git Bash; on native Windows use the manual steps above.

## Run the Project
- Backend only:
  - macOS: `python3 backend/app.py`
  - Windows: `python backend/app.py`
- Frontend only:
  - `cd frontend && npm start`
- Verify API health:
  - Open `http://localhost:8000/api/health` in a browser; expect `{"status":"healthy",...}`.

## API Usage
- Predict delay: `POST http://localhost:8000/api/predict`
  - Example body:
    ```json
    {"source":"Delhi","destination":"Mumbai","travel_date":"2025-01-15"}
    ```
- Recommendations: `POST http://localhost:8000/api/recommend`
  - Example body:
    ```json
    {"source":"Delhi","destination":"Mumbai","travel_date":"2025-01-15","preference":"fastest"}
    ```
- Weather: `GET http://localhost:8000/api/weather?city=Delhi`
- Trains: `GET http://localhost:8000/api/trains?source=Delhi&destination=Mumbai`

## Notes
- Weather data is mocked for demo; integrate OpenWeather by replacing logic in `backend/app.py:46`.
- Ensure ports `8000` (backend) and `3000` (frontend) are free before starting.
- If `model.pkl` is missing, run training via `train_model.py`.

### Imputation pipeline & model tuning 🔧
- Imputation models for missing RailRadar (`rr_mean` / `rr_std`) are trained from labeled RailRadar examples in `data/railradar_labels.csv` and feature set `data/railradar_features.csv`.
- Tuning is run via `python scripts/train_imputation_models.py`, which performs a RandomizedSearchCV (be aware some invalid `max_features` choices in older sklearns caused FitFailedWarning; see `scripts/train_imputation_models.py` for the current `param_distributions`). The tuned artifacts are saved to `models/rr_mean_model_tuned.joblib` and `models/rr_std_model_tuned.joblib` and search results are written to `models/`.
- Run the full imputation pipeline to update masters and flags (v4 → v7):
  - `python scripts/build_features.py`  # regenerate features
  - `python scripts/train_imputation_models.py`  # trains & tunes imputation models
  - `python scripts/apply_imputation.py`  # apply imputations to master (v4)
  - `python scripts/calibrate_and_flag_imputations.py`  # normalized-residual scalar calibration (v5)
  - `python scripts/conformal_intervals.py`  # split-conformal intervals and v6
  - `python scripts/apply_final_flags.py`  # final v7 with flags
- Quick checks / tests: `pytest -q scripts/test_imputation_pipeline.py` verifies tuned models, master v7, and conformal calibration file are present.

## Repository cleanup (2025-12-29) ✅
- Created top-level `archive_unused/` and moved unneeded intermediate or generated files into clear subfolders so the repo is easier to navigate.
  - `archive_unused/tmp_predict_response.json` — temporary test artifact (unused by code).
  - `archive_unused/data/archive/` — legacy data snapshots: `ap_trains_final.csv`, `ap_trains_final11.csv`, `ap_trains_final_modified.csv`, `README.txt` (not referenced by scripts).
  - `archive_unused/data/archive_national/train_data.csv` — legacy national archive snapshot.
  - `archive_unused/data/api-1.json` — old API export (no code references).
  - `archive_unused/frontend/build/` — generated frontend build artifacts (can be regenerated with `npm run build`).
  - `archive_unused/pycache/` and `archive_unused/other/` — removed compiled caches (`__pycache__` and `.pytest_cache`) to keep repo tidy.
- `.gitignore` was added with common ignores: `__pycache__/`, `*.pyc`, `.venv/`, `.pytest_cache/`, `frontend/build/`, `archive_unused/`, `tmp_predict_response.json`, `node_modules/`, `.vscode/`.
- Notes / rationale:
  - Only files with **no code references** were moved; the backup artifacts used by the app (e.g. `backend/model.pkl.bak`, `backend/season_encoder.pkl.bak`) were **left in place** because `backend/app.py` may restore from them at runtime.
  - If you want any archived items permanently deleted instead of archived, review `archive_unused/` and remove them manually or confirm and I will delete them.

## Pricing estimator fixes (2025-12-29) 🔧
- Fixed price estimation asymmetry and overly-high fallback prices:
  - When an exact (train, source, destination) lookup is missing, the estimator now checks the reversed pair and scales that fare by the requested segment distance (`lookup_reversed`). This reduces inconsistencies between `A->B` and `B->A` lookups.
  - A global median per‑km rate (`PRICE_GLOBAL_RATE`) is computed from `datasets/price_lookup.csv` and used as a safer fallback than the previous arbitrary multiplier. This reduces inflated distance fallbacks.
  - When possible, the estimator uses per‑train median per‑km rates, caps estimates relative to the train's full price to avoid unrealistic segment fares, and prefers estimating segment price from the train's known distances when available.
- Tests added: `tests/test_price_estimation_sources.py` includes checks for reversed-lookup scaling and global-rate fallback.

## Risk Score & Decision Advisor (2025-12-29) 💡
- Added a deterministic, rule-based **Risk Score** (0–100) and **Decision Advisor** to help users quickly assess travel risk.
  - Inputs: predicted delay (`predicted_delay_min`), calibrated prediction interval (`pred_rr_mean_conf_lower_95` / `pred_rr_mean_conf_upper_95` when available), imputation flags (`rr_imputation_flag_*`), and `distance_km`.
  - Output fields added to each train prediction:
    - `risk`: object containing `risk_score` (0–100), `confidence` (High/Medium/Low), `advice` (short textual advice), and `breakdown` (component contributions).
    - `recommendation.risk_advice` and `recommendation.risk_score` for easy consumption by the frontend.
  - Design: explainable weighted score (delay, uncertainty, imputation, distance) with configurable weights and mode (e.g., `exam`, `office`, `casual`) to adjust advice strictness.
  - Tests: `tests/test_risk_score.py` validates low/high risk cases and mode-dependent advice behavior.

If you prefer other weights, additional features, or a machine‑learned scoring model later, I can update the implementation and tests accordingly.

If you want, I can run the full test suite and create a commit/PR with these changes.
