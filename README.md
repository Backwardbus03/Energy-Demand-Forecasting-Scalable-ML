# Energy Demand Forecasting — Scalable MLOps

Production-grade time-series forecasting of next-day regional electricity demand across US balancing authorities using the **EIA API v2**.

**Status: Core MLOps Pipeline Complete (Phases 1–4 & Phase 10 Serving/UI). 78 offline tests passing.**

---

## Performance Summary (Unseen 15% Chronological Test Split)

| Tier | Approached Model | MAE (MWh) | SMAPE (%) | WMAPE (%) | R² | Win Rate vs. Naive |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Large** | **LightGBM (Ours)** | **46,276.7** | **3.39%** | **3.48%** | **0.9893** | **6/6 (100%)** |
| | EIA Day-Ahead Forecast | 45,688.0 | 4.48% | 3.43% | 0.9924 | — |
| | Naive Persistence Baseline | 53,485.1 | 4.07% | 4.02% | 0.9869 | — |
| **Medium** | **LightGBM (Ours)** | **7,811.0** | **4.59%** | **4.59%** | **0.9896** | **23/25 (92%)** |
| | EIA Day-Ahead Forecast | 11,254.2 | 10.48% | 6.60% | 0.9691 | — |
| | Naive Persistence Baseline | 8,662.1 | 5.15% | 5.08% | 0.9877 | — |
| **Small** | **LightGBM (Ours)** | **820.6** | **4.96%** | **4.12%** | **0.9914** | **15/21 (71%)** |
| | EIA Day-Ahead Forecast | 837.4 | 5.37% | 4.03% | 0.9839 | — |
| | Naive Persistence Baseline | 918.8 | 5.51% | 4.62% | 0.9899 | — |
| **Overall** | **LightGBM (Ours)** | **9,442.8** | **4.60%** | **3.87%** | **0.9969** | **44/52 (85%)** |
| | EIA Day-Ahead Forecast | 11,226.3 | 7.78% | 4.52% | 0.9968 | — |
| | Naive Persistence Baseline | 10,726.5 | 5.17% | 4.40% | 0.9962 | — |

---

## Core Features & Architecture

- **Idempotent Ingestion**: Resumable API fetcher with automatic 30-day overlap backoffs, local response caching, natural key deduplication, and zero data invention.
- **Validation Gate**: Structural checks (duplicate keys, mixed units, unpinned timezones) exit 1 and block downstream execution. Preserves physical negative values (`NG`/`TI`) while flagging invalid demand.
- **Temporal Integrity**: Full calendar daily reindexing prevents `shift()` from skipping over interior missing days. Chronological splits with zero data leakage (`train < val < test`).
- **Feature Engineering**: Demand lags ($t, t-1, t-2, t-6, t-7, t-13, t-14, t-364$), rolling stats (7, 14, 30 days), rolling trend slopes (7 and 30-day linear regression momentum), exogenous grid features (`net_generation`, `interchange`), and US federal holiday indicators (observed dates, before/after, bridge days).
- **Per-Tier LightGBM Models**: Avoids 52-model operational sprawl by clustering into **Large** (≥500k MWh/day), **Medium** (50k–500k), and **Small** (<50k).
- **Data-Driven Guardrail**: Evaluates validation error against naive persistence per region; automatically falls back to naive persistence for anomalous edge cases (`GVL`, `HST`, `SPA`).
- **FastAPI Serving & UI**: Interactive dark-themed web dashboard and REST API for real-time predictions, benchmark comparisons, and historical charting.

---

## Quickstart

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure EIA API key (free at https://www.eia.gov/opendata/register.php)
cp .env.example .env
```

### 2. Run Pipeline & Train Models

```bash
# Step 1: Validate ingested raw data (validation gate)
python scripts/validate_raw_data.py

# Step 2: Ingest historical data (or incremental update)
python scripts/fetch_raw_data.py

# Step 3: Train 3-tier models and evaluate on unseen test set
python scripts/train_and_evaluate.py
```

### 3. Launch Interactive Dashboard & API

```bash
# Start FastAPI application with live reload
uvicorn src.serving.app:app --host 127.0.0.1 --port 8000
```
Open **http://127.0.0.1:8000** in your browser to explore predictions across all 52 balancing authorities.

### 4. Run Test Suite

```bash
pytest tests/ -v
```
*(78 tests passing across ingestion, validation, feature leakage, models, and serving endpoints).*

---

## API Reference

- `GET /health`: Model status and cache availability.
- `GET /api/regions`: List of 52 balancing authorities grouped by tier.
- `GET /api/history?respondent=PJM&limit=60`: Historical demand and benchmark forecasts for charting.
- `POST /api/predict`: Real-time inference endpoint.
  ```json
  {
    "respondent": "PJM",
    "target_date": "2026-09-10"
  }
  ```

---

## Repository Layout

```
configs/
  ingestion.yaml            Tunable settings (pinned timezone, routes, batch sizes)
src/
  data/
    eia_client.py           HTTP client with retries, exponential backoff, rate limiting
    fetch_eia.py            Normalisation to partitioned Parquet store
    validate.py             Pipeline validation gate & dataset loader
  features/
    build_features.py       Leak-free lag, rolling, trend, and calendar features
    holidays.py             Zero-dependency US federal holiday calendar & indicators
  models/
    baseline.py             Naive persistence and EIA day-ahead benchmark
    tier_model.py           3-Tier LightGBM models + validation guardrails
  evaluation/
    metrics.py              MAE, SMAPE, WMAPE, R² evaluation engine
  serving/
    app.py                  FastAPI backend endpoints
    static/index.html       Interactive dark-themed dashboard (Chart.js)
scripts/
  fetch_raw_data.py         CLI for raw ingestion (backfill / incremental)
  validate_raw_data.py      Validation gate CLI runner
  train_and_evaluate.py     End-to-end feature build, training, and evaluation
tests/                      78 offline unit tests (no network or disk state required)
models/                     Trained model artifacts (model_tier_*.joblib)
data/                       Ignored in git (managed via partitioned Parquets)
```

---

## Roadmap

- [x] **Phase 1**: Raw ingestion (idempotent, resumable, timezone-pinned)
- [x] **Phase 2**: Validation gate + schema enforcement
- [x] **Phase 3**: Feature engineering (lags, rolling, trend, calendar, US holidays)
- [x] **Phase 4**: 3-Tier LightGBM models + Baselines + Fallback Guardrail
- [ ] **Phase 5**: Optuna hyperparameter optimization per tier
- [ ] **Phase 6**: MLflow experiment tracking
- [ ] **Phase 7**: MLflow model registry & versioning
- [ ] **Phase 8**: Apache Airflow DAG orchestration
- [ ] **Phase 9**: Docker Compose containerization
- [x] **Phase 10**: FastAPI serving (`/api/predict`) & Interactive UI
- [ ] **Phase 11**: Cloud deployment (AWS ECS / S3)
- [ ] **Phase 12**: Evidently AI drift & performance monitoring
- [ ] **Phase 13**: Automated retraining pipeline
