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
  promote_model.py          Compares candidate vs. live model, gates promotion
dags/
  retraining_pipeline.py    Airflow DAG: fetch -> validate -> train -> promote -> reload
tests/                      78 offline unit tests (no network or disk state required)
models/                     Live (promoted) model artifacts (model_tier_*.joblib)
models_candidate/           Freshly trained, not-yet-promoted artifacts (ignored in git)
data/                       Ignored in git (managed via partitioned Parquets)
```

---

## Automated Retraining Pipeline (Phase 8 & 13)

Orchestrated with **Apache Airflow**, run daily on a schedule:

```
fetch_raw_data.py --incremental
        |
        v
validate_raw_data.py                 (exit 1 = structural errors -> DAG fails, stops here)
        |
        v
train_and_evaluate.py --models-dir models_candidate/   (trains a candidate, doesn't touch live models/)
        |
        v
promote_model.py                      (compares candidate WMAPE vs. live models/ per tier;
        |                              only overwrites models/ if no tier regresses > 2pp WMAPE)
        v
POST /admin/reload-models             (tells the running FastAPI server to reload from disk)
```

A rejected promotion is not a pipeline failure — it means the guardrail worked and the
previous production model was correctly left in place.

### Status: pipeline logic verified, Airflow scheduler blocked on macOS (arm64)

Every stage of the pipeline has been run and verified end-to-end for real, individually,
against live data:

- `fetch_raw_data.py --incremental` — fetched real rows from the EIA API
- `validate_raw_data.py` — passed against 793,160 real rows, correct exit codes confirmed
- `train_and_evaluate.py --models-dir models_candidate/` — trained real LightGBM models,
  wrote real metrics to `evaluation_summary_candidate.json`, left `models/` untouched
- `promote_model.py` — correctly compared candidate vs. live WMAPE and promoted
- `POST /admin/reload-models` — confirmed live against a running `uvicorn` server

**What's not working yet: Apache Airflow's own scheduler/webserver processes crash on
this machine.** On this Mac (macOS arm64), Airflow 2.10.5's internal gunicorn-based
subprocesses (both the webserver and the scheduler's own internal API server) segfault
on fork (`SIGSEGV`) immediately on startup — before any task or request is even handled.
This reproduces with `sync` and `gthread` worker classes, with `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`,
and with `AIRFLOW__LOGGING__SERVE_LOGS=False`, so it is not a worker-class or logging
config issue — it looks like a broken native dependency (likely gunicorn/grpcio/a C
extension) in this specific Airflow + Python 3.12 install on arm64 macOS, not a bug in
this project's code or DAG.

**Until that's resolved (planned: try again on Windows), verify/run the pipeline via the
CLI directly, bypassing Airflow's scheduler:**

```bash
# Run each stage manually, in order, exactly as the DAG would:
.venv/bin/python scripts/fetch_raw_data.py --incremental
.venv/bin/python scripts/validate_raw_data.py
.venv/bin/python scripts/train_and_evaluate.py \
  --models-dir models_candidate --output-summary data/evaluation_summary_candidate.json
.venv/bin/python scripts/promote_model.py
curl -X POST http://127.0.0.1:8000/admin/reload-models   # only if the server is running
```

Or, once Airflow is stable in an environment, test/run it through Airflow as normal
(see below) — the DAG file itself does not need to change.

### Why Airflow runs in its own environment

The project's `.venv` is Python 3.13, and Apache Airflow does not yet officially support
3.13 (constraint files cap at 3.12). Airflow itself runs from a separate `.venv-airflow`
(Python 3.12) that only hosts the scheduler/webserver; every DAG task shells out to the
project's own `.venv` to run the actual scripts, so application code never runs under a
different Python version than it's tested with.

### One-time setup

```bash
# 1. Airflow's own environment (Python 3.12 required)
brew install python@3.12
python3.12 -m venv .venv-airflow
.venv-airflow/bin/pip install "apache-airflow==2.10.5" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt"

# 2. Point Airflow at this project and initialize its metadata DB
export AIRFLOW_HOME="$(pwd)/.airflow"
.venv-airflow/bin/airflow db migrate
# dags_folder is already set to ./dags and load_examples=False in .airflow/airflow.cfg

# 3. Create an admin user (first time only)
.venv-airflow/bin/airflow users create \
  --username admin --password admin --firstname A --lastname B \
  --role Admin --email you@example.com
```

### Running it

```bash
export AIRFLOW_HOME="$(pwd)/.airflow"

# Start the scheduler (in one terminal)
.venv-airflow/bin/airflow scheduler

# Start the webserver / UI at http://localhost:8080 (in another terminal)
.venv-airflow/bin/airflow webserver --port 8080

# Or trigger a single run manually without the scheduler
.venv-airflow/bin/airflow dags trigger energy_demand_retraining_pipeline

# Or test a single task in isolation (no DB run history needed)
.venv-airflow/bin/airflow tasks test energy_demand_retraining_pipeline validate_raw_data 2026-01-01
```

The DAG is scheduled for `0 6 * * *` (daily 06:00 America/New_York) — edit the `schedule`
in `dags/retraining_pipeline.py` to change cadence. The serving app (`uvicorn src.serving.app:app`)
should be running for the final `reload_live_models` step to take effect immediately; if it
isn't reachable, that step is skipped with a warning and the new model still loads on next
server start.

---

## Roadmap

- [x] **Phase 1**: Raw ingestion (idempotent, resumable, timezone-pinned)
- [x] **Phase 2**: Validation gate + schema enforcement
- [x] **Phase 3**: Feature engineering (lags, rolling, trend, calendar, US holidays)
- [x] **Phase 4**: 3-Tier LightGBM models + Baselines + Fallback Guardrail
- [ ] **Phase 5**: Optuna hyperparameter optimization per tier
- [ ] **Phase 6**: MLflow experiment tracking
- [ ] **Phase 7**: MLflow model registry & versioning
- [~] **Phase 8**: Apache Airflow DAG orchestration — DAG written & pipeline logic verified end-to-end via CLI; Airflow's own scheduler/webserver currently segfault on this Mac (arm64) — see [status note](#status-pipeline-logic-verified-airflow-scheduler-blocked-on-macos-arm64)
- [ ] **Phase 9**: Docker Compose containerization
- [x] **Phase 10**: FastAPI serving (`/api/predict`) & Interactive UI
- [ ] **Phase 11**: Cloud deployment (AWS ECS / S3)
- [ ] **Phase 12**: Evidently AI drift & performance monitoring
- [x] **Phase 13**: Automated retraining pipeline (gated promotion, no auto-regression) — logic proven, run manually via CLI until Airflow is stable
