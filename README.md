# Energy Demand Forecasting — MLOps

Forecasting next-day regional electricity demand from **EIA API v2**.

**Status: Phase 1 complete (793,160 rows). Phase 2 validation complete; EDA next.**

## Problem

For each US balancing authority (region), predict:

```
target[t] = demand[t + 1]
```

Only information available at or before `t` may be used.

## Data source

`electricity/rto/daily-region-data` — daily demand, day-ahead forecast,
net generation and interchange for 83 balancing authorities, 2019-01-01 to present.

| Series | Meaning | Role |
|--------|---------|------|
| `D`  | Demand | **target** |
| `DF` | EIA day-ahead demand forecast | benchmark to beat |
| `NG` | Net generation | feature |
| `TI` | Total interchange (net imports; negative = exporting) | feature |

Units are megawatthours throughout.

### The timezone trap

The `timezone` facet duplicates every observation five times (Arizona,
Eastern, Central, Mountain, Pacific all carry identical values).

```
unpinned : 1,830,719 rows   <- 5x inflated
pinned   :   366,144 rows   <- same information
```

`configs/ingestion.yaml` pins `timezone: Eastern`, and `load_config` refuses
to run without it. This is de-duplication, not a geographic filter.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then add your key
```

Free API key: https://www.eia.gov/opendata/register.php

## Usage

```bash
# Validate the raw data (exit 0 = safe to proceed)
python scripts/validate_raw_data.py

# Smoke test (3 regions, one month)
python scripts/fetch_raw_data.py --limit-respondents 3 --start 2025-01-01 --end 2025-01-31

# Full historical backfill (~366k rows, ~15-25 min)
python scripts/fetch_raw_data.py

# Incremental update: only days newer than what is stored
python scripts/fetch_raw_data.py --incremental
```

Run the tests:

```bash
pytest tests/ -q
```

## Layout

```
configs/ingestion.yaml     all tunable settings
src/utils/config.py        config + secret loading
src/utils/logging_utils.py console + file logging
src/data/eia_client.py     HTTP: retries, pagination, caching
src/data/fetch_eia.py      normalise -> partitioned Parquet
scripts/fetch_raw_data.py  CLI entry point
tests/                     offline unit tests

data/raw/eia/year=YYYY/    immutable raw Parquet
data/raw/_cache/           verbatim API responses
data/raw/_logs/            run logs + failures.jsonl
```

## Raw schema

| column | type | notes |
|--------|------|-------|
| `period` | datetime | observation day |
| `respondent` | string | balancing authority code |
| `respondent_name` | string | full name |
| `type` | string | `D` / `DF` / `NG` / `TI` |
| `type_name` | string | human-readable series name |
| `value` | float | **NaN when missing — never 0** |
| `value_units` | string | `megawatthours` |
| `timezone` | string | pinned; retained for provenance |
| `_ingested_at` | string | UTC ingestion timestamp |
| `_source_route` | string | originating API route |

Natural key: `(period, respondent, type)`.

## Design rules

- **Raw data is immutable.** Re-runs de-duplicate on the natural key; revisions
  overwrite by `_ingested_at`. Running twice produces identical output.
- **Nothing is invented.** A missing observation stays `NaN`. Regions that
  return no data are logged to `failures.jsonl`, never zero-filled.
- **No feature engineering here.** Lags and rolling windows are Phase 3.
- **Responses are cached.** A parsing bug never costs a re-fetch.

## Validation

`python scripts/validate_raw_data.py` gates the pipeline. Structural problems
(duplicate keys, mixed units, unpinned timezone) exit 1; statistical findings
are reported and left for a human decision — nothing is silently repaired.

Decisions encoded in `src/data/validate.py`:

- **Aggregates excluded.** `US48` and the 12 regional roll-ups sum other
  respondents. Verified: `corr(MIDA, PJM) = 1.0000`, `corr(TEX, ERCO) = 1.0000`.
  Keeping both would double-count the same electricity.
- **Modeling set = 51 balancing authorities** at ≥95% coverage.
- **Negative `TI` and `NG` are physical**, not errors (net export; station
  load exceeding output). Only negative demand is flagged.

## Roadmap

Phase 2 EDA · Phase 3 features ·
Phase 4 baseline + models · Phase 5 Optuna · Phase 6 MLflow ·
Phase 7 registry · Phase 8 Airflow · Phase 9 Docker · Phase 10 FastAPI ·
Phase 11 AWS · Phase 12 monitoring · Phase 13 retraining
