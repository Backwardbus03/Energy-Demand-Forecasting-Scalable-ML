# Problem Statement — FIXED

**Frozen 2026-08-21.** Changes require an explicit decision recorded in the
Changelog at the bottom. This document exists to prevent scope creep.

---

## 1. The problem

> Predict next-day electricity demand for US balancing authorities.

Given demand history for a balancing authority (BA) through day *t*, forecast
its demand on day *t+1*, using only information available at or before *t*.

```
target[t] = demand[t + 1]
```

## 2. Data

| | |
|---|---|
| Source | EIA API v2, `electricity/rto/daily-region-data` |
| Entities | 51 balancing authorities |
| Window | 2022-01-01 → present |
| Observations | 86,164 region-days |
| Target | `demand` (MWh) |
| Features available | `net_generation`, `interchange`, calendar, lags, rollings |
| Benchmark only | `demand_forecast` (EIA's own day-ahead forecast) |

Raw store retains 2019+ (793,160 rows). The 2022 window is a *view*; widening
it needs no re-fetch.

## 3. Model architecture — per-tier models

Regions are grouped into three tiers by mean daily demand. **One model per
tier**, three models total.

| tier | threshold | regions | rows | naive SMAPE |
|---|---|---|---|---|
| Large | ≥ 500,000 MWh/day | 6 | 10,151 | 4.06% |
| Medium | 50,000–500,000 | 25 | 42,246 | 4.63% |
| Small | < 50,000 | 20 | 33,767 | 4.87% |

**Large:** CISO, ERCO, MISO, PJM, SOCO, SWPP

**Medium:** AECI, AZPS, BPAT, CPLE, DUK, FMPP, FPC, FPL, IPCO, ISNE, LDWP,
LGEE, NEVP, NYIS, PACE, PACW, PGE, PSCO, PSEI, SC, SCEG, SRP, SW, TEC, TVA

**Small:** AVA, BANC, CHPD, CPLW, DOPD, EPE, GCPD, GVL, HST, IID, JEA, NWMT,
PNM, SCL, SPA, TAL, TEPC, TIDC, TPWR, WALC

Rationale: naive SMAPE rises monotonically across tiers, so difficulty tracks
scale at the tier level. Tier boundaries are fixed **now, before modelling**,
and are not to be re-tuned after seeing results.

All 51 regions are retained. No region is excluded.

## 4. Baselines to beat

1. **Naive persistence** — `prediction = demand[t]`
2. **EIA day-ahead forecast** — `demand_forecast[t+1]`

A model that does not beat (1) has learned nothing.

## 5. Metrics

Reported **per tier**, never as a single averaged number.

| metric | why |
|---|---|
| MAE (MWh) | honest magnitude |
| SMAPE (%) | scale-free, survives near-zero values |
| Weighted MAPE | weighted by region size |
| R² | variance explained |

**Plain MAPE is not a headline metric.** With a 2,504× size range and 8
zero-demand rows it is dominated by the smallest entities — this is why EIA's
professional forecast scored *worse* than naive persistence (6.87% vs 4.76%)
on unweighted MAPE. That was a metric artefact, not a forecasting result.

## 6. Validation

Chronological only. **Never** `train_test_split(shuffle=True)`.

```
train 70%  |  validation 15%  |  test 15%
```

Walk-forward validation for Optuna. Splits must not overlap in time.

## 7. Explicitly OUT OF SCOPE

Adding any of these requires a Changelog entry and a stated reason.

| excluded | why |
|---|---|
| Hourly data | 24× volume, DST handling, needs weather to model well |
| Weather features | External API, new ingestion pipeline, geographic joins |
| Multi-step horizons (t+2, t+7) | Different problem: changes target, metrics, validation |
| Deep learning (LSTM, Transformer) | Project is about the MLOps lifecycle, not model novelty |
| Probabilistic / interval forecasts | Different loss functions and evaluation |
| Per-region models (51) | Unmanageable registry, tuning, and serving overhead |
| Kubernetes | Docker Compose is sufficient |
| Additional EIA datasets | One well-understood dataset beats several shallow ones |

## 8. Deliverable

A reproducible MLOps pipeline:

```
ingestion -> validation -> features -> training -> tracking -> registry
          -> orchestration -> containers -> serving -> monitoring
```

The ML is deliberately simple. **The pipeline is the work.**

## 9. Definition of done

- [x] Phase 1 — raw ingestion, idempotent and resumable
- [x] Phase 2 — validation gate + EDA
- [ ] Phase 3 — features, leakage-tested
- [ ] Phase 4 — baselines + 3 tier models beating naive persistence
- [ ] Phase 5 — Optuna, time-series aware
- [ ] Phase 6 — MLflow tracking
- [ ] Phase 7 — model registry
- [ ] Phase 8 — Airflow DAG
- [ ] Phase 9 — Docker Compose
- [ ] Phase 10 — FastAPI `/predict`
- [ ] Phase 11 — AWS
- [ ] Phase 12 — monitoring
- [ ] Phase 13 — automated retraining

## Changelog

| date | change | reason |
|---|---|---|
| 2026-08-21 | Statement frozen | Scope fixed before Phase 3 |
