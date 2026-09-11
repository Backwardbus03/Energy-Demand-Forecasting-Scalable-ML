"""Feature Engineering Pipeline (Phase 3).

Transforms daily raw panel data into leak-free tabular features for 1-day-ahead
electricity demand forecasting:
    target[t] = demand[t + 1]

Strict temporal integrity:
  - All features for row t use only information available at or before t.
  - EIA day-ahead forecast (DF) is preserved strictly as a benchmark, NEVER an input feature.
  - Missing dates are accounted for via daily re-indexing before shifting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.features.holidays import build_holiday_features

logger = logging.getLogger("eia.features")


def rolling_slope(s: pd.Series, window: int) -> pd.Series:
    """Compute rolling linear regression slope over the past `window` days (ending at t)."""
    k = window
    i = np.arange(k)
    denom = k * (k**2 - 1) / 12.0
    w = (i - (k - 1) / 2.0) / denom
    vals = s.to_numpy(dtype=float)
    conv = np.convolve(vals, w[::-1], mode="full")[: len(vals)]
    conv[: k - 1] = np.nan
    return pd.Series(conv, index=s.index)

# ---------------------------------------------------------------------------
# Tier definitions locked in PROBLEM_STATEMENT.md
# ---------------------------------------------------------------------------
TIERS: dict[str, list[str]] = {
    "Large": [
        "CISO", "ERCO", "MISO", "PJM", "SOCO", "SWPP",
    ],
    "Medium": [
        "AECI", "AZPS", "BPAT", "CPLE", "DUK", "FMPP", "FPC", "FPL", "IPCO",
        "ISNE", "LDWP", "LGEE", "NEVP", "NYIS", "PACE", "PACW", "PGE", "PSCO",
        "PSEI", "SC", "SCEG", "SRP", "SW", "TEC", "TVA",
    ],
    "Small": [
        "AVA", "BANC", "CHPD", "CPLW", "DOPD", "EPE", "GCPD", "GVL", "HST",
        "IID", "JEA", "NWMT", "PNM", "SCL", "SEC", "SPA", "TAL", "TEPC", "TIDC",
        "TPWR", "WALC",
    ],
}

RESPONDENT_TO_TIER: dict[str, str] = {
    resp: tier for tier, resps in TIERS.items() for resp in resps
}

CATEGORICAL_FEATURES: list[str] = ["respondent"]

NUMERIC_FEATURES: list[str] = [
    # Demand history at t and prior
    "demand_t",
    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_6",
    "demand_lag_7",
    "demand_lag_13",
    "demand_lag_14",
    "demand_lag_364",
    # Rolling statistics on demand up to t
    "demand_rolling_mean_7",
    "demand_rolling_std_7",
    "demand_rolling_mean_14",
    "demand_rolling_mean_30",
    "demand_rolling_std_30",
    # Demand trend slope up to t
    "demand_trend_7",
    "demand_trend_30",
    # Exogenous variables at t and lags
    "net_generation_t",
    "net_generation_lag_1",
    "net_generation_lag_7",
    "interchange_t",
    "interchange_lag_1",
    "interchange_lag_7",
    # Calendar features for target day (t + 1)
    "target_dayofweek",
    "target_is_weekend",
    "target_month",
    "target_dayofyear",
    "sin_dayofweek",
    "cos_dayofweek",
    "sin_dayofyear",
    "cos_dayofyear",
    # US Federal Holiday features for target day (t + 1)
    "is_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
    "is_bridge_day",
]

FEATURE_COLUMNS: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

METADATA_COLUMNS: list[str] = [
    "forecast_date",
    "target_date",
    "respondent",
    "tier",
    "target",
    "benchmark_eia_forecast",
    "naive_persistence",
]


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_cols: list[str]
    target_col: str = "target"


def assign_tier(respondent: str) -> str:
    """Return the tier for a balancing authority code."""
    return RESPONDENT_TO_TIER.get(respondent, "Unknown")


def build_respondent_features(df_resp: pd.DataFrame) -> pd.DataFrame:
    """Build features for a single balancing authority's continuous series.

    Ensures continuity by reindexing to a full daily date range, guarding
    against shifts skipping over missing days.
    """
    if df_resp.empty:
        return pd.DataFrame()

    resp = df_resp["respondent"].iloc[0]
    tier = assign_tier(resp)

    # Sort and reindex to complete daily calendar range
    s = df_resp.sort_values("period").set_index("period")
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D", name="period")
    s = s.reindex(full_idx)

    # Ensure required series exist
    for col in ["demand", "demand_forecast", "net_generation", "interchange"]:
        if col not in s.columns:
            s[col] = np.nan

    out = pd.DataFrame(index=s.index)
    out["forecast_date"] = s.index
    out["target_date"] = s.index + pd.Timedelta(days=1)
    out["respondent"] = resp
    out["tier"] = tier

    # TARGET: demand at t + 1
    out["target"] = s["demand"].shift(-1)

    # BENCHMARKS (Never used as training features!)
    # EIA published forecast for t + 1:
    out["benchmark_eia_forecast"] = s["demand_forecast"].shift(-1)
    # Naive persistence: prediction = demand at t
    out["naive_persistence"] = s["demand"]

    # DEMAND LAGS: available at day t
    out["demand_t"] = s["demand"]
    out["demand_lag_1"] = s["demand"].shift(1)
    out["demand_lag_2"] = s["demand"].shift(2)
    out["demand_lag_6"] = s["demand"].shift(6)   # 7 days before target t+1
    out["demand_lag_7"] = s["demand"].shift(7)
    out["demand_lag_13"] = s["demand"].shift(13) # 14 days before target t+1
    out["demand_lag_14"] = s["demand"].shift(14)
    out["demand_lag_364"] = s["demand"].shift(364) # ~1 year before target t+1

    # ROLLING DEMAND STATISTICS (strictly shifted so day t+1 is excluded)
    out["demand_rolling_mean_7"] = s["demand"].rolling(window=7, min_periods=4).mean()
    out["demand_rolling_std_7"] = s["demand"].rolling(window=7, min_periods=4).std()
    out["demand_rolling_mean_14"] = s["demand"].rolling(window=14, min_periods=7).mean()
    out["demand_rolling_mean_30"] = s["demand"].rolling(window=30, min_periods=15).mean()
    out["demand_rolling_std_30"] = s["demand"].rolling(window=30, min_periods=15).std()

    # DEMAND TREND SLOPES (up to day t)
    out["demand_trend_7"] = rolling_slope(s["demand"], window=7)
    out["demand_trend_30"] = rolling_slope(s["demand"], window=30)

    # EXOGENOUS LAGS (known at t)
    out["net_generation_t"] = s["net_generation"]
    out["net_generation_lag_1"] = s["net_generation"].shift(1)
    out["net_generation_lag_7"] = s["net_generation"].shift(7)

    out["interchange_t"] = s["interchange"]
    out["interchange_lag_1"] = s["interchange"].shift(1)
    out["interchange_lag_7"] = s["interchange"].shift(7)

    # CALENDAR FEATURES FOR TARGET DAY (t + 1)
    target_dt = out["target_date"].dt
    out["target_dayofweek"] = target_dt.dayofweek
    out["target_is_weekend"] = target_dt.dayofweek.isin([5, 6]).astype(int)
    out["target_month"] = target_dt.month
    out["target_dayofyear"] = target_dt.dayofyear

    # Cyclical encodings
    out["sin_dayofweek"] = np.sin(2 * np.pi * out["target_dayofweek"] / 7.0)
    out["cos_dayofweek"] = np.cos(2 * np.pi * out["target_dayofweek"] / 7.0)
    out["sin_dayofyear"] = np.sin(2 * np.pi * out["target_dayofyear"] / 365.25)
    out["cos_dayofyear"] = np.cos(2 * np.pi * out["target_dayofyear"] / 365.25)

    # US FEDERAL HOLIDAY FEATURES FOR TARGET DAY (t + 1)
    hols = build_holiday_features(out["target_date"])
    for col in ["is_holiday", "is_day_before_holiday", "is_day_after_holiday", "is_bridge_day"]:
        out[col] = hols[col].to_numpy()

    return out.reset_index(drop=True)


def build_features(df_panel: pd.DataFrame, drop_na_target: bool = True) -> pd.DataFrame:
    """Transform panel dataframe into feature dataset across all respondents.

    Parameters
    ----------
    df_panel:
        Wide panel dataframe from ``load_modeling_frame(wide=True)``.
    drop_na_target:
        If True, drop the final row per respondent where target[t+1] is unobserved.
    """
    logger.info("building features for %d rows across respondents", len(df_panel))

    pieces = []
    for resp, group in df_panel.groupby("respondent"):
        feat = build_respondent_features(group)
        pieces.append(feat)

    if not pieces:
        return pd.DataFrame()

    features_df = pd.concat(pieces, ignore_index=True)

    if drop_na_target:
        # We need target for supervised training and historical evaluation
        features_df = features_df.dropna(subset=["target"]).reset_index(drop=True)

    logger.info(
        "built feature matrix: shape=%s, target_range=%s -> %s",
        features_df.shape,
        features_df["target_date"].min().date() if not features_df.empty else "n/a",
        features_df["target_date"].max().date() if not features_df.empty else "n/a",
    )
    return features_df


def split_time_series(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> DatasetSplits:
    """Split dataframe strictly chronologically on target_date.

    Guarantees:
        max(train.target_date) < min(val.target_date) < min(test.target_date)
    """
    if df.empty:
        raise ValueError("Cannot split empty dataframe")

    unique_dates = pd.Series(sorted(df["target_date"].unique()))
    n_dates = len(unique_dates)

    if n_dates < 10:
        raise ValueError(f"Insufficient dates for 3-way split: {n_dates}")

    train_end_idx = int(n_dates * train_frac)
    val_end_idx = int(n_dates * (train_frac + val_frac))

    train_end_date = unique_dates.iloc[train_end_idx - 1]
    val_end_date = unique_dates.iloc[val_end_idx - 1]

    train = df[df["target_date"] <= train_end_date].copy().reset_index(drop=True)
    val = (
        df[(df["target_date"] > train_end_date) & (df["target_date"] <= val_end_date)]
        .copy()
        .reset_index(drop=True)
    )
    test = df[df["target_date"] > val_end_date].copy().reset_index(drop=True)

    # Verification of non-overlapping temporal partitions
    assert train["target_date"].max() < val["target_date"].min(), (
        f"Train/Val leakage: train max {train['target_date'].max()} >= val min {val['target_date'].min()}"
    )
    assert val["target_date"].max() < test["target_date"].min(), (
        f"Val/Test leakage: val max {val['target_date'].max()} >= test min {test['target_date'].min()}"
    )

    logger.info(
        "chronological split: train=%d rows (%s..%s) | val=%d rows (%s..%s) | test=%d rows (%s..%s)",
        len(train), train["target_date"].min().date(), train["target_date"].max().date(),
        len(val), val["target_date"].min().date(), val["target_date"].max().date(),
        len(test), test["target_date"].min().date(), test["target_date"].max().date(),
    )

    return DatasetSplits(
        train=train,
        val=val,
        test=test,
        feature_cols=FEATURE_COLUMNS,
        target_col="target",
    )
