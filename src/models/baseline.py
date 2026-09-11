"""Baseline Models & Benchmarks (Phase 4).

Implements the two mandatory baselines codified in PROBLEM_STATEMENT.md:
  1. Naive persistence: y_hat[t + 1] = demand[t]
     (A model that does not beat this has learned nothing).
  2. EIA day-ahead forecast benchmark: published next-day forecast.
"""

from __future__ import annotations

import pandas as pd


class NaivePersistenceBaseline:
    """Predicts next-day demand using current day's observed demand."""

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if "naive_persistence" in df.columns:
            return df["naive_persistence"]
        if "demand_t" in df.columns:
            return df["demand_t"]
        raise KeyError("Dataframe must contain 'naive_persistence' or 'demand_t' column")


class EIADayAheadBenchmark:
    """Extracts EIA's published day-ahead forecast for comparison."""

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if "benchmark_eia_forecast" in df.columns:
            return df["benchmark_eia_forecast"]
        raise KeyError("Dataframe must contain 'benchmark_eia_forecast' column")
