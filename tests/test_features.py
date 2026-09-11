"""Unit tests for Phase 3 Feature Engineering.

Runs offline without network or disk dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.build_features import (
    FEATURE_COLUMNS,
    TIERS,
    assign_tier,
    build_features,
    build_respondent_features,
    split_time_series,
)


@pytest.fixture
def sample_panel_df() -> pd.DataFrame:
    """Continuous 100-day panel for 2 respondents (1 Large, 1 Small)."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    rows = []
    for d in dates:
        # PJM: Large (~600k MWh)
        rows.append({
            "period": d,
            "respondent": "PJM",
            "demand": 600000.0 + 50000.0 * np.sin(d.dayofyear / 10),
            "demand_forecast": 595000.0 + 49000.0 * np.sin(d.dayofyear / 10),
            "net_generation": 580000.0,
            "interchange": 20000.0,
        })
        # SCL: Small (~20k MWh)
        rows.append({
            "period": d,
            "respondent": "SCL",
            "demand": 20000.0 + 2000.0 * np.cos(d.dayofyear / 10),
            "demand_forecast": 19800.0 + 1950.0 * np.cos(d.dayofyear / 10),
            "net_generation": 21000.0,
            "interchange": -1000.0,
        })
    return pd.DataFrame(rows)


class TestTierAssignment:
    def test_known_large_assigned_correctly(self):
        assert assign_tier("PJM") == "Large"
        assert assign_tier("ERCO") == "Large"

    def test_known_medium_assigned_correctly(self):
        assert assign_tier("NYIS") == "Medium"
        assert assign_tier("TVA") == "Medium"

    def test_known_small_assigned_correctly(self):
        assert assign_tier("SCL") == "Small"
        assert assign_tier("AVA") == "Small"
        assert assign_tier("SEC") == "Small"

    def test_unknown_returns_unknown(self):
        assert assign_tier("NONEXISTENT") == "Unknown"

    def test_all_approved_respondents_mapped(self):
        import json
        from pathlib import Path
        report_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "_logs" / "validation_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            approved = report.get("modeling_respondents", [])
            assert len(approved) == 52
            for resp in approved:
                assert assign_tier(resp) in {"Large", "Medium", "Small"}, f"{resp} unmapped!"


class TestFeatureLeakageAndIntegrity:
    def test_target_is_strictly_next_day_demand(self, sample_panel_df):
        pjm = sample_panel_df[sample_panel_df["respondent"] == "PJM"].reset_index(drop=True)
        feat = build_respondent_features(pjm)

        # Row 0 target must equal row 1 original demand
        assert feat.loc[0, "target"] == pjm.loc[1, "demand"]
        assert feat.loc[0, "target_date"] == pjm.loc[1, "period"]

    def test_benchmark_forecast_not_in_features(self):
        assert "benchmark_eia_forecast" not in FEATURE_COLUMNS
        assert "demand_forecast" not in FEATURE_COLUMNS
        assert "target" not in FEATURE_COLUMNS

    def test_naive_persistence_is_current_day_demand(self, sample_panel_df):
        pjm = sample_panel_df[sample_panel_df["respondent"] == "PJM"].reset_index(drop=True)
        feat = build_respondent_features(pjm)

        # Naive persistence at day t must equal demand at day t
        assert feat.loc[5, "naive_persistence"] == pjm.loc[5, "demand"]
        assert feat.loc[5, "demand_t"] == pjm.loc[5, "demand"]

    def test_lag_alignment_respects_missing_calendar_days(self):
        """If 2024-01-02 is missing, 2024-01-03's lag_1 must be NaN, not 2024-01-01!"""
        df_gapped = pd.DataFrame([
            {
                "period": pd.Timestamp("2024-01-01"),
                "respondent": "PJM",
                "demand": 100.0,
                "demand_forecast": 105.0,
                "net_generation": 90.0,
                "interchange": 10.0,
            },
            # 2024-01-02 is missing!
            {
                "period": pd.Timestamp("2024-01-03"),
                "respondent": "PJM",
                "demand": 200.0,
                "demand_forecast": 195.0,
                "net_generation": 180.0,
                "interchange": 20.0,
            },
        ])
        feat = build_respondent_features(df_gapped)

        # Reindexing should have created a row for 2024-01-02 with NaN
        row_jan3 = feat[feat["forecast_date"] == pd.Timestamp("2024-01-03")].iloc[0]
        # Since Jan 2 was missing, demand_lag_1 for Jan 3 must be NaN (honest!), not Jan 1's 100.0!
        assert pd.isna(row_jan3["demand_lag_1"])
        # demand_lag_2 for Jan 3 is 2 days ago (Jan 1), which is 100.0
        assert row_jan3["demand_lag_2"] == 100.0

    def test_trend_and_holiday_features_present(self, sample_panel_df):
        pjm = sample_panel_df[sample_panel_df["respondent"] == "PJM"].reset_index(drop=True)
        feat = build_respondent_features(pjm)

        for col in [
            "demand_trend_7",
            "demand_trend_30",
            "is_holiday",
            "is_day_before_holiday",
            "is_day_after_holiday",
            "is_bridge_day",
        ]:
            assert col in feat.columns
            assert col in FEATURE_COLUMNS


class TestSplitting:
    def test_split_is_strictly_chronological(self, sample_panel_df):
        feat = build_features(sample_panel_df)
        splits = split_time_series(feat, train_frac=0.70, val_frac=0.15, test_frac=0.15)

        assert splits.train["target_date"].max() < splits.val["target_date"].min()
        assert splits.val["target_date"].max() < splits.test["target_date"].min()

    def test_all_tiers_represented_in_splits(self, sample_panel_df):
        feat = build_features(sample_panel_df)
        splits = split_time_series(feat, train_frac=0.70, val_frac=0.15, test_frac=0.15)

        for s in [splits.train, splits.val, splits.test]:
            assert set(s["tier"].unique()) == {"Large", "Small"}
