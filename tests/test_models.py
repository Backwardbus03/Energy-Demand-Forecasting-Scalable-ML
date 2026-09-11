"""Unit tests for Phase 4 Baseline & Tier Models.

Runs offline without network or disk dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import (
    compute_mae,
    compute_r2,
    compute_smape,
    compute_wmape,
    evaluate_dataframe,
    evaluate_series,
)
from src.features.build_features import build_features, split_time_series
from src.models.baseline import EIADayAheadBenchmark, NaivePersistenceBaseline
from src.models.tier_model import TierModel, predict_with_tier_models, train_all_tiers


class TestMetrics:
    def test_mae_exact(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        assert compute_mae(y_true, y_pred) == 10.0

    def test_smape_zero_error(self):
        y = np.array([100.0, 200.0])
        assert compute_smape(y, y) == 0.0

    def test_smape_symmetric(self):
        y1, y2 = np.array([100.0]), np.array([120.0])
        # |100 - 120| / (100 + 120) == 20 / 220
        assert pytest.approx(compute_smape(y1, y2)) == compute_smape(y2, y1)

    def test_wmape_calculation(self):
        y_true = np.array([100.0, 300.0])
        y_pred = np.array([120.0, 280.0])
        # total actual = 400, total abs error = 20 + 20 = 40 -> 10%
        assert pytest.approx(compute_wmape(y_true, y_pred), rel=1e-3) == 10.0

    def test_r2_perfect(self):
        y = np.array([10.0, 20.0, 30.0])
        assert compute_r2(y, y) == 1.0


class TestBaselines:
    def test_naive_persistence_extracts_demand_t(self):
        df = pd.DataFrame({"naive_persistence": [150.0, 250.0]})
        base = NaivePersistenceBaseline()
        preds = base.predict(df)
        assert list(preds) == [150.0, 250.0]

    def test_eia_benchmark_extracts_forecast(self):
        df = pd.DataFrame({"benchmark_eia_forecast": [155.0, 248.0]})
        bench = EIADayAheadBenchmark()
        preds = bench.predict(df)
        assert list(preds) == [155.0, 248.0]


class TestTierModelPipeline:
    def test_train_predict_save_load(self, tmp_path):
        # Generate synthetic data for Large, Medium, Small
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        rows = []
        for d in dates:
            # Large: PJM
            rows.append({
                "period": d,
                "respondent": "PJM",
                "demand": 600000.0 + 10000.0 * np.sin(d.dayofyear),
                "demand_forecast": 590000.0,
                "net_generation": 580000.0,
                "interchange": 20000.0,
            })
            # Medium: NYIS
            rows.append({
                "period": d,
                "respondent": "NYIS",
                "demand": 150000.0 + 5000.0 * np.sin(d.dayofyear),
                "demand_forecast": 149000.0,
                "net_generation": 140000.0,
                "interchange": 10000.0,
            })
            # Small: SCL
            rows.append({
                "period": d,
                "respondent": "SCL",
                "demand": 25000.0 + 1000.0 * np.sin(d.dayofyear),
                "demand_forecast": 24500.0,
                "net_generation": 24000.0,
                "interchange": 1000.0,
            })
        df_panel = pd.DataFrame(rows)
        feat = build_features(df_panel)
        splits = split_time_series(feat, train_frac=0.60, val_frac=0.20, test_frac=0.20)

        # Train all tiers
        models = train_all_tiers(splits.train, splits.val, models_dir=tmp_path)
        assert set(models.keys()) == {"Large", "Medium", "Small"}

        # Predict on test
        scored = predict_with_tier_models(splits.test, models)
        assert "model_prediction" in scored.columns
        assert not scored["model_prediction"].isna().any()

        # Check evaluation dataframe
        eval_results = evaluate_dataframe(scored, pred_col="model_prediction")
        assert "Large" in eval_results
        assert "Medium" in eval_results
        assert "Small" in eval_results
        assert "Overall" in eval_results
        assert eval_results["Large"]["n_samples"] > 0

    def test_fallback_guardrail_replaces_predictions(self):
        # Create a TierModel with a designated fallback region
        tm = TierModel(
            tier="Small",
            feature_cols=["demand_t"],
            fallback_regions=["BAD_REGION"],
        )
        # Mock a minimal dummy model or mock predict
        class DummyModel:
            def predict(self, X):
                return np.full(len(X), 999999.0)

        tm.model = DummyModel()

        df = pd.DataFrame({
            "respondent": ["BAD_REGION", "GOOD_REGION"],
            "tier": ["Small", "Small"],
            "demand_t": [50.0, 100.0],
            "naive_persistence": [50.0, 100.0],
        })

        # With fallback enabled (default)
        preds = tm.predict(df, enable_fallback=True)
        # BAD_REGION should be replaced with naive_persistence (50.0)
        assert preds[0] == 50.0
        # GOOD_REGION should remain model output (999999.0)
        assert preds[1] == 999999.0

        # With fallback disabled
        preds_no_fb = tm.predict(df, enable_fallback=False)
        assert preds_no_fb[0] == 999999.0
        assert preds_no_fb[1] == 999999.0
