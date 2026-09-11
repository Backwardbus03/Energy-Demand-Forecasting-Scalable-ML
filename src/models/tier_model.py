"""Per-Tier Model Training & Inference (Phase 4).

Trains one LightGBM model per tier (Large, Medium, Small), three models total,
avoiding per-region operational bloat while respecting scale divergence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.build_features import CATEGORICAL_FEATURES, FEATURE_COLUMNS

logger = logging.getLogger("eia.models")


@dataclass
class TierModel:
    tier: str
    feature_cols: list[str]
    model: lgb.LGBMRegressor | None = None
    best_iteration: int = 0
    best_val_score: float = float("inf")
    fallback_regions: list[str] = field(default_factory=list)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str = "target",
        params: dict[str, Any] | None = None,
    ) -> TierModel:
        """Train LightGBM Regressor with early stopping on the validation set."""
        # Filter rows by tier
        tr = train_df[train_df["tier"] == self.tier].dropna(subset=[target_col]).copy()
        va = val_df[val_df["tier"] == self.tier].dropna(subset=[target_col]).copy()

        if tr.empty:
            raise ValueError(f"No training rows for tier {self.tier}")
        if va.empty:
            raise ValueError(f"No validation rows for tier {self.tier}")

        # Ensure categoricals are category dtype
        for cat in CATEGORICAL_FEATURES:
            if cat in tr.columns:
                tr[cat] = tr[cat].astype("category")
            if cat in va.columns:
                va[cat] = va[cat].astype("category")

        X_tr = tr[self.feature_cols]
        y_tr = tr[target_col].to_numpy()

        X_va = va[self.feature_cols]
        y_va = va[target_col].to_numpy()

        default_params: dict[str, Any] = {
            "objective": "regression_l1",  # Optimise for MAE
            "metric": "l1",
            "n_estimators": 600,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        if params:
            default_params.update(params)

        self.model = lgb.LGBMRegressor(**default_params)

        callbacks = [
            lgb.early_stopping(stopping_rounds=40, verbose=False),
            lgb.log_evaluation(period=0),  # Silent
        ]

        logger.info(
            "training %s tier model on %d train rows, validating on %d val rows",
            self.tier, len(tr), len(va),
        )

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=callbacks,
            )

        self.best_iteration = int(getattr(self.model, "best_iteration_", self.model.n_estimators))
        self.best_val_score = float(self.model.best_score_["valid_0"]["l1"])

        logger.info(
            "%s tier model fitted: best_iteration=%d, best_val_mae=%.1f MWh",
            self.tier, self.best_iteration, self.best_val_score,
        )

        # Automated data-driven guardrail: detect regions where model underperforms naive on validation
        self.fallback_regions = self.compute_fallback_regions(val_df, target_col=target_col)
        if self.fallback_regions:
            logger.info(
                "%s tier guardrail enabled for %d regions: %s",
                self.tier, len(self.fallback_regions), self.fallback_regions,
            )

        return self

    def compute_fallback_regions(
        self,
        val_df: pd.DataFrame,
        target_col: str = "target",
    ) -> list[str]:
        """Identify regions where model MAE on validation exceeds naive persistence.

        These regions receive an automated guardrail fallback to naive persistence at inference time.
        """
        va = val_df[val_df["tier"] == self.tier].dropna(subset=[target_col]).copy()
        if va.empty or self.model is None or "respondent" not in va.columns:
            return []

        # Predict without fallback to measure raw model performance
        raw_preds = self.predict(va, enable_fallback=False)
        va["_raw_pred"] = raw_preds

        fallback: list[str] = []
        for resp, grp in va.groupby("respondent"):
            y_true = grp[target_col].to_numpy()
            y_pred = grp["_raw_pred"].to_numpy()

            if "naive_persistence" in grp.columns:
                y_naive = grp["naive_persistence"].to_numpy()
            elif "demand_t" in grp.columns:
                y_naive = grp["demand_t"].to_numpy()
            else:
                continue

            valid = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isnan(y_naive)
            if not valid.any():
                continue

            model_mae = float(np.mean(np.abs(y_true[valid] - y_pred[valid])))
            naive_mae = float(np.mean(np.abs(y_true[valid] - y_naive[valid])))

            if model_mae > naive_mae:
                fallback.append(resp)
                pct_worse = (model_mae - naive_mae) / (naive_mae + 1e-8) * 100.0
                logger.warning(
                    "guardrail activated for %s (%s tier): val model MAE=%.1f > naive MAE=%.1f (+%.1f%%)",
                    resp, self.tier, model_mae, naive_mae, pct_worse,
                )

        return sorted(fallback)

    def predict(self, df: pd.DataFrame, enable_fallback: bool = True) -> np.ndarray:
        """Predict for given dataframe using the trained tier model."""
        if self.model is None:
            raise RuntimeError(f"TierModel for {self.tier} has not been trained yet")

        df_copy = df.copy()
        for cat in CATEGORICAL_FEATURES:
            if cat in df_copy.columns:
                df_copy[cat] = df_copy[cat].astype("category")

        X = df_copy[self.feature_cols]
        preds = np.asarray(self.model.predict(X), dtype=float)

        # Apply fallback guardrail if enabled and respondent column is present
        if enable_fallback and self.fallback_regions and "respondent" in df_copy.columns:
            fallback_col = None
            if "naive_persistence" in df_copy.columns:
                fallback_col = "naive_persistence"
            elif "demand_t" in df_copy.columns:
                fallback_col = "demand_t"

            if fallback_col is not None:
                mask = df_copy["respondent"].isin(self.fallback_regions).to_numpy()
                if mask.any():
                    naive_vals = df_copy[fallback_col].to_numpy()
                    preds[mask] = naive_vals[mask]

        return preds

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("saved %s model to %s", self.tier, filepath)

    @classmethod
    def load(cls, filepath: Path) -> TierModel:
        obj = joblib.load(filepath)
        if not isinstance(obj, TierModel):
            raise TypeError(f"Loaded object from {filepath} is not a TierModel")
        return obj


def train_all_tiers(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLUMNS,
    target_col: str = "target",
    models_dir: Path | None = None,
) -> dict[str, TierModel]:
    """Train models for all three tiers ('Large', 'Medium', 'Small')."""
    tiers = ["Large", "Medium", "Small"]
    models: dict[str, TierModel] = {}

    for tier in tiers:
        tm = TierModel(tier=tier, feature_cols=feature_cols)
        tm.fit(train_df, val_df, target_col=target_col)
        models[tier] = tm

        if models_dir:
            model_path = models_dir / f"model_tier_{tier.lower()}.joblib"
            tm.save(model_path)

    return models


def predict_with_tier_models(
    df: pd.DataFrame,
    models: dict[str, TierModel],
    pred_col_name: str = "model_prediction",
    enable_fallback: bool = True,
) -> pd.DataFrame:
    """Run inference per row matching its tier to the corresponding TierModel."""
    out = df.copy()
    predictions = np.full(len(out), np.nan)

    for tier, tm in models.items():
        idx = out["tier"] == tier
        if idx.any():
            subset = out[idx]
            preds = tm.predict(subset, enable_fallback=enable_fallback)
            predictions[idx.to_numpy()] = preds

    out[pred_col_name] = predictions
    return out
