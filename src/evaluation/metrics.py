"""Evaluation Metrics (Phase 4).

Implements the exact evaluation criteria codified in PROBLEM_STATEMENT.md:
  - MAE (MWh): honest magnitude
  - SMAPE (%): symmetric percentage error, robust to small values
  - Weighted MAPE (%): error sum divided by actual sum, scale-aware
  - R²: proportion of variance explained

Metrics are evaluated and reported PER TIER (Large, Medium, Small),
preventing scale aggregation artefacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass(frozen=True)
class MetricResult:
    mae: float
    smape: float
    wmape: float
    r2: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mae": round(self.mae, 2),
            "smape": round(self.smape, 2),
            "wmape": round(self.wmape, 2),
            "r2": round(self.r2, 4),
            "n_samples": self.n_samples,
        }


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (MWh)."""
    return float(mean_absolute_error(y_true, y_pred))


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error (%).

    Formula: 200 * mean(|y - y_hat| / (|y| + |y_hat| + eps))
    """
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    diff = np.abs(y_true - y_pred)
    return float(np.mean(200.0 * diff / denom))


def compute_wmape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Weighted Mean Absolute Percentage Error (%).

    Formula: 100 * sum(|y - y_hat|) / (sum(y) + eps)
    """
    total_actual = float(np.sum(np.abs(y_true))) + eps
    total_abs_error = float(np.sum(np.abs(y_true - y_pred)))
    return float(100.0 * total_abs_error / total_actual)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R²)."""
    if len(y_true) < 2:
        return 0.0
    return float(r2_score(y_true, y_pred))


def evaluate_series(y_true: np.ndarray, y_pred: np.ndarray) -> MetricResult:
    """Compute all standard headline metrics for aligned true and predicted series."""
    # Filter out NaNs
    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y_t = y_true[valid]
    y_p = y_pred[valid]

    if len(y_t) == 0:
        return MetricResult(mae=np.nan, smape=np.nan, wmape=np.nan, r2=np.nan, n_samples=0)

    return MetricResult(
        mae=compute_mae(y_t, y_p),
        smape=compute_smape(y_t, y_p),
        wmape=compute_wmape(y_t, y_p),
        r2=compute_r2(y_t, y_p),
        n_samples=len(y_t),
    )


def evaluate_dataframe(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str = "target",
    tier_col: str = "tier",
) -> dict[str, dict[str, Any]]:
    """Evaluate predictions per tier and overall across a scored dataframe.

    Returns
    -------
    dict mapping tier name ('Large', 'Medium', 'Small', 'Overall') to metrics dict.
    """
    results: dict[str, dict[str, Any]] = {}

    for tier, group in df.groupby(tier_col):
        y_true = group[target_col].to_numpy()
        y_pred = group[pred_col].to_numpy()
        results[str(tier)] = evaluate_series(y_true, y_pred).to_dict()

    # Overall summary
    all_true = df[target_col].to_numpy()
    all_pred = df[pred_col].to_numpy()
    results["Overall"] = evaluate_series(all_true, all_pred).to_dict()

    return results
