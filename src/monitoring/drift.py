"""Evidently AI Drift & Performance Monitoring Engine (Phase 12)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    RegressionPreset,
)
from evidently.report import Report

import sys

_compat_mse: Any = None

# Ensure backwards compatibility for Evidently with scikit-learn>=1.6 (which removed `squared` kwarg)
try:
    import sklearn.metrics
    import sklearn.metrics._regression

    _orig_mse = sklearn.metrics._regression.mean_squared_error

    def _compat_mse_fn(*args: Any, **kwargs: Any) -> Any:
        squared = kwargs.pop("squared", True)
        res = _orig_mse(*args, **kwargs)
        if not squared:
            return res ** 0.5
        return res

    _compat_mse = _compat_mse_fn
    sklearn.metrics.mean_squared_error = _compat_mse
    sklearn.metrics._regression.mean_squared_error = _compat_mse

    for mod_name in list(sys.modules.keys()):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "mean_squared_error") and mod_name.startswith(("evidently", "sklearn")):
            setattr(mod, "mean_squared_error", _compat_mse)
except Exception:
    pass

from src.features.build_features import FEATURE_COLUMNS

logger = logging.getLogger("eia.monitoring")


@dataclass(frozen=True)
class DriftSummary:
    reference_start_date: str
    reference_end_date: str
    current_start_date: str
    current_end_date: str
    total_features_evaluated: int
    drifted_features_count: int
    share_of_drifted_features: float
    dataset_drift_detected: bool
    target_drift_detected: bool
    target_drift_score: float
    wmape_reference: float | None
    wmape_current: float | None


class ModelMonitor:
    def __init__(
        self,
        drift_share_threshold: float = 0.30,
        target_p_value_threshold: float = 0.05,
    ) -> None:
        self.drift_share_threshold = drift_share_threshold
        self.target_p_value_threshold = target_p_value_threshold

    def prepare_datasets(
        self,
        df: pd.DataFrame,
        reference_days: int = 90,
        current_days: int = 30,
        date_col: str = "target_date",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split DataFrame into non-overlapping reference and current windows.

        Reference: [max_date - current_days - reference_days, max_date - current_days]
        Current:   [max_date - current_days, max_date]
        """
        if df.empty or date_col not in df.columns:
            raise ValueError(f"DataFrame must contain non-empty '{date_col}' column")

        dates = pd.to_datetime(df[date_col]).sort_values().unique()
        if len(dates) < (reference_days + current_days) // 3:
            raise ValueError(f"Insufficient distinct dates ({len(dates)}) for drift monitoring")

        max_date = dates[-1]
        split_date = max_date - pd.Timedelta(days=current_days)
        ref_start = split_date - pd.Timedelta(days=reference_days)

        ref_df = df[(df[date_col] >= ref_start) & (df[date_col] < split_date)].copy().reset_index(drop=True)
        curr_df = df[df[date_col] >= split_date].copy().reset_index(drop=True)

        if ref_df.empty or curr_df.empty:
            raise ValueError(f"Empty split created: ref={len(ref_df)}, curr={len(curr_df)}")

        assert ref_df[date_col].max() < curr_df[date_col].min(), "Temporal overlap in monitoring split!"
        return ref_df, curr_df

    def build_report(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        target_col: str = "target",
        prediction_col: str = "prediction",
    ) -> tuple[Report, DriftSummary]:
        """Execute Evidently data drift, data quality, and regression evaluation."""
        cols = feature_cols or [c for c in FEATURE_COLUMNS if c in reference_df.columns]
        eval_cols = list(cols)

        if target_col in reference_df.columns and target_col in current_df.columns:
            eval_cols.append(target_col)
        if prediction_col in reference_df.columns and prediction_col in current_df.columns:
            eval_cols.append(prediction_col)

        ref_eval = reference_df[eval_cols].dropna(subset=[cols[0]]).copy() if cols else reference_df.copy()
        curr_eval = current_df[eval_cols].dropna(subset=[cols[0]]).copy() if cols else current_df.copy()

        metrics_list: list[Any] = [
            DataDriftPreset(drift_share=self.drift_share_threshold),
            DataQualityPreset(),
        ]

        has_predictions = (
            prediction_col in eval_cols
            and target_col in eval_cols
            and not ref_eval[prediction_col].isna().all()
            and not curr_eval[prediction_col].isna().all()
        )
        if has_predictions:
            from evidently.metrics import RegressionQualityMetric

            metrics_list.append(RegressionQualityMetric())

        # Ensure compatibility patch is applied to all loaded evidently submodules
        if _compat_mse is not None:
            try:
                for mod_name in list(sys.modules.keys()):
                    mod = sys.modules.get(mod_name)
                    if mod and hasattr(mod, "mean_squared_error") and mod_name.startswith("evidently"):
                        setattr(mod, "mean_squared_error", _compat_mse)
            except Exception:
                pass

        report = Report(metrics=metrics_list)
        report.run(reference_data=ref_eval, current_data=curr_eval)

        raw_metrics = report.as_dict()
        summary = self._extract_summary(
            raw_metrics=raw_metrics,
            reference_df=reference_df,
            current_df=current_df,
            date_col="target_date" if "target_date" in reference_df.columns else "date",
        )
        return report, summary

    def _extract_summary(
        self,
        raw_metrics: dict[str, Any],
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        date_col: str,
    ) -> DriftSummary:
        drift_data: dict[str, Any] = {}
        for m in raw_metrics.get("metrics", []):
            if m.get("metric") == "DatasetDriftMetric":
                drift_data = m.get("result", {})
                break

        n_features = drift_data.get("number_of_columns", 0)
        n_drifted = drift_data.get("number_of_drifted_columns", 0)
        share_drifted = drift_data.get("share_of_drifted_columns", 0.0)
        dataset_drift = bool(drift_data.get("dataset_drift", False))

        target_drift = False
        target_score = 0.0
        drift_by_col = drift_data.get("drift_by_columns", {})
        if "target" in drift_by_col:
            target_score = float(drift_by_col["target"].get("drift_score", 0.0))
            target_drift = bool(drift_by_col["target"].get("drift_detected", False))

        ref_start = str(pd.to_datetime(reference_df[date_col].min()).date()) if date_col in reference_df else "unknown"
        ref_end = str(pd.to_datetime(reference_df[date_col].max()).date()) if date_col in reference_df else "unknown"
        curr_start = str(pd.to_datetime(current_df[date_col].min()).date()) if date_col in current_df else "unknown"
        curr_end = str(pd.to_datetime(current_df[date_col].max()).date()) if date_col in current_df else "unknown"

        return DriftSummary(
            reference_start_date=ref_start,
            reference_end_date=ref_end,
            current_start_date=curr_start,
            current_end_date=curr_end,
            total_features_evaluated=n_features,
            drifted_features_count=n_drifted,
            share_of_drifted_features=round(share_drifted, 4),
            dataset_drift_detected=dataset_drift,
            target_drift_detected=target_drift,
            target_drift_score=round(target_score, 4),
            wmape_reference=None,
            wmape_current=None,
        )

    def save_reports(
        self,
        report: Report,
        summary: DriftSummary,
        html_path: Path,
        json_path: Path,
    ) -> None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        report.save_html(str(html_path))
        logger.info("Saved interactive HTML drift report to %s", html_path)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2)
        logger.info("Saved machine-readable drift metrics to %s", json_path)
