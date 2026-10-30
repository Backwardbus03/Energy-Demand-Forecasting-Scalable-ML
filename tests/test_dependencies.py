from __future__ import annotations

import pandas as pd


def test_pandas_version_is_below_3() -> None:
    major = int(pd.__version__.split(".")[0])
    assert major < 3, f"pandas version must be < 3.0 to support evidently 0.4.x, got {pd.__version__}"


def test_evidently_imports_successfully() -> None:
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset, RegressionPreset
    from evidently.report import Report

    assert Report is not None
    assert DataDriftPreset is not None
    assert DataQualityPreset is not None
    assert RegressionPreset is not None


def test_drift_module_importable() -> None:
    from src.monitoring.drift import DriftSummary, ModelMonitor

    assert ModelMonitor is not None
    assert DriftSummary is not None

