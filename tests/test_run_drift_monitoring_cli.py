from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from scripts import run_drift_monitoring
from src.monitoring.drift import DriftSummary


def test_drift_monitoring_missing_features_file_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    non_existent = tmp_path / "missing.parquet"
    monkeypatch.setattr("sys.argv", [
        "run_drift_monitoring.py",
        "--features-path", str(non_existent),
    ])

    exit_code = run_drift_monitoring.main()
    assert exit_code == 2


def test_drift_monitoring_success_exits_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    features_file = tmp_path / "features.parquet"
    html_file = tmp_path / "report.html"
    json_file = tmp_path / "report.json"

    # Create synthetic features DataFrame
    df = pd.DataFrame({
        "target_date": pd.date_range("2024-01-01", periods=120, freq="D"),
        "target": np.random.normal(1000, 50, 120),
        "prediction": np.random.normal(1000, 50, 120),
        "demand_lag_1": np.random.normal(1000, 50, 120),
    })
    df.to_parquet(features_file, index=False)

    monkeypatch.setattr("sys.argv", [
        "run_drift_monitoring.py",
        "--features-path", str(features_file),
        "--output-html", str(html_file),
        "--output-json", str(json_file),
        "--reference-days", "60",
        "--current-days", "30",
    ])

    dummy_summary = DriftSummary(
        reference_start_date="2024-01-01",
        reference_end_date="2024-03-01",
        current_start_date="2024-03-01",
        current_end_date="2024-04-01",
        total_features_evaluated=3,
        drifted_features_count=0,
        share_of_drifted_features=0.0,
        dataset_drift_detected=False,
        target_drift_detected=False,
        target_drift_score=0.99,
        wmape_reference=None,
        wmape_current=None,
    )

    mock_report = MagicMock()

    with patch("scripts.run_drift_monitoring.ModelMonitor.build_report", return_value=(mock_report, dummy_summary)):
        exit_code = run_drift_monitoring.main()
        assert exit_code == 0
        assert json_file.exists()
        loaded = json.loads(json_file.read_text())
        assert loaded["dataset_drift_detected"] is False


def test_drift_monitoring_fail_on_drift_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    features_file = tmp_path / "features.parquet"
    html_file = tmp_path / "report.html"
    json_file = tmp_path / "report.json"

    df = pd.DataFrame({
        "target_date": pd.date_range("2024-01-01", periods=120, freq="D"),
        "target": np.random.normal(1000, 50, 120),
    })
    df.to_parquet(features_file, index=False)

    monkeypatch.setattr("sys.argv", [
        "run_drift_monitoring.py",
        "--features-path", str(features_file),
        "--output-html", str(html_file),
        "--output-json", str(json_file),
        "--reference-days", "60",
        "--current-days", "30",
        "--fail-on-drift",
    ])

    drift_summary = DriftSummary(
        reference_start_date="2024-01-01",
        reference_end_date="2024-03-01",
        current_start_date="2024-03-01",
        current_end_date="2024-04-01",
        total_features_evaluated=3,
        drifted_features_count=2,
        share_of_drifted_features=0.66,
        dataset_drift_detected=True,
        target_drift_detected=False,
        target_drift_score=0.01,
        wmape_reference=None,
        wmape_current=None,
    )

    mock_report = MagicMock()

    with patch("scripts.run_drift_monitoring.ModelMonitor.build_report", return_value=(mock_report, drift_summary)):
        exit_code = run_drift_monitoring.main()
        assert exit_code == 1

