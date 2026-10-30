from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from scripts import validate_raw_data


def test_validate_raw_data_missing_data_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["validate_raw_data.py"])
    with patch("scripts.validate_raw_data.load_raw", side_effect=FileNotFoundError("No data found")):
        exit_code = validate_raw_data.main()
        assert exit_code == 2


def test_validate_raw_data_success_exits_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report_file = tmp_path / "val_report.json"
    monkeypatch.setattr("sys.argv", ["validate_raw_data.py", "--report", str(report_file)])

    # Mock valid dataframe and passed report
    mock_df = pd.DataFrame({
        "period": pd.date_range("2024-01-01", periods=10, freq="D"),
        "respondent": ["CISO"] * 10,
        "type": ["D"] * 10,
        "value": [100.0] * 10,
    })

    mock_report = MagicMock()
    mock_report.passed = True
    mock_report.row_count = 10
    mock_report.errors = []
    mock_report.warnings = []
    mock_report.modeling_respondents = ["CISO"]
    mock_report.findings = []

    with patch("scripts.validate_raw_data.load_raw", return_value=mock_df), \
         patch("scripts.validate_raw_data.validate", return_value=mock_report):
        exit_code = validate_raw_data.main()
        assert exit_code == 0
        mock_report.save.assert_called_once_with(report_file)


def test_validate_raw_data_failure_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report_file = tmp_path / "val_report.json"
    monkeypatch.setattr("sys.argv", ["validate_raw_data.py", "--report", str(report_file)])

    mock_df = pd.DataFrame({"period": [], "respondent": [], "type": [], "value": []})
    mock_report = MagicMock()
    mock_report.passed = False
    mock_report.row_count = 0
    mock_report.errors = ["Zero rows"]
    mock_report.warnings = []
    mock_report.modeling_respondents = []
    mock_report.findings = []

    with patch("scripts.validate_raw_data.load_raw", return_value=mock_df), \
         patch("scripts.validate_raw_data.validate", return_value=mock_report):
        exit_code = validate_raw_data.main()
        assert exit_code == 1

