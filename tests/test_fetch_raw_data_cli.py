from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from scripts import fetch_raw_data
from src.data.fetch_eia import IngestionResult


from src.utils.config import ConfigError


def test_fetch_raw_data_missing_api_key_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["fetch_raw_data.py"])
    with patch("scripts.fetch_raw_data.load_config", side_effect=ConfigError("EIA_API_KEY is not set")):
        exit_code = fetch_raw_data.main()
        assert exit_code == 2


def test_fetch_raw_data_success_exits_0(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIA_API_KEY", "dummy_key")
    monkeypatch.setattr("sys.argv", ["fetch_raw_data.py", "--limit-respondents", "1"])

    mock_res = IngestionResult(
        rows_written=500,
        respondents_ok=["CISO"],
        respondents_skipped=[],
        respondents_no_data=[],
        respondents_failed=[],
        api_calls=2,
    )
    with patch("scripts.fetch_raw_data.ingest", return_value=mock_res):
        exit_code = fetch_raw_data.main()
        assert exit_code == 0


def test_fetch_raw_data_incremental_already_up_to_date_exits_0(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIA_API_KEY", "dummy_key")
    monkeypatch.setattr("sys.argv", ["fetch_raw_data.py", "--incremental"])

    mock_res = IngestionResult(
        rows_written=0,
        respondents_ok=[],
        respondents_skipped=["CISO", "ERCO"],
        respondents_no_data=[],
        respondents_failed=[],
        api_calls=0,
    )
    with patch("scripts.fetch_raw_data.ingest", return_value=mock_res):
        exit_code = fetch_raw_data.main()
        assert exit_code == 0


def test_fetch_raw_data_no_rows_and_none_skipped_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIA_API_KEY", "dummy_key")
    monkeypatch.setattr("sys.argv", ["fetch_raw_data.py"])

    mock_res = IngestionResult(
        rows_written=0,
        respondents_ok=[],
        respondents_skipped=[],
        respondents_no_data=["CISO"],
        respondents_failed=[],
        api_calls=1,
    )
    with patch("scripts.fetch_raw_data.ingest", return_value=mock_res):
        exit_code = fetch_raw_data.main()
        assert exit_code == 1


def test_fetch_raw_data_network_failure_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIA_API_KEY", "dummy_key")
    monkeypatch.setattr("sys.argv", ["fetch_raw_data.py"])

    mock_res = IngestionResult(
        rows_written=10,
        respondents_ok=["CISO"],
        respondents_skipped=[],
        respondents_no_data=[],
        respondents_failed=["ERCO"],
        api_calls=2,
    )
    with patch("scripts.fetch_raw_data.ingest", return_value=mock_res):
        exit_code = fetch_raw_data.main()
        assert exit_code == 1
