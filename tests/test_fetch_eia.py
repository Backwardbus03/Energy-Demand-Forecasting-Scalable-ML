"""Unit tests for raw EIA ingestion.

These run offline: no test here touches the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetch_eia import (  # noqa: E402
    KEY_COLUMNS,
    RAW_COLUMNS,
    normalise_rows,
    resolve_incremental_start,
    write_partitioned,
)

ROUTE = "electricity/rto/daily-region-data"


def api_row(period: str, respondent: str = "PJM", type_: str = "D", value="1000"):
    """One row shaped exactly as the live EIA API returns it."""
    return {
        "period": period,
        "respondent": respondent,
        "respondent-name": "PJM Interconnection, LLC",
        "type": type_,
        "type-name": "Demand",
        "value": value,
        "value-units": "megawatthours",
        "timezone": "Eastern",
    }


class TestNormaliseRows:
    def test_maps_api_schema_to_raw_schema(self):
        df = normalise_rows([api_row("2024-01-01")], ROUTE)
        assert list(df.columns) == RAW_COLUMNS
        assert len(df) == 1
        assert df.loc[0, "respondent"] == "PJM"
        assert df.loc[0, "value"] == 1000.0

    def test_empty_input_keeps_schema(self):
        df = normalise_rows([], ROUTE)
        assert df.empty
        assert list(df.columns) == RAW_COLUMNS

    def test_null_value_becomes_nan_not_zero(self):
        """A missing observation must never be silently recorded as 0."""
        df = normalise_rows([api_row("2024-01-01", value=None)], ROUTE)
        assert pd.isna(df.loc[0, "value"])
        assert df.loc[0, "value"] != 0

    def test_unparseable_value_becomes_nan(self):
        df = normalise_rows([api_row("2024-01-01", value="n/a")], ROUTE)
        assert pd.isna(df.loc[0, "value"])

    def test_negative_values_preserved(self):
        """Interchange (TI) is legitimately negative when a region exports."""
        df = normalise_rows([api_row("2024-01-01", type_="TI", value="-5137")], ROUTE)
        assert df.loc[0, "value"] == -5137.0

    def test_period_parsed_as_datetime(self):
        df = normalise_rows([api_row("2024-03-15")], ROUTE)
        assert df.loc[0, "period"] == pd.Timestamp("2024-03-15")

    def test_provenance_columns_populated(self):
        df = normalise_rows([api_row("2024-01-01")], ROUTE)
        assert df.loc[0, "_source_route"] == ROUTE
        assert df.loc[0, "_ingested_at"]


class TestWritePartitioned:
    def test_writes_year_partition(self, tmp_path: Path):
        df = normalise_rows([api_row("2024-01-01"), api_row("2024-06-01")], ROUTE)
        n = write_partitioned(df, tmp_path)
        assert n == 2
        assert (tmp_path / "year=2024" / "data.parquet").exists()

    def test_splits_across_years(self, tmp_path: Path):
        df = normalise_rows([api_row("2023-12-31"), api_row("2024-01-01")], ROUTE)
        write_partitioned(df, tmp_path)
        assert (tmp_path / "year=2023" / "data.parquet").exists()
        assert (tmp_path / "year=2024" / "data.parquet").exists()

    def test_rerun_is_idempotent(self, tmp_path: Path):
        """Re-ingesting the same window must not duplicate rows."""
        df = normalise_rows([api_row("2024-01-01")], ROUTE)
        write_partitioned(df, tmp_path)
        write_partitioned(df, tmp_path)
        stored = pd.read_parquet(tmp_path / "year=2024" / "data.parquet")
        assert len(stored) == 1
        assert stored.duplicated(KEY_COLUMNS).sum() == 0

    def test_revision_overwrites_prior_value(self, tmp_path: Path):
        """EIA revises recent days; the newer observation must win."""
        first = normalise_rows([api_row("2024-01-01", value="1000")], ROUTE)
        write_partitioned(first, tmp_path)

        revised = normalise_rows([api_row("2024-01-01", value="2000")], ROUTE)
        revised["_ingested_at"] = "2099-01-01T00:00:00+00:00"  # strictly later
        write_partitioned(revised, tmp_path)

        stored = pd.read_parquet(tmp_path / "year=2024" / "data.parquet")
        assert len(stored) == 1
        assert stored.loc[0, "value"] == 2000.0

    def test_distinct_types_are_not_deduped(self, tmp_path: Path):
        """D/DF/NG/TI share a date+region but are separate observations."""
        rows = [api_row("2024-01-01", type_=t) for t in ("D", "DF", "NG", "TI")]
        write_partitioned(normalise_rows(rows, ROUTE), tmp_path)
        stored = pd.read_parquet(tmp_path / "year=2024" / "data.parquet")
        assert len(stored) == 4

    def test_empty_dataframe_writes_nothing(self, tmp_path: Path):
        assert write_partitioned(normalise_rows([], ROUTE), tmp_path) == 0
        assert not list(tmp_path.glob("year=*"))


class TestIncremental:
    def test_returns_none_when_no_data(self, tmp_path: Path):
        assert resolve_incremental_start(tmp_path, lookback_days=7) is None

    def test_backs_off_by_lookback_window(self, tmp_path: Path):
        df = normalise_rows([api_row("2024-06-10")], ROUTE)
        write_partitioned(df, tmp_path)
        assert resolve_incremental_start(tmp_path, lookback_days=7) == "2024-06-03"

    def test_uses_latest_period_across_partitions(self, tmp_path: Path):
        rows = [api_row("2023-05-01"), api_row("2024-06-10")]
        write_partitioned(normalise_rows(rows, ROUTE), tmp_path)
        assert resolve_incremental_start(tmp_path, lookback_days=0) == "2024-06-10"


class TestConfigSafety:
    def test_pinned_timezone_required(self, tmp_path: Path, monkeypatch):
        """An unpinned timezone facet would inflate the dataset 5x."""
        from src.utils.config import ConfigError, load_config

        cfg = Path("configs/ingestion.yaml").read_text().replace(
            'timezone: "Eastern"', "timezone: null"
        )
        bad = tmp_path / "bad.yaml"
        bad.write_text(cfg)
        monkeypatch.setenv("EIA_API_KEY", "x" * 40)

        with pytest.raises(ConfigError, match="timezone"):
            load_config(bad)


class TestResume:
    """An interrupted backfill must resume without re-fetching or duplicating."""

    def test_stored_respondents_empty_when_no_data(self, tmp_path: Path):
        from src.data.fetch_eia import stored_respondents

        assert stored_respondents(tmp_path) == set()

    def test_stored_respondents_lists_what_is_on_disk(self, tmp_path: Path):
        from src.data.fetch_eia import stored_respondents

        rows = [api_row("2024-01-01", respondent=r) for r in ("PJM", "CISO", "ERCO")]
        write_partitioned(normalise_rows(rows, ROUTE), tmp_path)
        assert stored_respondents(tmp_path) == {"PJM", "CISO", "ERCO"}

    def test_stored_respondents_spans_partitions(self, tmp_path: Path):
        from src.data.fetch_eia import stored_respondents

        write_partitioned(
            normalise_rows([api_row("2023-01-01", respondent="PJM")], ROUTE), tmp_path
        )
        write_partitioned(
            normalise_rows([api_row("2024-01-01", respondent="MISO")], ROUTE), tmp_path
        )
        assert stored_respondents(tmp_path) == {"PJM", "MISO"}


class TestFailureClassification:
    """A network outage must never be recorded as 'this region has no data'."""

    def test_network_failure_flagged_distinctly_from_no_data(self):
        from src.data.fetch_eia import IngestionResult

        outage = IngestionResult(
            rows_written=0,
            respondents_ok=[],
            respondents_failed=["PACW", "PJM"],
            api_calls=2,
            respondents_no_data=[],
        )
        assert outage.had_network_failures

        genuine = IngestionResult(
            rows_written=100,
            respondents_ok=["PJM"],
            respondents_failed=[],
            api_calls=2,
            respondents_no_data=["AEC"],
        )
        assert not genuine.had_network_failures


class TestSecretRedaction:
    def test_api_key_stripped_from_error_text(self):
        """requests embeds the key in exception URLs; it must not reach logs."""
        from src.data.eia_client import _redact

        key = "abc123SECRETkey456"
        msg = f"HTTPSConnectionPool: /v2/data/?api_key={key}&frequency=daily"
        out = _redact(msg, key)
        assert key not in out
        assert "<REDACTED>" in out
