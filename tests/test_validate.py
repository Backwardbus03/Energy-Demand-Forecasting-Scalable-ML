"""Unit tests for the Phase 2 validation gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.validate import (  # noqa: E402
    AGGREGATE_RESPONDENTS,
    Severity,
    build_coverage,
    validate,
)


def frame(rows: list[dict]) -> pd.DataFrame:
    """Build a raw-schema frame from partial row dicts."""
    base = {
        "period": "2024-01-01", "respondent": "PJM", "respondent_name": "PJM Inc.",
        "type": "D", "type_name": "Demand", "value": 1000.0,
        "value_units": "megawatthours", "timezone": "Eastern",
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["period"] = pd.to_datetime(df["period"])
    return df


def days(respondent: str, n: int, type_: str = "D", start: str = "2024-01-01", **kw):
    return [
        {
            "respondent": respondent,
            "type": type_,
            "period": (pd.Timestamp(start) + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            **kw,
        }
        for i in range(n)
    ]


def find(report, check: str):
    return [f for f in report.findings if f.check == check]


class TestStructuralChecks:
    def test_clean_data_passes(self):
        assert validate(frame(days("PJM", 30)), min_coverage_pct=50).passed

    def test_empty_input_fails(self):
        assert not validate(frame([]) if False else pd.DataFrame()).passed

    def test_duplicate_keys_fail_the_gate(self):
        df = frame([{"period": "2024-01-01"}, {"period": "2024-01-01"}])
        report = validate(df)
        assert not report.passed
        assert find(report, "duplicates")[0].severity is Severity.ERROR

    def test_unpinned_timezone_fails(self):
        """Multiple timezones mean the 5x duplication returned."""
        df = frame([
            {"period": "2024-01-01", "timezone": "Eastern"},
            {"period": "2024-01-02", "timezone": "Pacific"},
        ])
        report = validate(df)
        assert not report.passed
        assert find(report, "timezone")[0].severity is Severity.ERROR

    def test_mixed_units_fail(self):
        df = frame([
            {"period": "2024-01-01", "value_units": "megawatthours"},
            {"period": "2024-01-02", "value_units": "kilowatthours"},
        ])
        assert not validate(df).passed

    def test_missing_column_fails(self):
        df = frame(days("PJM", 3)).drop(columns=["value_units"])
        assert not validate(df).passed


class TestPhysicallyValidNegatives:
    """The dataset's real negatives must not raise false alarms."""

    def test_negative_interchange_is_not_flagged(self):
        """TI < 0 means the region exports power."""
        df = frame(days("PJM", 5, type_="TI", value=-5137.0))
        report = validate(df, min_coverage_pct=0)
        assert find(report, "negative_values")[0].severity is Severity.INFO

    def test_negative_net_generation_is_not_flagged(self):
        """NG < 0 for small BAs whose station load exceeds output."""
        df = frame(days("HGMA", 5, type_="NG", value=-16.0))
        report = validate(df, min_coverage_pct=0)
        assert find(report, "negative_values")[0].severity is Severity.INFO

    def test_negative_demand_is_flagged(self):
        """A region cannot consume less than zero."""
        df = frame(days("PJM", 5, type_="D", value=-100.0))
        report = validate(df, min_coverage_pct=0)
        assert find(report, "negative_values")[0].severity is Severity.WARNING


class TestNoSilentFixing:
    """Suspicious data is reported, never repaired."""

    def test_missing_values_reported_and_preserved(self):
        df = frame(days("PJM", 5))
        df.loc[2, "value"] = None
        report = validate(df, min_coverage_pct=0)
        assert find(report, "missing_values")[0].severity is Severity.WARNING
        assert pd.isna(df.loc[2, "value"])  # untouched

    def test_zero_demand_is_flagged_not_dropped(self):
        df = frame(days("PJM", 5))
        df.loc[1, "value"] = 0.0
        report = validate(df, min_coverage_pct=0)
        assert find(report, "zero_demand")[0].severity is Severity.WARNING
        assert len(df) == 5  # nothing removed

    def test_warnings_alone_do_not_fail_the_gate(self):
        """Statistical findings need a human decision, not a hard stop."""
        df = frame(days("PJM", 5))
        df.loc[0, "value"] = None
        report = validate(df, min_coverage_pct=0)
        assert report.warnings
        assert report.passed


class TestAggregateHandling:
    def test_aggregates_excluded_by_default(self):
        df = frame(days("PJM", 30) + days("MIDA", 30))
        report = validate(df, min_coverage_pct=50)
        assert "PJM" in report.modeling_respondents
        assert "MIDA" not in report.modeling_respondents

    def test_aggregates_kept_when_requested(self):
        df = frame(days("PJM", 30) + days("MIDA", 30))
        report = validate(df, min_coverage_pct=50, exclude_aggregates=False)
        assert "MIDA" in report.modeling_respondents

    def test_known_rollups_are_listed(self):
        for r in ("US48", "MIDA", "TEX", "CAL"):
            assert r in AGGREGATE_RESPONDENTS


class TestCoverage:
    def test_low_coverage_respondent_excluded(self):
        df = frame(days("PJM", 100) + days("BHBA", 5))
        report = validate(df, min_coverage_pct=95)
        assert "PJM" in report.modeling_respondents
        assert "BHBA" not in report.modeling_respondents

    def test_coverage_matrix_labels_roles(self):
        cov = build_coverage(frame(days("PJM", 10) + days("US48", 10)))
        assert cov.loc["PJM", "role"] == "balancing_authority"
        assert cov.loc["US48", "role"] == "aggregate"

    def test_interior_gaps_detected(self):
        rows = days("PJM", 5) + days("PJM", 5, start="2024-01-20")
        report = validate(frame(rows), min_coverage_pct=0)
        assert find(report, "interior_gaps")

    def test_report_serialises(self):
        report = validate(frame(days("PJM", 10)), min_coverage_pct=0)
        d = report.to_dict()
        assert "passed" in d and "findings" in d


class TestModelingFrameLoader:
    """The analysis window is a view; the raw store is never modified."""

    def _store(self, tmp_path: Path):
        from src.data.fetch_eia import normalise_rows, write_partitioned

        rows = []
        for year in (2021, 2022, 2023):
            for t in ("D", "DF", "NG", "TI"):
                rows.append({
                    "period": f"{year}-06-01", "respondent": "PJM",
                    "respondent-name": "PJM", "type": t, "type-name": t,
                    "value": "100", "value-units": "megawatthours",
                    "timezone": "Eastern",
                })
        rows.append({
            "period": "2022-06-01", "respondent": "MIDA", "respondent-name": "Mid-Atlantic",
            "type": "D", "type-name": "Demand", "value": "500",
            "value-units": "megawatthours", "timezone": "Eastern",
        })
        write_partitioned(normalise_rows(rows, "route"), tmp_path)
        return tmp_path

    def test_start_date_filters_window(self, tmp_path: Path):
        from src.data.validate import load_modeling_frame

        raw = self._store(tmp_path)
        panel = load_modeling_frame(raw, start="2022-01-01")
        assert panel["period"].min() >= pd.Timestamp("2022-01-01")
        assert not panel.empty

    def test_raw_store_unchanged_by_loading(self, tmp_path: Path):
        """Narrowing the window must not delete history from disk."""
        from src.data.validate import load_modeling_frame

        raw = self._store(tmp_path)
        before = sorted(p.name for p in raw.glob("year=*"))
        load_modeling_frame(raw, start="2023-01-01")
        assert sorted(p.name for p in raw.glob("year=*")) == before
        # 2021 still loadable afterwards
        assert not load_modeling_frame(raw, start="2021-01-01").empty

    def test_wide_panel_shape(self, tmp_path: Path):
        from src.data.validate import load_modeling_frame

        panel = load_modeling_frame(self._store(tmp_path), respondents=["PJM"])
        for col in ("demand", "demand_forecast", "net_generation", "interchange"):
            assert col in panel.columns
        # one row per (period, respondent), not one per observation
        assert len(panel) == panel[["period", "respondent"]].drop_duplicates().shape[0]

    def test_long_format_preserved(self, tmp_path: Path):
        from src.data.validate import load_modeling_frame

        long = load_modeling_frame(self._store(tmp_path), respondents=["PJM"], wide=False)
        assert "type" in long.columns and "value" in long.columns

    def test_respondent_filter(self, tmp_path: Path):
        from src.data.validate import load_modeling_frame

        panel = load_modeling_frame(self._store(tmp_path), respondents=["PJM"])
        assert set(panel["respondent"].unique()) == {"PJM"}

    def test_missing_store_raises(self, tmp_path: Path):
        from src.data.validate import load_modeling_frame

        with pytest.raises(FileNotFoundError):
            load_modeling_frame(tmp_path / "nope")
