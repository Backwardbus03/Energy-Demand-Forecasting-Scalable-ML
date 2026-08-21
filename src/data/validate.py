"""Validation gate for raw EIA data (Phase 2).

Runs between ingestion and feature engineering. In Phase 8 this becomes an
Airflow task whose failure stops the DAG before bad data reaches a model.

Two classes of check, deliberately kept apart:

  STRUCTURAL  - the pipeline is broken (duplicate keys, wrong dtypes,
                missing columns). These FAIL the run.
  STATISTICAL - the world is odd (gaps, outliers, short series). These are
                REPORTED, never silently repaired.

The distinction matters because this dataset contains values that look wrong
but are not: interchange (TI) is legitimately negative when a region exports
power, and several regions genuinely stop or start mid-window.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("eia.validate")

# Regional roll-ups that SUM other respondents in the same dataset.
# Verified empirically: corr(MIDA, PJM) = 1.0000, corr(TEX, ERCO) = 1.0000,
# and the regional aggregates recover ~97% of US48. Training on both an
# aggregate and its components double-counts the same electricity.
AGGREGATE_RESPONDENTS: frozenset[str] = frozenset(
    {"US48", "CAL", "CAR", "CENT", "FLA", "MIDA", "MIDW", "NE", "NW", "NY", "SE", "TEN", "TEX"}
)

# Series where a negative value is physically meaningful.
#   TI - net interchange: negative when a region exports more than it imports.
#   NG - net generation: negative for small BAs whose station load exceeds
#        output (pumped storage pumping, plant auxiliaries). Verified in this
#        dataset: HGMA/DEAA/GRIF are small BAs sitting persistently negative.
# Only D (demand) is truly non-negative - a region cannot consume < 0 MWh.
NEGATIVE_ALLOWED: frozenset[str] = frozenset({"TI", "NG"})

TARGET_TYPE = "D"
REQUIRED_COLUMNS = [
    "period", "respondent", "respondent_name", "type",
    "type_name", "value", "value_units", "timezone",
]
KEY_COLUMNS = ["period", "respondent", "type"]


class Severity(str, Enum):
    ERROR = "ERROR"      # structural: fails the run
    WARNING = "WARNING"  # statistical: reported, needs a human decision
    INFO = "INFO"        # documented fact


@dataclass
class Finding:
    check: str
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.value:7s}] {self.check}: {self.message}"


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)
    coverage: pd.DataFrame | None = None
    modeling_respondents: list[str] = field(default_factory=list)
    row_count: int = 0
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add(self, check: str, severity: Severity, message: str, **details: Any) -> None:
        self.findings.append(Finding(check, severity, message, details))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def passed(self) -> bool:
        """Structural errors fail the gate. Warnings never do."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "passed": self.passed,
            "row_count": self.row_count,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "modeling_respondents": self.modeling_respondents,
            "findings": [
                {
                    "check": f.check,
                    "severity": f.severity.value,
                    "message": f.message,
                    "details": f.details,
                }
                for f in self.findings
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        if self.coverage is not None:
            self.coverage.to_csv(path.with_name("coverage_matrix.csv"))


# --------------------------------------------------------------- structural

def check_schema(df: pd.DataFrame, report: ValidationReport) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.add("schema", Severity.ERROR, f"missing columns: {missing}", columns=missing)
        return

    if not pd.api.types.is_datetime64_any_dtype(df["period"]):
        report.add("schema", Severity.ERROR, "`period` is not datetime")
    if not pd.api.types.is_numeric_dtype(df["value"]):
        report.add("schema", Severity.ERROR, "`value` is not numeric")

    units = set(df["value_units"].dropna().unique())
    if len(units) > 1:
        report.add("units", Severity.ERROR, f"mixed units: {units}", units=sorted(units))
    else:
        report.add("units", Severity.INFO, f"single unit: {units.pop() if units else 'n/a'}")


def check_duplicates(df: pd.DataFrame, report: ValidationReport) -> None:
    """Each (period, respondent, type) must appear exactly once."""
    n = int(df.duplicated(KEY_COLUMNS).sum())
    if n:
        sample = df[df.duplicated(KEY_COLUMNS, keep=False)].head(5)
        report.add(
            "duplicates", Severity.ERROR,
            f"{n:,} duplicate rows on {KEY_COLUMNS}",
            count=n, sample=sample[KEY_COLUMNS].astype(str).to_dict("records"),
        )
    else:
        report.add("duplicates", Severity.INFO, "no duplicate keys")


def check_timezone_pinned(df: pd.DataFrame, report: ValidationReport) -> None:
    """More than one timezone means the 5x duplication crept back in."""
    tzs = set(df["timezone"].dropna().unique())
    if len(tzs) > 1:
        report.add(
            "timezone", Severity.ERROR,
            f"{len(tzs)} timezones present - dataset is inflated: {sorted(tzs)}",
            timezones=sorted(tzs),
        )
    else:
        report.add("timezone", Severity.INFO, f"timezone pinned to {tzs.pop() if tzs else 'n/a'}")


# -------------------------------------------------------------- statistical

def check_missing_values(df: pd.DataFrame, report: ValidationReport) -> None:
    n = int(df["value"].isna().sum())
    if n == 0:
        report.add("missing_values", Severity.INFO, "no missing values")
        return
    by_type = df[df["value"].isna()].groupby("type").size().to_dict()
    report.add(
        "missing_values", Severity.WARNING,
        f"{n:,} missing values ({n / len(df) * 100:.4f}%) - left as NaN, never zero-filled",
        count=n, by_type={k: int(v) for k, v in by_type.items()},
    )


def check_negative_values(df: pd.DataFrame, report: ValidationReport) -> None:
    """Negative TI is physical (net export); negative D/NG is not."""
    bad = df[(df["value"] < 0) & (~df["type"].isin(NEGATIVE_ALLOWED))]
    if bad.empty:
        report.add(
            "negative_values", Severity.INFO,
            "no invalid negatives (TI/NG negatives are physical and expected)",
        )
        return
    report.add(
        "negative_values", Severity.WARNING,
        f"{len(bad):,} negative values where negatives are not physical "
        f"(types: {sorted(bad['type'].unique())})",
        count=len(bad),
        by_type={k: int(v) for k, v in bad.groupby("type").size().items()},
        by_respondent={
            k: int(v) for k, v in
            bad.groupby("respondent").size().sort_values(ascending=False).head(10).items()
        },
    )


def check_zero_values(df: pd.DataFrame, report: ValidationReport) -> None:
    """Zero demand is suspicious - it is what a naive pipeline writes on failure."""
    z = df[(df["type"] == TARGET_TYPE) & (df["value"] == 0)]
    if z.empty:
        report.add("zero_demand", Severity.INFO, "no zero-valued demand observations")
        return
    report.add(
        "zero_demand", Severity.WARNING,
        f"{len(z):,} zero-demand observations - a region cannot consume 0 MWh, so "
        "these are likely reporting artefacts; excluded or imputed explicitly in Phase 3",
        count=len(z),
        by_respondent=z.groupby("respondent").size().sort_values(ascending=False).head(10).to_dict(),
    )


def build_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per-respondent completeness of the demand series."""
    d = df[df["type"] == TARGET_TYPE]
    if d.empty:
        return pd.DataFrame()

    span_start, span_end = df["period"].min(), df["period"].max()
    expected = (span_end - span_start).days + 1

    cov = d.groupby("respondent").agg(
        name=("respondent_name", "first"),
        days=("period", "nunique"),
        first_day=("period", "min"),
        last_day=("period", "max"),
        n_missing=("value", lambda s: int(s.isna().sum())),
    )
    cov["coverage_pct"] = (cov["days"] / expected * 100).round(2)
    # Interior gaps: days absent between a series' own first and last day.
    cov["interior_gaps"] = (
        (cov["last_day"] - cov["first_day"]).dt.days + 1 - cov["days"]
    ).clip(lower=0)
    cov["role"] = [
        "aggregate" if r in AGGREGATE_RESPONDENTS else "balancing_authority"
        for r in cov.index
    ]
    return cov.sort_values("coverage_pct", ascending=False)


def check_coverage(
    df: pd.DataFrame,
    report: ValidationReport,
    min_coverage_pct: float,
    exclude_aggregates: bool,
) -> None:
    """Build the coverage matrix and select the modeling set."""
    cov = build_coverage(df)
    report.coverage = cov
    if cov.empty:
        report.add("coverage", Severity.ERROR, "no demand (type=D) rows found")
        return

    eligible = cov
    if exclude_aggregates:
        aggs = cov[cov["role"] == "aggregate"]
        report.add(
            "aggregates", Severity.INFO,
            f"excluding {len(aggs)} aggregate respondents (they sum other respondents; "
            "keeping both double-counts the same electricity)",
            excluded=sorted(aggs.index.tolist()),
        )
        eligible = cov[cov["role"] == "balancing_authority"]

    keep = eligible[eligible["coverage_pct"] >= min_coverage_pct]
    drop = eligible[eligible["coverage_pct"] < min_coverage_pct]

    report.modeling_respondents = sorted(keep.index.tolist())

    if not drop.empty:
        report.add(
            "coverage", Severity.WARNING,
            f"{len(drop)} respondents below {min_coverage_pct}% coverage - excluded from "
            "modeling and documented (grid lifecycle events, not ingestion errors)",
            excluded={
                r: {
                    "coverage_pct": float(row["coverage_pct"]),
                    "days": int(row["days"]),
                    "first_day": str(row["first_day"].date()),
                    "last_day": str(row["last_day"].date()),
                }
                for r, row in drop.iterrows()
            },
        )

    gaps = keep[keep["interior_gaps"] > 0]
    if not gaps.empty:
        report.add(
            "interior_gaps", Severity.WARNING,
            f"{len(gaps)} modeling respondents have interior gaps - lag features will "
            "propagate NaN across these; handle explicitly in Phase 3",
            respondents={r: int(row["interior_gaps"]) for r, row in gaps.iterrows()},
        )

    report.add(
        "coverage", Severity.INFO,
        f"modeling set: {len(keep)} respondents at >= {min_coverage_pct}% coverage",
        n_selected=len(keep),
    )


def check_target_leakage_risk(df: pd.DataFrame, report: ValidationReport) -> None:
    """DF is EIA's own day-ahead forecast: a benchmark, never an input feature."""
    if TARGET_TYPE in df["type"].values and "DF" in df["type"].values:
        report.add(
            "leakage_watch", Severity.INFO,
            "DF (day-ahead forecast) present - treat as a BENCHMARK to beat, not as a "
            "feature for predicting D at t+1",
        )


# ------------------------------------------------------------------ runner

def validate(
    df: pd.DataFrame,
    min_coverage_pct: float = 95.0,
    exclude_aggregates: bool = True,
) -> ValidationReport:
    """Run every check and return the report."""
    report = ValidationReport(row_count=len(df))

    if df.empty:
        report.add("input", Severity.ERROR, "dataset is empty")
        return report

    check_schema(df, report)
    if report.errors:  # later checks assume a valid schema
        return report

    check_duplicates(df, report)
    check_timezone_pinned(df, report)
    check_missing_values(df, report)
    check_negative_values(df, report)
    check_zero_values(df, report)
    check_coverage(df, report, min_coverage_pct, exclude_aggregates)
    check_target_leakage_risk(df, report)

    return report


def load_raw(raw_dir: Path) -> pd.DataFrame:
    """Load every raw Parquet partition into one frame."""
    parts = sorted(raw_dir.glob("year=*/data.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"No raw data in {raw_dir}. Run scripts/fetch_raw_data.py first."
        )
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
