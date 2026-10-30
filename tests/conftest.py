from __future__ import annotations

import json
import os
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def dummy_eia_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure EIA_API_KEY is present in the environment for all tests."""
    monkeypatch.setenv("EIA_API_KEY", "test_dummy_key_12345")


@pytest.fixture
def synthetic_raw_store(tmp_path: Path) -> Path:
    """Creates a temporary partitioned raw store matching EIA Parquet layout."""
    raw_dir = tmp_path / "raw" / "eia"
    raw_dir.mkdir(parents=True, exist_ok=True)

    respondents = ["CISO", "DUK", "AVA"]
    types = ["D", "DF", "NG", "TI"]

    # Generate 140 days of daily data across 2 years
    date_range = pd.date_range("2023-10-01", "2024-03-01", freq="D")
    records: list[dict[str, Any]] = []

    np.random.seed(42)
    for dt in date_range:
        for resp in respondents:
            base_val = 50000.0 if resp == "CISO" else (25000.0 if resp == "DUK" else 5000.0)
            for t in types:
                noise = np.random.normal(0, 500)
                records.append({
                    "period": dt,
                    "respondent": resp,
                    "type": t,
                    "value": max(100.0, base_val + noise),
                })

    all_df = pd.DataFrame(records)

    # Partition by year
    for year, group in all_df.groupby(all_df["period"].dt.year):
        year_dir = raw_dir / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        group.to_parquet(year_dir / "data.parquet", index=False)

    # Create matching validation report
    logs_dir = tmp_path / "raw" / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_data = {
        "passed": True,
        "modeling_respondents": respondents,
        "summary": {
            "total_respondents_configured": len(respondents),
            "approved_count": len(respondents),
            "rejected_count": 0,
        },
        "respondents": {
            resp: {
                "passed": True,
                "coverage_pct": 100.0,
                "tier": "Large" if resp == "CISO" else ("Medium" if resp == "DUK" else "Small"),
            }
            for resp in respondents
        },
    }
    (logs_dir / "validation_report.json").write_text(json.dumps(report_data, indent=2))

    return raw_dir


@pytest.fixture
def run_cli_main() -> Callable[[Callable[[], int], list[str]], int]:
    """Helper to run a CLI script's main() with simulated sys.argv."""
    def _run(main_func: Callable[[], int], argv: list[str]) -> int:
        old_argv = sys.argv
        try:
            sys.argv = ["script"] + argv
            return main_func()
        finally:
            sys.argv = old_argv

    return _run
