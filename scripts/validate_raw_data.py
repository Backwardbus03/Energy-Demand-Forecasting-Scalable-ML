"""CLI for Phase 2 raw data validation.

Exit codes are meaningful so Airflow (Phase 8) can gate on them:
    0 = passed (warnings allowed)
    1 = structural errors - do not proceed to feature engineering
    2 = could not run (no data)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.validate import Severity, load_raw, validate  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate raw EIA data.")
    p.add_argument("--min-coverage", type=float, default=95.0,
                   help="Minimum %% coverage for the modeling set (default: 95).")
    p.add_argument("--include-aggregates", action="store_true",
                   help="Keep regional roll-ups (US48, MIDA, ...) in the modeling set.")
    p.add_argument("--report", type=Path, default=None,
                   help="Where to write the JSON report.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    logger = setup_logging(config.paths.log_dir, run_name="validate")

    try:
        df = load_raw(config.paths.raw_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    logger.info("validating %s rows", f"{len(df):,}")
    report = validate(
        df,
        min_coverage_pct=args.min_coverage,
        exclude_aggregates=not args.include_aggregates,
    )

    out = args.report or (config.paths.log_dir / "validation_report.json")
    report.save(out)

    print("\n" + "=" * 72)
    print("  DATA VALIDATION REPORT")
    print("=" * 72)
    print(f"  Rows            : {report.row_count:,}")
    print(f"  Errors          : {len(report.errors)}")
    print(f"  Warnings        : {len(report.warnings)}")
    print(f"  Modeling set    : {len(report.modeling_respondents)} respondents")
    print("-" * 72)

    for sev in (Severity.ERROR, Severity.WARNING, Severity.INFO):
        for f in [x for x in report.findings if x.severity is sev]:
            print(f"  {f}")

    print("-" * 72)
    print(f"  Report          : {out}")
    print(f"  Coverage matrix : {out.with_name('coverage_matrix.csv')}")
    print(f"  RESULT          : {'PASSED' if report.passed else 'FAILED'}")
    print("=" * 72)

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
