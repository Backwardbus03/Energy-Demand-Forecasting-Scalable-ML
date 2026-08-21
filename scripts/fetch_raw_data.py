"""CLI entry point for Phase 1 raw EIA ingestion.

Examples
--------
Smoke test (3 regions, ~1 month)::

    python scripts/fetch_raw_data.py --limit-respondents 3 --start 2025-01-01 --end 2025-01-31

Full historical backfill::

    python scripts/fetch_raw_data.py

Incremental update (only new days since last run)::

    python scripts/fetch_raw_data.py --incremental
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

# Make ``src`` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetch_eia import ingest, resolve_incremental_start  # noqa: E402
from src.utils.config import ConfigError, load_config  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch raw EIA electricity demand data.")
    p.add_argument("--config", type=Path, default=None, help="Path to ingestion.yaml")
    p.add_argument("--start", type=str, default=None, help="Override start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None, help="Override end date (YYYY-MM-DD)")
    p.add_argument(
        "--incremental",
        action="store_true",
        help="Fetch only days after the latest already stored (minus lookback).",
    )
    p.add_argument(
        "--limit-respondents",
        type=int,
        default=None,
        help="Only fetch the first N regions. Use for smoke tests.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-fetch regions already present in the raw store (default: skip them).",
    )
    p.add_argument("--no-cache", action="store_true", help="Bypass the response cache.")
    p.add_argument("--verbose", action="store_true", help="Debug logging.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
        return 2

    logger = setup_logging(config.paths.log_dir, level=10 if args.verbose else 20)

    # Resolve the effective date window.
    start, end = args.start, args.end
    if args.incremental:
        inferred = resolve_incremental_start(config.paths.raw_dir, config.lookback_days)
        if inferred:
            start = inferred
            logger.info("incremental mode: resuming from %s", start)
        else:
            logger.info("incremental mode: no existing data, full backfill")

    if start or end:
        query = dataclasses.replace(
            config.query,
            start=start or config.query.start,
            end=end or config.query.end,
        )
        config = dataclasses.replace(config, query=query)

    try:
        result = ingest(
            config,
            use_cache=not args.no_cache,
            limit_respondents=args.limit_respondents,
            resume=not args.no_resume,
        )
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingestion aborted: %s", exc)
        return 1

    print("\n" + "=" * 62)
    print("  INGESTION SUMMARY")
    print("=" * 62)
    print(f"  Rows written      : {result.rows_written:,}")
    print(f"  Regions fetched   : {len(result.respondents_ok)}")
    print(f"  Regions skipped   : {len(result.respondents_skipped)} (already stored)")
    print(f"  Regions no data   : {len(result.respondents_no_data)}")
    print(f"  Regions failed    : {len(result.respondents_failed)}")
    print(f"  API calls made    : {result.api_calls}")
    print(f"  Output            : {config.paths.raw_dir}")

    if result.respondents_no_data:
        print(f"\n  No data (genuine, not an error):")
        print(f"    {', '.join(result.respondents_no_data)}")

    if result.respondents_failed:
        print(f"\n  !! FAILED - retry before using this data:")
        print(f"    {', '.join(result.respondents_failed)}")
        print(f"    Details: {config.paths.log_dir / 'failures.jsonl'}")
        print(f"    Re-run the same command; completed regions are skipped.")
    print("=" * 62)

    # Fail loudly on network errors: an incomplete backfill must not be
    # mistaken for a complete one by the Phase 2 validation step.
    if result.had_network_failures:
        return 1
    return 0 if (result.rows_written > 0 or result.respondents_skipped) else 1


if __name__ == "__main__":
    raise SystemExit(main())
