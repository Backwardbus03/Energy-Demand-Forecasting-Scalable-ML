"""Raw EIA ingestion: API -> normalised rows -> partitioned Parquet.

Design rules (Phase 1):
  - Raw data is append-only and never edited in place.
  - Nothing is invented: a missing observation stays missing (NaN), never 0.
  - No feature engineering here. Lags/rollings belong to Phase 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data.eia_client import EIAApiError, EIAClient, EIATransientError
from src.utils.config import IngestionConfig
from src.utils.logging_utils import append_jsonl, utc_now_iso

logger = logging.getLogger("eia.fetch")

# Raw schema. Order is fixed so every Parquet part file is consistent.
RAW_COLUMNS: list[str] = [
    "period",
    "respondent",
    "respondent_name",
    "type",
    "type_name",
    "value",
    "value_units",
    "timezone",
    "_ingested_at",
    "_source_route",
]

# Natural key: one observation per region per day per series.
KEY_COLUMNS: list[str] = ["period", "respondent", "type"]


@dataclass
class IngestionResult:
    rows_written: int
    respondents_ok: list[str]
    respondents_failed: list[str]
    api_calls: int
    respondents_skipped: list[str] = field(default_factory=list)
    respondents_no_data: list[str] = field(default_factory=list)

    @property
    def had_network_failures(self) -> bool:
        """True if any region failed for reasons other than genuinely having no data.

        Phase 2 must not mistake an outage for missing coverage, so this is
        surfaced separately and drives a non-zero exit code.
        """
        return bool(self.respondents_failed)

    def summary(self) -> str:
        return (
            f"rows={self.rows_written:,} "
            f"ok={len(self.respondents_ok)} "
            f"skipped={len(self.respondents_skipped)} "
            f"no_data={len(self.respondents_no_data)} "
            f"failed={len(self.respondents_failed)} "
            f"api_calls={self.api_calls}"
        )


def normalise_rows(rows: Iterable[dict[str, Any]], route: str) -> pd.DataFrame:
    """Convert raw API dicts into the fixed raw schema.

    ``value`` arrives as a string and may be null; it is coerced to float with
    ``errors="coerce"`` so unparseable entries become NaN rather than crashing
    or silently becoming zero.
    """
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    out = pd.DataFrame(
        {
            "period": pd.to_datetime(df.get("period"), errors="coerce"),
            "respondent": df.get("respondent").astype("string"),
            "respondent_name": df.get("respondent-name").astype("string"),
            "type": df.get("type").astype("string"),
            "type_name": df.get("type-name").astype("string"),
            "value": pd.to_numeric(df.get("value"), errors="coerce"),
            "value_units": df.get("value-units").astype("string"),
            "timezone": df.get("timezone").astype("string"),
        }
    )
    out["_ingested_at"] = utc_now_iso()
    out["_source_route"] = route
    return out[RAW_COLUMNS]


def write_partitioned(df: pd.DataFrame, raw_dir: Path) -> int:
    """Write rows to ``data/raw/eia/year=YYYY/`` as Parquet.

    Existing partitions are merged and de-duplicated on the natural key,
    keeping the most recently ingested copy. This is what makes re-running
    the pipeline safe and makes incremental updates possible: EIA revises
    recent days, and a revision should overwrite rather than duplicate.
    """
    if df.empty:
        return 0

    written = 0
    df = df.copy()
    df["_year"] = df["period"].dt.year

    for year, chunk in df.groupby("_year", sort=True):
        part_dir = raw_dir / f"year={int(year)}"
        part_dir.mkdir(parents=True, exist_ok=True)
        part_file = part_dir / "data.parquet"

        chunk = chunk.drop(columns="_year")

        if part_file.exists():
            existing = pd.read_parquet(part_file)
            chunk = pd.concat([existing, chunk], ignore_index=True)

        before = len(chunk)
        chunk = (
            chunk.sort_values("_ingested_at")
            .drop_duplicates(subset=KEY_COLUMNS, keep="last")
            .sort_values(KEY_COLUMNS)
            .reset_index(drop=True)
        )
        if before != len(chunk):
            logger.debug("year=%s deduped %d -> %d", year, before, len(chunk))

        chunk.to_parquet(part_file, index=False, compression="snappy")
        written += len(chunk)

    return written


def discover_respondents(client: EIAClient) -> list[str]:
    """List every balancing authority the endpoint exposes."""
    facets = client.list_facet("respondent")
    ids = sorted(f["id"] for f in facets)
    logger.info("discovered %d respondents", len(ids))
    return ids


def stored_respondents(raw_dir: Path) -> set[str]:
    """Return respondents already present in the raw store.

    Used to resume an interrupted backfill without re-reading every cached
    page for regions that are already complete.
    """
    parts = sorted(raw_dir.glob("year=*/data.parquet"))
    if not parts:
        return set()
    seen: set[str] = set()
    for part in parts:
        seen.update(pd.read_parquet(part, columns=["respondent"])["respondent"].unique())
    return {str(r) for r in seen}


def resolve_incremental_start(raw_dir: Path, lookback_days: int) -> str | None:
    """Return the start date for an incremental run, or None if no data yet.

    Re-fetches a trailing window because EIA revises recently published days.
    """
    parts = sorted(raw_dir.glob("year=*/data.parquet"))
    if not parts:
        return None

    max_period = max(
        pd.read_parquet(p, columns=["period"])["period"].max() for p in parts
    )
    if pd.isna(max_period):
        return None

    start = (max_period - timedelta(days=lookback_days)).date()
    return start.isoformat()


def ingest(
    config: IngestionConfig,
    use_cache: bool = True,
    limit_respondents: int | None = None,
    resume: bool = True,
) -> IngestionResult:
    """Run the full ingestion across all configured respondents."""
    config.paths.ensure()
    client = EIAClient(config, use_cache=use_cache)
    failure_log = config.paths.log_dir / "failures.jsonl"

    respondents = config.query.respondents or discover_respondents(client)
    if limit_respondents:
        respondents = respondents[:limit_respondents]

    logger.info(
        "ingesting %d respondents | %s -> %s | types=%s | timezone=%s",
        len(respondents),
        config.query.start,
        config.query.end,
        ",".join(config.query.types),
        config.query.timezone,
    )

    already = stored_respondents(config.paths.raw_dir) if resume else set()
    if already:
        logger.info("resume: %d respondents already stored, skipping", len(already))

    total_rows = 0
    ok: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    no_data: list[str] = []

    for i, respondent in enumerate(respondents, start=1):
        if respondent in already:
            skipped.append(respondent)
            logger.info("[%d/%d] %s: already stored, skipping", i, len(respondents), respondent)
            continue
        try:
            frames = [
                normalise_rows(page, config.api.route)
                for page in client.iter_rows(respondent)
            ]
            if not frames:
                logger.warning("[%d/%d] %s: no data", i, len(respondents), respondent)
                # No data is a real finding, not a failure - record and move on.
                append_jsonl(
                    failure_log,
                    {
                        "ts": utc_now_iso(),
                        "respondent": respondent,
                        "reason": "no_data",
                        "category": "no_data",
                    },
                )
                no_data.append(respondent)
                continue

            df = pd.concat(frames, ignore_index=True)
            n = write_partitioned(df, config.paths.raw_dir)
            total_rows += len(df)
            ok.append(respondent)
            logger.info(
                "[%d/%d] %s: %d rows", i, len(respondents), respondent, len(df)
            )

        except EIAApiError:
            # Auth/query errors will affect every respondent - fail fast.
            raise
        except Exception as exc:  # noqa: BLE001 - one bad region must not kill the run
            category = "network" if isinstance(exc, EIATransientError) else "error"
            logger.error("[%d/%d] %s failed (%s): %s", i, len(respondents), respondent, category, exc)
            append_jsonl(
                failure_log,
                {
                    "ts": utc_now_iso(),
                    "respondent": respondent,
                    "reason": str(exc),
                    "category": category,
                },
            )
            failed.append(respondent)

    result = IngestionResult(
        rows_written=total_rows,
        respondents_ok=ok,
        respondents_failed=failed,
        api_calls=client.calls_made,
        respondents_skipped=skipped,
        respondents_no_data=no_data,
    )
    logger.info("ingestion complete: %s", result.summary())
    return result
