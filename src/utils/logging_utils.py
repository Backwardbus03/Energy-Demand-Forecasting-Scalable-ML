"""Logging setup shared across the pipeline.

Two sinks:
  - console, for the human watching the run
  - a rotating-by-run file under ``data/raw/_logs``, for post-hoc debugging
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(
    log_dir: Path | None = None,
    level: int = logging.INFO,
    run_name: str = "ingest",
) -> logging.Logger:
    """Configure root logging and return the pipeline logger."""
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fh = logging.FileHandler(log_dir / f"{run_name}_{stamp}.log")
        fh.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(fh)

    # requests/urllib3 are chatty at INFO and drown out our own messages.
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logging.getLogger("eia")


def utc_now_iso() -> str:
    """Timestamp for provenance columns."""
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record to a .jsonl file.

    Used for the failure log: append-only so a crash mid-run never truncates
    what earlier attempts recorded.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
