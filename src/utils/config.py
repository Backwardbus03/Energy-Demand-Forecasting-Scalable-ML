"""Configuration loading for the ingestion pipeline.

Keeps two concerns separate:
  - Tunable settings live in ``configs/*.yaml`` (committed, reviewable).
  - Secrets live in ``.env`` (never committed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
    """Raised when configuration is missing or malformed."""


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    route: str
    frequency: str
    page_size: int
    timeout_seconds: int
    max_retries: int
    request_delay: float


@dataclass(frozen=True)
class QueryConfig:
    start: str
    end: str
    timezone: str
    types: list[str]
    respondents: list[str] | None = None


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    cache_dir: Path
    log_dir: Path

    def ensure(self) -> None:
        for p in (self.raw_dir, self.cache_dir, self.log_dir):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class IngestionConfig:
    api: ApiConfig
    query: QueryConfig
    paths: Paths
    lookback_days: int
    api_key: str = field(repr=False)  # keep the secret out of logs/tracebacks

    @property
    def data_url(self) -> str:
        return f"{self.api.base_url}/{self.api.route}/data/"

    @property
    def facet_url_template(self) -> str:
        return f"{self.api.base_url}/{self.api.route}/facet/{{facet}}"


def load_api_key(env_var: str = "EIA_API_KEY") -> str:
    """Read the EIA API key from the environment or ``.env``."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv(env_var, "").strip()
    if not key:
        raise ConfigError(
            f"{env_var} is not set. Copy .env.example to .env and add your key.\n"
            "Register free at https://www.eia.gov/opendata/register.php"
        )
    return key


def load_config(path: str | Path | None = None) -> IngestionConfig:
    """Load and validate ``configs/ingestion.yaml``."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "ingestion.yaml"
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")

    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text())

    q = raw["query"]
    # `end: null` means "up to today" - resolved here so the rest of the
    # pipeline always sees a concrete date.
    end = q.get("end") or date.today().isoformat()

    if not q.get("timezone"):
        raise ConfigError(
            "query.timezone must be pinned. The EIA timezone facet duplicates "
            "every observation 5x; leaving it unset inflates the dataset."
        )

    paths = Paths(
        raw_dir=PROJECT_ROOT / raw["paths"]["raw_dir"],
        cache_dir=PROJECT_ROOT / raw["paths"]["cache_dir"],
        log_dir=PROJECT_ROOT / raw["paths"]["log_dir"],
    )

    return IngestionConfig(
        api=ApiConfig(**raw["api"]),
        query=QueryConfig(
            start=q["start"],
            end=end,
            timezone=q["timezone"],
            types=list(q["types"]),
            respondents=q.get("respondents"),
        ),
        paths=paths,
        lookback_days=int(raw["incremental"]["lookback_days"]),
        api_key=load_api_key(),
    )
