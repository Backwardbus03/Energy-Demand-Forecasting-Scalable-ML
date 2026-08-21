"""Thin, well-behaved client for the EIA API v2.

Responsibilities:
  - build correctly-encoded requests (EIA uses ``facets[x][]=`` array syntax)
  - retry transient failures with exponential backoff
  - paginate using the API's own ``total`` as ground truth
  - cache raw JSON responses so a parsing bug never costs a re-fetch

Deliberately does NOT reshape data - parsing/normalisation lives in
``fetch_eia.py`` so the network layer stays independently testable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.config import IngestionConfig

logger = logging.getLogger("eia.client")


def _redact(message: object, api_key: str) -> str:
    """Strip the API key out of text destined for logs."""
    text = str(message)
    if api_key:
        text = text.replace(api_key, "<REDACTED>")
    return text


class EIAApiError(RuntimeError):
    """Non-retryable API error (bad key, malformed query)."""


class EIATransientError(RuntimeError):
    """Retryable API error (429, 5xx, network blip)."""


class EIAClient:
    """Paginating, caching HTTP client for one EIA data route."""

    def __init__(self, config: IngestionConfig, use_cache: bool = True) -> None:
        self.config = config
        self.use_cache = use_cache
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "energy-forecasting-mlops/0.1"})
        self._calls_made = 0

    @property
    def calls_made(self) -> int:
        return self._calls_made

    # ---------------------------------------------------------------- caching

    def _cache_path(self, params: dict[str, Any]) -> Path:
        """Deterministic cache filename from the request parameters.

        The API key is excluded from the hash so rotating a key does not
        invalidate the entire cache.
        """
        safe = {k: v for k, v in params.items() if k != "api_key"}
        blob = json.dumps(safe, sort_keys=True, default=str)
        digest = hashlib.sha256(blob.encode()).hexdigest()[:20]
        return self.config.paths.cache_dir / f"{digest}.json"

    # ---------------------------------------------------------------- request

    @retry(
        retry=retry_if_exception_type(EIATransientError),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """Single GET with retry on transient failures."""
        try:
            resp = self.session.get(
                url, params=params, timeout=self.config.api.timeout_seconds
            )
        except requests.RequestException as exc:
            # `requests` embeds the full query string - including the API key -
            # in its exception text. Redact before it reaches logs or tracebacks.
            raise EIATransientError(f"network error: {_redact(exc, self.config.api_key)}") from exc

        self._calls_made += 1

        # 403/401 mean a bad key - retrying only burns quota.
        if resp.status_code in (401, 403):
            raise EIAApiError(
                f"HTTP {resp.status_code}: authentication failed. "
                "Check EIA_API_KEY in .env."
            )
        if resp.status_code == 429:
            raise EIATransientError("HTTP 429: rate limited")
        if resp.status_code >= 500:
            raise EIATransientError(f"HTTP {resp.status_code}: server error")
        if resp.status_code != 200:
            raise EIAApiError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        try:
            payload = resp.json()
        except ValueError as exc:
            raise EIATransientError(f"invalid JSON: {exc}") from exc

        # EIA returns errors inside a 200 body, so status code alone is not enough.
        if "response" not in payload:
            raise EIAApiError(f"unexpected payload shape: {str(payload)[:300]}")

        return payload

    def _get_cached(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET with an on-disk cache layer."""
        cache_file = self._cache_path(params)
        if self.use_cache and cache_file.exists():
            logger.debug("cache hit %s", cache_file.name)
            return json.loads(cache_file.read_text())

        payload = self._get(url, params)

        if self.use_cache:
            cache_file.write_text(json.dumps(payload))

        time.sleep(self.config.api.request_delay)
        return payload

    # ------------------------------------------------------------------ facet

    def list_facet(self, facet: str) -> list[dict[str, str]]:
        """List available values for a facet (e.g. all respondents)."""
        url = self.config.facet_url_template.format(facet=facet)
        payload = self._get_cached(url, {"api_key": self.config.api_key})
        return payload["response"]["facets"]

    # ------------------------------------------------------------- data pages

    def _build_params(self, offset: int, respondent: str | None) -> dict[str, Any]:
        cfg = self.config
        params: dict[str, Any] = {
            "api_key": cfg.api_key,
            "frequency": cfg.api.frequency,
            "data[0]": "value",
            "start": cfg.query.start,
            "end": cfg.query.end,
            # Pinning timezone removes the 5x duplication in this endpoint.
            "facets[timezone][]": cfg.query.timezone,
            "facets[type][]": cfg.query.types,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": cfg.api.page_size,
        }
        if respondent:
            params["facets[respondent][]"] = respondent
        return params

    def count_rows(self, respondent: str | None = None) -> int:
        """Ask the API how many rows the query matches, without fetching them."""
        params = self._build_params(offset=0, respondent=respondent)
        params["length"] = 1
        payload = self._get_cached(self.config.data_url, params)
        return int(payload["response"].get("total", 0))

    def iter_rows(self, respondent: str | None = None) -> Iterator[list[dict[str, Any]]]:
        """Yield pages of raw row dicts for one respondent (or all).

        Pagination is driven by the API's reported ``total`` rather than by
        "keep going until a short page", because EIA silently caps ``length``
        at 5000 - a short page is not a reliable end-of-data signal.
        """
        total = self.count_rows(respondent)
        label = respondent or "ALL"
        if total == 0:
            logger.warning("no rows for respondent=%s", label)
            return

        page_size = self.config.api.page_size
        fetched = 0
        offset = 0

        while offset < total:
            params = self._build_params(offset=offset, respondent=respondent)
            payload = self._get_cached(self.config.data_url, params)
            rows = payload["response"].get("data", [])

            if not rows:
                logger.warning(
                    "empty page at offset=%d for %s (expected %d total); stopping",
                    offset, label, total,
                )
                break

            fetched += len(rows)
            yield rows
            offset += len(rows)

        if fetched != total:
            # Not fatal - EIA can publish rows mid-run - but it must be visible.
            logger.warning(
                "row count mismatch for %s: fetched %d, API reported %d",
                label, fetched, total,
            )
