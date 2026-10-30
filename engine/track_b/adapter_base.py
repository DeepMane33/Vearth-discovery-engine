"""Base class for Track B dataset adapters.

Each adapter implements:
  fetch(target, start_date, end_date) -> raw data
  parse(raw) -> structured records
  quality_flag(records) -> flagged records
  cache_key(target, ...) -> cache filename
  save_cache(data) / load_cache(key) -> disk caching

All adapters must respect rate limits and cache all downloads on disk.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

from config import DATA_DIR


class DatasetAdapter(ABC):
    """Base adapter for a planetary science dataset."""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir or DATA_DIR / "track_b_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_request_time = 0.0
        self.min_interval = 1.0  # seconds between requests (rate limit)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    # ------------------------------------------------------------------
    # Disk caching
    # ------------------------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() else "_" for c in key)
        return self.cache_dir / f"{safe}.json"

    def load_cache(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def save_cache(self, key: str, data: dict) -> None:
        p = self._cache_path(key)
        try:
            p.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                         encoding="utf-8")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def fetch(self, target: str, start_date: str, end_date: str) -> dict:
        """Download raw data for target between dates. Returns dict."""
        raise NotImplementedError

    @abstractmethod
    def parse(self, raw: dict) -> list[dict]:
        """Parse raw API response into list of structured records."""
        raise NotImplementedError

    @abstractmethod
    def quality_flag(self, records: list[dict]) -> list[dict]:
        """Add quality flags to each record. Returns flagged records."""
        raise NotImplementedError

    @abstractmethod
    def get_cache_key(self, target: str, start_date: str,
                      end_date: str) -> str:
        """Return a unique cache key for this query."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience: fetch + parse + quality flag + cache
    # ------------------------------------------------------------------
    def get_data(self, target: str, start_date: str,
                 end_date: str, use_cache: bool = True) -> dict:
        """Full pipeline: check cache, fetch, parse, quality-flag, cache.

        Returns dict with keys: records, quality_summary, source, target,
        start_date, end_date, n_records, cached (bool).
        """
        key = self.get_cache_key(target, start_date, end_date)

        if use_cache:
            cached = self.load_cache(key)
            if cached is not None:
                cached["cached"] = True
                return cached

        self._rate_limit()
        raw = self.fetch(target, start_date, end_date)
        records = self.parse(raw)
        records = self.quality_flag(records)

        result = {
            "records": records,
            "quality_summary": self._quality_summary(records),
            "source": self.source_name(),
            "target": target,
            "start_date": start_date,
            "end_date": end_date,
            "n_records": len(records),
            "cached": False,
        }

        if use_cache:
            self.save_cache(key, result)

        return result

    @abstractmethod
    def source_name(self) -> str:
        """Human-readable source name."""
        raise NotImplementedError

    def _quality_summary(self, records: list[dict]) -> dict:
        """Summary of quality flags across records."""
        total = len(records)
        flagged = sum(1 for r in records if r.get("quality_flag", "ok") != "ok")
        return {
            "total_records": total,
            "flagged_records": flagged,
            "clean_records": total - flagged,
        }
