"""Persistent bar cache — build once, reuse across all runs.

Cached bars live at data/<name>/<name>.parquet. The name encodes the
bar source and configuration so different bar types don't collide.

Usage:
    from src.data.bar_cache import BarCache

    # Check / load / save
    df = BarCache.load("l1_5s")          # returns pl.DataFrame or None
    BarCache.save("l1_5s", df)           # writes to data/l1_5s/l1_5s.parquet
    path = BarCache.path("l1_5s")        # data/l1_5s/l1_5s.parquet

    # Enriched bars (with signal columns)
    df = BarCache.load("enriched_5s_atr_spread_vwap_session")
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

CACHE_ROOT = Path("data")


def _cache_path(name: str) -> Path:
    return CACHE_ROOT / name / f"{name}.parquet"


class BarCache:
    """Simple file-based bar cache."""

    @staticmethod
    def path(name: str) -> Path:
        return _cache_path(name)

    @staticmethod
    def exists(name: str) -> bool:
        return _cache_path(name).is_file()

    @staticmethod
    def load(name: str) -> pl.DataFrame | None:
        """Load cached bars. Returns None if cache miss."""
        p = _cache_path(name)
        if not p.is_file():
            return None
        return pl.read_parquet(p)

    @staticmethod
    def save(name: str, df: pl.DataFrame) -> Path:
        """Save bars to cache. Returns the path written."""
        p = _cache_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(p)
        size_mb = p.stat().st_size / 1e6
        print(f"  [BarCache] Saved {name}: {len(df):,} rows, {size_mb:.1f} MB -> {p}")
        return p

    @staticmethod
    def bar_name(
        source: str = "parquet",
        freq: str | None = None,
        l1_seconds: int | None = None,
        dollar_threshold: float | None = None,
    ) -> str:
        """Build a canonical cache name from bar config.

        Examples:
            bar_name(freq="5s")               -> "bars_5s"
            bar_name(freq="1m")               -> "bars_1m"
            bar_name(source="l1", l1_seconds=5) -> "bars_l1_5s"
            bar_name(dollar_threshold=50000)  -> "bars_dollar_50k"
        """
        if dollar_threshold is not None:
            if dollar_threshold >= 1000:
                return f"bars_dollar_{int(dollar_threshold // 1000)}k"
            return f"bars_dollar_{int(dollar_threshold)}"
        if source == "l1" and l1_seconds is not None:
            return f"bars_l1_{l1_seconds}s"
        if freq:
            return f"bars_{freq}"
        return "bars_1s"

    @staticmethod
    def enriched_name(freq: str, signal_names: list[str]) -> str:
        """Build cache name for enriched bars (with signal columns).

        Examples:
            enriched_name("5s", ["atr", "spread"]) -> "enriched_5s_atr_spread"
        """
        sig_tag = "_".join(sorted(signal_names)) if signal_names else "none"
        return f"enriched_{freq}_{sig_tag}"
