"""Tests for persistent bar cache."""

import polars as pl
import pytest
from unittest.mock import patch
from pathlib import Path

from src.data.bar_cache import BarCache


class TestBarCacheNaming:
    def test_bar_name_freq(self):
        assert BarCache.bar_name(freq="5s") == "bars_5s"
        assert BarCache.bar_name(freq="1m") == "bars_1m"

    def test_bar_name_l1(self):
        assert BarCache.bar_name(source="l1", l1_seconds=5) == "bars_l1_5s"

    def test_bar_name_dollar(self):
        assert BarCache.bar_name(dollar_threshold=50000) == "bars_dollar_50k"
        assert BarCache.bar_name(dollar_threshold=500) == "bars_dollar_500"

    def test_bar_name_default(self):
        assert BarCache.bar_name() == "bars_1s"

    def test_enriched_name(self):
        name = BarCache.enriched_name("5s", ["atr", "spread", "vwap_session"])
        assert name == "enriched_5s_atr_spread_vwap_session"

    def test_enriched_name_sorted(self):
        # Signal names should be sorted for deterministic cache keys
        n1 = BarCache.enriched_name("1m", ["spread", "atr"])
        n2 = BarCache.enriched_name("1m", ["atr", "spread"])
        assert n1 == n2

    def test_enriched_name_no_signals(self):
        assert BarCache.enriched_name("5s", []) == "enriched_5s_none"


class TestBarCacheIO:
    def test_save_and_load(self, tmp_path):
        df = pl.DataFrame({"timestamp": [1, 2, 3], "close": [100.0, 101.0, 102.0]})
        with patch("src.data.bar_cache.CACHE_ROOT", tmp_path):
            path = BarCache.save("test_bars", df)
            assert path.exists()
            assert path == tmp_path / "test_bars" / "test_bars.parquet"

            loaded = BarCache.load("test_bars")
            assert loaded is not None
            assert len(loaded) == 3
            assert loaded["close"].to_list() == [100.0, 101.0, 102.0]

    def test_load_miss(self, tmp_path):
        with patch("src.data.bar_cache.CACHE_ROOT", tmp_path):
            assert BarCache.load("nonexistent") is None

    def test_exists(self, tmp_path):
        with patch("src.data.bar_cache.CACHE_ROOT", tmp_path):
            assert not BarCache.exists("test_bars")
            df = pl.DataFrame({"x": [1]})
            BarCache.save("test_bars", df)
            assert BarCache.exists("test_bars")
