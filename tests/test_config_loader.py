"""Tests for config loader — new build_signal_engine / build_filter_engine."""

from config.loader import build_filter_engine, build_signal_engine


class TestBuildSignalEngine:
    def test_no_signals(self):
        engine = build_signal_engine({})
        assert engine is None

    def test_empty_signals(self):
        engine = build_signal_engine({"signals": []})
        assert engine is None

    def test_with_signals(self):
        engine = build_signal_engine({"signals": ["spread", "atr"]})
        assert engine is not None
        assert engine.signal_names == ["spread", "atr"]

    def test_with_signal_configs(self):
        cfg = {
            "signals": ["atr"],
            "signal_configs": {"atr": {"lookback_bars": 10}},
        }
        engine = build_signal_engine(cfg)
        assert engine is not None


class TestBuildFilterEngine:
    def test_no_filters(self):
        fe = build_filter_engine({})
        assert fe.is_empty

    def test_empty_filters(self):
        fe = build_filter_engine({"filters": []})
        assert fe.is_empty

    def test_with_filters(self):
        fe = build_filter_engine({
            "filters": [{"signal": "spread", "expr": "< 2.0", "seq": 1}]
        })
        assert not fe.is_empty
        assert len(fe.rules) == 1

    def test_orb_style_config(self):
        """Test a config matching the actual orb.yaml."""
        cfg = {
            "signals": ["atr", "vwap_session", "relative_volume", "spread"],
            "filters": [{"signal": "spread", "expr": "< 2.0", "seq": 1}],
        }
        engine = build_signal_engine(cfg)
        fe = build_filter_engine(cfg)
        assert engine is not None
        assert len(engine.signal_names) == 4
        assert len(fe.rules) == 1

    def test_multi_seq_filters(self):
        cfg = {
            "filters": [
                {"signal": "spread", "expr": "< 2.0", "seq": 1},
                {"signal": "spread", "expr": "< 3.0", "seq": 2},
            ]
        }
        fe = build_filter_engine(cfg)
        assert len(fe.rules) == 2
        assert fe.sequences == [1, 2]
