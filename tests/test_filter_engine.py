"""Tests for FilterEngine — declarative list-driven filter evaluation."""

import pytest

from src.filters.filter_engine import FilterEngine, FilterResult, FilterRule, parse_rules
from src.signals.base import SignalResult
from src.signals.signal_bundle import SignalBundle


def _bundle(**kwargs: float) -> SignalBundle:
    """Helper: build a SignalBundle from name=value pairs."""
    results = {}
    for name, val in kwargs.items():
        if isinstance(val, dict):
            # val is {"value": ..., "metadata": {...}, ...}
            results[name] = SignalResult(
                value=val.get("value", 0.0),
                passes=val.get("passes", val.get("value", 0.0) > 0),
                direction=val.get("direction", "none"),
                metadata=val.get("metadata", {}),
            )
        else:
            results[name] = SignalResult(value=val, passes=val > 0, direction="none")
    return SignalBundle(results=results, bar_count=1)


def _rule(signal: str, expr: str, seq: int = 1) -> dict:
    """Helper: build a filter rule dict."""
    return {"signal": signal, "expr": expr, "seq": seq}


# ── parse_rules tests ────────────────────────────────────────────────


class TestParseRules:
    def test_less_than(self):
        rules = parse_rules([_rule("spread", "< 2.0")])
        assert len(rules) == 1
        assert rules[0] == FilterRule("spread", "<", 2.0)

    def test_greater_than(self):
        rules = parse_rules([_rule("volume", "> 1.5")])
        assert rules[0] == FilterRule("volume", ">", 1.5)

    def test_less_than_or_equal(self):
        rules = parse_rules([_rule("atr", "<= 5.0")])
        assert rules[0] == FilterRule("atr", "<=", 5.0)

    def test_greater_than_or_equal(self):
        rules = parse_rules([_rule("rsi", ">= 30")])
        assert rules[0] == FilterRule("rsi", ">=", 30.0)

    def test_equals(self):
        rules = parse_rules([_rule("mode", "== 1.0")])
        assert rules[0] == FilterRule("mode", "==", 1.0)

    def test_not_equals(self):
        rules = parse_rules([_rule("atr", "!= 0")])
        assert rules[0] == FilterRule("atr", "!=", 0.0)

    def test_boolean_true_becomes_passes(self):
        rules = parse_rules([{"signal": "ema_crossover", "expr": True}])
        assert rules[0] == FilterRule("ema_crossover", "passes", 0.0)

    def test_boolean_false_skipped(self):
        rules = parse_rules([{"signal": "disabled", "expr": False}])
        assert len(rules) == 0

    def test_bare_number_becomes_equals(self):
        rules = parse_rules([_rule("threshold", "42.5")])
        assert rules[0] == FilterRule("threshold", "==", 42.5)

    def test_numeric_value_becomes_equals(self):
        rules = parse_rules([{"signal": "count", "expr": 3}])
        assert rules[0] == FilterRule("count", "==", 3.0)

    def test_multiple_rules(self):
        rules = parse_rules([
            _rule("spread", "< 2.0"),
            _rule("volume", "> 1.5"),
            {"signal": "ema", "expr": True},
        ])
        assert len(rules) == 3

    def test_invalid_expression_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_rules([_rule("bad", "foobar")])

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            parse_rules([{"signal": "bad", "expr": [1, 2, 3]}])

    def test_empty_list(self):
        rules = parse_rules([])
        assert len(rules) == 0

    def test_seq_default_is_1(self):
        rules = parse_rules([{"signal": "spread", "expr": "< 2.0"}])
        assert rules[0].seq == 1

    def test_seq_explicit(self):
        rules = parse_rules([_rule("spread", "< 2.0", seq=2)])
        assert rules[0].seq == 2

    def test_multi_seq_rules(self):
        rules = parse_rules([
            _rule("spread", "< 2.0", seq=1),
            _rule("spread", "< 3.0", seq=2),
            _rule("atr", "> 1.0", seq=1),
        ])
        assert len(rules) == 3
        assert rules[0].seq == 1
        assert rules[1].seq == 2
        assert rules[2].seq == 1

    def test_bar_field_parsed(self):
        rules = parse_rules([{"signal": "spread", "expr": "< 2.0", "bar": "5m"}])
        assert rules[0].bar == "5m"

    def test_bar_field_default_none(self):
        rules = parse_rules([_rule("spread", "< 2.0")])
        assert rules[0].bar is None

    def test_bar_field_with_seq(self):
        rules = parse_rules([
            {"signal": "spread", "expr": "< 2.0", "seq": 1, "bar": "5m"},
            {"signal": "spread", "expr": "< 2.0", "seq": 2, "bar": "15m"},
        ])
        assert rules[0].bar == "5m"
        assert rules[1].bar == "15m"

    def test_non_dict_item_raises(self):
        with pytest.raises(ValueError, match="must be dicts"):
            parse_rules(["bad"])


# ── FilterEngine tests ───────────────────────────────────────────────


class TestFilterEngine:
    def test_empty_engine_passes(self):
        fe = FilterEngine()
        result = fe.evaluate(_bundle(spread=5.0))
        assert result.passes is True
        assert result.block_reasons == []

    def test_from_list_none(self):
        fe = FilterEngine.from_list(None)
        assert fe.is_empty

    def test_from_list_empty(self):
        fe = FilterEngine.from_list([])
        assert fe.is_empty

    def test_less_than_passes(self):
        fe = FilterEngine.from_list([_rule("spread", "< 2.0")])
        result = fe.evaluate(_bundle(spread=1.5))
        assert result.passes is True

    def test_less_than_blocks(self):
        fe = FilterEngine.from_list([_rule("spread", "< 2.0")])
        result = fe.evaluate(_bundle(spread=2.5))
        assert result.passes is False
        assert len(result.block_reasons) == 1
        assert "spread" in result.block_reasons[0]

    def test_greater_than_passes(self):
        fe = FilterEngine.from_list([_rule("relative_volume", "> 1.5")])
        result = fe.evaluate(_bundle(relative_volume=2.0))
        assert result.passes is True

    def test_greater_than_blocks(self):
        fe = FilterEngine.from_list([_rule("relative_volume", "> 1.5")])
        result = fe.evaluate(_bundle(relative_volume=1.0))
        assert result.passes is False

    def test_passes_op_with_passing_signal(self):
        fe = FilterEngine.from_list([{"signal": "ema_crossover", "expr": True}])
        result = fe.evaluate(_bundle(ema_crossover=1.0))
        assert result.passes is True

    def test_passes_op_with_failing_signal(self):
        fe = FilterEngine.from_list([{"signal": "ema_crossover", "expr": True}])
        result = fe.evaluate(_bundle(ema_crossover=-1.0))
        assert result.passes is False

    def test_missing_signal_skipped(self):
        fe = FilterEngine.from_list([_rule("vpin", "< 0.5")])
        result = fe.evaluate(_bundle(spread=1.0))
        assert result.passes is True

    def test_multiple_rules_all_pass(self):
        fe = FilterEngine.from_list([
            _rule("spread", "< 2.0"),
            _rule("relative_volume", "> 1.5"),
        ])
        result = fe.evaluate(_bundle(spread=1.0, relative_volume=2.0))
        assert result.passes is True

    def test_multiple_rules_one_blocks(self):
        fe = FilterEngine.from_list([
            _rule("spread", "< 2.0"),
            _rule("relative_volume", "> 1.5"),
        ])
        result = fe.evaluate(_bundle(spread=3.0, relative_volume=2.0))
        assert result.passes is False
        assert len(result.block_reasons) == 1

    def test_multiple_rules_all_block(self):
        fe = FilterEngine.from_list([
            _rule("spread", "< 2.0"),
            _rule("relative_volume", "> 1.5"),
        ])
        result = fe.evaluate(_bundle(spread=3.0, relative_volume=0.5))
        assert result.passes is False
        assert len(result.block_reasons) == 2

    def test_rules_property(self):
        fe = FilterEngine.from_list([
            _rule("spread", "< 2.0"),
            _rule("atr", "> 1.0"),
        ])
        assert len(fe.rules) == 2

    def test_signal_names_property(self):
        fe = FilterEngine.from_list([
            _rule("spread", "< 2.0"),
            _rule("atr", "> 1.0"),
        ])
        names = fe.signal_names
        assert "spread" in names
        assert "atr" in names

    def test_is_empty_property(self):
        assert FilterEngine().is_empty
        assert not FilterEngine.from_list([_rule("spread", "< 2.0")]).is_empty


# ── evaluate_seq tests ───────────────────────────────────────────────


class TestEvaluateSeq:
    def test_evaluate_seq_filters_by_seq(self):
        fe = FilterEngine.from_list([
            _rule("spread", "< 2.0", seq=1),
            _rule("atr", "> 5.0", seq=2),
        ])
        # seq=1 only checks spread
        r1 = fe.evaluate_seq(_bundle(spread=1.0, atr=1.0), seq=1)
        assert r1.passes is True

        # seq=2 only checks atr — atr=1.0 fails "> 5.0"
        r2 = fe.evaluate_seq(_bundle(spread=1.0, atr=1.0), seq=2)
        assert r2.passes is False

    def test_evaluate_seq_empty_seq_passes(self):
        fe = FilterEngine.from_list([_rule("spread", "< 2.0", seq=1)])
        # seq=2 has no rules — passes
        result = fe.evaluate_seq(_bundle(spread=5.0), seq=2)
        assert result.passes is True

    def test_evaluate_seq_same_signal_different_thresholds(self):
        fe = FilterEngine.from_list([
            _rule("spread", "< 2.0", seq=1),
            _rule("spread", "< 3.0", seq=2),
        ])
        # spread=2.5: fails seq=1 (not < 2.0), passes seq=2 (< 3.0)
        r1 = fe.evaluate_seq(_bundle(spread=2.5), seq=1)
        assert r1.passes is False

        r2 = fe.evaluate_seq(_bundle(spread=2.5), seq=2)
        assert r2.passes is True

    def test_sequences_property(self):
        fe = FilterEngine.from_list([
            _rule("spread", "< 2.0", seq=1),
            _rule("atr", "> 1.0", seq=2),
            _rule("volume", "> 100", seq=1),
        ])
        assert fe.sequences == [1, 2]

    def test_sequences_empty(self):
        fe = FilterEngine()
        assert fe.sequences == []

    def test_bar_freqs_property(self):
        fe = FilterEngine.from_list([
            {"signal": "spread", "expr": "< 2.0", "seq": 1, "bar": "5m"},
            {"signal": "atr", "expr": "> 1.0", "seq": 2, "bar": "15m"},
        ])
        assert fe.bar_freqs == {1: "5m", 2: "15m"}

    def test_bar_freqs_empty_when_no_bar(self):
        fe = FilterEngine.from_list([_rule("spread", "< 2.0")])
        assert fe.bar_freqs == {}

    def test_bar_freqs_partial(self):
        fe = FilterEngine.from_list([
            {"signal": "spread", "expr": "< 2.0", "seq": 1, "bar": "5m"},
            {"signal": "atr", "expr": "> 1.0", "seq": 2},
        ])
        assert fe.bar_freqs == {1: "5m"}


# ── metadata field access tests ─────────────────────────────────────


class TestFieldAccess:
    def test_field_reads_metadata(self):
        """field: key reads from result.metadata instead of result.value."""
        fe = FilterEngine.from_list([
            {"signal": "vwap_session", "field": "session_age_bars", "expr": ">= 30"},
        ])
        bundle = _bundle(vwap_session={"value": -2.0, "metadata": {"session_age_bars": 50}})
        assert fe.evaluate(bundle).passes is True

    def test_field_blocks_when_below_threshold(self):
        fe = FilterEngine.from_list([
            {"signal": "vwap_session", "field": "session_age_bars", "expr": ">= 30"},
        ])
        bundle = _bundle(vwap_session={"value": -2.0, "metadata": {"session_age_bars": 10}})
        result = fe.evaluate(bundle)
        assert result.passes is False
        assert "session_age_bars" in result.block_reasons[0]

    def test_field_missing_skipped(self):
        """Missing metadata field is silently skipped (passes)."""
        fe = FilterEngine.from_list([
            {"signal": "vwap_session", "field": "nonexistent", "expr": ">= 1.0"},
        ])
        bundle = _bundle(vwap_session={"value": -2.0, "metadata": {"vwap": 5600}})
        assert fe.evaluate(bundle).passes is True

    def test_parse_field_rule(self):
        rules = parse_rules([
            {"signal": "vwap_session", "field": "slope", "expr": "< 0.5"},
        ])
        assert len(rules) == 1
        assert rules[0].field == "slope"
        assert rules[0].signal_name == "vwap_session"


# ── abs prefix tests ────────────────────────────────────────────────


class TestAbsPrefix:
    def test_abs_with_positive_value(self):
        fe = FilterEngine.from_list([
            {"signal": "vwap_session", "field": "slope", "expr": "abs <= 0.5"},
        ])
        bundle = _bundle(vwap_session={"value": 0.0, "metadata": {"slope": 0.3}})
        assert fe.evaluate(bundle).passes is True

    def test_abs_with_negative_value(self):
        fe = FilterEngine.from_list([
            {"signal": "vwap_session", "field": "slope", "expr": "abs <= 0.5"},
        ])
        bundle = _bundle(vwap_session={"value": 0.0, "metadata": {"slope": -0.3}})
        assert fe.evaluate(bundle).passes is True

    def test_abs_blocks_large_negative(self):
        fe = FilterEngine.from_list([
            {"signal": "vwap_session", "field": "slope", "expr": "abs <= 0.5"},
        ])
        bundle = _bundle(vwap_session={"value": 0.0, "metadata": {"slope": -1.5}})
        result = fe.evaluate(bundle)
        assert result.passes is False
        assert "abs" in result.block_reasons[0]

    def test_abs_on_value_not_field(self):
        """abs works on .value too, not just metadata fields."""
        fe = FilterEngine.from_list([
            {"signal": "deviation", "expr": "abs >= 2.0"},
        ])
        assert fe.evaluate(_bundle(deviation=-2.5)).passes is True
        assert fe.evaluate(_bundle(deviation=2.5)).passes is True
        assert fe.evaluate(_bundle(deviation=-1.0)).passes is False

    def test_parse_abs_expr(self):
        rules = parse_rules([
            {"signal": "x", "field": "sd", "expr": "abs >= 2.0"},
        ])
        assert rules[0].use_abs is True
        assert rules[0].op == ">="
        assert rules[0].threshold == 2.0
        assert rules[0].field == "sd"

    def test_non_abs_default(self):
        rules = parse_rules([{"signal": "x", "expr": ">= 2.0"}])
        assert rules[0].use_abs is False
