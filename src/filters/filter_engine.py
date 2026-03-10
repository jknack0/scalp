"""FilterEngine — declarative YAML-driven filter evaluation.

Parses filter rules from strategy YAML configs and evaluates them against
a pre-computed SignalBundle. Supports multi-timeframe pipelines via sequence
numbers (seq) — each seq is evaluated against a different SignalBundle
from a different bar timeframe.

YAML syntax (list format):
    filters:
      - signal: spread
        expr: "< 2.0"
        seq: 1
        bar: 5m              # compute signals on 5m bars
      - signal: vwap_session
        field: deviation_sd   # access metadata field instead of .value
        expr: "abs >= 2.0"    # abs prefix takes absolute value before comparing
      - signal: hmm_regime
        expr: "!= 3"
        seq: 1
        bar: 5m
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field

from src.signals.signal_bundle import SignalBundle


# Supported comparison operators
_OPS: dict[str, callable] = {
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass(frozen=True)
class FilterRule:
    """A single filter rule: signal_name op threshold.

    seq controls evaluation order in multi-timeframe pipelines.
    seq=1 rules are evaluated first (e.g. on 5m signals),
    seq=2 rules are evaluated second (e.g. on 15m signals), etc.
    If any seq fails, later seqs are not evaluated.

    field: if set, read this metadata key instead of .value.
    use_abs: if True, take abs() of the value before comparing.
    """

    signal_name: str
    op: str  # "<", ">", "<=", ">=", "==", "!=", "passes"
    threshold: float = 0.0
    seq: int = 1
    bar: str | None = None  # bar timeframe, e.g. "5m", "15m", "1h"
    field: str | None = None  # metadata key, e.g. "deviation_sd", "slope"
    use_abs: bool = False  # take abs() before comparing


@dataclass(frozen=True)
class FilterResult:
    """Result of evaluating filter rules."""

    passes: bool
    block_reasons: list[str] = field(default_factory=list)


def _parse_expr(
    signal_name: str,
    expr,
    seq: int = 1,
    bar: str | None = None,
    metadata_field: str | None = None,
) -> FilterRule | None:
    """Parse a single filter expression into a FilterRule.

    Returns None if the expression disables the filter (e.g. false).
    Supports "abs" prefix: "abs >= 2.0" takes absolute value before comparing.
    """
    if isinstance(expr, bool):
        if expr:
            return FilterRule(signal_name, "passes", seq=seq, bar=bar, field=metadata_field)
        return None

    if isinstance(expr, str):
        expr = expr.strip()

        # "passes" as string expression
        if expr == "passes":
            return FilterRule(signal_name, "passes", seq=seq, bar=bar, field=metadata_field)

        # Check for "abs" prefix
        use_abs = False
        if expr.startswith("abs ") or expr.startswith("abs\t"):
            use_abs = True
            expr = expr[4:].strip()

        for op_str in ("<=", ">=", "!=", "=="):
            if expr.startswith(op_str):
                threshold = float(expr[len(op_str):].strip())
                return FilterRule(signal_name, op_str, threshold, seq=seq, bar=bar,
                                  field=metadata_field, use_abs=use_abs)
        for op_str in ("<", ">"):
            if expr.startswith(op_str):
                threshold = float(expr[len(op_str):].strip())
                return FilterRule(signal_name, op_str, threshold, seq=seq, bar=bar,
                                  field=metadata_field, use_abs=use_abs)

        if not use_abs:
            try:
                threshold = float(expr)
                return FilterRule(signal_name, "==", threshold, seq=seq, bar=bar,
                                  field=metadata_field)
            except ValueError:
                pass

        raise ValueError(
            f"Cannot parse filter expression for '{signal_name}': {expr!r}"
        )

    if isinstance(expr, (int, float)):
        return FilterRule(signal_name, "==", float(expr), seq=seq, bar=bar,
                          field=metadata_field)

    raise ValueError(
        f"Unsupported filter expression type for '{signal_name}': {type(expr).__name__}"
    )


def parse_rules(filter_config: list) -> list[FilterRule]:
    """Parse a YAML filter list into FilterRule objects.

    Each item is a dict with keys: signal, expr, seq (optional), field (optional).

    Example:
        [
            {"signal": "spread", "expr": "< 2.0", "seq": 1},
            {"signal": "vwap_session", "field": "deviation_sd", "expr": "abs >= 2.0"},
            {"signal": "hmm_regime", "expr": "!= 3", "seq": 1},
        ]
    """
    rules: list[FilterRule] = []

    for item in filter_config:
        if not isinstance(item, dict):
            raise ValueError(f"Filter list items must be dicts, got {type(item).__name__}")
        signal_name = item["signal"]
        expr = item["expr"]
        seq = int(item.get("seq", 1))
        bar = item.get("bar")  # e.g. "5m", "15m", None
        metadata_field = item.get("field")  # e.g. "deviation_sd", "slope"
        rule = _parse_expr(signal_name, expr, seq=seq, bar=bar, metadata_field=metadata_field)
        if rule is not None:
            rules.append(rule)

    return rules


class FilterEngine:
    """Evaluates filter rules against SignalBundles.

    Supports multi-timeframe filtering via sequence numbers. Each seq
    is evaluated against a different SignalBundle (from a different
    bar timeframe). If any seq fails, later seqs are skipped.

    Usage (single timeframe):
        fe = FilterEngine.from_list([{"signal": "spread", "expr": "< 2.0"}])
        result = fe.evaluate(bundle_5m)

    Usage (metadata field + abs):
        fe = FilterEngine.from_list([
            {"signal": "vwap_session", "field": "slope", "expr": "abs <= 0.5"},
            {"signal": "adx", "expr": "< 20.0"},
        ])
        result = fe.evaluate(bundle)
    """

    def __init__(self, rules: list[FilterRule] | None = None) -> None:
        self._rules = rules or []

    @classmethod
    def from_list(cls, filter_list: list | None) -> FilterEngine:
        """Build from a YAML filter list (the 'filters:' section)."""
        if not filter_list:
            return cls([])
        return cls(parse_rules(filter_list))

    def evaluate(self, bundle: SignalBundle) -> FilterResult:
        """Evaluate all rules (all seqs) against a single bundle.

        Use this for single-timeframe pipelines where all rules
        apply to the same signal source.
        """
        return self._evaluate_rules(self._rules, bundle)

    def evaluate_seq(self, bundle: SignalBundle, seq: int) -> FilterResult:
        """Evaluate only rules at a specific sequence number.

        Use this for multi-timeframe pipelines where each seq corresponds
        to a different bar timeframe / signal engine.
        """
        seq_rules = [r for r in self._rules if r.seq == seq]
        return self._evaluate_rules(seq_rules, bundle)

    def _evaluate_rules(self, rules: list[FilterRule], bundle: SignalBundle) -> FilterResult:
        """Evaluate a list of rules against a bundle."""
        if not rules:
            return FilterResult(passes=True)

        block_reasons: list[str] = []

        for rule in rules:
            result = bundle.get(rule.signal_name)

            if result is None:
                continue

            # Signal declared data unavailable — block, don't silently pass
            if result.metadata.get("unavailable"):
                block_reasons.append(
                    f"{rule.signal_name}: data unavailable (missing required columns)"
                )
                continue

            if rule.op == "passes":
                if not result.passes:
                    block_reasons.append(
                        f"{rule.signal_name}: did not pass (value={result.value:.4f})"
                    )
                continue

            # Get the value to compare: either .value or metadata[field]
            if rule.field is not None:
                raw_value = result.metadata.get(rule.field)
                if raw_value is None:
                    continue  # field not present, skip
                value = float(raw_value)
                label = f"{rule.signal_name}.{rule.field}"
            else:
                value = result.value
                label = rule.signal_name

            if rule.use_abs:
                value = abs(value)

            op_fn = _OPS.get(rule.op)
            if op_fn is None:
                block_reasons.append(f"{label}: unknown op '{rule.op}'")
                continue
            if not op_fn(value, rule.threshold):
                abs_str = "abs " if rule.use_abs else ""
                block_reasons.append(
                    f"{label}: {abs_str}{value:.4f} not {rule.op} {rule.threshold}"
                )

        return FilterResult(
            passes=len(block_reasons) == 0,
            block_reasons=block_reasons,
        )

    @property
    def rules(self) -> list[FilterRule]:
        return list(self._rules)

    @property
    def signal_names(self) -> list[str]:
        """Signal names referenced by filter rules."""
        return list({r.signal_name for r in self._rules})

    @property
    def sequences(self) -> list[int]:
        """Sorted list of unique sequence numbers in the rules."""
        return sorted({r.seq for r in self._rules}) if self._rules else []

    @property
    def bar_freqs(self) -> dict[int, str]:
        """Map seq -> bar frequency string (e.g. {1: "5m", 2: "15m"}).

        Only includes seqs where at least one rule has a bar field set.
        If rules within the same seq disagree, the first one wins.
        """
        freqs: dict[int, str] = {}
        for r in self._rules:
            if r.bar is not None and r.seq not in freqs:
                freqs[r.seq] = r.bar
        return freqs

    @property
    def is_empty(self) -> bool:
        return len(self._rules) == 0
