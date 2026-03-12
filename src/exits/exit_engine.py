"""ExitEngine — declarative YAML-driven exit condition evaluation.

Parallel to FilterEngine (which gates entries), ExitEngine evaluates exit
conditions each bar while a position is open. Any single condition firing
triggers an exit (OR logic). All conditions are YAML-configured with numeric
parameters suitable for sweep-based tuning.

YAML schema:
    exits:
      - type: static_target
        enabled: true
        atr_multiple: 1.5
      - type: static_stop
        enabled: true
        atr_multiple: 1.0
      - type: trailing_stop
        enabled: false
        atr_multiple: 1.2
        activate_after_ticks: 4
      - type: time_stop
        enabled: true
        max_bars: 12
      - type: vwap_reversion_target
        enabled: true
        target_sd_band: 0.5
      - type: adverse_signal_exit
        enabled: true
        signal: vwap_session
        field: slope
        long_threshold: -0.25
        short_threshold: 0.25
      - type: regime_exit
        enabled: true
        hmm_signal: hmm_regime
        hostile_regimes_long: [1]
        hostile_regimes_short: [0]
        min_bars_before_active: 2
      - type: volatility_expansion_exit
        enabled: true
        atr_signal: atr
        expansion_multiple: 1.8
        min_bars_before_active: 3

Supported exit types:
    static_target              — ATR-multiple take-profit from fill price
    static_stop                — ATR-multiple stop-loss from fill price
    trailing_stop              — ATR-multiple trailing stop with activation threshold
    time_stop                  — Exit after max_bars in trade
    vwap_reversion_target      — Exit when price reverts to within N SD of VWAP
    adverse_signal_exit        — Exit when a signal field flips adverse
    regime_exit                — Exit when HMM regime becomes hostile
    volatility_expansion_exit  — Exit when ATR expands beyond entry ATR × multiple
    adverse_momentum           — Exit if unrealized loss exceeds ATR/points threshold
    signal_bound_exit          — Exit if signal outside [lower, upper] bounds
    price_vs_signal_exit       — Exit if price crosses dynamic signal-derived level
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.core.events import BarEvent
from src.signals.signal_bundle import SignalBundle

TICK_SIZE = 0.25


@dataclass
class ExitContext:
    """Context passed to exit conditions each bar.

    Populated by the OMS/backtest engine from position state + current bar.
    """

    bar: BarEvent
    bundle: SignalBundle
    direction: str  # "LONG" or "SHORT"
    fill_price: float
    bars_in_trade: int
    entry_snapshot: dict = field(default_factory=dict)

    # Mutable state for trailing stop — managed by ExitEngine
    peak_price: float = 0.0


@dataclass(frozen=True)
class ExitResult:
    """Result of evaluating all exit conditions."""

    should_exit: bool
    reason: str | None = None
    # Price override: if set, use this instead of bar.close for market exit
    # Used by static_target and static_stop which have exact price levels
    exit_price: float | None = None


class ExitCondition(ABC):
    """Base class for a single exit condition."""

    @abstractmethod
    def evaluate(self, ctx: ExitContext) -> str | None:
        """Evaluate this condition. Returns exit reason string or None."""
        ...

    @property
    @abstractmethod
    def exit_type(self) -> str:
        """Short name for this exit type."""
        ...


class StaticTarget(ExitCondition):
    """ATR-multiple take-profit from fill price."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.atr_multiple = float(cfg.get("atr_multiple", 1.5))

    @property
    def exit_type(self) -> str:
        return "tp:static_target"

    def evaluate(self, ctx: ExitContext) -> str | None:
        atr = ctx.entry_snapshot.get("atr", 0.0)
        if atr <= 0:
            return None
        distance = atr * self.atr_multiple
        if ctx.direction == "LONG":
            if ctx.bar.high >= ctx.fill_price + distance:
                return self.exit_type
        else:
            if ctx.bar.low <= ctx.fill_price - distance:
                return self.exit_type
        return None


class StaticStop(ExitCondition):
    """ATR-multiple stop-loss from fill price."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.atr_multiple = float(cfg.get("atr_multiple", 1.0))

    @property
    def exit_type(self) -> str:
        return "stop:static_stop"

    def evaluate(self, ctx: ExitContext) -> str | None:
        atr = ctx.entry_snapshot.get("atr", 0.0)
        if atr <= 0:
            return None
        distance = atr * self.atr_multiple
        if ctx.direction == "LONG":
            if ctx.bar.low <= ctx.fill_price - distance:
                return self.exit_type
        else:
            if ctx.bar.high >= ctx.fill_price + distance:
                return self.exit_type
        return None


class TrailingStop(ExitCondition):
    """ATR-multiple trailing stop. Tracks peak price, exits on pullback."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.atr_multiple = float(cfg.get("atr_multiple", 1.2))
        self.activate_after_ticks = float(cfg.get("activate_after_ticks", 4))

    @property
    def exit_type(self) -> str:
        return "stop:trailing"

    def evaluate(self, ctx: ExitContext) -> str | None:
        atr = ctx.entry_snapshot.get("atr", 0.0)
        if atr <= 0:
            return None

        # Check if in enough profit to activate
        if ctx.direction == "LONG":
            pnl_ticks = (ctx.bar.close - ctx.fill_price) / TICK_SIZE
        else:
            pnl_ticks = (ctx.fill_price - ctx.bar.close) / TICK_SIZE

        if pnl_ticks < self.activate_after_ticks:
            return None

        trail_distance = atr * self.atr_multiple

        # Update peak and check trail
        if ctx.direction == "LONG":
            if ctx.bar.high > ctx.peak_price:
                ctx.peak_price = ctx.bar.high
            if ctx.peak_price > 0 and ctx.bar.close <= ctx.peak_price - trail_distance:
                return self.exit_type
        else:
            if ctx.peak_price == 0 or ctx.bar.low < ctx.peak_price:
                ctx.peak_price = ctx.bar.low
            if ctx.peak_price > 0 and ctx.bar.close >= ctx.peak_price + trail_distance:
                return self.exit_type

        return None


class TimeStop(ExitCondition):
    """Exit after max_bars in trade."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.max_bars = int(cfg.get("max_bars", 12))

    @property
    def exit_type(self) -> str:
        return "stop:time"

    def evaluate(self, ctx: ExitContext) -> str | None:
        if ctx.bars_in_trade >= self.max_bars:
            return self.exit_type
        return None


class VWAPReversionTarget(ExitCondition):
    """Exit when price reverts to within target_sd_band of VWAP."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.target_sd_band = float(cfg.get("target_sd_band", 0.5))
        self.vwap_signal = cfg.get("vwap_signal", "vwap_session")
        self.deviation_field = cfg.get("deviation_field", "deviation_sd")

    @property
    def exit_type(self) -> str:
        return "tp:reversion_target"

    def evaluate(self, ctx: ExitContext) -> str | None:
        result = ctx.bundle.get(self.vwap_signal)
        if result is None:
            return None

        meta = result.metadata
        vwap = meta.get("vwap", 0.0)
        sd = meta.get("sd", 0.0)
        if vwap == 0.0 or sd == 0.0:
            return None

        # Use the most favorable price during the bar (high for LONG, low for SHORT)
        # to detect if price touched the target band at any point, not just at close
        if ctx.direction == "LONG":
            best_price = ctx.bar.high  # closest to VWAP (above entry)
        else:
            best_price = ctx.bar.low  # closest to VWAP (below entry)

        best_deviation_sd = (best_price - vwap) / sd
        if abs(best_deviation_sd) <= self.target_sd_band:
            return self.exit_type
        return None


class AdverseSignalExit(ExitCondition):
    """Exit when a signal field moves adversely beyond threshold."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.signal_name = cfg.get("signal", "")
        self.field_name = cfg.get("field", None)
        self.long_threshold = float(cfg.get("long_threshold", 0.0))
        self.short_threshold = float(cfg.get("short_threshold", 0.0))

    @property
    def exit_type(self) -> str:
        suffix = f"{self.signal_name}"
        if self.field_name:
            suffix += f"_{self.field_name}"
        return f"early:adverse_{suffix}"

    def evaluate(self, ctx: ExitContext) -> str | None:
        result = ctx.bundle.get(self.signal_name)
        if result is None:
            return None

        if self.field_name:
            value = result.metadata.get(self.field_name)
            if value is None:
                return None
            value = float(value)
        else:
            value = result.value

        if ctx.direction == "LONG" and value < self.long_threshold:
            return self.exit_type
        if ctx.direction == "SHORT" and value > self.short_threshold:
            return self.exit_type
        return None


class RegimeExit(ExitCondition):
    """Exit when HMM regime becomes hostile to the position direction."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.hmm_signal = cfg.get("hmm_signal", "hmm_regime")
        self.hostile_long = set(cfg.get("hostile_regimes_long", []))
        self.hostile_short = set(cfg.get("hostile_regimes_short", []))
        self.min_bars = int(cfg.get("min_bars_before_active", 2))

    @property
    def exit_type(self) -> str:
        return "early:regime_flip"

    def evaluate(self, ctx: ExitContext) -> str | None:
        if ctx.bars_in_trade < self.min_bars:
            return None

        result = ctx.bundle.get(self.hmm_signal)
        if result is None:
            return None

        regime = int(result.value)

        if ctx.direction == "LONG" and regime in self.hostile_long:
            return self.exit_type
        if ctx.direction == "SHORT" and regime in self.hostile_short:
            return self.exit_type
        return None


class VolatilityExpansionExit(ExitCondition):
    """Exit when live ATR expands beyond entry ATR × multiple."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.atr_signal = cfg.get("atr_signal", "atr")
        self.expansion_multiple = float(cfg.get("expansion_multiple", 1.8))
        self.min_bars = int(cfg.get("min_bars_before_active", 3))

    @property
    def exit_type(self) -> str:
        return "early:vol_expansion"

    def evaluate(self, ctx: ExitContext) -> str | None:
        if ctx.bars_in_trade < self.min_bars:
            return None

        entry_atr = ctx.entry_snapshot.get("atr", 0.0)
        if entry_atr <= 0:
            return None

        result = ctx.bundle.get(self.atr_signal)
        if result is None:
            return None

        live_atr = result.metadata.get("atr_raw", result.value)
        if live_atr > entry_atr * self.expansion_multiple:
            return self.exit_type
        return None


class BracketTarget(ExitCondition):
    """Exit at the Signal's original target_price (from ExitBuilder geometry).

    Reads target_price from entry_snapshot, which is captured at fill time
    from the Signal object. Works with any ExitBuilder geometry type
    (or_width, sd_band, fixed_ticks, atr_multiple, etc.).

    YAML:
        - type: bracket_target
          enabled: true
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        pass

    @property
    def exit_type(self) -> str:
        return "tp:bracket_target"

    def evaluate(self, ctx: ExitContext) -> str | None:
        target = ctx.entry_snapshot.get("target_price", 0.0)
        if target == 0.0:
            return None
        if ctx.direction == "LONG" and ctx.bar.high >= target:
            return self.exit_type
        if ctx.direction == "SHORT" and ctx.bar.low <= target:
            return self.exit_type
        return None


class BracketStop(ExitCondition):
    """Exit at the Signal's original stop_price (from ExitBuilder geometry).

    YAML:
        - type: bracket_stop
          enabled: true
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        pass

    @property
    def exit_type(self) -> str:
        return "stop:bracket_stop"

    def evaluate(self, ctx: ExitContext) -> str | None:
        stop = ctx.entry_snapshot.get("stop_price", 0.0)
        if stop == 0.0:
            return None
        if ctx.direction == "LONG" and ctx.bar.low <= stop:
            return self.exit_type
        if ctx.direction == "SHORT" and ctx.bar.high >= stop:
            return self.exit_type
        return None


class AdverseMomentum(ExitCondition):
    """Exit if unrealized loss exceeds threshold within first N bars.

    YAML:
        - type: adverse_momentum
          enabled: true
          atr_multiple: 1.0        # exit if loss > 1.0x ATR
          max_bars: 3              # only check in first 3 bars (0 = always)
          threshold_points: 0      # alternative: fixed points instead of ATR
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.atr_multiple = float(cfg.get("atr_multiple", 1.0))
        self.max_bars = int(cfg.get("max_bars", 0))
        self.threshold_points = float(cfg.get("threshold_points", 0.0))

    @property
    def exit_type(self) -> str:
        return "early:adverse_momentum"

    def evaluate(self, ctx: ExitContext) -> str | None:
        if self.max_bars > 0 and ctx.bars_in_trade > self.max_bars:
            return None

        if self.threshold_points > 0:
            threshold = self.threshold_points
        else:
            atr = ctx.entry_snapshot.get("atr", 0.0)
            if atr <= 0:
                return None
            threshold = atr * self.atr_multiple

        if ctx.direction == "LONG":
            loss = ctx.fill_price - ctx.bar.close
        else:
            loss = ctx.bar.close - ctx.fill_price

        if loss > threshold:
            return self.exit_type
        return None


class SignalBoundExit(ExitCondition):
    """Exit if a signal value goes outside [lower, upper] bounds.

    Non-directional: fires for both LONG and SHORT positions.

    YAML:
        - type: signal_bound_exit
          enabled: true
          signal: adx
          field: null            # optional: read from metadata field
          upper_bound: 30.0      # exit if value > 30 (null = don't check)
          lower_bound: null      # exit if value < X (null = don't check)
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.signal_name = cfg.get("signal", "")
        self.field_name = cfg.get("field", None)
        self.upper_bound = cfg.get("upper_bound", None)
        self.lower_bound = cfg.get("lower_bound", None)
        if self.upper_bound is not None:
            self.upper_bound = float(self.upper_bound)
        if self.lower_bound is not None:
            self.lower_bound = float(self.lower_bound)

    @property
    def exit_type(self) -> str:
        suffix = self.signal_name
        if self.field_name:
            suffix += f"_{self.field_name}"
        return f"early:bound_{suffix}"

    def evaluate(self, ctx: ExitContext) -> str | None:
        result = ctx.bundle.get(self.signal_name)
        if result is None:
            return None

        if self.field_name:
            value = result.metadata.get(self.field_name)
            if value is None:
                return None
            value = float(value)
        else:
            value = result.value

        if self.upper_bound is not None and value > self.upper_bound:
            return self.exit_type
        if self.lower_bound is not None and value < self.lower_bound:
            return self.exit_type
        return None


class PriceVsSignalExit(ExitCondition):
    """Exit if price crosses a dynamic signal-derived level.

    Used for: keltner reentry, donchian trailing, beyond-EMA, level breaches.

    YAML:
        - type: price_vs_signal_exit
          enabled: true
          signal: keltner_channel
          long_field: upper       # LONG exits if close < signal.metadata[upper]
          short_field: lower      # SHORT exits if close > signal.metadata[lower]
          offset: 0.0             # points offset (positive = more room before exit)
          max_bars: 0             # only check in first N bars (0 = always)
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.signal_name = cfg.get("signal", "")
        self.long_field = cfg.get("long_field", None)
        self.short_field = cfg.get("short_field", None)
        self.offset = float(cfg.get("offset", 0.0))
        self.max_bars = int(cfg.get("max_bars", 0))

    @property
    def exit_type(self) -> str:
        return f"early:price_vs_{self.signal_name}"

    def evaluate(self, ctx: ExitContext) -> str | None:
        if self.max_bars > 0 and ctx.bars_in_trade > self.max_bars:
            return None

        result = ctx.bundle.get(self.signal_name)
        if result is None:
            return None
        meta = result.metadata

        if ctx.direction == "LONG" and self.long_field:
            level = meta.get(self.long_field)
            if level is not None and ctx.bar.close < float(level) - self.offset:
                return self.exit_type
        elif ctx.direction == "SHORT" and self.short_field:
            level = meta.get(self.short_field)
            if level is not None and ctx.bar.close > float(level) + self.offset:
                return self.exit_type
        return None


# ── Condition registry ──────────────────────────────────────────────

_CONDITION_TYPES: dict[str, type[ExitCondition]] = {
    "static_target": StaticTarget,
    "static_stop": StaticStop,
    "trailing_stop": TrailingStop,
    "time_stop": TimeStop,
    "vwap_reversion_target": VWAPReversionTarget,
    "adverse_signal_exit": AdverseSignalExit,
    "regime_exit": RegimeExit,
    "volatility_expansion_exit": VolatilityExpansionExit,
    "adverse_momentum": AdverseMomentum,
    "signal_bound_exit": SignalBoundExit,
    "price_vs_signal_exit": PriceVsSignalExit,
    "bracket_target": BracketTarget,
    "bracket_stop": BracketStop,
}


def build_condition(cfg: dict[str, Any]) -> ExitCondition:
    """Build an ExitCondition from a YAML config dict."""
    exit_type = cfg.get("type", "")
    cls = _CONDITION_TYPES.get(exit_type)
    if cls is None:
        raise ValueError(f"Unknown exit condition type: {exit_type!r}. "
                         f"Available: {list(_CONDITION_TYPES.keys())}")
    return cls(cfg)


# ── ExitEngine ──────────────────────────────────────────────────────

class ExitEngine:
    """Evaluates declarative exit conditions against position state + signals.

    Parallel to FilterEngine: YAML-configured, sweep-friendly, OR logic.
    Any single condition firing triggers an exit.

    Usage:
        engine = ExitEngine.from_list(yaml_config["exits"])
        ctx = ExitContext(bar=bar, bundle=bundle, direction="LONG",
                          fill_price=5600.0, bars_in_trade=3,
                          entry_snapshot={"atr": 3.5, "vwap": 5610.0})
        result = engine.evaluate(ctx)
        if result.should_exit:
            close_position(reason=result.reason)
    """

    def __init__(self, conditions: list[ExitCondition] | None = None) -> None:
        self._conditions = conditions or []

    @classmethod
    def from_list(cls, exit_list: list[dict] | None) -> ExitEngine:
        """Build from the 'exits:' section of a strategy YAML.

        Only builds conditions where enabled is True (default).
        """
        if not exit_list:
            return cls([])
        conditions = []
        for cfg in exit_list:
            if not isinstance(cfg, dict):
                continue
            if not cfg.get("enabled", True):
                continue
            conditions.append(build_condition(cfg))
        return cls(conditions)

    def evaluate(self, ctx: ExitContext) -> ExitResult:
        """Evaluate all conditions. First non-None fires (OR logic).

        Conditions are evaluated in declaration order. Put higher-priority
        exits (stops) before lower-priority ones (early exits) in YAML.
        """
        for condition in self._conditions:
            reason = condition.evaluate(ctx)
            if reason is not None:
                return ExitResult(should_exit=True, reason=reason)
        return ExitResult(should_exit=False)

    def get_bracket_prices(self, fill_price: float, direction: str,
                           entry_snapshot: dict) -> tuple[float, float]:
        """Compute static target/stop prices for bracket orders.

        Used for live trading where we need upfront bracket prices.
        Reads from static_target and static_stop conditions if present.
        Returns (target_price, stop_price). Falls back to (0, 0) if not configured.
        """
        target = 0.0
        stop = 0.0
        atr = entry_snapshot.get("atr", 0.0)
        if atr <= 0:
            return target, stop

        for cond in self._conditions:
            if isinstance(cond, StaticTarget):
                distance = atr * cond.atr_multiple
                if direction == "LONG":
                    target = fill_price + distance
                else:
                    target = fill_price - distance
            elif isinstance(cond, StaticStop):
                distance = atr * cond.atr_multiple
                if direction == "LONG":
                    stop = fill_price - distance
                else:
                    stop = fill_price + distance

        return target, stop

    @property
    def conditions(self) -> list[ExitCondition]:
        return list(self._conditions)

    @property
    def is_empty(self) -> bool:
        return len(self._conditions) == 0

    def has_type(self, exit_type: str) -> bool:
        """Check if a condition of this type is configured."""
        return any(
            isinstance(c, _CONDITION_TYPES.get(exit_type, type(None)))
            for c in self._conditions
        )
