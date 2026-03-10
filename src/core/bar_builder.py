"""Per-strategy bar builders: tick → bar conversion with configurable bar types.

Each strategy declares what bar type and timeframe it wants. The BarBuilder
protocol defines the interface; concrete implementations handle time-based,
dollar-volume, and tick-count bars. A BarBuilderFactory reads strategy config
and produces the right builder.

Usage (live pipeline):
    builder = BarBuilderFactory.from_config(strategy_config["bar"])
    # on each tick:
    bar = builder.on_tick(tick)
    if bar is not None:
        strategy.on_bar(bar)

Usage (backtest):
    builder = BarBuilderFactory.from_config(strategy_config["bar"])
    for tick in tick_stream:
        bar = builder.on_tick(tick)
        if bar is not None:
            yield bar
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Protocol

from src.core.events import BarEvent, TickEvent
from src.models.dollar_bar import DollarBar

# MES contract
_MES_POINT_VALUE = 5.0
_TICK_SIZE = 0.25


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class BarBuilder(Protocol):
    """Protocol for tick-to-bar conversion."""

    @property
    def bar_type_label(self) -> str:
        """Human-readable bar type (e.g. '5s', 'dollar_50k', '100tick')."""
        ...

    def on_tick(self, tick: TickEvent) -> BarEvent | None:
        """Accumulate a tick. Returns a completed bar or None."""
        ...

    def flush(self) -> BarEvent | None:
        """Force-emit the current partial bar (e.g. at session close)."""
        ...

    def reset(self) -> None:
        """Reset all state (e.g. on session open)."""
        ...


# ---------------------------------------------------------------------------
# Shared accumulator
# ---------------------------------------------------------------------------

@dataclass
class _Accumulator:
    """Working state for a bar being built."""

    symbol: str = ""
    open: float = 0.0
    high: float = -math.inf
    low: float = math.inf
    close: float = 0.0
    volume: int = 0
    tick_count: int = 0
    start_ns: int = 0

    # L1 approximation
    bid_sum: float = 0.0
    ask_sum: float = 0.0
    bid_size_sum: float = 0.0
    ask_size_sum: float = 0.0
    bid_count: int = 0
    ask_count: int = 0
    buy_vol: int = 0
    sell_vol: int = 0

    # Dollar bar enrichment
    cum_dollar_vol: float = 0.0
    cum_pv: float = 0.0  # price * volume for VWAP

    def is_empty(self) -> bool:
        return self.tick_count == 0

    def add_tick(self, tick: TickEvent) -> None:
        price = tick.last_price
        size = tick.last_size

        if self.is_empty():
            self.symbol = tick.symbol
            self.open = price
            self.start_ns = tick.timestamp_ns

        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += size
        self.tick_count += 1

        # Dollar volume
        self.cum_dollar_vol += price * size * _MES_POINT_VALUE
        self.cum_pv += price * size

        # L1
        if tick.bid > 0:
            self.bid_sum += tick.bid
            self.bid_count += 1
        if tick.ask > 0:
            self.ask_sum += tick.ask
            self.ask_count += 1

        # Aggressive classification (Lee-Ready at BBO)
        if tick.ask > 0 and price >= tick.ask:
            self.buy_vol += size
        elif tick.bid > 0 and price <= tick.bid:
            self.sell_vol += size

    def to_bar_event(self, bar_type: str) -> BarEvent:
        """Emit a plain BarEvent."""
        return BarEvent(
            symbol=self.symbol,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            bar_type=bar_type,
            timestamp_ns=self.start_ns,
            avg_bid_price=self.bid_sum / self.bid_count if self.bid_count > 0 else 0.0,
            avg_ask_price=self.ask_sum / self.ask_count if self.ask_count > 0 else 0.0,
            aggressive_buy_vol=float(self.buy_vol),
            aggressive_sell_vol=float(self.sell_vol),
        )

    def to_dollar_bar(
        self,
        bar_type: str,
        session_vwap: float = 0.0,
        prior_day_vwap: float = 0.0,
        session_open_time: datetime | None = None,
    ) -> DollarBar:
        """Emit a DollarBar with enriched fields."""
        return DollarBar(
            symbol=self.symbol,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            bar_type=bar_type,
            timestamp_ns=self.start_ns,
            avg_bid_price=self.bid_sum / self.bid_count if self.bid_count > 0 else 0.0,
            avg_ask_price=self.ask_sum / self.ask_count if self.ask_count > 0 else 0.0,
            aggressive_buy_vol=float(self.buy_vol),
            aggressive_sell_vol=float(self.sell_vol),
            session_vwap=session_vwap,
            prior_day_vwap=prior_day_vwap,
            buy_volume=self.buy_vol,
            sell_volume=self.sell_vol,
            session_open_time=session_open_time,
        )


# ---------------------------------------------------------------------------
# Time-based bar builder
# ---------------------------------------------------------------------------

class TimeBarBuilder:
    """Emits bars at fixed time intervals.

    Config:
        interval_seconds: float (e.g. 1.0, 5.0, 60.0)
        emit_dollar_bar: bool (if True, emits DollarBar with session VWAP)
    """

    def __init__(
        self,
        interval_seconds: float = 5.0,
        emit_dollar_bar: bool = False,
        prior_day_vwap: float = 0.0,
        session_open_time: datetime | None = None,
    ) -> None:
        self._interval_ns = int(interval_seconds * 1_000_000_000)
        self._emit_dollar_bar = emit_dollar_bar
        self._prior_day_vwap = prior_day_vwap
        self._session_open_time = session_open_time
        self._acc = _Accumulator()
        self._boundary_ns: int = 0  # next bar boundary

        # Session VWAP tracking
        self._session_pv: float = 0.0
        self._session_vol: int = 0

    @property
    def bar_type_label(self) -> str:
        secs = self._interval_ns / 1_000_000_000
        if secs < 60:
            return f"{int(secs)}s"
        return f"{int(secs // 60)}m"

    def on_tick(self, tick: TickEvent) -> BarEvent | None:
        if tick.last_price <= 0:
            return None

        # Initialize boundary from first tick
        if self._boundary_ns == 0:
            self._boundary_ns = tick.timestamp_ns + self._interval_ns

        # Check if this tick crosses the boundary
        if tick.timestamp_ns >= self._boundary_ns:
            bar = self._flush_and_emit()
            # Advance boundary
            while self._boundary_ns <= tick.timestamp_ns:
                self._boundary_ns += self._interval_ns
            # Start new bar with this tick
            self._acc.add_tick(tick)
            self._update_session_vwap(tick)
            return bar

        self._acc.add_tick(tick)
        self._update_session_vwap(tick)
        return None

    def flush(self) -> BarEvent | None:
        if self._acc.is_empty():
            return None
        return self._flush_and_emit()

    def reset(self) -> None:
        self._acc = _Accumulator()
        self._boundary_ns = 0
        self._session_pv = 0.0
        self._session_vol = 0

    def _update_session_vwap(self, tick: TickEvent) -> None:
        self._session_pv += tick.last_price * tick.last_size
        self._session_vol += tick.last_size

    def _session_vwap(self) -> float:
        if self._session_vol == 0:
            return 0.0
        return self._session_pv / self._session_vol

    def _flush_and_emit(self) -> BarEvent | None:
        if self._acc.is_empty():
            return None

        label = self.bar_type_label

        if self._emit_dollar_bar:
            bar = self._acc.to_dollar_bar(
                bar_type=label,
                session_vwap=self._session_vwap(),
                prior_day_vwap=self._prior_day_vwap,
                session_open_time=self._session_open_time,
            )
        else:
            bar = self._acc.to_bar_event(bar_type=label)

        self._acc = _Accumulator()
        return bar


# ---------------------------------------------------------------------------
# Dollar bar builder
# ---------------------------------------------------------------------------

class DollarBarBuilder:
    """Emits bars when cumulative dollar volume crosses a threshold.

    Config:
        dollar_threshold: float (e.g. 50_000)
    """

    def __init__(
        self,
        dollar_threshold: float = 50_000.0,
        prior_day_vwap: float = 0.0,
        session_open_time: datetime | None = None,
    ) -> None:
        self._threshold = dollar_threshold
        self._prior_day_vwap = prior_day_vwap
        self._session_open_time = session_open_time
        self._acc = _Accumulator()

        # Session VWAP
        self._session_pv: float = 0.0
        self._session_vol: int = 0

    @property
    def bar_type_label(self) -> str:
        if self._threshold >= 1_000_000:
            return f"dollar_{self._threshold / 1_000_000:.0f}M"
        if self._threshold >= 1_000:
            return f"dollar_{self._threshold / 1_000:.0f}k"
        return f"dollar_{self._threshold:.0f}"

    def on_tick(self, tick: TickEvent) -> DollarBar | None:
        if tick.last_price <= 0:
            return None

        self._acc.add_tick(tick)
        self._session_pv += tick.last_price * tick.last_size
        self._session_vol += tick.last_size

        if self._acc.cum_dollar_vol >= self._threshold:
            bar = self._acc.to_dollar_bar(
                bar_type=self.bar_type_label,
                session_vwap=self._session_vwap(),
                prior_day_vwap=self._prior_day_vwap,
                session_open_time=self._session_open_time,
            )
            self._acc = _Accumulator()
            return bar

        return None

    def flush(self) -> DollarBar | None:
        if self._acc.is_empty():
            return None
        bar = self._acc.to_dollar_bar(
            bar_type=self.bar_type_label,
            session_vwap=self._session_vwap(),
            prior_day_vwap=self._prior_day_vwap,
            session_open_time=self._session_open_time,
        )
        self._acc = _Accumulator()
        return bar

    def reset(self) -> None:
        self._acc = _Accumulator()
        self._session_pv = 0.0
        self._session_vol = 0

    def _session_vwap(self) -> float:
        if self._session_vol == 0:
            return 0.0
        return self._session_pv / self._session_vol


# ---------------------------------------------------------------------------
# Volume bar builder
# ---------------------------------------------------------------------------

class VolumeBarBuilder:
    """Emits bars when cumulative volume crosses a threshold.

    Config:
        volume_threshold: int (e.g. 500 contracts)
        emit_dollar_bar: bool
    """

    def __init__(
        self,
        volume_threshold: int = 500,
        emit_dollar_bar: bool = False,
        prior_day_vwap: float = 0.0,
        session_open_time: datetime | None = None,
    ) -> None:
        self._threshold = volume_threshold
        self._emit_dollar_bar = emit_dollar_bar
        self._prior_day_vwap = prior_day_vwap
        self._session_open_time = session_open_time
        self._acc = _Accumulator()

        # Session VWAP
        self._session_pv: float = 0.0
        self._session_vol: int = 0

    @property
    def bar_type_label(self) -> str:
        return f"{self._threshold}vol"

    def on_tick(self, tick: TickEvent) -> BarEvent | None:
        if tick.last_price <= 0:
            return None

        self._acc.add_tick(tick)
        self._session_pv += tick.last_price * tick.last_size
        self._session_vol += tick.last_size

        if self._acc.volume >= self._threshold:
            bar = self._emit()
            self._acc = _Accumulator()
            return bar

        return None

    def flush(self) -> BarEvent | None:
        if self._acc.is_empty():
            return None
        bar = self._emit()
        self._acc = _Accumulator()
        return bar

    def reset(self) -> None:
        self._acc = _Accumulator()
        self._session_pv = 0.0
        self._session_vol = 0

    def _session_vwap(self) -> float:
        if self._session_vol == 0:
            return 0.0
        return self._session_pv / self._session_vol

    def _emit(self) -> BarEvent:
        label = self.bar_type_label
        if self._emit_dollar_bar:
            return self._acc.to_dollar_bar(
                bar_type=label,
                session_vwap=self._session_vwap(),
                prior_day_vwap=self._prior_day_vwap,
                session_open_time=self._session_open_time,
            )
        return self._acc.to_bar_event(bar_type=label)


# ---------------------------------------------------------------------------
# Tick bar builder
# ---------------------------------------------------------------------------

class TickBarBuilder:
    """Emits bars every N ticks.

    Config:
        tick_count: int (e.g. 100)
        emit_dollar_bar: bool
    """

    def __init__(
        self,
        tick_count: int = 100,
        emit_dollar_bar: bool = False,
        prior_day_vwap: float = 0.0,
        session_open_time: datetime | None = None,
    ) -> None:
        self._threshold = tick_count
        self._emit_dollar_bar = emit_dollar_bar
        self._prior_day_vwap = prior_day_vwap
        self._session_open_time = session_open_time
        self._acc = _Accumulator()

        self._session_pv: float = 0.0
        self._session_vol: int = 0

    @property
    def bar_type_label(self) -> str:
        return f"{self._threshold}tick"

    def on_tick(self, tick: TickEvent) -> BarEvent | None:
        if tick.last_price <= 0:
            return None

        self._acc.add_tick(tick)
        self._session_pv += tick.last_price * tick.last_size
        self._session_vol += tick.last_size

        if self._acc.tick_count >= self._threshold:
            bar = self._emit()
            self._acc = _Accumulator()
            return bar

        return None

    def flush(self) -> BarEvent | None:
        if self._acc.is_empty():
            return None
        bar = self._emit()
        self._acc = _Accumulator()
        return bar

    def reset(self) -> None:
        self._acc = _Accumulator()
        self._session_pv = 0.0
        self._session_vol = 0

    def _session_vwap(self) -> float:
        if self._session_vol == 0:
            return 0.0
        return self._session_pv / self._session_vol

    def _emit(self) -> BarEvent:
        label = self.bar_type_label
        if self._emit_dollar_bar:
            return self._acc.to_dollar_bar(
                bar_type=label,
                session_vwap=self._session_vwap(),
                prior_day_vwap=self._prior_day_vwap,
                session_open_time=self._session_open_time,
            )
        return self._acc.to_bar_event(bar_type=label)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class BarBuilderFactory:
    """Construct a BarBuilder from a config dict.

    Expected config shape (from strategy YAML ``bar`` section):

        bar:
          type: time          # "time", "dollar", "volume", "tick"
          interval_seconds: 5 # time bars only
          dollar_threshold: 50000  # dollar bars only
          volume_threshold: 500    # volume bars only
          tick_count: 100          # tick bars only
          enrich: true        # emit DollarBar with session_vwap etc.
    """

    _BUILDERS = {
        "time": TimeBarBuilder,
        "dollar": DollarBarBuilder,
        "volume": VolumeBarBuilder,
        "tick": TickBarBuilder,
    }

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        prior_day_vwap: float = 0.0,
        session_open_time: datetime | None = None,
    ) -> TimeBarBuilder | DollarBarBuilder | VolumeBarBuilder | TickBarBuilder:
        """Build a bar builder from a config dict.

        Args:
            config: The ``bar`` section of a strategy YAML.
            prior_day_vwap: Prior day's closing VWAP (set externally).
            session_open_time: Session open timestamp (set externally).
        """
        bar_type = config.get("type", "time").lower()
        enrich = config.get("enrich", False)

        if bar_type == "time":
            return TimeBarBuilder(
                interval_seconds=float(config.get("interval_seconds", 5.0)),
                emit_dollar_bar=enrich,
                prior_day_vwap=prior_day_vwap,
                session_open_time=session_open_time,
            )

        if bar_type == "dollar":
            return DollarBarBuilder(
                dollar_threshold=float(config.get("dollar_threshold", 50_000)),
                prior_day_vwap=prior_day_vwap,
                session_open_time=session_open_time,
            )

        if bar_type == "volume":
            return VolumeBarBuilder(
                volume_threshold=int(config.get("volume_threshold", 500)),
                emit_dollar_bar=enrich,
                prior_day_vwap=prior_day_vwap,
                session_open_time=session_open_time,
            )

        if bar_type == "tick":
            return TickBarBuilder(
                tick_count=int(config.get("tick_count", 100)),
                emit_dollar_bar=enrich,
                prior_day_vwap=prior_day_vwap,
                session_open_time=session_open_time,
            )

        raise ValueError(
            f"Unknown bar type: {bar_type!r}. "
            f"Available: {list(cls._BUILDERS.keys())}"
        )

    @classmethod
    def available_types(cls) -> list[str]:
        return list(cls._BUILDERS.keys())
