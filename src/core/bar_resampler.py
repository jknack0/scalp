"""Live bar resampler — accumulates 1s bars into clock-aligned higher-timeframe bars.

Sits between TickAggregator (1s bars) and SignalHandler. When a complete
higher-timeframe bar is ready, fires a callback with the aggregated BarEvent.

Usage:
    resampler = BarResampler(freq_seconds=300, callback=handler.on_bar)
    bus.subscribe(EventType.BAR, resampler.on_bar)
"""

from __future__ import annotations

from src.core.events import BarEvent
from src.core.logging import get_logger

logger = get_logger("bar_resampler")


def _freq_to_seconds(freq: str) -> int:
    """Convert a frequency string like '5m', '1m', '15s' to seconds."""
    freq = freq.strip().lower()
    if freq.endswith("m"):
        return int(freq[:-1]) * 60
    elif freq.endswith("s"):
        return int(freq[:-1])
    elif freq.endswith("h"):
        return int(freq[:-1]) * 3600
    raise ValueError(f"Unsupported frequency: {freq!r}")


class BarResampler:
    """Accumulates 1s bars into clock-aligned higher-timeframe bars.

    Aligns windows to UTC clock boundaries (e.g. 5m bars start at
    :00, :05, :10, etc.). When a new window starts, the completed
    bar is emitted via the callback.
    """

    def __init__(
        self,
        freq_seconds: int,
        callback,
    ) -> None:
        self._freq_ns = freq_seconds * 1_000_000_000
        self._freq_seconds = freq_seconds
        self._callback = callback

        # Current window state
        self._window_start_ns: int | None = None
        self._open: float = 0.0
        self._high: float = -float("inf")
        self._low: float = float("inf")
        self._close: float = 0.0
        self._volume: int = 0
        self._symbol: str = ""
        self._bar_count: int = 0
        self._has_data: bool = False

        # L1 aggregation
        self._agg_buy_vol: float = 0.0
        self._agg_sell_vol: float = 0.0

    def _window_for(self, ts_ns: int) -> int:
        """Return the window start timestamp for a given bar timestamp."""
        return ts_ns - (ts_ns % self._freq_ns)

    async def on_bar(self, bar: BarEvent) -> None:
        """Ingest a 1s bar. Emit a resampled bar when a window completes."""
        window_start = self._window_for(bar.timestamp_ns)

        if self._window_start_ns is None:
            # First bar ever
            self._window_start_ns = window_start
            self._symbol = bar.symbol

        if window_start > self._window_start_ns:
            # New window started — emit the completed bar
            if self._has_data:
                await self._emit()
            self._reset(window_start, bar.symbol)

        # Accumulate into current window
        if not self._has_data:
            self._open = bar.open
            self._has_data = True
        self._high = max(self._high, bar.high)
        self._low = min(self._low, bar.low)
        self._close = bar.close
        self._volume += bar.volume
        self._agg_buy_vol += bar.aggressive_buy_vol
        self._agg_sell_vol += bar.aggressive_sell_vol

    async def _emit(self) -> None:
        """Emit the completed resampled bar."""
        freq_s = self._freq_seconds
        if freq_s >= 3600:
            bar_type = f"{freq_s // 3600}h"
        elif freq_s >= 60:
            bar_type = f"{freq_s // 60}m"
        else:
            bar_type = f"{freq_s}s"

        resampled = BarEvent(
            symbol=self._symbol,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            bar_type=bar_type,
            timestamp_ns=self._window_start_ns,
            aggressive_buy_vol=self._agg_buy_vol,
            aggressive_sell_vol=self._agg_sell_vol,
        )

        self._bar_count += 1
        logger.info(
            "resampled_bar",
            bar_num=self._bar_count,
            bar_type=bar_type,
            open=round(resampled.open, 2),
            close=round(resampled.close, 2),
            volume=resampled.volume,
        )

        await self._callback(resampled)

    def _reset(self, window_start_ns: int, symbol: str) -> None:
        """Reset accumulator for a new window."""
        self._window_start_ns = window_start_ns
        self._symbol = symbol
        self._open = 0.0
        self._high = -float("inf")
        self._low = float("inf")
        self._close = 0.0
        self._volume = 0
        self._has_data = False
        self._agg_buy_vol = 0.0
        self._agg_sell_vol = 0.0

    async def flush(self) -> None:
        """Emit any partial bar (e.g. at session close)."""
        if self._has_data:
            await self._emit()
            self._has_data = False
