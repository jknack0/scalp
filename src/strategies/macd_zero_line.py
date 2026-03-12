"""Strategy 5: MACD Zero-Line Rejection.

Trades MACD histogram zero-line rejections with triple alignment
(MACD direction + SMA trend + VWAP bias must agree). Uses ATR-based
exits and early exit on histogram zero-cross or adverse momentum.

All entry gates except triple alignment are declarative filters in the
YAML config, evaluated by FilterEngine before the strategy runs.

The strategy handles:
1. Triple alignment check (macd.direction, sma_trend.direction, vwap_bias.direction)
2. Exit geometry computation (ATR-based target/stop, time stop)
3. Early exit on histogram zero-cross or adverse momentum
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import yaml

from src.core.events import BarEvent
from src.core.logging import get_logger
from src.exits.exit_builder import ExitBuilder, ExitContext
from src.filters.filter_engine import FilterEngine
from src.models.hmm_regime import RegimeState
from src.signals.signal_bundle import EMPTY_BUNDLE, SignalBundle
from src.strategies.base import Direction, Signal

from zoneinfo import ZoneInfo

logger = get_logger("macd_zero_line")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25


class MACDZeroLineStrategy:
    """MACD zero-line rejection strategy -- standalone, duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "macd_zero_line")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 3)

        exit_cfg = config.get("exit", {})
        self._exit_builder = ExitBuilder.from_yaml(exit_cfg)
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 45)

        # Build FilterEngine from YAML filters
        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        # State
        self._signals_today = 0
        self._current_regime = RegimeState.RANGE_BOUND

    @classmethod
    def from_yaml(cls, path: str) -> MACDZeroLineStrategy:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg)

    def on_bar(self, bar: BarEvent, bundle: SignalBundle = EMPTY_BUNDLE) -> Signal | None:
        now = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=_ET)

        if self._signals_today >= self._max_signals_per_day:
            logger.debug("blocked_daily_limit", time=now.strftime("%H:%M"),
                         signals_today=self._signals_today)
            return None

        # Run all declarative filters (session_time, adx, relative_volume, macd passes)
        filter_result = self._filter_engine.evaluate(bundle)
        if not filter_result.passes:
            logger.debug("blocked_filters", time=now.strftime("%H:%M"),
                         close=bar.close,
                         reasons=filter_result.block_reasons[:3])
            return None

        # Get MACD signal for direction
        macd_result = bundle.get("macd")
        if macd_result is None:
            logger.debug("blocked_no_macd", time=now.strftime("%H:%M"))
            return None

        macd_dir = macd_result.direction  # "long" or "short"
        if macd_dir == "none":
            logger.debug("blocked_macd_no_direction", time=now.strftime("%H:%M"))
            return None

        # Triple alignment: macd, sma_trend, vwap_bias must all agree
        sma_result = bundle.get("sma_trend")
        vwap_bias_result = bundle.get("vwap_bias")

        sma_dir = sma_result.direction if sma_result else "none"
        vwap_dir = vwap_bias_result.direction if vwap_bias_result else "none"

        if not (macd_dir == sma_dir == vwap_dir):
            logger.debug("blocked_alignment", time=now.strftime("%H:%M"),
                         close=bar.close,
                         macd_dir=macd_dir, sma_dir=sma_dir, vwap_dir=vwap_dir)
            return None

        # Direction from MACD
        direction = Direction.LONG if macd_dir == "long" else Direction.SHORT

        # Log that filters + alignment passed
        adx_result = bundle.get("adx")
        rvol_result = bundle.get("relative_volume")
        atr_result = bundle.get("atr")
        adx_val = adx_result.value if adx_result else 0.0
        rvol_val = rvol_result.value if rvol_result else 0.0
        histogram = macd_result.metadata.get("histogram", 0.0)

        logger.info("filters_passed", time=now.strftime("%H:%M"),
                    close=bar.close, direction=direction.value,
                    histogram=round(histogram, 4),
                    macd_dir=macd_dir, sma_dir=sma_dir, vwap_dir=vwap_dir,
                    adx=round(adx_val, 1), rvol=round(rvol_val, 1))

        # Entry at current price
        entry_price = bar.close

        # ATR for exit geometry
        atr_raw = 0.0
        if atr_result is not None:
            atr_raw = atr_result.metadata.get("atr_raw", 0.0)

        # Compute exit via ExitBuilder
        ctx = ExitContext(
            entry_price=entry_price,
            direction=direction.value,
            atr=atr_raw,
        )
        geo = self._exit_builder.compute(ctx)

        target = geo.target_price
        stop = geo.stop_price

        # Geometry sanity check
        if direction == Direction.LONG:
            if not (stop < entry_price < target):
                logger.info("blocked_geometry",
                            time=now.strftime("%H:%M"),
                            direction="LONG", entry=entry_price,
                            target=round(target, 2), stop=round(stop, 2),
                            reason="stop >= entry or entry >= target")
                return None
        else:
            if not (stop > entry_price > target):
                logger.info("blocked_geometry",
                            time=now.strftime("%H:%M"),
                            direction="SHORT", entry=entry_price,
                            target=round(target, 2), stop=round(stop, 2),
                            reason="stop <= entry or entry <= target")
                return None

        expiry = now + timedelta(minutes=self._time_stop_minutes)

        # Confidence from histogram strength
        confidence = min(0.6 + abs(histogram) * 0.1, 0.9)

        signal = Signal(
            strategy_id=self.strategy_id,
            direction=direction,
            entry_price=entry_price,
            target_price=target,
            stop_price=stop,
            signal_time=now,
            expiry_time=expiry,
            confidence=confidence,
            regime_state=self._current_regime,
            metadata={
                "histogram": histogram,
                "macd_line": macd_result.metadata.get("macd_line", 0.0),
                "signal_line": macd_result.metadata.get("signal_line", 0.0),
                "prev_histogram": macd_result.metadata.get("prev_histogram", 0.0),
                "atr": atr_raw,
                "adx": adx_val,
                "rvol": rvol_val,
                "sma_dir": sma_dir,
                "vwap_dir": vwap_dir,
            },
        )

        self._signals_today += 1
        logger.info(
            "signal_generated",
            component=self.strategy_id,
            direction=direction.value,
            entry=entry_price,
            target=round(target, 2),
            stop=round(stop, 2),
            histogram=round(histogram, 4),
            adx=round(adx_val, 1),
            rvol=round(rvol_val, 1),
            atr=round(atr_raw, 2),
            confidence=round(confidence, 2),
            signal_id=signal.id,
        )
        return signal

    def reset(self) -> None:
        self._signals_today = 0
