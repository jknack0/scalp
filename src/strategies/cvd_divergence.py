"""Strategy 3: CVD Divergence Fade.

Fades moves that lack conviction — when price makes a new high but CVD
declines (bearish divergence), or price makes a new low but CVD rises
(bullish divergence). Combined with volume exhaustion and VWAP bias
confirmation for high-probability reversals.

All entry gates are declarative filters in the YAML config, evaluated by
FilterEngine before the strategy runs.

The strategy only handles:
1. Direction from cvd_divergence signal (fade the failing move)
2. VWAP bias cross-validation (must agree with trade direction)
3. Minimum divergence strength check
4. Exit geometry computation (target=VWAP ± 1 SD, stop=1.5 ATR, 10 min time stop)
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

logger = get_logger("cvd_divergence")

_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25


class CVDDivergenceStrategy:
    """CVD divergence fade strategy — standalone, duck-typed on_bar/reset."""

    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id: str = strat.get("strategy_id", "cvd_divergence")
        self._max_signals_per_day: int = strat.get("max_signals_per_day", 6)

        # Minimum divergence strength to act on
        self._min_divergence_strength: float = strat.get("min_divergence_strength", 0.3)

        exit_cfg = config.get("exit", {})
        self._exit_builder = ExitBuilder.from_yaml(exit_cfg)
        self._time_stop_minutes: int = exit_cfg.get("time_stop_minutes", 10)

        # Parse early exit conditions from YAML
        self._early_exits = exit_cfg.get("early_exit", [])

        # Build FilterEngine from YAML filters
        self._filter_engine = FilterEngine.from_list(config.get("filters"))

        # State
        self._signals_today = 0
        self._current_regime = RegimeState.RANGE_BOUND

    @classmethod
    def from_yaml(cls, path: str) -> CVDDivergenceStrategy:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg)

    def on_bar(self, bar: BarEvent, bundle: SignalBundle = EMPTY_BUNDLE) -> Signal | None:
        now = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=_ET)

        if self._signals_today >= self._max_signals_per_day:
            logger.debug("blocked_daily_limit", time=now.strftime("%H:%M"),
                         signals_today=self._signals_today)
            return None

        # Run all declarative filters
        filter_result = self._filter_engine.evaluate(bundle)
        if not filter_result.passes:
            logger.debug("blocked_filters", time=now.strftime("%H:%M"),
                         close=bar.close,
                         reasons=filter_result.block_reasons[:3])
            return None

        # --- CVD divergence signal: direction + strength ---
        cvd_result = bundle.get("cvd_divergence")
        if cvd_result is None:
            logger.debug("blocked_no_cvd", time=now.strftime("%H:%M"))
            return None

        # Minimum divergence strength
        if cvd_result.value < self._min_divergence_strength:
            logger.debug("blocked_divergence_weak", time=now.strftime("%H:%M"),
                         strength=round(cvd_result.value, 3),
                         min_required=self._min_divergence_strength)
            return None

        # Direction = fade the failing move (cvd_divergence.direction already
        # points in the fade direction: "long" if bullish divergence, "short"
        # if bearish divergence)
        if cvd_result.direction == "long":
            direction = Direction.LONG
        elif cvd_result.direction == "short":
            direction = Direction.SHORT
        else:
            logger.debug("blocked_no_direction", time=now.strftime("%H:%M"))
            return None

        # --- VWAP bias cross-validation ---
        vwap_bias_result = bundle.get("vwap_bias")
        if vwap_bias_result is None:
            logger.debug("blocked_no_vwap_bias", time=now.strftime("%H:%M"))
            return None

        # VWAP bias direction must AGREE with trade direction
        if direction == Direction.LONG and vwap_bias_result.direction != "long":
            logger.info("blocked_vwap_bias_disagree",
                        time=now.strftime("%H:%M"),
                        trade_dir="LONG",
                        vwap_bias_dir=vwap_bias_result.direction,
                        reason="VWAP bias does not agree with LONG")
            return None
        if direction == Direction.SHORT and vwap_bias_result.direction != "short":
            logger.info("blocked_vwap_bias_disagree",
                        time=now.strftime("%H:%M"),
                        trade_dir="SHORT",
                        vwap_bias_dir=vwap_bias_result.direction,
                        reason="VWAP bias does not agree with SHORT")
            return None

        # --- Gather signal values for logging + metadata ---
        adx_result = bundle.get("adx")
        adx_val = adx_result.value if adx_result else 0.0

        spread_result = bundle.get("spread")
        spread_val = spread_result.value if spread_result else 0.0

        vol_exh_result = bundle.get("volume_exhaustion")
        vol_exh_val = vol_exh_result.value if vol_exh_result else 0.0

        logger.info("filters_passed", time=now.strftime("%H:%M"),
                    close=bar.close, direction=direction.value,
                    cvd_strength=round(cvd_result.value, 3),
                    cvd_dir=cvd_result.direction,
                    vwap_bias_dir=vwap_bias_result.direction,
                    adx=round(adx_val, 1),
                    spread=round(spread_val, 3),
                    vol_exhaustion=round(vol_exh_val, 3))

        # --- Entry price ---
        entry_price = bar.close

        # --- ATR for stop computation ---
        atr_raw = 0.0
        atr_result = bundle.get("atr")
        if atr_result is not None:
            atr_raw = atr_result.metadata.get("atr_raw", 0.0)

        # --- VWAP session data for sd_band target ---
        vwap = 0.0
        sd = 0.0
        vwap_session_result = bundle.get("vwap_session")
        if vwap_session_result is not None:
            meta = vwap_session_result.metadata
            vwap = meta.get("vwap", 0.0)
            sd = meta.get("sd", 0.0)

        if vwap == 0.0 or sd == 0.0:
            logger.debug("blocked_vwap_zero", time=now.strftime("%H:%M"))
            return None

        # --- Compute exit via ExitBuilder ---
        ctx = ExitContext(
            entry_price=entry_price,
            direction=direction.value,
            atr=atr_raw,
            vwap=vwap,
            vwap_sd=sd,
        )
        geo = self._exit_builder.compute(ctx)

        target = geo.target_price
        stop = geo.stop_price

        # --- Geometry sanity check ---
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

        # Confidence based on divergence strength
        confidence = min(0.5 + cvd_result.value * 0.3, 0.9)

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
                "cvd_strength": cvd_result.value,
                "cvd_direction": cvd_result.direction,
                "vwap_bias_dir": vwap_bias_result.direction,
                "vwap": vwap,
                "sd": sd,
                "atr": atr_raw,
                "adx": adx_val,
                "spread": spread_val,
                "vol_exhaustion": vol_exh_val,
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
            cvd_strength=round(cvd_result.value, 3),
            adx=round(adx_val, 1),
            spread=round(spread_val, 3),
            atr=round(atr_raw, 2),
            confidence=round(confidence, 2),
            signal_id=signal.id,
        )
        return signal

    def check_early_exit(
        self,
        bar: BarEvent,
        bundle: SignalBundle,
        bars_in_trade: int,
        direction: Direction,
        fill_price: float,
    ) -> str | None:
        """Check if any early exit condition fires (OR logic).

        Called by the backtest engine on each bar while a position is open.
        Returns an exit reason string (e.g. "early:adverse_momentum") or None.
        """
        for cond in self._early_exits:
            reason = self._eval_early_exit(cond, bar, bundle, bars_in_trade, direction, fill_price)
            if reason is not None:
                return reason
        return None

    def _eval_early_exit(
        self,
        cond: dict,
        bar: BarEvent,
        bundle: SignalBundle,
        bars_in_trade: int,
        direction: Direction,
        fill_price: float,
    ) -> str | None:
        """Evaluate a single early exit condition."""
        exit_type = cond.get("type", "")

        if exit_type == "adverse_momentum":
            return self._check_adverse_momentum_exit(cond, bar, bundle, bars_in_trade, direction, fill_price)

        return None

    def _check_adverse_momentum_exit(
        self,
        cond: dict,
        bar: BarEvent,
        bundle: SignalBundle,
        bars_in_trade: int,
        direction: Direction,
        fill_price: float,
    ) -> str | None:
        """Exit if unrealized loss exceeds ATR multiple within first N bars.

        Catches trades that go immediately wrong — the faster you cut, the less damage.
        """
        max_bars = cond.get("bars", 2)
        atr_mult = cond.get("atr_multiple", 1.0)

        if bars_in_trade > max_bars:
            return None  # Only applies to early bars

        atr_result = bundle.get("atr")
        if atr_result is None:
            return None
        atr_raw = atr_result.metadata.get("atr_raw", 0.0)
        if atr_raw <= 0:
            return None

        # Unrealized P&L (using bar close as mark)
        if direction == Direction.LONG:
            unrealized = bar.close - fill_price
        else:
            unrealized = fill_price - bar.close

        threshold = -atr_mult * atr_raw
        if unrealized < threshold:
            logger.info("early_exit_adverse_momentum",
                        direction=direction.value,
                        bars_in_trade=bars_in_trade,
                        unrealized=round(unrealized, 2),
                        threshold=round(threshold, 2))
            return "early:adverse_momentum"
        return None

    def reset(self) -> None:
        self._signals_today = 0
