"""ExitBuilder — declarative exit geometry from YAML config.

Strategies detect patterns and decide *when* and *which direction* to trade.
ExitBuilder decides *where* target and stop go based on YAML exit config.
OMS executes the bracket mechanically (unchanged).

YAML schema:
    exit:
      target:
        type: or_width          # or: fixed_ticks, atr_multiple, vwap, sd_band
        multiplier: 1.0         # meaning depends on type
      stop:
        type: first_break       # or: fixed_ticks, atr_multiple, or_width, sd_band
        buffer_ticks: 1         # extra ticks beyond reference
      time_stop_minutes: 390
      session_close: "15:55"

Supported target types:
    or_width        — entry ± (or_width × multiplier)
    fixed_ticks     — entry ± (ticks × tick_size)
    atr_multiple    — entry ± (atr × multiplier)
    vwap            — target at current VWAP level
    sd_band         — target at vwap ± (sd × multiplier)

Supported stop types:
    first_break     — first_break_extreme ± buffer_ticks
    fixed_ticks     — entry ∓ (ticks × tick_size)
    atr_multiple    — entry ∓ (atr × multiplier)
    or_width        — entry ∓ (or_width × multiplier)
    sd_band         — stop at vwap ∓ (sd × multiplier)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TICK_SIZE = 0.25


@dataclass
class ExitContext:
    """Raw context from the strategy — ExitBuilder picks what it needs.

    Strategies populate whichever fields are relevant; ExitBuilder only
    reads the fields required by the configured exit type.
    """

    entry_price: float = 0.0
    direction: str = "LONG"  # "LONG" or "SHORT"

    # ORB-specific
    or_width: float = 0.0
    first_break_extreme: float = 0.0

    # General
    atr: float = 0.0
    vwap: float = 0.0
    vwap_sd: float = 0.0  # standard deviation of price from VWAP


@dataclass(frozen=True)
class ExitGeometry:
    """Computed target and stop prices."""

    target_price: float
    stop_price: float


def _parse_target(cfg: dict[str, Any]) -> dict[str, Any]:
    """Normalize target config to dict with type + params."""
    if isinstance(cfg, str):
        # Simple string like "vwap"
        return {"type": cfg}
    return dict(cfg)


def _parse_stop(cfg: dict[str, Any]) -> dict[str, Any]:
    """Normalize stop config to dict with type + params."""
    if isinstance(cfg, (int, float)):
        # Legacy: bare number = fixed ticks
        return {"type": "fixed_ticks", "ticks": float(cfg)}
    if isinstance(cfg, str):
        return {"type": cfg}
    return dict(cfg)


class ExitBuilder:
    """Computes exit geometry from YAML config + strategy context.

    Usage:
        builder = ExitBuilder.from_yaml(yaml_config["exit"])
        ctx = ExitContext(entry_price=5600.0, direction="LONG", or_width=10.0, ...)
        geo = builder.compute(ctx)
        # geo.target_price, geo.stop_price
    """

    def __init__(
        self,
        target_cfg: dict[str, Any],
        stop_cfg: dict[str, Any],
        time_stop_minutes: int = 60,
        session_close: str | None = None,
        slippage_ticks: int = 0,
    ) -> None:
        self._target_cfg = target_cfg
        self._stop_cfg = stop_cfg
        self.time_stop_minutes = time_stop_minutes
        self.session_close = session_close
        self.slippage_ticks = slippage_ticks

    @classmethod
    def from_yaml(cls, exit_config: dict[str, Any], orb_config: dict[str, Any] | None = None) -> ExitBuilder:
        """Build from the 'exit:' section of a strategy YAML.

        Args:
            exit_config: The exit: section from YAML.
            orb_config: Optional orb: section for legacy slippage_ticks.
        """
        target_raw = exit_config.get("target", {"type": "fixed_ticks", "ticks": 8})
        stop_raw = exit_config.get("stop", {"type": "fixed_ticks", "ticks": 8})

        target_cfg = _parse_target(target_raw)
        stop_cfg = _parse_stop(stop_raw)

        time_stop = int(exit_config.get("time_stop_minutes", 60))
        session_close = exit_config.get("session_close")

        # Slippage: check exit config first, then orb config
        slippage = int(exit_config.get("slippage_ticks", 0))
        if slippage == 0 and orb_config:
            slippage = int(orb_config.get("slippage_ticks", 0))

        return cls(
            target_cfg=target_cfg,
            stop_cfg=stop_cfg,
            time_stop_minutes=time_stop,
            session_close=session_close,
            slippage_ticks=slippage,
        )

    def compute(self, ctx: ExitContext) -> ExitGeometry:
        """Compute target and stop prices from context."""
        is_long = ctx.direction == "LONG"

        target = self._compute_target(ctx, is_long)
        stop = self._compute_stop(ctx, is_long)

        return ExitGeometry(target_price=target, stop_price=stop)

    # ── Target computation ─────────────────────────────────────────

    def _compute_target(self, ctx: ExitContext, is_long: bool) -> float:
        t = self._target_cfg.get("type", "fixed_ticks")
        mult = float(self._target_cfg.get("multiplier", 1.0))

        if t == "or_width":
            distance = ctx.or_width * mult
            return ctx.entry_price + distance if is_long else ctx.entry_price - distance

        if t == "fixed_ticks":
            ticks = float(self._target_cfg.get("ticks", 8))
            distance = ticks * TICK_SIZE
            return ctx.entry_price + distance if is_long else ctx.entry_price - distance

        if t == "atr_multiple":
            distance = ctx.atr * mult
            return ctx.entry_price + distance if is_long else ctx.entry_price - distance

        if t == "vwap":
            # Target is the VWAP level itself
            return ctx.vwap if ctx.vwap > 0 else ctx.entry_price

        if t == "sd_band":
            distance = ctx.vwap_sd * mult
            if is_long:
                return ctx.vwap + distance if ctx.vwap > 0 else ctx.entry_price + distance
            else:
                return ctx.vwap - distance if ctx.vwap > 0 else ctx.entry_price - distance

        raise ValueError(f"Unknown target type: {t}")

    # ── Stop computation ───────────────────────────────────────────

    def _compute_stop(self, ctx: ExitContext, is_long: bool) -> float:
        t = self._stop_cfg.get("type", "fixed_ticks")

        if t == "first_break":
            buffer = float(self._stop_cfg.get("buffer_ticks", 1)) * TICK_SIZE
            if is_long:
                return ctx.first_break_extreme - buffer
            else:
                return ctx.first_break_extreme + buffer

        if t == "fixed_ticks":
            ticks = float(self._stop_cfg.get("ticks", 8))
            distance = ticks * TICK_SIZE
            return ctx.entry_price - distance if is_long else ctx.entry_price + distance

        if t == "atr_multiple":
            mult = float(self._stop_cfg.get("multiplier", 2.0))
            distance = ctx.atr * mult
            return ctx.entry_price - distance if is_long else ctx.entry_price + distance

        if t == "or_width":
            mult = float(self._stop_cfg.get("multiplier", 1.0))
            distance = ctx.or_width * mult
            return ctx.entry_price - distance if is_long else ctx.entry_price + distance

        if t == "sd_band":
            mult = float(self._stop_cfg.get("multiplier", 2.0))
            distance = ctx.vwap_sd * mult
            if is_long:
                return ctx.vwap - distance if ctx.vwap > 0 else ctx.entry_price - distance
            else:
                return ctx.vwap + distance if ctx.vwap > 0 else ctx.entry_price + distance

        raise ValueError(f"Unknown stop type: {t}")
