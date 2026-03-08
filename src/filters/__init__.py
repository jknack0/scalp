"""Regime filters for trade gating."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class L2Snapshot:
    """Simplified L2 order book snapshot used by filter modules.

    bids and asks are lists of (price, size) tuples, best level first.
    """

    timestamp: datetime
    bids: list[tuple[float, int]]
    asks: list[tuple[float, int]]


from src.filters.hidden_liquidity import (
    HiddenLevel,
    HiddenLiquidityConfig,
    HiddenLiquidityDetector,
    HiddenLiquidityMap,
    HiddenTradeEvent,
)
from src.filters.iceberg_absorption import (
    AbsorptionDetector,
    AbsorptionSignal,
    IcebergConfig,
    IcebergDetector,
    IcebergSignal,
    TradeEvent,
    infer_aggressor,
)
from src.filters.depth_monitor import (
    DepthConfig,
    DepthMonitor,
    DepthSignal,
)
from src.filters.mid_momentum import (
    MidMomentumMonitor,
    MidSnapshot,
    MomentumConfig,
    MomentumSignal,
)
from src.filters.quote_fade import (
    FadeConfig,
    FadeResult,
    OrderRouteEvent,
    QuoteEvent,
    QuoteFadeDetector,
)
from src.filters.weighted_mid import (
    WeightedMidConfig,
    WeightedMidMonitor,
    WeightedMidSignal,
    WeightedMidSnapshot,
)
from src.filters.spread_monitor import (
    SpreadConfig,
    SpreadMonitor,
    SpreadSnapshot,
    SpreadState,
)
from src.filters.vpin_monitor import (
    VPINConfig,
    VPINMonitor,
    VPINState,
)

__all__ = [
    "AbsorptionDetector",
    "AbsorptionSignal",
    "DepthConfig",
    "DepthMonitor",
    "DepthSignal",
    "IcebergConfig",
    "IcebergDetector",
    "IcebergSignal",
    "HiddenLevel",
    "HiddenLiquidityConfig",
    "HiddenLiquidityDetector",
    "HiddenLiquidityMap",
    "HiddenTradeEvent",
    "FadeConfig",
    "FadeResult",
    "L2Snapshot",
    "MidMomentumMonitor",
    "MidSnapshot",
    "MomentumConfig",
    "MomentumSignal",
    "OrderRouteEvent",
    "QuoteEvent",
    "QuoteFadeDetector",
    "TradeEvent",
    "infer_aggressor",
    "SpreadConfig",
    "SpreadMonitor",
    "SpreadSnapshot",
    "SpreadState",
    "VPINConfig",
    "VPINMonitor",
    "VPINState",
    "WeightedMidConfig",
    "WeightedMidMonitor",
    "WeightedMidSignal",
    "WeightedMidSnapshot",
]
