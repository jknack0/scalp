"""Stateless signal computations for trade gating.

Each signal takes a list of BarEvent and returns a SignalResult.
Signals are registered via @SignalRegistry.register for config-driven
construction.
"""

from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry

# Import all signals to trigger registration
from src.signals.adx import ADXConfig, ADXSignal
from src.signals.atr import ATRConfig, ATRSignal
from src.signals.cvd_divergence import CVDDivergenceConfig, CVDDivergenceSignal
from src.signals.ema_crossover import EMACrossoverConfig, EMACrossoverSignal
from src.signals.hmm_regime import HMMRegimeConfig, HMMRegimeSignal
from src.signals.orb_breakout import ORBBreakoutConfig, ORBBreakoutSignal
from src.signals.orb_range_size import ORBRangeSizeConfig, ORBRangeSizeSignal
from src.signals.poc_distance import POCDistanceConfig, POCDistanceSignal
from src.signals.prior_day_bias import PriorDayBiasConfig, PriorDayBiasSignal
from src.signals.relative_volume import RelativeVolumeConfig, RelativeVolumeSignal
from src.signals.rsi_momentum import RSIMomentumConfig, RSIMomentumSignal
from src.signals.session_time import SessionTimeSignal
from src.signals.spread import SpreadConfig, SpreadSignal
from src.signals.volume_exhaustion import VolumeExhaustionConfig, VolumeExhaustionSignal
from src.signals.vpin import VPINConfig, VPINSignal
from src.signals.vwap_bias import VWAPBiasConfig, VWAPBiasSignal
from src.signals.vwap_deviation import VWAPDeviationConfig, VWAPDeviationSignal
from src.signals.vwap_session import VWAPSessionConfig, VWAPSessionSignal
from src.signals.vwap_slope import VWAPSlopeConfig, VWAPSlopeSignal
from src.signals.bollinger import BollingerConfig, BollingerSignal
from src.signals.sma_trend import SMATrendConfig, SMATrendSignal

__all__ = [
    "SignalBase",
    "SignalResult",
    "SignalRegistry",
    "ADXConfig",
    "ADXSignal",
    "ATRConfig",
    "ATRSignal",
    "CVDDivergenceConfig",
    "CVDDivergenceSignal",
    "EMACrossoverConfig",
    "EMACrossoverSignal",
    "HMMRegimeConfig",
    "HMMRegimeSignal",
    "ORBBreakoutConfig",
    "ORBBreakoutSignal",
    "ORBRangeSizeConfig",
    "ORBRangeSizeSignal",
    "POCDistanceConfig",
    "POCDistanceSignal",
    "PriorDayBiasConfig",
    "PriorDayBiasSignal",
    "RelativeVolumeConfig",
    "RelativeVolumeSignal",
    "RSIMomentumConfig",
    "RSIMomentumSignal",
    "SessionTimeSignal",
    "SpreadConfig",
    "SpreadSignal",
    "VolumeExhaustionConfig",
    "VolumeExhaustionSignal",
    "VPINConfig",
    "VPINSignal",
    "VWAPBiasConfig",
    "VWAPBiasSignal",
    "VWAPDeviationConfig",
    "VWAPDeviationSignal",
    "VWAPSessionConfig",
    "VWAPSessionSignal",
    "VWAPSlopeConfig",
    "VWAPSlopeSignal",
    "BollingerConfig",
    "BollingerSignal",
    "SMATrendConfig",
    "SMATrendSignal",
]
