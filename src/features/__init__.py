"""Real-time feature calculators for MES scalping strategies.

Stateful classes that maintain running accumulators, updating incrementally
on each bar or tick. Suitable for both live trading and historical replay.

Classes:
    VWAPCalculator — Session VWAP with SD bands and first-kiss detection.
    ATRCalculator — Rolling ATR with volatility regime classification.
    CVDCalculator — Cumulative Volume Delta with trade classification.
    VolumeProfileTracker — Live developing profile + prior session reference.
    FeatureHub — Composition layer tying all calculators together.
    FeatureVector — Frozen snapshot of all feature values.
"""

from src.features.atr import ATRCalculator
from src.features.cvd import CVDCalculator
from src.features.feature_hub import FeatureHub, FeatureVector
from src.features.volume_profile import VolumeProfileTracker
from src.features.vwap import VWAPCalculator

__all__ = [
    "ATRCalculator",
    "CVDCalculator",
    "FeatureHub",
    "FeatureVector",
    "VolumeProfileTracker",
    "VWAPCalculator",
]
