"""Regime Detector V2 signal — 3-state HMM with T2 features + PCA.

Wraps RegimeDetectorV2 to produce a SignalResult compatible with
FilterEngine (passes gate) and ExitEngine (regime_exit via int value).

Feeds bars incrementally — only new bars are ingested on each compute() call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.core.events import BarEvent
from src.models.regime_detector_v2 import RegimeDetectorV2, RegimeLabel
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeV2Config:
    model_path: str = "models/regime_v2"
    pass_states: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Convert string state names to RegimeLabel enums for matching
        converted: list[RegimeLabel] = []
        for s in self.pass_states:
            if isinstance(s, str):
                converted.append(RegimeLabel[s])
            elif isinstance(s, int):
                converted.append(RegimeLabel(s))
            else:
                converted.append(s)
        object.__setattr__(self, "_pass_labels", converted)

    @property
    def pass_labels(self) -> list[RegimeLabel]:
        return self._pass_labels  # type: ignore[attr-defined]


def _bar_to_dict(bar: BarEvent) -> dict:
    """Convert BarEvent to dict for the detector's online API."""
    return {
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
    }


@SignalRegistry.register
class RegimeV2Signal(SignalBase):
    """Regime Detector V2 signal (online, stateful).

    Feeds bars incrementally to RegimeDetectorV2.update().
    Returns SignalResult with:
      - value: RegimeLabel int (0=TRENDING, 1=RANGING, 2=HIGH_VOL)
      - passes: True if current regime is in pass_states
      - metadata: probabilities, confidence, position_size, transition, whipsaw
    """

    name = "regime_v2"

    def __init__(
        self,
        config: RegimeV2Config | None = None,
        detector: RegimeDetectorV2 | None = None,
    ) -> None:
        self.config = config or RegimeV2Config()
        self._detector = detector
        self._bars_fed: int = 0

        # Auto-load detector from model_path if not injected
        if self._detector is None and self.config.model_path:
            from pathlib import Path

            model_dir = Path(self.config.model_path)
            if (model_dir / "regime_v2.joblib").exists():
                self._detector = RegimeDetectorV2.load(str(model_dir))
                _logger.info(
                    "RegimeDetectorV2 auto-loaded from %s", self.config.model_path
                )

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        # No detector loaded — pass-through
        if self._detector is None:
            return SignalResult(
                value=0.0,
                passes=True,
                direction="none",
                metadata={"reason": "no_detector"},
            )

        if not bars:
            return SignalResult(
                value=0.0,
                passes=True,
                direction="none",
                metadata={"reason": "no_bars"},
            )

        # Feed only new bars (incremental).
        # The bar window is a sliding window capped at 500 — old bars get
        # trimmed from the front, so _bars_fed can exceed len(bars).
        # When that happens, only the last bar (just appended) is new.
        if self._bars_fed >= len(bars):
            new_bars = bars[-1:]
        else:
            new_bars = bars[self._bars_fed:]
        proba = None
        for bar in new_bars:
            proba = self._detector.update(_bar_to_dict(bar))
            self._bars_fed += 1

        # Still warming up — detector returns None until enough bars
        if proba is None:
            return SignalResult(
                value=0.0,
                passes=True,  # Don't block during warmup
                direction="none",
                metadata={
                    "reason": "warmup",
                    "bars_fed": self._bars_fed,
                },
            )

        # Check pass condition
        pass_labels = self.config.pass_labels
        if pass_labels:
            passes = proba.regime in pass_labels
        else:
            passes = True

        # Also block on whipsaw halt — if detector says halt, don't trade
        if proba.whipsaw_halt:
            passes = False

        # Also block on flat position size — confidence too low
        if proba.position_size == "flat":
            passes = False

        # Build probability dict
        prob_dict = {
            label.name: float(proba.probabilities[label.value])
            for label in RegimeLabel
        }

        return SignalResult(
            value=float(proba.regime.value),
            passes=passes,
            direction="none",  # Regime is a gate, not directional
            metadata={
                "regime": proba.regime.name,
                "regime_value": proba.regime.value,
                "probabilities": prob_dict,
                "confidence": proba.confidence,
                "position_size": proba.position_size,
                "transition_signal": proba.transition_signal,
                "bars_in_regime": proba.bars_in_regime,
                "whipsaw_halt": proba.whipsaw_halt,
                "pass_states": [l.name for l in pass_labels],
            },
        )
