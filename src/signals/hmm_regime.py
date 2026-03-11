"""HMM regime classification signal.

Wraps the stateful HMMRegimeClassifier to produce a SignalResult.
Feeds bars incrementally — only new bars are ingested on each compute() call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.core.events import BarEvent
from src.models.hmm_regime import (
    HMMRegimeClassifier,
    NotReadyError,
    RegimeState,
)
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HMMRegimeConfig:
    pass_states: list[RegimeState] = field(default_factory=list)
    model_path: str = "models/hmm/v1"

    def __post_init__(self):
        # Convert string state names to RegimeState enums
        if self.pass_states and isinstance(self.pass_states[0], str):
            converted = [RegimeState[s] for s in self.pass_states]
            object.__setattr__(self, "pass_states", converted)


def _bar_to_dict(bar: BarEvent) -> dict:
    """Convert BarEvent to dict for the classifier's online API."""
    return {
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
    }


@SignalRegistry.register
class HMMRegimeSignal(SignalBase):
    """HMM regime classification signal (online, stateful).

    Feeds bars incrementally to the classifier. Only new bars (beyond what
    was previously ingested) are processed on each compute() call.
    """

    name = "hmm_regime"

    def __init__(
        self,
        config: HMMRegimeConfig | None = None,
        classifier: HMMRegimeClassifier | None = None,
    ) -> None:
        self.config = config or HMMRegimeConfig()
        self._classifier = classifier
        self._bars_fed: int = 0

        # Auto-load classifier from model_path if not injected
        if self._classifier is None and self.config.model_path:
            from pathlib import Path

            model_dir = Path(self.config.model_path)
            if (model_dir / "hmm_model.joblib").exists():
                self._classifier = HMMRegimeClassifier.load(model_dir)
                _logger.info(
                    "HMM classifier auto-loaded from %s", self.config.model_path
                )

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        # No classifier loaded — pass-through
        if self._classifier is None or self._classifier.model is None:
            return SignalResult(
                value=0.0,
                passes=True,
                direction="none",
                metadata={"reason": "no_classifier"},
            )

        if not bars:
            return SignalResult(
                value=0.0,
                passes=True,
                direction="none",
                metadata={"reason": "no_bars"},
            )

        # Feed only new bars (incremental)
        new_bars = bars[self._bars_fed :]
        state = None
        for bar in new_bars:
            try:
                state_int = self._classifier.predict(_bar_to_dict(bar))
                state = RegimeState(state_int)
            except NotReadyError:
                pass
            self._bars_fed += 1

        # Still warming up
        if state is None and self._classifier._current_regime is None:
            return SignalResult(
                value=0.0,
                passes=True,
                direction="none",
                metadata={
                    "reason": "warmup",
                    "bars_fed": self._bars_fed,
                },
            )

        # Use latest regime (either from this batch or previous)
        if state is None:
            state = self._classifier._current_regime

        # Check pass condition
        pass_states = self.config.pass_states
        if pass_states:
            passes = state in pass_states
        else:
            passes = True

        # Build probability dict
        proba = self._classifier.last_proba
        prob_dict = (
            {rs.name: float(proba[rs.value]) for rs in RegimeState}
            if proba is not None
            else {}
        )

        return SignalResult(
            value=float(state.value),
            passes=passes,
            direction="none",  # HMM is a gate, not directional
            metadata={
                "regime": state.name,
                "regime_value": state.value,
                "probabilities": prob_dict,
                "pass_states": [s.name for s in pass_states],
            },
        )
