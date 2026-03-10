"""Regime detector protocol.

Injected into strategies that need HMM regime awareness.
Concrete implementations live in src/models/hmm_regime.py or test mocks.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable


RegimeLabel = Literal["trending", "ranging", "volatile"]


@runtime_checkable
class RegimeDetector(Protocol):
    """Protocol for regime classification backends."""

    def current_regime(self) -> RegimeLabel:
        """Return the current market regime label."""
        ...
