"""SignalBundle and SignalEngine — compute signals once, share across strategies."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.core.events import BarEvent
from src.signals.base import UNAVAILABLE_RESULT, SignalBase, SignalResult
from src.signals.registry import SignalRegistry

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalBundle:
    """Frozen container of pre-computed signal results for a bar window.

    Strategies read values via accessor methods instead of maintaining
    their own calculators.
    """

    results: dict[str, SignalResult] = field(default_factory=dict)
    bar_count: int = 0

    def get(self, name: str) -> SignalResult | None:
        """Get a signal result by name, or None if not computed."""
        return self.results.get(name)

    def value(self, name: str, default: float = 0.0) -> float:
        """Get a signal's numeric value, with fallback default."""
        r = self.results.get(name)
        return r.value if r is not None else default

    def passes(self, name: str) -> bool:
        """Check if a signal passes (True if signal not found)."""
        r = self.results.get(name)
        return r.passes if r is not None else True

    def direction(self, name: str) -> str:
        """Get a signal's direction ('long', 'short', or 'none')."""
        r = self.results.get(name)
        return r.direction if r is not None else "none"

    def metadata(self, name: str) -> dict:
        """Get a signal's metadata dict (empty dict if not found)."""
        r = self.results.get(name)
        return r.metadata if r is not None else {}

    def has(self, name: str) -> bool:
        """Check if a signal was computed."""
        return name in self.results


# Empty singleton for strategies that don't use signals yet
EMPTY_BUNDLE = SignalBundle()


class SignalEngine:
    """Computes a set of signals for a bar window, returns a SignalBundle.

    Instantiated once at startup with a list of signal names.
    Each call to compute() runs all signals and returns results.
    """

    def __init__(
        self,
        signal_names: list[str],
        signal_configs: dict[str, dict] | None = None,
    ) -> None:
        self._signals: list[SignalBase] = []
        self._warned: set[str] = set()
        configs = signal_configs or {}
        for name in signal_names:
            kwargs = configs.get(name, {})
            self._signals.append(SignalRegistry.build(name, **kwargs))

    def compute(self, bars: list[BarEvent]) -> SignalBundle:
        """Compute all registered signals for the given bar window.

        Signals that require data not present in the bars (e.g. spread needs
        L1 bid/ask) are skipped and return UNAVAILABLE_RESULT.
        """
        if not bars:
            return EMPTY_BUNDLE

        results: dict[str, SignalResult] = {}
        for signal in self._signals:
            if not signal.has_required_data(bars):
                if signal.name not in self._warned:
                    _log.warning(
                        "Signal '%s' requires columns %s but bar data is missing "
                        "— returning unavailable. This warning is shown once.",
                        signal.name,
                        sorted(signal.required_columns),
                    )
                    self._warned.add(signal.name)
                results[signal.name] = UNAVAILABLE_RESULT
            else:
                results[signal.name] = signal.compute(bars)

        return SignalBundle(results=results, bar_count=len(bars))

    @property
    def signal_names(self) -> list[str]:
        """Names of all signals this engine computes."""
        return [s.name for s in self._signals]

    @property
    def requires_l1(self) -> bool:
        """True if any signal needs L1 enrichment (bid/ask prices)."""
        l1_fields = {"avg_bid_price", "avg_ask_price", "avg_bid_size", "avg_ask_size"}
        return any(s.required_columns & l1_fields for s in self._signals)

    @property
    def all_required_columns(self) -> frozenset[str]:
        """Union of all required columns across all signals."""
        cols: set[str] = set()
        for s in self._signals:
            cols |= s.required_columns
        return frozenset(cols)
