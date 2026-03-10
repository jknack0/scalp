"""Signal registry for config-driven construction."""

from __future__ import annotations

import inspect
from typing import Any

from src.signals.base import SignalBase


def _unwrap_optional(annotation: Any) -> type | None:
    """Extract the non-None type from Optional[X] or X | None."""
    import typing

    origin = getattr(annotation, "__origin__", None)
    if origin is typing.Union:
        args = [a for a in annotation.__args__ if a is not type(None)]
        return args[0] if len(args) == 1 else None
    # Python 3.10+ union syntax: X | None
    if hasattr(annotation, "__args__"):
        args = [a for a in annotation.__args__ if a is not type(None)]
        return args[0] if len(args) == 1 else None
    # Plain type
    if isinstance(annotation, type):
        return annotation
    return None


class SignalRegistry:
    """Global registry so filters can be built from config by name."""

    _registry: dict[str, type[SignalBase]] = {}

    @classmethod
    def register(cls, signal_class: type[SignalBase]) -> type[SignalBase]:
        """Decorator: @SignalRegistry.register"""
        cls._registry[signal_class.name] = signal_class
        return signal_class

    @classmethod
    def get(cls, name: str) -> type[SignalBase]:
        """Get signal class by name."""
        if name not in cls._registry:
            raise KeyError(
                f"Unknown signal: {name}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> SignalBase:
        """Instantiate a signal by name with config kwargs.

        Inspects the signal's __init__ to find its config parameter type,
        constructs the config dataclass from kwargs, then passes it.
        If no kwargs are given, uses default config.
        """
        signal_cls = cls.get(name)

        if not kwargs:
            return signal_cls()

        # Find the config parameter type from __init__ signature
        # Use get_type_hints to resolve string annotations (from __future__ import annotations)
        import typing
        try:
            hints = typing.get_type_hints(signal_cls.__init__)
        except Exception:
            hints = {}
        for param_name, annotation in hints.items():
            if param_name in ("self", "classifier", "return"):
                continue
            config_type = _unwrap_optional(annotation)
            if config_type is not None and hasattr(config_type, "__dataclass_fields__"):
                config = config_type(**kwargs)
                return signal_cls(config=config)

        # Fallback: pass kwargs directly
        return signal_cls(**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """List all registered signal names."""
        return list(cls._registry.keys())
