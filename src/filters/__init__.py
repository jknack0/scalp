"""Filters: declarative FilterEngine for signal gating."""

from src.filters.filter_engine import (
    FilterEngine,
    FilterResult,
    FilterRule,
    parse_rules,
)

__all__ = [
    "FilterEngine",
    "FilterResult",
    "FilterRule",
    "parse_rules",
]
