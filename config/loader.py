"""Strategy configuration loader.

Loads strategy YAML configs from config/strategies/ and builds typed
dataclass instances with nested signal/filter configurations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_strategy_config(
    strategy_name: str,
    config_dir: str | Path = "config/strategies",
) -> dict[str, Any]:
    """Load a strategy YAML config by name.

    Args:
        strategy_name: Strategy identifier (e.g., "orb", "vwap_reversion").
        config_dir: Directory containing strategy YAML files.

    Returns:
        Parsed YAML as a dict.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    path = Path(config_dir) / f"{strategy_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Strategy config not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data



def build_signal_engine(config: dict[str, Any]) -> Any:
    """Build a SignalEngine from a strategy YAML config.

    Reads the top-level ``signals`` list:
        signals: [atr, vwap_session, relative_volume, spread]

    Returns a SignalEngine, or None if no signals declared.
    """
    from src.signals.signal_bundle import SignalEngine

    signal_names = config.get("signals", [])
    if not signal_names:
        return None
    signal_configs = config.get("signal_configs", {})
    return SignalEngine(signal_names, signal_configs)


def build_filter_engine(config: dict[str, Any]) -> Any:
    """Build a FilterEngine from a strategy YAML config.

    Reads the top-level ``filters`` list:
        filters:
          - signal: spread
            expr: "< 2.0"
            seq: 1
          - signal: spread
            expr: "< 2.0"
            seq: 2

    Returns a FilterEngine (may be empty if no filters declared).
    """
    from src.filters.filter_engine import FilterEngine

    filter_list = config.get("filters", [])
    return FilterEngine.from_list(filter_list)
