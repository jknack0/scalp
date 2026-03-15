# MES Scalping Bot — Strategy Design Prompt

You are designing a new trading strategy for an MES (micro E-mini S&P 500 futures) scalping bot. The bot is event-driven async Python running on Tradovate. You must output:

1. A **YAML config file** (`config/strategies/<strategy_id>.yaml`)
2. A **Python strategy class** (`src/strategies/<strategy_id>.py`)
3. Any **new signal classes** (`src/signals/<signal_name>.py`) if the strategy needs indicators not already available

Follow the exact conventions below. Do NOT deviate from this structure.

---

## Instrument

- **MES**: Micro E-mini S&P 500 futures
- Tick size: 0.25 index points
- Tick value: $1.25 ($5 multiplier × 0.25)
- Commission: $0.295/side ($0.59 round trip)
- RTH session: 9:30 AM – 4:00 PM Eastern

---

## YAML Config Structure

Every strategy YAML has these top-level sections:

```yaml
# config/strategies/<strategy_id>.yaml

strategy:
  strategy_id: my_strategy          # unique snake_case id
  max_signals_per_day: 5            # daily signal cap
  session_start: "09:30"            # Eastern time
  session_end: "16:00"

bar:
  type: time                        # "time" or "dollar"
  freq: 5m                          # time bars: 1s, 5s, 1m, 5m, 15m, 1h
  enrich: false                     # true if signals need L1 bid/ask data

# Signals to pre-compute on each bar window (list of registered signal names)
signals: [atr, vwap_session, rsi_momentum, adx, relative_volume, session_time]

# Override default signal parameters (optional)
signal_configs:
  rsi_momentum:
    period: 14
    long_threshold: 30.0
    short_threshold: 70.0
  adx:
    period: 14
    threshold: 25.0

# Declarative entry filters — ALL must pass before strategy.on_bar() runs
filters:
  # Simple value comparison
  - signal: adx
    expr: "< 25.0"

  # Metadata field access
  - signal: vwap_session
    field: deviation_sd
    expr: "abs >= 2.0"              # "abs" prefix takes absolute value

  # Time-of-day filter (minutes since midnight ET)
  - signal: session_time
    expr: ">= 585"                  # 9:45 AM = 9*60+45 = 585

  # Boolean pass/fail
  - signal: hmm_regime
    expr: passes

# Exit geometry
exit:
  target:
    type: vwap                      # see "Exit Types" below
    # multiplier: 1.0               # used by atr_multiple, or_width, sd_band
    # ticks: 8                      # used by fixed_ticks
  stop:
    type: atr_multiple
    multiplier: 2.0
  time_stop_minutes: 15             # close trade after N minutes if no fill
  # session_close: "15:55"          # optional hard session exit time

  # Optional: early exit conditions (OR logic — any one triggers exit)
  early_exit:
    - type: vwap_slope
      threshold: 0.3
    - type: adverse_momentum
      bars: 2
      atr_multiple: 1.0
    - type: rsi_failure
      bars: 3
      long_min: 25.0
      short_max: 75.0

# Strategy-specific params go in their own section
# Example for a hypothetical breakout strategy:
# breakout:
#   lookback_bars: 20
#   min_range_ticks: 8
```

### Available Signals (built-in)

These are already registered in `src/signals/` and can be used directly in the YAML `signals:` array:

| Signal Name | What It Computes | Key Metadata Fields |
|---|---|---|
| `atr` | Average True Range | `.metadata["atr_raw"]` |
| `adx` | Average Directional Index | `.value` = ADX value |
| `rsi_momentum` | RSI with configurable period | `.value` = RSI (0-100) |
| `vwap_session` | Session VWAP + bands | `.metadata["vwap", "sd", "deviation_sd", "slope", "session_age_bars"]` |
| `relative_volume` | Current vol vs average | `.value` = ratio (1.0 = average) |
| `session_time` | Minutes since midnight ET | `.value` = minutes |
| `hmm_regime` | HMM regime classifier | `.passes` = True if in allowed state |
| `spread` | Bid-ask spread (needs L1) | `.value` = spread |
| `vwap_deviation` | Price deviation from VWAP | `.value` = deviation |
| `vwap_slope` | VWAP trend slope | `.value` = slope |
| `vwap_bias` | VWAP directional bias | `.direction` = "long"/"short" |
| `bollinger` | Bollinger Bands | `.metadata` |
| `sma_trend` | SMA trend direction | `.direction` = "long"/"short" |
| `ema_crossover` | EMA crossover signal | `.direction`, `.passes` |
| `cvd_divergence` | Cumulative volume delta | `.value`, `.direction` |
| `volume_exhaustion` | Volume exhaustion signal | `.passes`, `.value` |
| `prior_day_bias` | Prior session bias | `.direction` |
| `poc_distance` | Distance from point of control | `.value` |
| `orb_range_size` | Opening range bar size | `.value` |
| `orb_breakout` | ORB breakout detection | `.passes`, `.direction` |
| `vpin` | Volume-sync informed trading | `.value` |

### Creating New Signals

If your strategy needs an indicator not listed above, create a new signal class in `src/signals/`. New signals must be **established, well-known technical indicators** (e.g. Keltner Channels, MACD, Stochastic, OBV, MFI, Donchian Channels, etc.) — do not invent novel indicators.

Every signal follows this pattern:

```python
# src/signals/<signal_name>.py

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.events import BarEvent
from src.signals.base import SignalBase, SignalResult
from src.signals.registry import SignalRegistry


@dataclass(frozen=True)
class MySignalConfig:
    """Frozen config dataclass — fields match signal_configs YAML keys."""
    period: int = 20
    threshold: float = 1.5


@SignalRegistry.register
class MySignal(SignalBase):
    """One-line description of the signal."""

    name = "my_signal"  # must match the YAML signals list entry

    # Declare required BarEvent fields beyond OHLCV (leave empty if OHLCV-only)
    # required_columns = frozenset({"avg_bid_price", "avg_ask_price"})  # L1 example

    def __init__(self, config: MySignalConfig | None = None) -> None:
        self.config = config or MySignalConfig()

    def compute(self, bars: list[BarEvent]) -> SignalResult:
        """Compute signal from bar history (oldest first).

        Args:
            bars: List of BarEvent objects, oldest first.

        Returns:
            SignalResult with:
              - value: primary numeric output (float)
              - passes: True/False gate for filters
              - direction: "long", "short", or "none"
              - metadata: dict of additional computed values
        """
        if len(bars) < self.config.period:
            return SignalResult(value=0.0, passes=False, direction="none",
                                metadata={"reason": "insufficient_bars"})

        # Compute the indicator from bars...
        closes = np.array([b.close for b in bars], dtype=np.float64)
        computed_value = float(np.mean(closes[-self.config.period:]))  # example

        return SignalResult(
            value=computed_value,
            passes=True,
            direction="none",
            metadata={"raw_value": computed_value},
        )
```

**Key rules for new signals:**
- Decorate with `@SignalRegistry.register` — this auto-registers by the `name` class attribute
- Config is a frozen `@dataclass` — field names must match YAML `signal_configs` keys
- `__init__` takes `config: MyConfig | None = None` — the registry auto-constructs it from YAML kwargs
- `compute()` receives `bars: list[BarEvent]` (oldest first) and returns a `SignalResult`
- Set `required_columns` class var if the signal needs L1 fields (e.g. `avg_bid_price`)
- Use numpy for vectorized math, not loops over bars
- The signal module must be imported somewhere to trigger registration (add an import in `src/signals/__init__.py`)

### Filter Expression Syntax

```
"< 2.0"           # less than
"> 1.5"           # greater than
"<= 10.0"         # less than or equal
">= 585"          # greater than or equal
"== 1.0"          # equals
"!= 3"            # not equals
"abs >= 2.0"      # absolute value, then compare
"passes"           # signal's .passes field must be True
```

Filters can reference `.value` (default) or a metadata field via `field: "key_name"`.

### Exit Target Types

| Type | Description | Uses |
|---|---|---|
| `fixed_ticks` | Entry ± (ticks × 0.25) | `ticks` param |
| `atr_multiple` | Entry ± (ATR × multiplier) | `multiplier` param |
| `vwap` | Target = current VWAP | — |
| `sd_band` | Target = VWAP ± (SD × mult) | `multiplier` param |
| `or_width` | Entry ± (OR width × mult) | `multiplier` param (ORB only) |

### Exit Stop Types

| Type | Description | Uses |
|---|---|---|
| `fixed_ticks` | Entry ∓ (ticks × 0.25) | `ticks` param |
| `atr_multiple` | Entry ∓ (ATR × multiplier) | `multiplier` param |
| `first_break` | First break extreme ± buffer | `buffer_ticks` param (ORB only) |
| `or_width` | Entry ∓ (OR width × mult) | `multiplier` param (ORB only) |
| `sd_band` | VWAP ∓ (SD × multiplier) | `multiplier` param |

---

## Python Strategy Class

Every strategy is a standalone class (no inheritance required) that duck-types two methods: `on_bar()` and `reset()`.

### Required Interface

```python
class MyStrategy:
    def __init__(self, config: dict[str, Any]) -> None:
        """Parse YAML config dict."""
        ...

    @classmethod
    def from_yaml(cls, path: str) -> MyStrategy:
        """Construct from YAML file path."""
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg)

    def on_bar(self, bar: BarEvent, bundle: SignalBundle = EMPTY_BUNDLE) -> Signal | None:
        """Process one bar. Return a Signal to trade, or None.

        FilterEngine has ALREADY run before this is called.
        Only put logic here that can't be expressed as a declarative filter
        (e.g. direction-dependent checks, state machines, multi-bar patterns).
        """
        ...

    def reset(self) -> None:
        """Reset daily state. Called at session open."""
        ...

    # Optional: for strategies with early exit logic
    def check_early_exit(
        self, bar: BarEvent, bundle: SignalBundle,
        bars_in_trade: int, direction: Direction, fill_price: float
    ) -> str | None:
        """Return exit reason string or None. Called each bar while in a position."""
        ...
```

### Key Imports

```python
from src.core.events import BarEvent
from src.core.logging import get_logger
from src.exits.exit_builder import ExitBuilder, ExitContext
from src.filters.filter_engine import FilterEngine
from src.signals.signal_bundle import EMPTY_BUNDLE, SignalBundle
from src.strategies.base import Direction, Signal
from src.models.hmm_regime import RegimeState
```

### BarEvent Fields

```python
@dataclass(frozen=True, slots=True)
class BarEvent:
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    bar_type: str           # "1s", "1m", "5m", "dollar", etc.
    timestamp_ns: int       # nanosecond epoch
    # L1 fields (0.0 when not enriched)
    avg_bid_size: float
    avg_ask_size: float
    avg_bid_price: float
    avg_ask_price: float
    aggressive_buy_vol: float
    aggressive_sell_vol: float
```

### Signal Dataclass (what on_bar returns)

```python
@dataclass(frozen=True)
class Signal:
    strategy_id: str
    direction: Direction          # Direction.LONG or Direction.SHORT
    entry_price: float
    target_price: float
    stop_price: float
    signal_time: datetime
    expiry_time: datetime         # time stop — OMS closes position after this
    confidence: float             # 0.0 to 1.0
    regime_state: RegimeState
    metadata: dict                # strategy-specific debug info
    id: str                       # auto-generated UUID
```

### SignalBundle (pre-computed indicators)

```python
bundle.get("atr")              # -> SignalResult | None
bundle.value("adx")            # -> float (0.0 if missing)
bundle.passes("hmm_regime")    # -> bool (True if missing)
bundle.metadata("vwap_session") # -> dict

# SignalResult fields:
#   .value: float
#   .passes: bool
#   .direction: "long" | "short" | "none"
#   .metadata: dict
```

### ExitBuilder Usage (for YAML-driven exits)

```python
# In __init__:
self._exit_builder = ExitBuilder.from_yaml(config.get("exit", {}))

# In on_bar:
ctx = ExitContext(
    entry_price=bar.close,
    direction=direction.value,     # "LONG" or "SHORT"
    atr=atr_raw,                   # from bundle.get("atr").metadata["atr_raw"]
    vwap=vwap,                     # from bundle.get("vwap_session").metadata["vwap"]
    vwap_sd=sd,                    # from bundle.get("vwap_session").metadata["sd"]
    # or_width=...,                # ORB strategies only
    # first_break_extreme=...,     # ORB strategies only
)
geo = self._exit_builder.compute(ctx)
target = geo.target_price
stop = geo.stop_price
```

### FilterEngine Usage (for YAML-driven filters)

```python
# In __init__:
self._filter_engine = FilterEngine.from_list(config.get("filters"))

# In on_bar:
filter_result = self._filter_engine.evaluate(bundle)
if not filter_result.passes:
    return None  # blocked by declarative filters
```

---

## Mandatory Rules

### Signal Geometry (ALWAYS validate before returning a Signal)

```python
# LONG: stop < entry < target
if direction == Direction.LONG:
    if not (stop < entry_price < target):
        return None  # invalid geometry

# SHORT: stop > entry > target
if direction == Direction.SHORT:
    if not (stop > entry_price > target):
        return None  # invalid geometry
```

### Sanity Checks

- If win rate > 90% or Sharpe > 10 in backtest, something is broken — investigate
- Always draw the number line with concrete values before coding entry/target/stop
- Test both LONG and SHORT directions

### Style

- Python 3.13, use `from __future__ import annotations`
- Type hints on all function signatures
- Use `structlog` via `get_logger("strategy_name")`
- Frozen dataclasses for immutable state
- `zoneinfo.ZoneInfo("US/Eastern")` for timezone handling
- All prices in index points (not dollars)
- Tick-align prices to 0.25 increments when needed

---

## Example: Complete Strategy (VWAP Band Reversion)

### YAML (`config/strategies/vwap_band_reversion.yaml`)

```yaml
strategy:
  strategy_id: vwap_band_reversion
  max_signals_per_day: 999
  session_start: "09:30"
  session_end: "16:00"

bar:
  type: time
  freq: 5m
  enrich: false

signals: [vwap_session, rsi_momentum, adx, atr, relative_volume, session_time]
signal_configs:
  rsi_momentum:
    period: 2
    long_threshold: 30.0
    short_threshold: 70.0
  adx:
    period: 14
    threshold: 25.0

filters:
  - signal: session_time
    expr: ">= 585"
  - signal: session_time
    expr: "<= 900"
  - signal: vwap_session
    field: session_age_bars
    expr: ">= 30"
  - signal: vwap_session
    field: slope
    expr: "abs <= 0.5"
  - signal: vwap_session
    field: deviation_sd
    expr: "abs >= 2.5"
  - signal: adx
    expr: "< 25.0"
  - signal: relative_volume
    expr: ">= 0.5"

exit:
  target:
    type: vwap
  stop:
    type: atr_multiple
    multiplier: 2.0
  time_stop_minutes: 15
  early_exit:
    - type: vwap_slope
      threshold: 0.3
```

### Python (`src/strategies/vwap_band_reversion.py`)

```python
"""VWAP Band Reversion — mean reversion to VWAP on >= 2.5 SD deviation."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import yaml

from src.core.events import BarEvent
from src.core.logging import get_logger
from src.exits.exit_builder import ExitBuilder, ExitContext
from src.filters.filter_engine import FilterEngine
from src.models.hmm_regime import RegimeState
from src.signals.signal_bundle import EMPTY_BUNDLE, SignalBundle
from src.strategies.base import Direction, Signal
from zoneinfo import ZoneInfo

logger = get_logger("vwap_band_reversion")
_ET = ZoneInfo("US/Eastern")
TICK_SIZE = 0.25


class VWAPBandReversionStrategy:
    def __init__(self, config: dict[str, Any]) -> None:
        strat = config.get("strategy", {})
        self.strategy_id = strat.get("strategy_id", "vwap_band_reversion")
        self._max_signals_per_day = strat.get("max_signals_per_day", 3)

        sig_cfgs = config.get("signal_configs", {})
        rsi_cfg = sig_cfgs.get("rsi_momentum", {})
        self._rsi_long_max = rsi_cfg.get("long_threshold", 20.0)
        self._rsi_short_min = rsi_cfg.get("short_threshold", 80.0)

        exit_cfg = config.get("exit", {})
        self._exit_builder = ExitBuilder.from_yaml(exit_cfg)
        self._time_stop_minutes = exit_cfg.get("time_stop_minutes", 30)

        self._filter_engine = FilterEngine.from_list(config.get("filters"))
        self._signals_today = 0
        self._current_regime = RegimeState.LOW_VOL_RANGE

    @classmethod
    def from_yaml(cls, path: str) -> VWAPBandReversionStrategy:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(config=cfg)

    def on_bar(self, bar: BarEvent, bundle: SignalBundle = EMPTY_BUNDLE) -> Signal | None:
        now = datetime.fromtimestamp(bar.timestamp_ns / 1e9, tz=_ET)

        if self._signals_today >= self._max_signals_per_day:
            return None

        # Declarative filters already checked — but we run them here too
        filter_result = self._filter_engine.evaluate(bundle)
        if not filter_result.passes:
            return None

        # VWAP data for direction + exits
        vwap_result = bundle.get("vwap_session")
        if vwap_result is None:
            return None
        meta = vwap_result.metadata
        vwap = meta.get("vwap", 0.0)
        sd = meta.get("sd", 0.0)
        deviation_sd = meta.get("deviation_sd", 0.0)

        if vwap == 0.0 or sd == 0.0:
            return None

        # Direction from VWAP deviation (price below VWAP = long, above = short)
        direction = Direction.LONG if deviation_sd < 0 else Direction.SHORT

        # RSI directional check (can't be a simple filter — depends on direction)
        rsi_result = bundle.get("rsi_momentum")
        if rsi_result is not None:
            rsi = rsi_result.value
            if direction == Direction.LONG and rsi > self._rsi_long_max:
                return None
            if direction == Direction.SHORT and rsi < self._rsi_short_min:
                return None

        # Compute exits via ExitBuilder
        atr_raw = 0.0
        atr_result = bundle.get("atr")
        if atr_result is not None:
            atr_raw = atr_result.metadata.get("atr_raw", 0.0)

        entry_price = bar.close
        ctx = ExitContext(
            entry_price=entry_price,
            direction=direction.value,
            atr=atr_raw,
            vwap=vwap,
            vwap_sd=sd,
        )
        geo = self._exit_builder.compute(ctx)
        target = geo.target_price
        stop = geo.stop_price

        # Geometry sanity check
        if direction == Direction.LONG and not (stop < entry_price < target):
            return None
        if direction == Direction.SHORT and not (stop > entry_price > target):
            return None

        expiry = now + timedelta(minutes=self._time_stop_minutes)
        confidence = min(0.6 + abs(deviation_sd - 2.0) * 0.1, 0.9)

        signal = Signal(
            strategy_id=self.strategy_id,
            direction=direction,
            entry_price=entry_price,
            target_price=target,
            stop_price=stop,
            signal_time=now,
            expiry_time=expiry,
            confidence=confidence,
            regime_state=self._current_regime,
            metadata={"deviation_sd": deviation_sd, "vwap": vwap, "atr": atr_raw},
        )
        self._signals_today += 1
        logger.info("signal_generated", direction=direction.value,
                     entry=entry_price, target=round(target, 2),
                     stop=round(stop, 2), signal_id=signal.id)
        return signal

    def reset(self) -> None:
        self._signals_today = 0
```

---

## Your Task

Design a new MES scalping strategy. Provide:

1. **Thesis** — 2-3 sentences on the market behavior being exploited
2. **YAML config** — complete, valid, ready to save to `config/strategies/`
3. **Python strategy class** — complete, valid, ready to save to `src/strategies/`
4. **New signal classes** (if needed) — only for established, well-known technical indicators not already in the built-in list. Each signal goes in `src/signals/<name>.py` following the pattern above.
5. **Number line example** — concrete LONG and SHORT examples with real prices proving geometry is correct:
   - LONG: `entry=5600, target=5610, stop=5590 → 5590 < 5600 < 5610 ✓`
   - SHORT: `entry=5600, target=5590, stop=5610 → 5610 > 5600 > 5590 ✓`

Use built-in signals when possible. Only create new signals for established indicators (MACD, Stochastic, Keltner, OBV, MFI, Donchian, etc.) — do not invent novel indicators.
