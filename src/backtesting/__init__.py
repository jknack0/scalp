"""Backtesting engine — bar-replay simulation with realistic fill modeling."""

from src.backtesting.decision_engine import (
    DecisionConfig,
    DecisionEngine,
    ValidationDecision,
    ValidationSummary,
)
from src.backtesting.cpcv import (
    CPCVConfig,
    CPCVFold,
    CPCVResult,
    CPCVValidator,
)
from src.backtesting.dsr import (
    DSRConfig,
    DSRResult,
    DeflatedSharpeCalculator,
)
from src.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    PendingOrder,
    SimulatedOMS,
)
from src.backtesting.metrics import (
    BacktestMetrics,
    BacktestResult,
    MetricsCalculator,
    Trade,
)
from src.backtesting.slippage import (
    SlippageResult,
    VolatilitySlippageModel,
)
from src.backtesting.wfa import (
    WFAConfig,
    WFACycle,
    WFAResult,
    WFARunner,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestMetrics",
    "BacktestResult",
    "CPCVConfig",
    "DecisionConfig",
    "DecisionEngine",
    "CPCVFold",
    "CPCVResult",
    "CPCVValidator",
    "DSRConfig",
    "DSRResult",
    "DeflatedSharpeCalculator",
    "MetricsCalculator",
    "PendingOrder",
    "SimulatedOMS",
    "SlippageResult",
    "Trade",
    "ValidationDecision",
    "ValidationSummary",
    "VolatilitySlippageModel",
    "WFAConfig",
    "WFACycle",
    "WFAResult",
    "WFARunner",
]
