"""Tests for Strategy Kill & Survivor decision engine.

All synthetic — manual ValidationSummary construction, no Parquet needed.
"""

import numpy as np
import pytest

from src.backtesting.decision_engine import (
    DecisionConfig,
    DecisionEngine,
    ValidationDecision,
    ValidationSummary,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_summary(
    strategy_id: str = "test_strategy",
    cpcv_pbo: float = 0.03,
    dsr: float = 0.98,
    wfa_efficiency: float = 0.65,
    wfa_is_oos_correlation: float = 0.40,
    param_stability_score: float = 0.80,
    total_oos_trades: int = 100,
    oos_sharpe: float = 1.2,
    oos_win_rate: float = 0.55,
    oos_profit_factor: float = 1.5,
) -> ValidationSummary:
    """Create a ValidationSummary with sensible defaults (all gates pass)."""
    return ValidationSummary(
        strategy_id=strategy_id,
        cpcv_pbo=cpcv_pbo,
        dsr=dsr,
        wfa_efficiency=wfa_efficiency,
        wfa_is_oos_correlation=wfa_is_oos_correlation,
        param_stability_score=param_stability_score,
        total_oos_trades=total_oos_trades,
        oos_sharpe=oos_sharpe,
        oos_win_rate=oos_win_rate,
        oos_profit_factor=oos_profit_factor,
    )


# ── Tests ────────────────────────────────────────────────────────────


def test_all_gates_pass_proceeds():
    """All metrics above thresholds → PROCEED, all passed_* True, no failures."""
    engine = DecisionEngine()
    summary = _make_summary()
    decision = engine.evaluate(summary, locked_params={"target_ticks": 8})

    assert decision.decision == "PROCEED"
    assert decision.passed_pbo is True
    assert decision.passed_dsr is True
    assert decision.passed_wfa is True
    assert decision.passed_stability is True
    assert decision.failure_modes == []
    assert decision.research_hypothesis == ""
    assert decision.locked_parameters == {"target_ticks": 8}


def test_pbo_fail_retires():
    """PBO above threshold → RETIRE with PBO failure mode."""
    engine = DecisionEngine()
    summary = _make_summary(cpcv_pbo=0.35)
    decision = engine.evaluate(summary)

    assert decision.decision == "RETIRE"
    assert decision.passed_pbo is False
    assert decision.passed_dsr is True
    assert decision.passed_wfa is True
    assert decision.passed_stability is True
    assert len(decision.failure_modes) == 1
    assert "PBO" in decision.failure_modes[0]


def test_dsr_fail_retires():
    """DSR below threshold → RETIRE with DSR failure mode."""
    engine = DecisionEngine()
    summary = _make_summary(dsr=0.50)
    decision = engine.evaluate(summary)

    assert decision.decision == "RETIRE"
    assert decision.passed_dsr is False
    assert decision.passed_pbo is True
    assert len(decision.failure_modes) == 1
    assert "DSR" in decision.failure_modes[0]


def test_wfa_fail_retires():
    """WFA efficiency below threshold → RETIRE with WFA failure mode."""
    engine = DecisionEngine()
    summary = _make_summary(wfa_efficiency=0.30)
    decision = engine.evaluate(summary)

    assert decision.decision == "RETIRE"
    assert decision.passed_wfa is False
    assert decision.passed_pbo is True
    assert decision.passed_dsr is True
    assert len(decision.failure_modes) == 1
    assert "WFA" in decision.failure_modes[0]


def test_stability_fail_retires():
    """Parameter stability below threshold → RETIRE."""
    engine = DecisionEngine()
    summary = _make_summary(param_stability_score=0.40)
    decision = engine.evaluate(summary)

    assert decision.decision == "RETIRE"
    assert decision.passed_stability is False
    assert len(decision.failure_modes) == 1
    assert "stability" in decision.failure_modes[0].lower()


def test_param_stability_score_calculation():
    """Test compute_param_stability() directly with known inputs."""
    # All identical values → CV=0 → score=1.0
    drift_identical = {"p1": [2.0, 2.0, 2.0], "p2": [5.0, 5.0, 5.0]}
    assert DecisionEngine.compute_param_stability(drift_identical) == 1.0

    # Known CV: mean=2, std=1 → CV=0.5 → score ≈ 0.5
    drift_known = {"p1": [1.0, 2.0, 3.0]}  # mean=2, std=1, CV=0.5
    score = DecisionEngine.compute_param_stability(drift_known)
    assert abs(score - 0.5) < 0.01

    # Empty dict → 1.0
    assert DecisionEngine.compute_param_stability({}) == 1.0


def test_high_correlation_retires_weaker():
    """Two PROCEED strategies with correlated P&L → weaker one retired."""
    engine = DecisionEngine()

    # Both pass all gates
    summary_a = _make_summary(strategy_id="strong", dsr=0.99)
    summary_b = _make_summary(strategy_id="weak", dsr=0.96)

    # Daily P&L: highly correlated (r ≈ 0.99)
    np.random.seed(42)
    base = np.random.randn(100)
    pnl_a = base + np.random.randn(100) * 0.1
    pnl_b = base + np.random.randn(100) * 0.1

    decisions = engine.evaluate_all(
        summaries=[summary_a, summary_b],
        locked_params_map={
            "strong": {"target": 8},
            "weak": {"target": 6},
        },
        daily_pnl_map={"strong": pnl_a, "weak": pnl_b},
    )

    decision_map = {d.strategy_id: d for d in decisions}

    # Strong survives, weak retired due to correlation
    assert decision_map["strong"].decision == "PROCEED"
    assert decision_map["weak"].decision == "RETIRE"
    assert any("correlation" in fm.lower() for fm in decision_map["weak"].failure_modes)
