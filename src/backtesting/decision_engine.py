"""Strategy Kill & Survivor decision engine.

Aggregates CPCV, DSR, and WFA validation results for each strategy. Applies
four independent gates (PBO, DSR, WFA efficiency, parameter stability). All
must pass for PROCEED; any failure triggers RETIRE. A second pass retires
correlated survivors (daily P&L correlation > threshold) keeping the stronger.

Generates a markdown validation report and locked-params YAML for Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import yaml

from src.backtesting.cpcv import CPCVResult
from src.backtesting.dsr import DSRResult
from src.backtesting.wfa import WFAResult


@dataclass
class DecisionConfig:
    """Thresholds for the four validation gates."""

    pbo_threshold: float = 0.10
    dsr_threshold: float = 0.95
    wfa_threshold: float = 0.50
    stability_threshold: float = 0.65
    correlation_threshold: float = 0.70


@dataclass(frozen=True)
class ValidationSummary:
    """Aggregated validation metrics for one strategy."""

    strategy_id: str
    cpcv_pbo: float
    dsr: float
    wfa_efficiency: float
    wfa_is_oos_correlation: float
    param_stability_score: float
    total_oos_trades: int
    oos_sharpe: float
    oos_win_rate: float
    oos_profit_factor: float


@dataclass(frozen=True)
class ValidationDecision:
    """PROCEED / RETIRE decision for one strategy."""

    strategy_id: str
    decision: str  # "PROCEED" or "RETIRE"
    passed_pbo: bool
    passed_dsr: bool
    passed_wfa: bool
    passed_stability: bool
    failure_modes: list[str]
    research_hypothesis: str
    locked_parameters: dict
    correlation_with_survivors: dict


# Map failure gate → research suggestion
_HYPOTHESIS_MAP = {
    "PBO": "Reduce parameter count or simplify entry conditions to lower overfitting risk.",
    "DSR": "Increase sample size or reduce strategy variants tested to improve deflated Sharpe.",
    "WFA": "Narrow parameter grid or switch to adaptive parameters for better OOS transfer.",
    "stability": "Anchor key parameters or use regime-conditional defaults to reduce drift.",
}


class DecisionEngine:
    """Evaluates strategies against validation gates and generates reports."""

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self._config = config or DecisionConfig()

    def evaluate(
        self,
        summary: ValidationSummary,
        locked_params: dict | None = None,
    ) -> ValidationDecision:
        """Apply four independent gates to a single strategy.

        All must pass for PROCEED. Returns a ValidationDecision with
        per-gate results, failure descriptions, and research hints.
        """
        cfg = self._config

        passed_pbo = summary.cpcv_pbo < cfg.pbo_threshold
        passed_dsr = summary.dsr >= cfg.dsr_threshold
        passed_wfa = summary.wfa_efficiency >= cfg.wfa_threshold
        passed_stability = summary.param_stability_score >= cfg.stability_threshold

        failure_modes: list[str] = []
        if not passed_pbo:
            failure_modes.append(
                f"PBO {summary.cpcv_pbo:.2f} >= {cfg.pbo_threshold:.2f} — "
                "high probability of backtest overfitting"
            )
        if not passed_dsr:
            failure_modes.append(
                f"DSR {summary.dsr:.2f} < {cfg.dsr_threshold:.2f} — "
                "Sharpe not significant after multiple-testing correction"
            )
        if not passed_wfa:
            failure_modes.append(
                f"WFA efficiency {summary.wfa_efficiency:.2f} < {cfg.wfa_threshold:.2f} — "
                "poor out-of-sample transfer"
            )
        if not passed_stability:
            failure_modes.append(
                f"Param stability {summary.param_stability_score:.2f} < "
                f"{cfg.stability_threshold:.2f} — excessive parameter drift"
            )

        decision = "PROCEED" if not failure_modes else "RETIRE"

        # Build research hypothesis from failure pattern
        hypothesis = ""
        if failure_modes:
            hints = []
            if not passed_pbo:
                hints.append(_HYPOTHESIS_MAP["PBO"])
            if not passed_dsr:
                hints.append(_HYPOTHESIS_MAP["DSR"])
            if not passed_wfa:
                hints.append(_HYPOTHESIS_MAP["WFA"])
            if not passed_stability:
                hints.append(_HYPOTHESIS_MAP["stability"])
            hypothesis = " ".join(hints)

        return ValidationDecision(
            strategy_id=summary.strategy_id,
            decision=decision,
            passed_pbo=passed_pbo,
            passed_dsr=passed_dsr,
            passed_wfa=passed_wfa,
            passed_stability=passed_stability,
            failure_modes=failure_modes,
            research_hypothesis=hypothesis,
            locked_parameters=locked_params or {} if decision == "PROCEED" else {},
            correlation_with_survivors={},
        )

    def evaluate_all(
        self,
        summaries: list[ValidationSummary],
        locked_params_map: dict[str, dict] | None = None,
        daily_pnl_map: dict[str, np.ndarray] | None = None,
    ) -> list[ValidationDecision]:
        """Evaluate all strategies, then apply correlation filter among survivors.

        1. Run evaluate() on each summary independently.
        2. If daily_pnl_map is provided, compute pairwise correlation among
           PROCEED strategies. If any pair exceeds correlation_threshold,
           retire the one with lower DSR.
        3. Return final decisions.
        """
        params_map = locked_params_map or {}

        # First pass: individual gate evaluation
        decisions = [
            self.evaluate(s, params_map.get(s.strategy_id))
            for s in summaries
        ]

        # Second pass: correlation filter
        if daily_pnl_map is None:
            return decisions

        survivors = [d for d in decisions if d.decision == "PROCEED"]
        if len(survivors) < 2:
            return decisions

        # Build lookup: strategy_id → summary (for DSR comparison)
        summary_map = {s.strategy_id: s for s in summaries}

        # Compute pairwise correlation among survivors
        corr_matrix = self.compute_correlation(
            {sid: daily_pnl_map[sid] for sid in [d.strategy_id for d in survivors]
             if sid in daily_pnl_map}
        )

        # Find pairs exceeding threshold
        to_retire: set[str] = set()
        for (sid_a, sid_b), corr_val in corr_matrix.items():
            if corr_val > self._config.correlation_threshold:
                dsr_a = summary_map[sid_a].dsr
                dsr_b = summary_map[sid_b].dsr
                weaker = sid_b if dsr_a >= dsr_b else sid_a
                to_retire.add(weaker)

        # Rebuild decisions with correlation info and retirements
        final: list[ValidationDecision] = []
        for d in decisions:
            # Add correlation info for survivors
            corr_info: dict[str, float] = {}
            for (sid_a, sid_b), corr_val in corr_matrix.items():
                if sid_a == d.strategy_id:
                    corr_info[sid_b] = corr_val
                elif sid_b == d.strategy_id:
                    corr_info[sid_a] = corr_val

            if d.strategy_id in to_retire:
                # Retire due to correlation
                failure_modes = list(d.failure_modes) + [
                    f"Retired due to high correlation (>{self._config.correlation_threshold:.2f}) "
                    "with a stronger survivor"
                ]
                final.append(ValidationDecision(
                    strategy_id=d.strategy_id,
                    decision="RETIRE",
                    passed_pbo=d.passed_pbo,
                    passed_dsr=d.passed_dsr,
                    passed_wfa=d.passed_wfa,
                    passed_stability=d.passed_stability,
                    failure_modes=failure_modes,
                    research_hypothesis="Highly correlated with a stronger strategy. "
                    "Consider differentiating entry signals or trading different sessions.",
                    locked_parameters={},
                    correlation_with_survivors=corr_info,
                ))
            else:
                # Keep with correlation info attached
                final.append(ValidationDecision(
                    strategy_id=d.strategy_id,
                    decision=d.decision,
                    passed_pbo=d.passed_pbo,
                    passed_dsr=d.passed_dsr,
                    passed_wfa=d.passed_wfa,
                    passed_stability=d.passed_stability,
                    failure_modes=d.failure_modes,
                    research_hypothesis=d.research_hypothesis,
                    locked_parameters=d.locked_parameters,
                    correlation_with_survivors=corr_info,
                ))

        return final

    @staticmethod
    def compute_correlation(
        daily_pnl_map: dict[str, np.ndarray],
    ) -> dict[tuple[str, str], float]:
        """Compute pairwise correlation of daily P&L arrays.

        Truncates to shorter array length if lengths differ.
        Returns {(id_a, id_b): correlation} for all unique pairs.
        """
        ids = sorted(daily_pnl_map.keys())
        result: dict[tuple[str, str], float] = {}

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = daily_pnl_map[ids[i]]
                b = daily_pnl_map[ids[j]]
                min_len = min(len(a), len(b))
                if min_len < 2:
                    result[(ids[i], ids[j])] = 0.0
                    continue
                corr = np.corrcoef(a[:min_len], b[:min_len])
                val = float(corr[0, 1])
                result[(ids[i], ids[j])] = 0.0 if np.isnan(val) else val

        return result

    @staticmethod
    def compute_param_stability(param_drift: dict[str, list]) -> float:
        """Compute parameter stability score from WFA param drift.

        For each parameter: CV = std / |mean|.
        Score = 1 - mean(CVs), clamped to [0, 1].
        Empty drift or all-identical values → 1.0.
        """
        if not param_drift:
            return 1.0

        cvs: list[float] = []
        for values in param_drift.values():
            # Filter out None and non-numeric values (e.g. time strings)
            numeric = []
            for v in values:
                if v is None:
                    continue
                try:
                    numeric.append(float(v))
                except (TypeError, ValueError):
                    continue
            if len(numeric) < 2:
                cvs.append(0.0)
                continue
            arr = np.array(numeric, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1))
            if abs(mean) < 1e-15:
                # If mean is ~0 but there's variance, treat as unstable
                cvs.append(std if std > 1e-15 else 0.0)
            else:
                cvs.append(std / abs(mean))

        if not cvs:
            return 1.0

        score = 1.0 - float(np.mean(cvs))
        return max(0.0, min(1.0, score))

    @staticmethod
    def from_results(
        cpcv: CPCVResult,
        dsr: DSRResult,
        wfa: WFAResult,
    ) -> ValidationSummary:
        """Build a ValidationSummary from CPCV, DSR, and WFA results."""
        stability = DecisionEngine.compute_param_stability(wfa.param_drift)

        # Aggregate OOS metrics from WFA cycles
        total_oos_trades = sum(c.oos_trades for c in wfa.cycles)
        cycles_with_trades = [c for c in wfa.cycles if c.oos_trades > 0]
        if cycles_with_trades:
            # Weighted average by trade count
            total = sum(c.oos_trades for c in cycles_with_trades)
            avg_win_rate = sum(c.oos_win_rate * c.oos_trades for c in cycles_with_trades) / total
            avg_pf = sum(c.oos_profit_factor * c.oos_trades for c in cycles_with_trades) / total
        else:
            avg_win_rate = 0.0
            avg_pf = 0.0

        return ValidationSummary(
            strategy_id=cpcv.strategy_id,
            cpcv_pbo=cpcv.pbo,
            dsr=dsr.dsr,
            wfa_efficiency=wfa.efficiency_ratio,
            wfa_is_oos_correlation=wfa.is_oos_correlation,
            param_stability_score=stability,
            total_oos_trades=total_oos_trades,
            oos_sharpe=wfa.avg_oos_sharpe,
            oos_win_rate=avg_win_rate,
            oos_profit_factor=avg_pf,
        )

    def generate_report(
        self,
        decisions: list[ValidationDecision],
        summaries: list[ValidationSummary],
        correlation_matrix: dict[tuple[str, str], float] | None = None,
    ) -> str:
        """Generate a markdown validation report.

        Sections: executive summary, validation table, per-strategy assessment,
        correlation matrix (if 2+ survivors), locked params YAML, Phase 5 criteria.
        """
        lines: list[str] = []
        summary_map = {s.strategy_id: s for s in summaries}

        survivors = [d for d in decisions if d.decision == "PROCEED"]
        retired = [d for d in decisions if d.decision == "RETIRE"]

        # Header
        lines.append("# Phase 4 Validation Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

        # Executive summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(f"- **Strategies evaluated**: {len(decisions)}")
        lines.append(f"- **Survivors (PROCEED)**: {len(survivors)}")
        lines.append(f"- **Retired**: {len(retired)}")
        lines.append("")

        if survivors:
            names = ", ".join(d.strategy_id for d in survivors)
            lines.append(f"Advancing to Phase 5: **{names}**")
        else:
            lines.append("**No strategies survived validation.** "
                         "Review research hypotheses below before proceeding.")
        lines.append("")

        # Validation table
        lines.append("## Validation Gate Results")
        lines.append("")
        lines.append(
            "| Strategy | PBO | DSR | WFA Eff | Stability | Decision |"
        )
        lines.append(
            "|----------|-----|-----|---------|-----------|----------|"
        )
        for d in decisions:
            s = summary_map.get(d.strategy_id)
            if s is None:
                continue
            pbo_mark = "PASS" if d.passed_pbo else "FAIL"
            dsr_mark = "PASS" if d.passed_dsr else "FAIL"
            wfa_mark = "PASS" if d.passed_wfa else "FAIL"
            stab_mark = "PASS" if d.passed_stability else "FAIL"
            lines.append(
                f"| {d.strategy_id} | {pbo_mark} ({s.cpcv_pbo:.2f}) | "
                f"{dsr_mark} ({s.dsr:.2f}) | "
                f"{wfa_mark} ({s.wfa_efficiency:.2f}) | "
                f"{stab_mark} ({s.param_stability_score:.2f}) | "
                f"**{d.decision}** |"
            )
        lines.append("")

        # Per-strategy assessment
        lines.append("## Per-Strategy Assessment")
        lines.append("")
        for d in decisions:
            s = summary_map.get(d.strategy_id)
            lines.append(f"### {d.strategy_id} — {d.decision}")
            lines.append("")
            if s:
                lines.append(f"- OOS Sharpe: {s.oos_sharpe:.3f}")
                lines.append(f"- OOS Trades: {s.total_oos_trades}")
                lines.append(f"- IS/OOS Correlation: {s.wfa_is_oos_correlation:.2f}")
            if d.failure_modes:
                lines.append("")
                lines.append("**Failure modes:**")
                for fm in d.failure_modes:
                    lines.append(f"- {fm}")
            if d.research_hypothesis:
                lines.append("")
                lines.append(f"**Research hypothesis:** {d.research_hypothesis}")
            if d.locked_parameters:
                lines.append("")
                lines.append("**Locked parameters:**")
                lines.append("```yaml")
                lines.append(yaml.dump(d.locked_parameters, default_flow_style=False).strip())
                lines.append("```")
            lines.append("")

        # Correlation matrix (if 2+ survivors)
        if correlation_matrix and len(survivors) >= 2:
            lines.append("## Survivor Correlation Matrix")
            lines.append("")
            survivor_ids = [d.strategy_id for d in survivors]
            lines.append("| | " + " | ".join(survivor_ids) + " |")
            lines.append("|" + "---|" * (len(survivor_ids) + 1))
            for sid_a in survivor_ids:
                row = f"| {sid_a} |"
                for sid_b in survivor_ids:
                    if sid_a == sid_b:
                        row += " 1.00 |"
                    else:
                        key = (min(sid_a, sid_b), max(sid_a, sid_b))
                        val = correlation_matrix.get(key, 0.0)
                        row += f" {val:.2f} |"
                lines.append(row)
            lines.append("")

        # Locked params YAML
        if survivors:
            lines.append("## Locked Parameters for Phase 5")
            lines.append("")
            lines.append("```yaml")
            lines.append(self.generate_locked_params_yaml(decisions))
            lines.append("```")
            lines.append("")

        # Phase 5 entry criteria
        lines.append("## Phase 5 Entry Criteria")
        lines.append("")
        lines.append("Before proceeding to Phase 5 (Live Paper Trading):")
        lines.append("")
        lines.append("- [ ] All surviving strategies have locked parameters (above)")
        lines.append("- [ ] No parameter changes allowed without re-running Phase 4")
        lines.append("- [ ] Ensemble weights computed from OOS Sharpe ratios")
        lines.append("- [ ] Risk limits configured per strategy")
        lines.append("- [ ] Paper trading infrastructure verified")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def generate_locked_params_yaml(decisions: list[ValidationDecision]) -> str:
        """Generate YAML for locked parameters of PROCEED strategies."""
        locked: dict[str, dict] = {}
        for d in decisions:
            if d.decision == "PROCEED" and d.locked_parameters:
                locked[d.strategy_id] = d.locked_parameters

        if not locked:
            return "# No strategies survived validation"

        # Build YAML with DO NOT CHANGE header
        lines = ["# DO NOT CHANGE — Phase 4 validated parameters",
                 f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                 ""]
        lines.append(yaml.dump(locked, default_flow_style=False).strip())
        return "\n".join(lines)
