"""Reconstruct SignalBundle from enriched row dicts.

The enriched DataFrame has pre-computed ``sig_*`` columns from
:mod:`src.signals.vectorized`.  This module converts row values
into SignalBundle objects inline during the engine's existing
row iteration — no bulk pre-allocation needed.
"""

from __future__ import annotations

from src.signals.base import SignalResult
from src.signals.signal_bundle import SignalBundle


def bundle_from_row(
    row: dict,
    signal_names: list[str],
) -> SignalBundle:
    """Build a single SignalBundle from an enriched row dict.

    Called per-row inside BacktestEngine.run() — avoids allocating
    millions of bundles upfront.

    Args:
        row: Row dict from ``iter_rows(named=True)`` with ``sig_*`` keys.
        signal_names: Signal names that were computed.

    Returns:
        SignalBundle with results populated from pre-computed columns.
    """
    results: dict[str, SignalResult] = {}

    if "atr" in signal_names and "sig_atr_value" in row:
        atr_val = float(row["sig_atr_value"])
        results["atr"] = SignalResult(
            value=atr_val,
            passes=True,
            direction="none",
            metadata={
                "atr_ticks": atr_val,
                "atr_raw": float(row["sig_atr_raw"]),
                "vol_regime": row["sig_atr_vol_regime"],
                "atr_percentile": float(row["sig_atr_percentile"]),
            },
        )

    if "vwap_session" in signal_names and "sig_vwap_value" in row:
        vwap_val = float(row["sig_vwap_value"])
        results["vwap_session"] = SignalResult(
            value=vwap_val,
            passes=True,
            direction=row["sig_vwap_direction"],
            metadata={
                "vwap": float(row["sig_vwap_vwap"]),
                "sd": float(row["sig_vwap_sd"]),
                "slope": float(row["sig_vwap_slope"]),
                "deviation_sd": vwap_val,
                "mode": row["sig_vwap_mode"],
                "first_kiss": row["sig_vwap_first_kiss"],
                "session_age_bars": int(row["sig_vwap_session_age"]),
            },
        )

    if "relative_volume" in signal_names and "sig_rvol_value" in row:
        rvol_val = float(row["sig_rvol_value"])
        results["relative_volume"] = SignalResult(
            value=rvol_val,
            passes=bool(row["sig_rvol_passes"]),
            direction="none",
            metadata={"rvol": rvol_val},
        )

    if "spread" in signal_names and "sig_spread_value" in row:
        is_unavailable = bool(row.get("sig_spread_unavailable", False))
        results["spread"] = SignalResult(
            value=float(row["sig_spread_value"]),
            passes=bool(row["sig_spread_passes"]),
            direction="none",
            metadata={
                "z_score": float(row["sig_spread_value"]),
                **({"unavailable": True, "reason": "missing_data"} if is_unavailable else {}),
            },
        )

    if "adx" in signal_names and "sig_adx_value" in row:
        adx_val = float(row["sig_adx_value"])
        results["adx"] = SignalResult(
            value=adx_val,
            passes=bool(row["sig_adx_passes"]),
            direction=row["sig_adx_direction"],
            metadata={
                "adx": adx_val,
                "plus_di": float(row["sig_adx_plus_di"]),
                "minus_di": float(row["sig_adx_minus_di"]),
            },
        )

    if "donchian_channel" in signal_names and "sig_dc_value" in row:
        dc_val = float(row["sig_dc_value"])
        results["donchian_channel"] = SignalResult(
            value=dc_val,
            passes=bool(row["sig_dc_passes"]),
            direction=row["sig_dc_direction"],
            metadata={
                "entry_upper": float(row["sig_dc_entry_upper"]) if row["sig_dc_entry_upper"] is not None else 0.0,
                "entry_lower": float(row["sig_dc_entry_lower"]) if row["sig_dc_entry_lower"] is not None else 0.0,
                "exit_upper": float(row["sig_dc_exit_upper"]) if row["sig_dc_exit_upper"] is not None else 0.0,
                "exit_lower": float(row["sig_dc_exit_lower"]) if row["sig_dc_exit_lower"] is not None else 0.0,
                "width": dc_val,
                "mid": (float(row["sig_dc_entry_upper"] or 0) + float(row["sig_dc_entry_lower"] or 0)) / 2.0,
            },
        )

    if "session_time" in signal_names and "sig_session_time_value" in row:
        st_val = float(row["sig_session_time_value"])
        results["session_time"] = SignalResult(
            value=st_val,
            passes=True,
            direction="none",
            metadata={},
        )

    if "regime_v2" in signal_names and "sig_regime_v2_value" in row:
        regime_val = int(row["sig_regime_v2_value"])
        halt = bool(row.get("sig_regime_v2_whipsaw_halt", False))
        pos_size = row.get("sig_regime_v2_position_size", "flat")
        results["regime_v2"] = SignalResult(
            value=float(regime_val),
            passes=not halt and pos_size != "flat",
            direction="none",
            metadata={
                "regime": regime_val,
                "confidence": float(row["sig_regime_v2_confidence"]),
                "position_size": pos_size,
                "p_trending": float(row["sig_regime_v2_p_trending"]),
                "p_ranging": float(row["sig_regime_v2_p_ranging"]),
                "p_high_vol": float(row["sig_regime_v2_p_high_vol"]),
            },
        )

    return SignalBundle(results=results, bar_count=len(signal_names))
