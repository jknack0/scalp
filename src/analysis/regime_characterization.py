"""Mean reversion / momentum transition analysis across timescales.

Maps where MES mean-reverts vs trends across horizons (30s–60m) and
times of day. Produces the foundational analysis for Phase 3 strategy
parameter selection.

Key analyses:
- ACF profile of returns across lags
- Hurst exponent by resampled window size
- AR(1) mean reversion coefficient by timescale
- Crossing point: timescale where reversion → momentum transition occurs
- Time-of-day segmentation (open / midday / close)
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from statsmodels.tsa.stattools import acf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from src.analysis.bar_statistics import compute_log_returns, hurst_exponent
from src.data.bars import resample_bars


# ── Result dataclasses ───────────────────────────────────────────────

@dataclass(frozen=True)
class AutocorrelationProfile:
    """ACF values across lags with confidence intervals."""
    lags: list[int]
    acf_values: list[float]
    confidence_lower: list[float]
    confidence_upper: list[float]
    significant_lags: list[int]  # lags where ACF is outside CI


@dataclass(frozen=True)
class HurstByWindow:
    """Hurst exponent computed at multiple resampled window sizes."""
    window_seconds: list[int]
    window_labels: list[str]
    hurst_values: list[float]
    classifications: list[str]


@dataclass(frozen=True)
class AR1Result:
    """AR(1) mean reversion coefficient at one timescale.

    Convention: mean_reversion_coeff = -AR1_coefficient_on_returns.
    Positive → mean reversion (returns tend to reverse).
    Negative → momentum (returns tend to persist).
    """
    timescale_label: str
    timescale_seconds: int
    mean_reversion_coeff: float
    ar1_raw: float
    p_value: float


@dataclass(frozen=True)
class CrossingPoint:
    """Interpolated timescale where mean reversion → momentum transition occurs."""
    timescale_seconds: float
    timescale_label: str
    below_regime: str  # "mean-reverting"
    above_regime: str  # "momentum"


@dataclass(frozen=True)
class RegimeReport:
    """Full regime characterization for one data segment."""
    label: str
    n_bars: int
    acf_profile: AutocorrelationProfile
    hurst_by_window: HurstByWindow
    ar1_results: list[AR1Result]
    crossing_point: CrossingPoint | None


# ── Window size helpers ──────────────────────────────────────────────

DEFAULT_WINDOWS = [30, 120, 300, 900, 1800, 3600]  # seconds

WINDOW_LABELS = {
    1: "1s",
    30: "30s",
    120: "2m",
    300: "5m",
    900: "15m",
    1800: "30m",
    3600: "60m",
}

POLARS_FREQ = {
    1: "1s",
    30: "30s",
    120: "2m",
    300: "5m",
    900: "15m",
    1800: "30m",
    3600: "60m",
}

DEFAULT_TOD_SEGMENTS = {
    "open": (9, 30, 11, 0),    # 9:30 AM – 11:00 AM
    "midday": (11, 0, 14, 0),  # 11:00 AM – 2:00 PM
    "close": (14, 0, 16, 0),   # 2:00 PM – 4:00 PM
}


# ── Core functions ───────────────────────────────────────────────────

def compute_acf_profile(
    returns: np.ndarray,
    max_lag: int = 200,
    confidence: float = 0.95,
) -> AutocorrelationProfile:
    """Compute autocorrelation function with confidence intervals.

    Args:
        returns: 1-D array of returns.
        max_lag: Maximum lag to compute. Clamped to len(returns) // 2.
        confidence: Confidence level for intervals (default 95%).

    Returns:
        AutocorrelationProfile with ACF values and significance flags.
    """
    max_lag = min(max_lag, len(returns) // 2 - 1)
    if max_lag < 1:
        return AutocorrelationProfile(
            lags=[], acf_values=[], confidence_lower=[],
            confidence_upper=[], significant_lags=[],
        )

    acf_values, conf_intervals = acf(
        returns, nlags=max_lag, alpha=1 - confidence, fft=True,
    )

    # Skip lag 0 (always 1.0)
    lags = list(range(1, max_lag + 1))
    acf_vals = [float(acf_values[i]) for i in range(1, max_lag + 1)]
    lower = [float(conf_intervals[i, 0]) for i in range(1, max_lag + 1)]
    upper = [float(conf_intervals[i, 1]) for i in range(1, max_lag + 1)]

    # Significant if the CI excludes zero (i.e., zero is not in [lower, upper])
    significant = [
        lag for lag, lo, hi in zip(lags, lower, upper)
        if lo > 0 or hi < 0
    ]

    return AutocorrelationProfile(
        lags=lags,
        acf_values=acf_vals,
        confidence_lower=lower,
        confidence_upper=upper,
        significant_lags=significant,
    )


def compute_hurst_by_window(
    df_1s: pl.DataFrame,
    windows: list[int] | None = None,
) -> HurstByWindow:
    """Compute Hurst exponent at multiple resampled window sizes.

    Args:
        df_1s: 1-second bar DataFrame with timestamp and close columns.
        windows: List of window sizes in seconds. Defaults to
                 [30, 120, 300, 900, 1800, 3600].

    Returns:
        HurstByWindow with H values and classifications per window.
    """
    if windows is None:
        windows = DEFAULT_WINDOWS

    h_values = []
    classifications = []
    labels = []

    for w in windows:
        freq = POLARS_FREQ.get(w, f"{w}s")
        label = WINDOW_LABELS.get(w, f"{w}s")
        labels.append(label)

        resampled = resample_bars(df_1s, freq)
        if len(resampled) < 50:
            h_values.append(float("nan"))
            classifications.append("insufficient-data")
            continue

        returns = compute_log_returns(resampled["close"].to_numpy())
        if len(returns) < 100:
            h_values.append(float("nan"))
            classifications.append("insufficient-data")
            continue

        try:
            result = hurst_exponent(returns)
            h_values.append(result.H)
            classifications.append(result.classification)
        except ValueError:
            h_values.append(float("nan"))
            classifications.append("insufficient-data")

    return HurstByWindow(
        window_seconds=windows,
        window_labels=labels,
        hurst_values=h_values,
        classifications=classifications,
    )


def compute_ar1_by_timescale(
    df_1s: pl.DataFrame,
    timescales: list[int] | None = None,
) -> list[AR1Result]:
    """Fit AR(1) on returns at each timescale, extract mean reversion coefficient.

    Convention: mean_reversion_coeff = -AR1_coefficient.
    Positive → returns tend to reverse (mean reversion).
    Negative → returns tend to persist (momentum).

    Args:
        df_1s: 1-second bar DataFrame.
        timescales: Window sizes in seconds. Defaults to DEFAULT_WINDOWS.

    Returns:
        List of AR1Result, one per timescale.
    """
    if timescales is None:
        timescales = DEFAULT_WINDOWS

    results = []
    for ts in timescales:
        freq = POLARS_FREQ.get(ts, f"{ts}s")
        label = WINDOW_LABELS.get(ts, f"{ts}s")

        resampled = resample_bars(df_1s, freq)
        if len(resampled) < 50:
            results.append(AR1Result(
                timescale_label=label, timescale_seconds=ts,
                mean_reversion_coeff=float("nan"), ar1_raw=float("nan"),
                p_value=float("nan"),
            ))
            continue

        returns = compute_log_returns(resampled["close"].to_numpy())
        if len(returns) < 30:
            results.append(AR1Result(
                timescale_label=label, timescale_seconds=ts,
                mean_reversion_coeff=float("nan"), ar1_raw=float("nan"),
                p_value=float("nan"),
            ))
            continue

        # Fit AR(1): r[t] = a + b * r[t-1] + e
        y = returns[1:]
        x = add_constant(returns[:-1])
        model = OLS(y, x).fit()
        ar1_coeff = float(model.params[1])
        p_val = float(model.pvalues[1])

        results.append(AR1Result(
            timescale_label=label,
            timescale_seconds=ts,
            mean_reversion_coeff=-ar1_coeff,
            ar1_raw=ar1_coeff,
            p_value=p_val,
        ))

    return results


def find_crossing_point(ar1_results: list[AR1Result]) -> CrossingPoint | None:
    """Find the timescale where mean reversion coefficient crosses zero.

    Uses linear interpolation between the last positive and first negative
    AR1 result (sorted by timescale).

    Returns:
        CrossingPoint or None if no sign change is found.
    """
    # Filter out NaN results and sort by timescale
    valid = [r for r in ar1_results if not np.isnan(r.mean_reversion_coeff)]
    valid.sort(key=lambda r: r.timescale_seconds)

    if len(valid) < 2:
        return None

    for i in range(len(valid) - 1):
        c1 = valid[i].mean_reversion_coeff
        c2 = valid[i + 1].mean_reversion_coeff

        if c1 > 0 and c2 <= 0:
            # Linear interpolation
            t1 = valid[i].timescale_seconds
            t2 = valid[i + 1].timescale_seconds
            crossing_t = t1 + (t2 - t1) * (c1 / (c1 - c2))

            if crossing_t < 60:
                label = f"{crossing_t:.0f}s"
            else:
                label = f"{crossing_t / 60:.1f}m"

            return CrossingPoint(
                timescale_seconds=crossing_t,
                timescale_label=label,
                below_regime="mean-reverting",
                above_regime="momentum",
            )

    return None


def segment_by_time_of_day(
    df: pl.DataFrame,
    segments: dict[str, tuple[int, int, int, int]] | None = None,
) -> dict[str, pl.DataFrame]:
    """Filter DataFrame into time-of-day segments.

    Args:
        df: DataFrame with a timestamp column.
        segments: Dict of name → (start_hour, start_min, end_hour, end_min).
                  Defaults to open/midday/close RTH segments.

    Returns:
        Dict of segment name → filtered DataFrame.
    """
    if segments is None:
        segments = DEFAULT_TOD_SEGMENTS

    result = {}
    for name, (sh, sm, eh, em) in segments.items():
        start_minutes = sh * 60 + sm
        end_minutes = eh * 60 + em
        total_minutes = (
            pl.col("timestamp").dt.hour().cast(pl.Int64) * 60
            + pl.col("timestamp").dt.minute().cast(pl.Int64)
        )
        filtered = df.filter(
            (total_minutes >= start_minutes) & (total_minutes < end_minutes)
        )
        result[name] = filtered

    return result


def run_full_analysis(
    df_1s: pl.DataFrame,
    label: str = "all",
    max_acf_lag: int = 200,
) -> RegimeReport:
    """Run all regime characterization analyses on a 1s bar DataFrame.

    Args:
        df_1s: 1-second bar DataFrame.
        label: Label for this data segment (e.g., "all", "open", "midday").
        max_acf_lag: Maximum ACF lag (in 1s bars).

    Returns:
        RegimeReport bundling all results.
    """
    closes = df_1s["close"].to_numpy()
    returns = compute_log_returns(closes)

    acf_prof = compute_acf_profile(returns, max_lag=max_acf_lag)
    hurst_win = compute_hurst_by_window(df_1s)
    ar1_res = compute_ar1_by_timescale(df_1s)
    crossing = find_crossing_point(ar1_res)

    return RegimeReport(
        label=label,
        n_bars=len(df_1s),
        acf_profile=acf_prof,
        hurst_by_window=hurst_win,
        ar1_results=ar1_res,
        crossing_point=crossing,
    )


# ── Reporting ────────────────────────────────────────────────────────

def print_regime_report(reports: list[RegimeReport]) -> None:
    """Print formatted summary of regime characterization results."""
    for report in reports:
        print(f"\n{'=' * 70}")
        print(f"REGIME ANALYSIS: {report.label.upper()} ({report.n_bars:,} bars)")
        print(f"{'=' * 70}")

        # ACF summary
        n_sig = len(report.acf_profile.significant_lags)
        total = len(report.acf_profile.lags)
        print(f"\n  ACF Profile: {n_sig}/{total} lags significant at 95%")
        if report.acf_profile.significant_lags:
            first_5 = report.acf_profile.significant_lags[:5]
            print(f"  First significant lags: {first_5}")

        # Hurst by window
        print(f"\n  {'Window':<8} {'Hurst':>7} {'Classification':<16}")
        print(f"  {'-' * 35}")
        hw = report.hurst_by_window
        for lbl, h, cls in zip(hw.window_labels, hw.hurst_values, hw.classifications):
            h_str = f"{h:.3f}" if not np.isnan(h) else "N/A"
            print(f"  {lbl:<8} {h_str:>7} {cls:<16}")

        # AR(1) coefficients
        print(f"\n  {'Timescale':<10} {'MR Coeff':>10} {'AR1 Raw':>10} {'p-value':>10}")
        print(f"  {'-' * 45}")
        for ar in report.ar1_results:
            if np.isnan(ar.mean_reversion_coeff):
                print(f"  {ar.timescale_label:<10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            else:
                sign = "+" if ar.mean_reversion_coeff > 0 else ""
                print(
                    f"  {ar.timescale_label:<10} {sign}{ar.mean_reversion_coeff:>9.4f} "
                    f"{ar.ar1_raw:>10.4f} {ar.p_value:>10.4f}"
                )

        # Crossing point
        if report.crossing_point:
            cp = report.crossing_point
            print(f"\n  CROSSING POINT: {cp.timescale_label} ({cp.timescale_seconds:.0f}s)")
            print(f"    Below: {cp.below_regime}")
            print(f"    Above: {cp.above_regime}")
        else:
            print("\n  CROSSING POINT: Not found (no sign change detected)")

    print()
