# The definitive Donchian Channel guide for MES scalping bots

**Donchian Channel strategies failed catastrophically in your 10-year sweep—0 out of 2,304 configurations profitable—because naive channel breakouts are structurally incompatible with S&P 500 futures microstructure.** Equity index futures are mean-reverting at intraday timeframes (Hurst exponent ~0.49),  meaning the majority of Donchian breakouts are false signals that immediately reverse. Academic backtests confirm: raw 20-period Donchian breakout achieves only a **35% win rate** across 4,887 trades and 360 years of exchange data, yielding approximately $0.05 profit per dollar invested—before commissions.  In MES, where your round-trip cost is $0.59 per contract against $1.25-per-tick profit targets,   that edge evaporates entirely.

The fix is not abandoning Donchian—it's weaponizing it with regime gating, volume confirmation, session filters, and deploying all four flavors (breakout, fade, midline, squeeze) in the correct market conditions. Research shows HMM regime gating alone lifts Sharpe from ~0.5 to **1.1–1.9**,  volume confirmation (RVOL ≥ 1.5×) increases breakout follow-through by **40%**,  and ADX filtering eliminates the choppy low-conviction entries that destroyed your P&L. This guide provides the exact filter combinations, YAML configs, and parameter sweep ranges to make each flavor production-ready.

-----

## Why your 2,304-config sweep produced zero winners

The failure wasn't a parameter problem—it was a structural mismatch. Understanding the three root causes is essential before building any Donchian strategy for MES.

**Root cause 1: Equity indices are mean-reverting, not trending, at intraday scales.** Robert Carver's research demonstrates that at timescales of minutes to hours, S&P 500 futures exhibit negative autocorrelation—meaning breakout signals are systematically wrong.  The Price Action Lab documented that S&P 500 daily return autocorrelation shifted from positive (pre-1990s) to negative in recent decades.  At 1-minute and 5-minute resolution, the mean-reverting tendency is even stronger. Commodities and FX, where Donchian breakout was designed and where Turtle traders made $175 million, exhibit the opposite behavior: persistent trending.

**Root cause 2: Algorithmic market-making creates systematic stop hunts.** Institutional algorithms probe beyond key levels (including Donchian channel boundaries) to trigger stop-losses and gather liquidity before reversing. TradingStats.net data across 5,519 ES trading days shows **~33% of Initial Balance first breakouts fail**, with downside breakouts failing 53.2% of the time due to institutional dip-buying.  When your Donchian bot enters a breakout, it's frequently providing liquidity to these counter-traders.

**Root cause 3: Unfiltered breakout strategies take trades in every regime.** Your sweep likely tested every combination of period length and ATR multiple, but without regime gating, every configuration took breakout trades during range-bound conditions—which represent the majority of intraday MES price action. A 20-period Donchian breakout in a range-bound lunch session is guaranteed to lose. The studies on HMM regime detection show that simply blocking trades during high-volatility choppy or low-volatility range-bound regimes eliminates **60-70% of losing trades** while preserving the winners.

The solution architecture uses four distinct strategy flavors, each deployed only in its optimal regime via the `regime_v2` signal, with volume, momentum, and session filters providing additional confirmation layers.

-----

## Flavor 1: Breakout entries for trend-following

### The edge thesis

Donchian breakout works when price genuinely transitions from equilibrium to trend. In MES, this happens during **macro catalyst sessions** (FOMC, CPI, NFP), **afternoon breakouts of range-bound days** (2:00–2:45 PM ET),  and **post-IB range extensions**. The edge comes from capturing the ~25-35% of breakouts that produce outsized moves while filtering out the 65-75% that fail. Bulkowski's research quantifies this: breakouts with volume ≥50% above the 20-day average achieve a **65% success rate**  versus 39% without volume confirmation—a 26-percentage-point improvement that transforms a losing strategy into a winning one.

### Entry logic with exact conditions

The breakout entry requires five simultaneous conditions, implemented as filter expressions in the SignalBundle:

```yaml
strategy_id: donchian_breakout_trend
bar_size: 1m
direction: LONG  # mirror for SHORT

filters:
  entry:
    - signal: donchian_channel
      field: upper_break
      op: "=="
      value: true

    - signal: regime_v2
      field: state
      op: in
      value: ["TRENDING"]  # HMM must confirm trending regime

    - signal: adx
      field: value
      op: ">="
      value: 22  # trend strength gate; 20-25 range for intraday

    - signal: relative_volume
      field: ratio
      op: ">="
      value: 1.5  # RVOL must be 1.5x+ vs time-of-day average

    - signal: session_time
      field: minutes_since_open
      op: ">="
      value: 30  # block first 30 min (IB still forming)

    - signal: session_time
      field: minutes_until_close
      op: ">="
      value: 15  # no entries in final 15 min

    - signal: vwap_bias
      field: direction
      op: "=="
      value: "BULLISH"  # price must be above VWAP for longs

    - signal: sma_trend
      field: direction
      op: "=="
      value: "UP"  # higher-timeframe trend alignment
```

For **volume-confirmed breakouts with CVD**, add the divergence filter to reject breakouts lacking order flow support:

```yaml
    - signal: cvd_divergence
      field: bearish_divergence  # for LONG entries
      op: "=="
      value: false  # reject if CVD diverges bearishly from price
```

### Exit strategy recommendations

Use a layered exit combining bracket protection with a trailing mechanism. The ATR-based bracket provides worst-case risk management while the trailing stop captures trend continuation.

```yaml
exits:
  bracket_stop:
    atr_multiplier: 1.75  # initial stop: 1.75x 1-min ATR (~2-5 pts on MES)
    atr_period: 14

  bracket_target:
    atr_multiplier: 3.5   # target: 3.5x ATR → 2:1 R:R minimum
    atr_period: 14

  trailing_stop:
    atr_multiplier: 1.5   # once in profit, trail at 1.5x ATR
    activation_atr: 1.0   # activate trailing after 1x ATR profit
    atr_period: 14

  time_stop:
    max_bars: 30          # exit after 30 1-min bars if no resolution

  regime_exit:
    exit_on_state: ["RANGE_BOUND", "VOLATILE_CHOP"]  # exit if regime shifts

  adverse_momentum:
    adx_below: 18         # exit if ADX drops below 18 (trend dying)
```

### Parameter ranges for sweep

|Parameter             |Min |Max|Step|Notes                                                 |
|----------------------|----|---|----|------------------------------------------------------|
|`donchian_period`     |10  |40 |5   |20 is default; test 10, 15, 20, 25, 30, 35, 40        |
|`adx_threshold`       |18  |30 |2   |Lower for 1m, higher for 5m                           |
|`adx_period`          |7   |14 |7   |7 for fast intraday, 14 for standard                  |
|`rvol_threshold`      |1.2 |2.5|0.3 |1.5 is literature consensus                           |
|`bracket_stop_atr`    |1.25|2.5|0.25|Study shows 2.0x reduces max DD by 32%                |
|`bracket_target_atr`  |2.5 |5.0|0.5 |Must maintain ≥1.5:1 R:R                              |
|`trailing_atr`        |1.0 |2.0|0.25|Tighter = more exits, looser = more trend capture     |
|`time_stop_bars`      |15  |60 |15  |Research shows 8-10 bars optimal for MR; longer for TF|
|`session_start_offset`|15  |45 |15  |Minutes after open to begin trading                   |

### Red flags and failure modes

The breakout flavor fails most dramatically during **lunch hours (12:00–1:30 PM ET)** when volume drops 40-60% and MES becomes a whipsaw machine. It also fails during **low-ADX consolidation periods** where the channel boundaries are repeatedly tested without follow-through. Monitor for **win rate below 30%** over any 50-trade window—this signals a regime the bot should not be trading. The most insidious failure mode is **consecutive small losses from choppy ranging days** that individually seem harmless but compound into significant drawdown over weeks.

-----

## Flavor 2: Mean reversion fades from channel extremes

### The edge thesis

Since MES is structurally mean-reverting at intraday timeframes, fading Donchian channel extremes aligns with the market's dominant statistical property. Mean reversion strategies on equity indices typically achieve **65-75% win rates** with smaller individual gains. The edge is most pronounced during range-bound regimes (which account for 60-70% of trading sessions) and during lunch hours when the mean-reverting tendency peaks. Research on S&P 500 IBS (Internal Bar Strength) shows the probability of an up day after a low-IBS reading differs from high-IBS by **11.2%**—a statistically significant edge that Donchian band fades exploit.

### Why this is Donchian's best MES application

The same mean-reverting microstructure that kills breakout strategies **feeds** fade strategies. When price touches the Donchian upper band in a range-bound session, institutional algorithms, market makers, and mean-reversion traders all converge to push price back toward equilibrium. The Donchian band becomes a high-probability fade level because it represents a statistically extreme price relative to recent history.

### Entry logic with exact conditions

```yaml
strategy_id: donchian_fade_reversion
bar_size: 1m
direction: LONG  # fading the lower band touch

filters:
  entry:
    - signal: donchian_channel
      field: lower_touch
      op: "=="
      value: true  # price at or below lower Donchian band

    - signal: regime_v2
      field: state
      op: in
      value: ["RANGE_BOUND"]  # ONLY in range-bound regime

    - signal: adx
      field: value
      op: "<"
      value: 20  # weak trend confirms range-bound conditions

    - signal: rsi_momentum
      field: value
      op: "<"
      value: 30  # RSI oversold confirms extreme

    - signal: stochastic
      field: k_value
      op: "<"
      value: 20  # stochastic oversold double-confirmation

    - signal: cvd_divergence
      field: bullish_divergence  # CVD showing buying vs falling price
      op: "=="
      value: true

    - signal: vwap_deviation
      field: std_devs_below
      op: ">="
      value: 1.5  # price is 1.5+ std devs below VWAP

    - signal: session_time
      field: window
      op: in
      value: ["MIDDAY", "AFTERNOON"]  # best for MR: 11:00 AM - 3:00 PM
```

For SHORT fades (at upper band), mirror all conditions: `upper_touch`, RSI > 70, stochastic > 80, bearish CVD divergence, VWAP deviation above +1.5 SDs.

### Exit strategy for fades

Fades require tight, asymmetric exits: quick targets (the move back to mean is fast) and defensive stops (if the band break is genuine, exit immediately).

```yaml
exits:
  vwap_reversion_target:
    target: "vwap"       # primary target: revert to session VWAP

  bracket_target:
    atr_multiplier: 1.5  # secondary target: 1.5x ATR (modest)
    atr_period: 14

  bracket_stop:
    atr_multiplier: 1.25 # tight stop: 1.25x ATR beyond the band
    atr_period: 14

  time_stop:
    max_bars: 20         # if no reversion in 20 bars, it's a trend → exit

  volatility_expansion_exit:
    atr_expansion: 1.5   # exit if ATR expands 50%+ (breakout beginning)

  regime_exit:
    exit_on_state: ["TRENDING"]  # regime shift to trending kills the fade
```

### The Turtle Soup integration

Linda Raschke's Turtle Soup strategy is precisely this concept applied to Donchian. The exact implementation for your bot:

**Turtle Soup Long setup conditions:**

1. Price makes a new 20-bar low (touches/breaks the Donchian lower band)
1. The previous 20-bar low occurred at least **3 bars earlier** (not a continuation low)
1. On the next bar, place a buy stop at the previous 20-bar low level
1. If filled (price reverses back through the old low), the false breakout is confirmed

```yaml
strategy_id: donchian_turtle_soup
bar_size: 1m

filters:
  entry:
    - signal: donchian_channel
      field: lower_break_fresh  # new 20-bar low, prior low ≥3 bars ago
      op: "=="
      value: true

    - signal: regime_v2
      field: state
      op: in
      value: ["RANGE_BOUND", "TRENDING"]  # works in both; avoid VOLATILE_CHOP

    - signal: relative_volume
      field: ratio
      op: "<"
      value: 2.0  # LOW volume on breakout → likely false (key differentiator)

    - signal: rsi_momentum
      field: value
      op: "<"
      value: 35  # oversold but not extreme washout
```

The critical insight: Turtle Soup **inverts** the volume filter. For breakouts, you want high RVOL to confirm. For Turtle Soup, you want **low RVOL** on the breakout—indicating the move lacks institutional conviction and is likely to fail.

### Parameter ranges for sweep

|Parameter                     |Min|Max |Step|Notes                                    |
|------------------------------|---|----|----|-----------------------------------------|
|`donchian_period`             |15 |30  |5   |20 is standard for Turtle Soup           |
|`rsi_oversold`                |20 |35  |5   |Lower = more selective, higher win rate  |
|`rsi_overbought`              |65 |80  |5   |For short fades                          |
|`stochastic_threshold`        |15 |25  |5   |Oversold confirmation                    |
|`vwap_deviation_threshold`    |1.0|2.0 |0.25|Standard deviations from VWAP            |
|`bracket_stop_atr`            |1.0|1.75|0.25|Tighter stops for fades                  |
|`bracket_target_atr`          |1.0|2.0 |0.25|Modest targets for MR                    |
|`time_stop_bars`              |10 |30  |5   |8-10 bars is optimal per research        |
|`adx_ceiling`                 |18 |25  |2   |Max ADX for fade entries                 |
|`min_bars_since_prior_extreme`|3  |10  |1   |Turtle Soup: separation between breakouts|

-----

## Flavor 3: Channel midline as dynamic support and resistance

### The edge thesis

The Donchian midline—`(highest_high + lowest_low) / 2`—functions as a dynamic equilibrium price.  In trending conditions, price oscillates between the midline and the trend-side band: during uptrends, pullbacks to the midline find support before resuming; during downtrends, rallies stall at midline resistance.  This creates a **pullback entry strategy** with defined risk (below midline for longs) and clear targets (the upper band). The midline represents the 50% retracement of the channel range, which frequently aligns with Fibonacci 50% levels and VWAP, creating confluence zones.

### Entry logic with exact conditions

The midline strategy requires confirmed trend direction on the 5-minute timeframe, then enters pullbacks to the 1-minute midline:

```yaml
strategy_id: donchian_midline_pullback
bar_size: 1m
direction: LONG

filters:
  entry:
    - signal: donchian_channel
      field: midline_touch
      op: "=="
      value: true  # price pulls back to midline

    - signal: donchian_channel
      field: trend  # derived from band slope or price vs midline on 5m
      op: "=="
      value: "UP"

    - signal: regime_v2
      field: state
      op: in
      value: ["TRENDING"]  # only in trending regimes

    - signal: adx
      field: value
      op: ">="
      value: 22  # confirmed trend, but mid-strength (pullback phase)

    - signal: ema_crossover
      field: bullish
      op: "=="
      value: true  # short EMA above long EMA confirms trend

    - signal: vwap_bias
      field: direction
      op: "=="
      value: "BULLISH"  # above VWAP for longs

    - signal: hmm_regime
      field: trend_strength
      op: ">="
      value: 0.6  # regime model confidence in trend state ≥60%

    - signal: session_time
      field: minutes_since_open
      op: ">="
      value: 60  # post-IB: trend established
```

### Multi-timeframe implementation

The most powerful midline configuration uses **5-minute Donchian for trend direction** and **1-minute Donchian for entry timing**. This is the recommended multi-timeframe architecture:

**5-minute layer (context/regime):**

- 5m Donchian period: 20 bars (covers ~100 minutes of price history)
- If price is above 5m midline → bullish bias
- If 5m channel is expanding → trend strengthening
- If 5m ADX > 25 → confirmed trend environment

**1-minute layer (entry signal):**

- 1m Donchian period: 20 bars (covers ~20 minutes)
- Wait for pullback to 1m midline within the 5m trend direction
- Enter when price bounces off 1m midline with volume confirmation
- Stop below the 1m lower band (for longs)

```yaml
strategy_id: donchian_mtf_midline
bar_size: 1m

filters:
  entry:
    # 5m context layer
    - signal: donchian_channel
      timeframe: 5m
      field: above_midline
      op: "=="
      value: true

    - signal: donchian_channel
      timeframe: 5m
      field: channel_expanding
      op: "=="
      value: true

    # 1m entry layer
    - signal: donchian_channel
      timeframe: 1m
      field: midline_bounce
      op: "=="
      value: true

    - signal: relative_volume
      field: ratio
      op: ">="
      value: 1.2  # moderate volume on bounce

exits:
  bracket_stop:
    reference: "donchian_lower_1m"  # stop at 1m lower band
    buffer_atr: 0.5                 # plus 0.5 ATR cushion

  bracket_target:
    reference: "donchian_upper_1m"  # target: 1m upper band

  trailing_stop:
    atr_multiplier: 1.25
    activation_atr: 0.75

  signal_bound_exit:
    signal: donchian_channel
    field: below_midline
    timeframe: 5m  # exit if 5m trend reverses
```

### Parameter ranges for sweep

|Parameter                |Min |Max |Step|Notes                                 |
|-------------------------|----|----|----|--------------------------------------|
|`donchian_period_1m`     |15  |35  |5   |Entry-timeframe period                |
|`donchian_period_5m`     |15  |30  |5   |Context-timeframe period              |
|`adx_threshold`          |20  |28  |2   |Trend confirmation                    |
|`ema_fast`               |8   |13  |1   |For crossover trend filter            |
|`ema_slow`               |21  |34  |1   |For crossover trend filter            |
|`bracket_stop_buffer_atr`|0.25|1.0 |0.25|Cushion below lower band              |
|`trailing_atr`           |1.0 |1.75|0.25|Moderate trailing for pullback entries|
|`rvol_threshold`         |1.0 |1.5 |0.25|Lower bar than breakout flavor        |

### Red flags

The midline strategy fails when the channel is **very narrow** (squeeze condition)—in this case, the midline, upper band, and lower band converge and provide no meaningful differentiation. It also fails during **V-shaped reversals** where price blows through the midline without pausing. Monitor channel width relative to ATR: if `channel_width < 1.5 × ATR`, the midline is too close to both bands to be useful.

-----

## Flavor 4: Squeeze and contraction setups

### The edge thesis

Volatility is mean-reverting. Periods of extreme channel compression (Donchian channel width at multi-period lows) reliably precede volatility expansion.  The squeeze strategy doesn't predict direction—it identifies **when** a large move is imminent, then uses momentum and order flow to determine direction.  Research on the TTM Squeeze (Bollinger Bands inside Keltner Channels) shows squeezes lasting **10-20 periods minimum** typically precede moves of **5-10%+** on daily charts. On 5-minute MES bars, this translates to multi-point directional moves following extended contractions.

### Detecting the squeeze

The primary squeeze detection method is **channel width percentile ranking**: calculate current Donchian channel width and rank it against the past 100-120 periods. When width drops below the **10th percentile**, a squeeze condition is active.

```yaml
strategy_id: donchian_squeeze_breakout
bar_size: 5m  # squeeze detection works better on 5m

filters:
  # Phase 1: Squeeze detection
  squeeze_active:
    - signal: donchian_channel
      field: width_percentile
      op: "<="
      value: 10  # channel width in bottom 10th percentile

    - signal: bollinger
      field: bandwidth_percentile
      op: "<="
      value: 15  # Bollinger BandWidth also compressed

    - signal: atr
      field: percentile_rank
      op: "<="
      value: 15  # ATR at low percentile confirms low vol

  # Phase 2: Breakout from squeeze
  entry:
    - signal: donchian_channel
      field: upper_break  # or lower_break for shorts
      op: "=="
      value: true

    - signal: donchian_channel
      field: squeeze_fired  # width expanding from squeeze
      op: "=="
      value: true

    - signal: relative_volume
      field: ratio
      op: ">="
      value: 2.0  # HIGHER threshold for squeeze breakout

    - signal: macd
      field: histogram_positive  # momentum confirms direction
      op: "=="
      value: true

    - signal: vpin
      field: cdf
      op: ">="
      value: 0.6  # elevated informed trading flow

    - signal: keltner_channel
      field: outside_upper  # BB expanding outside KC = squeeze fired
      op: "=="
      value: true
```

### Combining squeeze with Initial Balance

The IB range provides essential context for squeeze setups. Research shows **70-75% probability** that if price breaks and closes a 30-minute period outside the IB, the trend continues.   When a Donchian squeeze coincides with price coiling within a narrow IB range, the subsequent breakout is particularly powerful.

```yaml
    # IB integration for squeeze context
    - signal: initial_balance
      field: width_relative  # IB width vs average IB
      op: "<="
      value: 0.75  # narrow IB day → breakout day likely

    - signal: initial_balance
      field: price_vs_ib
      op: in
      value: ["ABOVE_IBH", "BELOW_IBL"]  # price has broken IB range

    - signal: orb_breakout
      field: confirmed
      op: "=="
      value: true  # opening range breakout confirmed
```

### Exit strategy for squeeze breakouts

Squeeze breakouts demand **wider stops** (the initial volatility expansion creates noise) but also **wider targets** (these moves tend to be larger than typical breakouts):

```yaml
exits:
  bracket_stop:
    atr_multiplier: 2.5   # wider stop for expansion phase
    atr_period: 14

  bracket_target:
    atr_multiplier: 5.0   # 5x ATR target (2:1 R:R with 2.5x stop)
    atr_period: 14

  trailing_stop:
    atr_multiplier: 2.0   # wider trailing for vol expansion
    activation_atr: 2.0   # activate after 2x ATR profit
    atr_period: 14

  time_stop:
    max_bars: 40           # squeeze breakouts need time to develop

  volatility_expansion_exit:
    atr_contraction: 0.7   # exit if ATR drops 30% (expansion fading)
```

### ATR-normalized channel width: why it matters for MES

Raw Donchian channel width is meaningless without volatility context. A 10-point channel width means something very different when ATR is 2 points versus 8 points. **ATR-normalized channel width** = `channel_width / ATR(14)`. This ratio provides a regime-independent squeeze/expansion metric:

- **Ratio < 2.0**: Extreme squeeze. Channel width is less than 2× current ATR. High probability of imminent expansion.
- **Ratio 2.0–3.5**: Normal contraction. Watch for squeeze development.
- **Ratio 3.5–5.0**: Normal/average channel. Standard signal conditions.
- **Ratio > 5.0**: Expanded channel. Trending conditions. Breakout and midline flavors preferred.

Implementing this as a signal filter:

```yaml
    - signal: donchian_channel
      field: atr_normalized_width  # channel_width / ATR(14)
      op: "<="
      value: 2.0  # squeeze threshold
```

### Parameter ranges for sweep

|Parameter                       |Min|Max|Step|Notes                           |
|--------------------------------|---|---|----|--------------------------------|
|`donchian_period`               |15 |40 |5   |For squeeze detection           |
|`width_percentile_threshold`    |5  |20 |5   |Lower = more selective          |
|`width_percentile_lookback`     |60 |120|20  |Historical ranking window       |
|`atr_normalized_width_threshold`|1.5|2.5|0.25|Squeeze trigger                 |
|`rvol_threshold`                |1.5|3.0|0.5 |Higher for squeeze breakouts    |
|`bracket_stop_atr`              |2.0|3.0|0.5 |Wider stops for expansion       |
|`bracket_target_atr`            |4.0|6.0|0.5 |Wider targets for expansion     |
|`min_squeeze_bars`              |5  |20 |5   |Min bars in squeeze before valid|
|`vpin_cdf_threshold`            |0.5|0.8|0.1 |Informed flow confirmation      |

-----

## HMM regime gating: the single most impactful improvement

Regime detection is not optional—it is the **primary determinant** of whether your Donchian strategies survive. The QuantConnect study demonstrated that a 3-state HMM on equity returns produced a **Sharpe of 1.9** with only 7.3% maximum drawdown.  The LSEG study confirmed HMM provides the "best identification of market regime shifts" for S&P 500 futures compared to k-means, GMM, and agglomerative clustering.

### Strategy-regime mapping

This is the critical configuration that tells your SignalEngine which Donchian flavor to activate in each regime:

|`regime_v2` State|Donchian Breakout|Mean Reversion Fade|Midline Pullback|Squeeze Breakout  |
|-----------------|-----------------|-------------------|----------------|------------------|
|**TRENDING**     |✅ Active         |❌ Blocked          |✅ Active        |✅ Active (on fire)|
|**RANGE_BOUND**  |❌ Blocked        |✅ Active           |❌ Blocked       |⚠️ Prepare only    |
|**VOLATILE_CHOP**|❌ Blocked        |❌ Blocked          |❌ Blocked       |❌ Blocked         |
|**LOW_VOL_DRIFT**|⚠️ Reduced size   |✅ Active           |✅ Active        |⚠️ Prepare only    |

The `regime_v2` signal should map to these states using HMM with **2-3 hidden states** (2 is most robust). The primary distinction is between trending (positive autocorrelation, HMM low-vol-trend state) and range-bound (negative autocorrelation, HMM choppy state). **Volatile chop—high volatility without direction—is the strategy graveyard where all flavors lose.** Block everything.

### Implementation in the filter engine

```yaml
# Global regime gate applied to all strategies
regime_gate:
  signal: regime_v2

  # Breakout-only gate
  breakout_allowed:
    field: state
    op: in
    value: ["TRENDING"]

  # Fade-only gate
  fade_allowed:
    field: state
    op: in
    value: ["RANGE_BOUND", "LOW_VOL_DRIFT"]

  # Universal kill switch
  all_blocked:
    field: state
    op: in
    value: ["VOLATILE_CHOP"]
```

### HMM retraining considerations

Research recommends a **4-year rolling window** for HMM training on S&P 500 data. However, for intraday 1m/5m bars, the effective lookback should be calibrated to capture recent regime transitions—typically **20-60 trading days** of bar data. Shu et al. (2024) found that Jump Models outperform HMMs because HMMs tend to identify "transient and short-lived regimes" leading to sub-optimal strategy performance. If your `regime_v2` implementation uses HMM, consider adding a **minimum regime duration filter** of 5-10 bars to prevent rapid oscillation between states.

-----

## CVD divergence for volume-confirmed breakouts

CVD divergence is the highest-value order flow confirmation for Donchian breakout entries. The logic is intuitive: if price breaks above the Donchian upper band but CVD (cumulative buying minus selling) is flat or declining, the breakout lacks aggressive buyer participation and is likely false.

### Four CVD-Donchian integration patterns

**Pattern 1 — Confirmed breakout (trade it):** Price breaks Donchian upper band while CVD is making new highs. Aggressive buyers are driving the move. Enter long with standard breakout parameters.

**Pattern 2 — Divergent breakout (fade it or skip it):** Price breaks Donchian upper band but CVD makes a lower high. This is bearish divergence at the channel extreme—buyers are exhausted. Either skip the breakout entry entirely, or activate the Turtle Soup fade.

**Pattern 3 — Hidden accumulation (prepare for breakout):** Price is range-bound within the Donchian channel, but CVD is steadily rising. Aggressive buyers are accumulating below the surface. When the Donchian upper band finally breaks, enter with high conviction and tighter-than-normal stops.

**Pattern 4 — Absorption at support (mean reversion entry):** Price touches Donchian lower band while CVD is flat or slightly positive. Despite selling pressure pushing price to the band, buying absorption is holding. Enter the mean reversion fade with CVD confirmation.

```yaml
# CVD-confirmed breakout filter
- signal: cvd_divergence
  field: price_cvd_aligned  # price and CVD both making new highs/lows
  op: "=="
  value: true

# CVD divergence for Turtle Soup fade
- signal: cvd_divergence
  field: bearish_divergence  # price new high, CVD lower high
  op: "=="
  value: true  # enables the fade entry
```

-----

## Initial Balance integration with Donchian signals

The Initial Balance (first 60 minutes, 9:30–10:30 AM ET) provides the day's structural framework. Research shows a **70-75% probability** of trend continuation when price closes a 30-minute period outside the IB, and a **70-75% probability** of reversal to the opposite IB boundary when the breakout fails.

### IB-Donchian synergy patterns

**Narrow IB + Donchian squeeze:** When both the IB range and Donchian channel width are below their 25th percentile, the day is coiling for a large directional move. Use the squeeze breakout flavor with enhanced conviction once the IB boundary breaks.

**Wide IB + Donchian range-bound:** When the IB range is above its 75th percentile and Donchian confirms range-bound conditions, the day is likely rotational. Activate the mean reversion fade at IB High and IB Low, using Donchian bands as additional confirmation.

**IB breakout + Donchian breakout alignment:** When price breaks above both the IB high AND the Donchian upper band simultaneously, the signal is exceptionally strong. Both structural (IB) and statistical (Donchian) breakout levels agree.

```yaml
# Narrow IB squeeze day detection
- signal: initial_balance
  field: width_percentile
  op: "<="
  value: 25

- signal: orb_range_size
  field: relative_to_atr
  op: "<="
  value: 0.75  # IB range is small relative to recent ATR
```

-----

## Session timing: when each flavor performs best

Session timing is a non-negotiable filter for MES. The data is clear on when different strategies work and fail.

**9:30–9:45 AM (first 15 minutes): BLOCK ALL ENTRIES.** Chaotic price action, false signals, elevated slippage. Let the market stabilize.

**9:45–10:30 AM (IB formation): Observe only.** Collect IB range data. No Donchian entries. Exception: if a Donchian squeeze fired pre-market and the open triggers a massive directional move with RVOL > 3.0×, the squeeze breakout flavor may enter.

**10:30 AM–12:00 PM (post-IB, morning session): Breakout and midline flavors.** This is the window where genuine trend continuation from the IB breakout develops. Breakout entries have the highest follow-through here. Midline pullback entries work well as the trend pauses and resumes.

**12:00–1:30 PM (lunch): Mean reversion fades only.** Volume drops 40-60%. Breakout signals during lunch are noise. The mean reversion fade at Donchian channel extremes is ideal—price oscillates in a tighter range, and band-to-midline moves are predictable.

**1:30–2:00 PM (transition): Reduced activity.** Market rebuilds participation. Monitor for squeeze conditions developing.

**2:00–2:45 PM (afternoon breakout window): Squeeze breakout and breakout flavors.** Research identifies this as the **optimal breakout window**—if the market has been range-bound all day, the afternoon breakout frequently carries into the close.

**3:00–4:00 PM (power hour): Trend-following only. DO NOT FADE.** The last-hour trend has strong follow-through as day traders liquidate and overnight traders establish positions. Midline pullback entries align with power hour moves. Fading a power hour trend is the most reliable way to lose money.

```yaml
# Session window implementation
session_windows:
  morning_breakout:
    start: "10:30"
    end: "12:00"
    allowed_flavors: [breakout, midline, squeeze]

  lunch_reversion:
    start: "12:00"
    end: "13:30"
    allowed_flavors: [fade]

  afternoon_breakout:
    start: "14:00"
    end: "15:00"
    allowed_flavors: [breakout, squeeze, midline]

  power_hour:
    start: "15:00"
    end: "15:45"
    allowed_flavors: [breakout, midline]  # NO fades

  blocked:
    start: "09:30"
    end: "10:30"
    allowed_flavors: []  # IB formation, no entries
```

-----

## The complete 2-tier tuning pipeline

### Tier 1: Coarse sweep (identify viable parameter regions)

Run all four flavors across the full parameter space with wide steps. The goal is to identify which parameter regions produce positive expectancy, not to find the optimal point.

**Tier 1 sweep dimensions (per flavor):**

|Dimension              |Values                          |Count|
|-----------------------|--------------------------------|-----|
|`donchian_period`      |10, 15, 20, 25, 30, 35, 40      |7    |
|`atr_stop_multiplier`  |1.25, 1.75, 2.25, 2.75          |4    |
|`atr_target_multiplier`|2.0, 3.0, 4.0, 5.0              |4    |
|`adx_threshold`        |18, 22, 26                      |3    |
|`rvol_threshold`       |1.2, 1.5, 2.0                   |3    |
|`regime_filter`        |[TRENDING], [RANGE_BOUND], [ALL]|3    |

**Total Tier 1 configs per flavor:** 7 × 4 × 4 × 3 × 3 × 3 = **3,024**
**Across 4 flavors:** ~12,096 configs

Evaluate on: net P&L, Sharpe ratio, max drawdown, win rate, profit factor. **Discard any config with Sharpe < 0.5 or profit factor < 1.1.**

### Tier 2: Fine-grained optimization (refine the survivors)

Take the top 5% of Tier 1 configs per flavor. Narrow the parameter ranges to ±2 steps around the winning values, decrease step sizes by 50%, and add secondary parameters:

|Secondary Parameter      |Values                   |
|-------------------------|-------------------------|
|`trailing_stop_atr`      |1.0, 1.25, 1.5, 1.75, 2.0|
|`trailing_activation_atr`|0.5, 0.75, 1.0, 1.5      |
|`time_stop_bars`         |10, 15, 20, 30, 45       |
|`session_start_offset`   |15, 30, 45, 60 min       |
|`rsi_threshold`          |25, 30, 35 (for fades)   |
|`vpin_cdf_threshold`     |0.5, 0.6, 0.7, 0.8       |
|`cvd_divergence_enabled` |true, false              |

**Walk-forward validation:** Use 70% in-sample / 30% out-of-sample with rolling 6-month windows. Any config that degrades by >30% out-of-sample is overfitted—discard it.

### Robustness checks

After Tier 2, run three robustness tests on final candidates:

1. **Parameter neighborhood stability:** Shift each parameter ±1 step. If performance drops >20%, the parameter is overfit to noise.
1. **Regime stress test:** Run the config exclusively on 2020 (COVID crash), 2022 (rate hikes), and 2024 (low-vol melt-up). It should not catastrophically fail in any single regime.
1. **Commission sensitivity:** Double the commission assumption from $0.295 to $0.59 per side. If the strategy turns unprofitable, the edge is too thin for production.

-----

## MES-specific constants and their strategic implications

These numbers should be hardcoded in your bot's configuration:

```yaml
instrument:
  symbol: MES
  tick_size: 0.25        # minimum price increment
  tick_value: 1.25       # dollar value per tick per contract
  point_value: 5.00      # dollar value per full point per contract
  commission_per_side: 0.295  # per contract per side
  round_trip_cost: 0.59  # per contract per round trip

  # Derived minimum edge requirements:
  # 1-tick scalp profit: $1.25 - $0.59 = $0.66 net (47% of gross eaten by commissions)
  # 2-tick scalp profit: $2.50 - $0.59 = $1.91 net (24% commission drag)
  # 4-tick scalp profit: $5.00 - $0.59 = $4.41 net (12% commission drag)
  # 8-tick scalp profit: $10.00 - $0.59 = $9.41 net (6% commission drag)
```

The commission math reveals a critical constraint: **1-tick targets are nonviable** (47% commission drag). Even 2-tick targets surrender 24% to commissions. The minimum viable profit target for MES scalping is **4 ticks (1 point, $5.00)** where commission drag drops to 12%. This directly impacts strategy design—ultra-tight scalping is structurally uneconomic on MES.

For stop losses, the minimum practical stop is **4 ticks (1 point)**—anything tighter produces excessive stop-outs from normal 1-minute noise (ATR on 1m is typically 1-3 points). The sweet spot for MES intraday strategies based on the research is:

- **Profit target:** 4-12 ticks (1-3 points, $5-$15)
- **Stop loss:** 6-12 ticks (1.5-3 points, $7.50-$15)
- **R:R ratio:** 1:1 to 1:2 (mean reversion) or 1:2 to 1:3 (breakout/squeeze)

-----

## Conclusion: from zero profitable configs to a production system

The path from 0/2,304 to a profitable Donchian system requires three architectural shifts. First, **regime gating is mandatory**—never run a single Donchian flavor in all market conditions. The research consistently shows HMM regime detection improves Sharpe ratios by **0.4–1.4** and reduces maximum drawdown by **30-50%**. Second, **volume confirmation transforms breakout reliability** from 39% to 65%—a single filter that nearly doubles your hit rate. Third, **deploying all four flavors in their correct conditions** turns Donchian from a one-trick breakout strategy into a complete market-state-adaptive system.

The most important insight from this research is counterintuitive: **Donchian's best MES application is not breakout—it's mean reversion fading**. The channel extremes, combined with RSI/stochastic oversold readings, CVD divergence confirmation, and range-bound regime gating, produce the highest win-rate (65-75%) and most consistent P&L in the mean-reverting microstructure of equity index futures. Breakout and squeeze flavors capture larger moves but with lower frequency and win rate. The midline pullback provides steady, moderate returns in trending conditions.

Your 2-tier sweep should prioritize the fade and midline flavors—these are structurally aligned with MES microstructure. Breakout and squeeze flavors are the high-beta additions that capture outsized moves on the relatively infrequent genuine trend days. Run all four in parallel, gated by `regime_v2`, and the system becomes what the Turtle Trading system was for commodities in the 1980s: a systematic edge adapted to its instrument's statistical DNA.
