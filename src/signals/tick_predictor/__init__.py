"""TickDirectionPredictor — Phase 13 signal.

Predicts MES tick direction 15 bars (15 seconds) ahead from OHLCV-1s order flow
features.  Output: signed float encoding direction + confidence for use in
FilterEngine and ExitEngine.

Architecture::

    Databento OHLCV-1s
      -> BarEvent (EventBus)
        -> SignalEngine.compute(bar_window)
          -> TickPredictorSignal.compute()
            -> FeatureBuilder.on_bar() -> feature_vector (18,)
              -> LightGBM.predict() -> raw_proba (3,)
                -> TemperatureCalibrator -> cal_proba (3,)
                  -> SignalResult(value=signed_confidence, metadata={...})

Value Encoding
--------------
SignalResult.value encodes direction + confidence as a single float:
  value > 0  -> UP,   magnitude = confidence  (e.g.  0.72 = UP   72%)
  value < 0  -> DOWN, magnitude = confidence  (e.g. -0.68 = DOWN 68%)
  value ~ 0  -> FLAT

Strategy YAML Integration
-------------------------
Recommended ExitEngine patterns (zero changes to ExitEngine required):

  # Pattern 1 -- Adverse signal exit (exit if predictor flips against position)
  exits:
    - type: adverse_signal_exit
      enabled: true
      signal: tick_predictor
      adverse_threshold: -0.60

  # Pattern 2 -- Signal bound exit (exit if confidence collapses)
  exits:
    - type: signal_bound_exit
      enabled: true
      signal: tick_predictor
      lower_bound: -0.30
      upper_bound: 0.30

  # Pattern 3 -- FilterEngine gate (only trade when predictor is aligned)
  filters:
    - signal: tick_predictor
      expr: "> 0.55"
"""

from src.signals.tick_predictor.signal import TickPredictorSignal

__all__ = ["TickPredictorSignal"]
