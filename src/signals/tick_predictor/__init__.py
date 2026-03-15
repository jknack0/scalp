"""TickDirectionPredictor — Phase 13 signal (EV output).

Predicts MES tick direction ahead from OHLCV-1s features and outputs normalized
expected value.  Positive EV = long edge, negative = short edge.

Architecture::

    Databento OHLCV-1s
      -> BarEvent (EventBus)
        -> SignalEngine.compute(bar_window)
          -> TickPredictorSignal.compute()
            -> FeatureBuilder.on_bar() -> feature_vector (26,)
              -> LightGBM.predict() -> raw_proba (3,)
                -> TemperatureCalibrator -> cal_proba (3,)
                  -> EV = (p_up * tp) - (p_down * sl) - cost
                    -> SignalResult(value=ev_normalized, metadata={...})

Value Encoding
--------------
SignalResult.value = ev_normalized = EV / tp_ticks:
  value > 0  -> long edge,  magnitude = fraction of TP  (e.g. 0.20 = 20% of TP)
  value < 0  -> short edge, magnitude = fraction of TP  (e.g. -0.15 = 15% of TP)
  value ~ 0  -> no edge (cost > expected gain)

Strategy YAML Integration
-------------------------
Standalone entry (strategy YAML):

  filters:
    # Take LONG only when model shows positive EV above entry threshold
    - signal: tick_predictor
      expr: "> 0.15"

    # Take SHORT only when model shows negative EV
    - signal: tick_predictor
      expr: "< -0.15"

    # High conviction only
    - signal: tick_predictor
      expr: "> 0.30"

As filter for existing strategies (e.g. gate a LONG ORB entry):

  filters:
    # Block LONG entry if model sees negative EV
    - signal: tick_predictor
      expr: "> 0.0"

ExitEngine patterns (zero changes to ExitEngine required):

  # Adverse signal exit (exit LONG if EV flips negative)
  exits:
    - type: adverse_signal_exit
      enabled: true
      signal: tick_predictor
      adverse_threshold: -0.15

  # Signal bound exit (exit if EV collapses toward zero)
  exits:
    - type: signal_bound_exit
      enabled: true
      signal: tick_predictor
      lower_bound: -0.10
      upper_bound: 0.10
"""

from src.signals.tick_predictor.signal import TickPredictorSignal

__all__ = ["TickPredictorSignal"]
