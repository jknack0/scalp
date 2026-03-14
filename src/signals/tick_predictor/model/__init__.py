"""LightGBM model training and calibration for TickDirectionPredictor.

Imports are lazy to avoid requiring lightgbm at bot startup.
The live bot only needs TemperatureCalibrator (pure numpy);
ModelTrainer is only used by scripts/tick_predictor/train.py.
"""


def __getattr__(name: str):
    if name == "TemperatureCalibrator":
        from src.signals.tick_predictor.model.calibrator import TemperatureCalibrator
        return TemperatureCalibrator
    if name == "ModelTrainer":
        from src.signals.tick_predictor.model.trainer import ModelTrainer
        return ModelTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ModelTrainer", "TemperatureCalibrator"]
