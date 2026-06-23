"""Point-prediction and interval-calibration metrics for RUL models."""

import numpy as np


class CalibrationEvaluator:
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
        return float(np.mean((y_true >= lower) & (y_true <= upper)))

    @staticmethod
    def sharpness(lower: np.ndarray, upper: np.ndarray) -> float:
        return float(np.mean(upper - lower))
