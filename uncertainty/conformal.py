"""Split conformal prediction intervals for RUL point predictors."""

import numpy as np

from uncertainty.base import BaseUncertaintyWrapper


class ConformalPredictor(BaseUncertaintyWrapper):
    def __init__(self):
        self.scores = None

    def calibrate(self, model, X_calib, y_calib):
        predictions = model.predict(X_calib)
        self.scores = np.abs(y_calib - predictions)

    def predict_interval(self, model, X, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.scores)
        quantile_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        q = np.quantile(self.scores, quantile_level)
        predictions = model.predict(X)
        return predictions - q, predictions + q
