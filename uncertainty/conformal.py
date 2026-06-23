"""Split conformal prediction intervals for RUL point predictors."""

import numpy as np

from uncertainty.base import BaseUncertaintyWrapper


class ConformalPredictor(BaseUncertaintyWrapper):
    def __init__(self):
        self.scores = None

    def calibrate(self, model, X_calib, y_calib):
        last_window_mask = self._last_window_per_engine_mask(y_calib)
        X_calib_last = X_calib[last_window_mask]
        y_calib_last = y_calib[last_window_mask]

        predictions = model.predict(X_calib_last)
        self.scores = np.abs(y_calib_last - predictions)

    @staticmethod
    def _last_window_per_engine_mask(y_calib: np.ndarray) -> np.ndarray:
        # X_calib/y_calib hold every sliding window from every calibration
        # engine, grouped contiguously per engine in cycle order. Within an
        # engine RUL never increases window-to-window; it only jumps back up
        # where the next engine's windows begin. The last window of each
        # engine is therefore the one right before such a jump (or the very
        # last window overall) -- exactly matching the single last-window-only
        # setup used for the test set, so calibration and test errors are
        # drawn from the same distribution.
        is_last = np.ones(len(y_calib), dtype=bool)
        is_last[:-1] = y_calib[1:] > y_calib[:-1]
        return is_last

    def predict_interval(self, model, X, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        q = np.quantile(self.scores, 1 - alpha)
        predictions = model.predict(X)
        return predictions - q, predictions + q
