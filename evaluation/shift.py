"""Compares calibration quality across test subsets, e.g. under distribution shift."""

import pandas as pd

from evaluation.metrics import CalibrationEvaluator
from models.base import BaseRULModel
from uncertainty.base import BaseUncertaintyWrapper


class DistributionShiftAnalyzer:
    def __init__(self, model: BaseRULModel, uncertainty_wrapper: BaseUncertaintyWrapper):
        self.model = model
        self.uncertainty_wrapper = uncertainty_wrapper

    def run_shift_test(self, subset_name: str, X_test, y_test, alpha: float = 0.1) -> dict:
        preds = self.model.predict(X_test)
        lower, upper = self.uncertainty_wrapper.predict_interval(self.model, X_test, alpha=alpha)

        return {
            "subset": subset_name,
            "rmse": CalibrationEvaluator.rmse(y_test, preds),
            "picp": CalibrationEvaluator.picp(y_test, lower, upper),
            "sharpness": CalibrationEvaluator.sharpness(lower, upper),
        }

    def compare_subsets(self, results: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(results, columns=["subset", "rmse", "picp", "sharpness"])
