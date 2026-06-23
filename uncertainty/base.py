"""Abstract base class defining the interface for uncertainty wrappers."""

from abc import ABC, abstractmethod

import numpy as np


class BaseUncertaintyWrapper(ABC):
    @abstractmethod
    def calibrate(self, model, X_calib, y_calib):
        raise NotImplementedError

    @abstractmethod
    def predict_interval(self, model, X, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
