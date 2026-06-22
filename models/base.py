"""Abstract base class defining the interface for RUL prediction models."""

from abc import ABC, abstractmethod

import numpy as np


class BaseRULModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError
