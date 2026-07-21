"""Benchmark baseline models for financial time-series prediction."""

import numpy as np
import pandas as pd


class PersistenceBaseline:
    """Predicts zero future return (or previous return value) as a benchmark."""

    def __init__(self, constant_value: float = 0.0):
        self.constant_value = constant_value
        self.is_fitted = True

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "PersistenceBaseline":
        """No parameters to fit."""
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return constant zero return prediction array."""
        n = len(X) if isinstance(X, (np.ndarray, list)) else len(X)
        return np.full(n, self.constant_value)


class HistoricalMeanBaseline:
    """Predicts historical mean return of the training target."""

    def __init__(self):
        self.mean_value = 0.0
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "HistoricalMeanBaseline":
        """Fit mean value on training target."""
        self.mean_value = float(np.mean(y_train))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return training mean return prediction array."""
        n = len(X) if isinstance(X, (np.ndarray, list)) else len(X)
        return np.full(n, self.mean_value)
