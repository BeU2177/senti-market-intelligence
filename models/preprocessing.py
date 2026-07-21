"""Preprocessing pipeline fitted exclusively on training set to prevent data leakage."""

import numpy as np
import pandas as pd
from typing import List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from utils.logging_config import get_logger

logger = get_logger(__name__)


class FeaturePreprocessor:
    """Feature imputation and scaling pipeline strictly fitted on training set."""

    def __init__(self, impute_strategy: str = "median", scale: bool = True):
        self.imputer = SimpleImputer(strategy=impute_strategy)
        self.scaler = StandardScaler() if scale else None
        self.is_fitted = False
        self.feature_names: List[str] = []

    def fit(self, X_train: pd.DataFrame) -> "FeaturePreprocessor":
        """Fits imputer and scaler EXCLUSIVELY on training data."""
        self.feature_names = list(X_train.columns)
        
        # Fit imputer on X_train
        X_imp = self.imputer.fit_transform(X_train)

        # Fit scaler on X_train imputer output
        if self.scaler:
            self.scaler.fit(X_imp)

        self.is_fitted = True
        logger.info(f"FeaturePreprocessor fitted on X_train with {len(self.feature_names)} features.")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms validation or test DataFrame using fitted imputer and scaler parameters."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePreprocessor must be fitted on X_train before calling transform()!")

        # Ensure correct column order
        X_reordered = X[self.feature_names]
        
        X_imp = self.imputer.transform(X_reordered)
        if self.scaler:
            return self.scaler.transform(X_imp)
        return X_imp

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit exclusively on X_train and return transformed X_train."""
        return self.fit(X_train).transform(X_train)
