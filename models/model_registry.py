"""Model registry factory mapping estimator keys to instantiations."""

from typing import Dict, Any, Type, Callable
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from models.baselines import PersistenceBaseline, HistoricalMeanBaseline
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Attempt to import XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBRegressor = None


class ModelRegistry:
    """Factory registry for instantiating baseline and classical ML regression models."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def get_available_models(self) -> list[str]:
        """Return list of supported model keys."""
        models = [
            "persistence",
            "historical_mean",
            "linear",
            "ridge",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
        ]
        if HAS_XGBOOST:
            models.append("xgboost")
        return models

    def get_model(self, model_name: str, **custom_params) -> Any:
        """Instantiate a requested model by name."""
        name_clean = model_name.lower().strip()

        if name_clean == "persistence":
            return PersistenceBaseline(**custom_params)

        if name_clean == "historical_mean":
            return HistoricalMeanBaseline()

        if name_clean == "linear":
            return LinearRegression(**custom_params)

        if name_clean == "ridge":
            params = {"alpha": 1.0, "random_state": self.random_seed}
            params.update(custom_params)
            return Ridge(**params)

        if name_clean == "random_forest":
            params = {"n_estimators": 100, "max_depth": 5, "random_state": self.random_seed, "n_jobs": -1}
            params.update(custom_params)
            return RandomForestRegressor(**params)

        if name_clean == "extra_trees":
            params = {"n_estimators": 100, "max_depth": 5, "random_state": self.random_seed, "n_jobs": -1}
            params.update(custom_params)
            return ExtraTreesRegressor(**params)

        if name_clean == "gradient_boosting":
            params = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "random_state": self.random_seed}
            params.update(custom_params)
            return GradientBoostingRegressor(**params)

        if name_clean == "xgboost":
            if HAS_XGBOOST:
                params = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "random_state": self.random_seed}
                params.update(custom_params)
                return XGBRegressor(**params)
            else:
                logger.warning("XGBoost package unavailable. Falling back to GradientBoostingRegressor.")
                params = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "random_state": self.random_seed}
                params.update(custom_params)
                return GradientBoostingRegressor(**params)

        raise ValueError(f"Unknown model name '{model_name}'. Available: {self.get_available_models()}")
