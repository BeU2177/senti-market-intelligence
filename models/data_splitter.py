"""Chronological and walk-forward time-series data splitters with purging and embargo."""

from typing import Tuple, List, Dict, Any, Generator
import numpy as np
import pandas as pd
from utils.logging_config import get_logger

logger = get_logger(__name__)


class TimeSplitter:
    """Chronological time-series data splitter (Train 70%, Validation 15%, Test 15%)."""

    def __init__(self, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Split ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(
        self, df: pd.DataFrame, feature_cols: List[str], target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Chronologically split DataFrame into X_train, y_train, X_val, y_val, X_test, y_test."""
        df_clean = df.dropna(subset=[target_col]).sort_values("timestamp").reset_index(drop=True)

        n = len(df_clean)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_df = df_clean.iloc[:train_end]
        val_df = df_clean.iloc[train_end:val_end]
        test_df = df_clean.iloc[val_end:]

        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_val, y_val = val_df[feature_cols], val_df[target_col]
        X_test, y_test = test_df[feature_cols], test_df[target_col]

        logger.info(
            f"TimeSplitter: Total={n} -> Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} "
            f"(Target: '{target_col}')"
        )

        return X_train, y_train, X_val, y_val, X_test, y_test


class WalkForwardSplitter:
    """Expanding-window walk-forward splitter with target purging and embargo windows."""

    def __init__(self, n_splits: int = 4, min_train_size: int = 50, purge_window: int = 7, embargo_window: int = 2):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.purge_window = purge_window
        self.embargo_window = embargo_window

    def split_indices(self, n_samples: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate (train_indices, val_indices) for expanding-window validation with purging."""
        if n_samples < self.min_train_size + self.n_splits * 10:
            # Fallback for small datasets
            val_size = max(5, int(n_samples * 0.15))
            train_idx = np.arange(0, n_samples - val_size)
            val_idx = np.arange(n_samples - val_size, n_samples)
            yield train_idx, val_idx
            return

        val_size = int((n_samples - self.min_train_size) / self.n_splits)

        for i in range(self.n_splits):
            train_end = self.min_train_size + i * val_size
            val_start = train_end + self.purge_window + self.embargo_window
            val_end = min(val_start + val_size, n_samples)

            if val_start >= n_samples or val_start >= val_end:
                break

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)

            yield train_idx, val_idx
