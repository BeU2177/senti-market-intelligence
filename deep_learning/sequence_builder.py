"""Sequence tensor builder for 3D time-series inputs with strict split boundary isolation."""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from utils.logging_config import get_logger

logger = get_logger(__name__)


class SequenceBuilder:
    """Builds 3D feature sequence tensors [N, seq_len, F] and multi-horizon target arrays [N, H]."""

    def __init__(self, sequence_length: int = 30, target_horizons: Optional[List[int]] = None):
        self.sequence_length = sequence_length
        self.target_horizons = target_horizons or [1, 3, 5, 7]
        self.target_cols = [f"future_return_{h}d" for h in self.target_horizons]

    def build_sequences(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Constructs 3D sequences [N, seq_len, F] and 2D targets [N, H] from a contiguous DataFrame slice."""
        req_cols = feature_cols + [c for c in self.target_cols if c in df.columns]
        df_clean = df.dropna(subset=self.target_cols).sort_values("timestamp").reset_index(drop=True)

        n_rows = len(df_clean)
        if n_rows < self.sequence_length:
            logger.warning(
                f"DataFrame slice has only {n_rows} rows, which is less than sequence_length {self.sequence_length}."
            )
            return np.empty((0, self.sequence_length, len(feature_cols))), np.empty((0, len(self.target_cols)))

        X_data = df_clean[feature_cols].values
        y_data = df_clean[self.target_cols].values

        X_sequences = []
        y_targets = []

        for i in range(self.sequence_length - 1, n_rows):
            # Window: i - sequence_length + 1 -> i (inclusive)
            seq_window = X_data[i - self.sequence_length + 1 : i + 1]
            target_val = y_data[i]

            X_sequences.append(seq_window)
            y_targets.append(target_val)

        X_arr = np.array(X_sequences, dtype=np.float32)
        y_arr = np.array(y_targets, dtype=np.float32)

        logger.info(
            f"Built sequence tensors: X.shape={X_arr.shape}, y.shape={y_arr.shape} "
            f"(seq_len={self.sequence_length}, features={len(feature_cols)}, horizons={len(self.target_cols)})"
        )
        return X_arr, y_arr

    def build_split_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Chronologically splits DataFrame into Train/Val/Test slices BEFORE generating sequences to guarantee zero boundary leakage."""
        df_clean = df.dropna(subset=self.target_cols).sort_values("timestamp").reset_index(drop=True)

        n = len(df_clean)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df_clean.iloc[:train_end]
        val_df = df_clean.iloc[train_end:val_end]
        test_df = df_clean.iloc[val_end:]

        X_tr, y_tr = self.build_sequences(train_df, feature_cols)
        X_v, y_v = self.build_sequences(val_df, feature_cols)
        X_te, y_te = self.build_sequences(test_df, feature_cols)

        return X_tr, y_tr, X_v, y_v, X_te, y_te
