"""Unit tests for chronological time-series splitters and walk-forward validation."""

import pytest
import pandas as pd
import numpy as np

from models.data_splitter import TimeSplitter, WalkForwardSplitter


@pytest.fixture
def sample_split_df():
    """Generates 100 chronological rows of sample data."""
    dates = pd.date_range("2023-01-01", periods=100, tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "feature_1": np.arange(100, dtype=float),
            "feature_2": np.arange(100, 200, dtype=float),
            "future_return_1d": np.random.normal(0, 0.02, 100),
        }
    )


def test_time_splitter_chronological_ratios(sample_split_df):
    """Test TimeSplitter splits chronologically into 70% Train, 15% Val, 15% Test without overlap."""
    splitter = TimeSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    X_tr, y_tr, X_v, y_v, X_te, y_te = splitter.split(
        sample_split_df, feature_cols=["feature_1", "feature_2"], target_col="future_return_1d"
    )

    assert len(X_tr) == 70
    assert len(X_v) == 15
    assert len(X_te) == 15

    # Verify chronological order (Max train feature_1 < Min val feature_1 < Min test feature_1)
    assert X_tr["feature_1"].max() < X_v["feature_1"].min()
    assert X_v["feature_1"].max() < X_te["feature_1"].min()


def test_walk_forward_splitter_expanding(sample_split_df):
    """Test WalkForwardSplitter generates expanding window folds with purging."""
    splitter = WalkForwardSplitter(n_splits=3, min_train_size=40, purge_window=5, embargo_window=1)
    folds = list(splitter.split_indices(len(sample_split_df)))

    assert len(folds) >= 1
    for train_idx, val_idx in folds:
        # Train indices come strictly before validation indices + purge window
        assert train_idx.max() < val_idx.min()
        # Purging gap enforced
        assert val_idx.min() - train_idx.max() > 5
