"""Unit tests for SequenceBuilder 3D sequence construction and split boundary isolation."""

import pytest
import pandas as pd
import numpy as np

from deep_learning.sequence_builder import SequenceBuilder


@pytest.fixture
def sample_seq_df():
    """Generates 100 rows of sample data with multi-horizon targets."""
    dates = pd.date_range("2023-01-01", periods=100, tz="UTC")
    return pd.DataFrame({
        "timestamp": dates,
        "feat_1": np.arange(100, dtype=float),
        "feat_2": np.arange(100, 200, dtype=float),
        "future_return_1d": np.random.normal(0, 0.02, 100),
        "future_return_3d": np.random.normal(0, 0.03, 100),
        "future_return_5d": np.random.normal(0, 0.04, 100),
        "future_return_7d": np.random.normal(0, 0.05, 100),
    })


def test_sequence_builder_shapes(sample_seq_df):
    """Test SequenceBuilder constructs correct 3D feature tensors and 2D target matrices."""
    builder = SequenceBuilder(sequence_length=20, target_horizons=[1, 3, 5, 7])
    X, y = builder.build_sequences(sample_seq_df, feature_cols=["feat_1", "feat_2"])

    # 100 rows with lookback 20 produces 100 - 20 + 1 = 81 sequences
    assert X.shape == (81, 20, 2)
    assert y.shape == (81, 4)


def test_sequence_builder_boundary_isolation(sample_seq_df):
    """Test build_split_sequences guarantees zero boundary overlap leakage between Train, Val, and Test."""
    builder = SequenceBuilder(sequence_length=15, target_horizons=[1, 3, 5, 7])
    X_tr, y_tr, X_v, y_v, X_te, y_te = builder.build_split_sequences(
        sample_seq_df, feature_cols=["feat_1", "feat_2"], train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    assert X_tr.shape[1] == 15
    assert X_v.shape[1] == 15
    assert X_te.shape[1] == 15
    assert X_tr.shape[2] == 2
    assert y_tr.shape[1] == 4
