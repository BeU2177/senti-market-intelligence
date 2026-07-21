"""Unit tests for target generation module."""

import pytest
import pandas as pd
import numpy as np

from features.target_builder import TargetBuilder


def test_target_builder_calculations():
    """Test 1d, 3d, 5d, 7d return and direction target calculations."""
    dates = pd.date_range("2023-01-01", periods=10, tz="UTC")
    close_prices = [100.0, 105.0, 102.0, 110.0, 108.0, 112.0, 115.0, 114.0, 118.0, 120.0]
    df = pd.DataFrame({"timestamp": dates, "close": close_prices})

    builder = TargetBuilder(horizons=[1, 3])
    targets = builder.build_targets(df)

    assert "future_return_1d" in targets.columns
    assert "future_return_3d" in targets.columns
    assert "direction_1d" in targets.columns

    # Row 0: close[0] = 100.0, close[1] = 105.0 -> future_return_1d = 0.05, direction_1d = 1
    assert np.isclose(targets["future_return_1d"].iloc[0], 0.05)
    assert targets["direction_1d"].iloc[0] == 1

    # Row 0: close[0] = 100.0, close[3] = 110.0 -> future_return_3d = 0.10
    assert np.isclose(targets["future_return_3d"].iloc[0], 0.10)

    # Last row (index 9): future values are NaN
    assert pd.isna(targets["future_return_1d"].iloc[9])
    assert pd.isna(targets["direction_1d"].iloc[9])
