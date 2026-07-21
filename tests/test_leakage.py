"""Unit tests for automated lookahead and target leakage detector."""

import pytest
import pandas as pd
import numpy as np

from features.leakage_validator import LeakageValidator


def test_leakage_validator_clean_dataset():
    """Test validator approves a clean dataset without lookahead features."""
    dates = pd.date_range("2023-01-01", periods=20, tz="UTC")
    close = pd.Series(np.arange(100.0, 120.0), index=dates)

    df = pd.DataFrame({
        "timestamp": dates,
        "close": close,
        "sma_5": close.rolling(5).mean(),
        "future_return_1d": (close.shift(-1) / close) - 1.0,
    })

    validator = LeakageValidator()
    report = validator.validate_dataset(
        df=df,
        feature_cols=["sma_5"],
        target_cols=["future_return_1d"],
    )

    assert report.is_clean is True
    assert len(report.leakage_errors) == 0


def test_leakage_validator_detects_target_in_features():
    """Test validator flags target column accidentally placed in feature list."""
    dates = pd.date_range("2023-01-01", periods=20, tz="UTC")
    close = pd.Series(np.arange(100.0, 120.0), index=dates)

    df = pd.DataFrame({
        "timestamp": dates,
        "close": close,
        "future_return_1d": (close.shift(-1) / close) - 1.0,
    })

    validator = LeakageValidator()
    report = validator.validate_dataset(
        df=df,
        feature_cols=["future_return_1d"],  # Intentionally passing target as feature
        target_cols=["future_return_1d"],
    )

    assert report.is_clean is False
    assert any("Target column" in err or "Overlap detected" in err for err in report.leakage_errors)


def test_leakage_validator_detects_future_price_feature():
    """Test validator detects feature with synthetic future close price correlation."""
    dates = pd.date_range("2023-01-01", periods=30, tz="UTC")
    np.random.seed(42)
    close = pd.Series(100.0 + np.cumsum(np.random.normal(0, 1, 30)), index=dates)

    # Intentionally create a feature using future close price shift(-1)
    leaked_feature = close.shift(-1) * 1.05

    df = pd.DataFrame({
        "timestamp": dates,
        "close": close,
        "bad_feature_future": leaked_feature,
        "future_return_1d": (close.shift(-1) / close) - 1.0,
    })

    validator = LeakageValidator()
    report = validator.validate_dataset(
        df=df,
        feature_cols=["bad_feature_future"],
        target_cols=["future_return_1d"],
    )

    assert report.is_clean is False
    assert any("suspicious correlation" in err for err in report.leakage_errors)
