"""Unit tests for DataQualityValidator module."""

import pytest
import pandas as pd
from data.validation.data_quality import DataQualityValidator


@pytest.fixture
def sample_valid_df():
    """Returns a valid market data DataFrame."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=5, tz="UTC"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [102.0, 103.0, 104.0, 105.0, 106.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            "symbol": ["TEST"] * 5,
            "data_source": ["unit_test"] * 5,
        }
    )


def test_validator_valid_data(sample_valid_df):
    """Test validator passes clean valid DataFrame."""
    validator = DataQualityValidator()
    result = validator.validate(sample_valid_df)

    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.row_count == 5
    assert result.duplicate_timestamps == 0


def test_validator_empty_df():
    """Test validator handles empty DataFrame correctly."""
    validator = DataQualityValidator()
    result = validator.validate(pd.DataFrame())

    assert result.is_valid is False
    assert "DataFrame is empty or None" in result.errors[0]


def test_validator_missing_columns(sample_valid_df):
    """Test validator flags missing required columns."""
    df_incomplete = sample_valid_df.drop(columns=["high", "volume"])
    validator = DataQualityValidator()
    result = validator.validate(df_incomplete)

    assert result.is_valid is False
    assert any("high" in err for err in result.errors)


def test_validator_duplicate_timestamps(sample_valid_df):
    """Test validator flags duplicate timestamps."""
    sample_valid_df.loc[1, "timestamp"] = sample_valid_df.loc[0, "timestamp"]
    validator = DataQualityValidator()
    result = validator.validate(sample_valid_df)

    assert result.is_valid is False
    assert result.duplicate_timestamps == 1
    assert any("duplicate timestamp" in err for err in result.errors)


def test_validator_invalid_ohlc(sample_valid_df):
    """Test validator flags invalid OHLC relationships (High < Low, High < Open)."""
    sample_valid_df.loc[0, "high"] = 90.0  # High lower than Open (100.0) and Low (99.0)
    validator = DataQualityValidator()
    result = validator.validate(sample_valid_df)

    assert result.is_valid is False
    assert any("High < Low" in err for err in result.errors)
    assert any("High < Open" in err for err in result.errors)


def test_validator_negative_values(sample_valid_df):
    """Test validator flags negative prices and volume."""
    sample_valid_df.loc[0, "close"] = -10.0
    sample_valid_df.loc[1, "volume"] = -500.0
    validator = DataQualityValidator()
    result = validator.validate(sample_valid_df)

    assert result.is_valid is False
    assert any("negative price" in err for err in result.errors)
    assert any("negative volume" in err for err in result.errors)


def test_validator_zero_price_warning(sample_valid_df):
    """Test validator emits warning for zero prices."""
    sample_valid_df.loc[0, "open"] = 0.0
    sample_valid_df.loc[0, "low"] = 0.0
    validator = DataQualityValidator()
    result = validator.validate(sample_valid_df)

    assert result.is_valid is True  # Warning does not invalidate unless errors exist
    assert len(result.warnings) > 0
    assert any("zero price" in warn for warn in result.warnings)


def test_validator_freshness_classification(sample_valid_df):
    """Test freshness status classification for historical data."""
    validator = DataQualityValidator()
    result = validator.validate(sample_valid_df)

    # Sample dates are 2023-01-01 -> Classified as STALE relative to present
    assert result.freshness_status.value == "STALE"


def test_validator_data_gap_detection(sample_valid_df):
    """Test validator detects unusually large timestamp gaps exceeding normal weekends."""
    # Insert a 10-day gap between row 2 and row 3
    dates = [
        pd.Timestamp("2023-01-01", tz="UTC"),
        pd.Timestamp("2023-01-02", tz="UTC"),
        pd.Timestamp("2023-01-15", tz="UTC"),  # 13 day gap
        pd.Timestamp("2023-01-16", tz="UTC"),
        pd.Timestamp("2023-01-17", tz="UTC"),
    ]
    sample_valid_df["timestamp"] = dates

    validator = DataQualityValidator()
    result = validator.validate(sample_valid_df, interval="1d")

    assert len(result.data_gaps) == 1
    assert result.data_gaps[0]["gap_days"] >= 10.0

