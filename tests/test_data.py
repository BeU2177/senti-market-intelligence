"""Unit tests for market data schemas, providers, and service layer."""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime, timezone

from data.schemas.market_data import (
    MarketRecord,
    ProviderMetadata,
    DataValidationResult,
    MarketDataResponse,
)
from data.providers.yfinance_provider import YFinanceProvider
from services.market_service import MarketService


def test_market_record_schema():
    """Verify MarketRecord schema instantiation and field validation."""
    record = MarketRecord(
        timestamp=datetime.now(timezone.utc),
        open=150.0,
        high=155.0,
        low=149.0,
        close=152.0,
        volume=1000000.0,
        symbol="AAPL",
        data_source="yfinance",
    )
    assert record.symbol == "AAPL"
    assert record.close == 152.0
    assert record.data_source == "yfinance"


def test_provider_metadata_schema():
    """Verify ProviderMetadata schema structure."""
    metadata = ProviderMetadata(
        provider_name="yfinance",
        description="Yahoo Finance data provider",
        supports_realtime=False,
        supported_intervals=["1d", "1wk"],
    )
    assert metadata.provider_name == "yfinance"
    assert "1d" in metadata.supported_intervals


@patch("yfinance.Ticker")
def test_yfinance_provider_normalization(mock_ticker):
    """Test YFinanceProvider normalizes columns and datetimes accurately."""
    mock_instance = MagicMock()
    # Create mock history DataFrame resembling yfinance output
    dates = pd.date_range("2023-01-01", periods=3, tz="UTC")
    raw_df = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 104.0],
            "High": [105.0, 106.0, 107.0],
            "Low": [99.0, 101.0, 103.0],
            "Close": [103.0, 105.0, 106.0],
            "Volume": [10000, 12000, 15000],
        },
        index=dates,
    )
    raw_df.index.name = "Date"
    mock_instance.history.return_value = raw_df
    mock_ticker.return_value = mock_instance

    provider = YFinanceProvider()
    df = provider.fetch_market_data("AAPL", period="1mo", interval="1d")

    assert not df.empty
    expected_cols = ["timestamp", "open", "high", "low", "close", "volume", "symbol", "data_source"]
    assert list(df.columns) == expected_cols
    assert df["symbol"].iloc[0] == "AAPL"
    assert df["data_source"].iloc[0] == "yfinance"
    assert df["close"].iloc[0] == 103.0


@patch("yfinance.Ticker")
def test_yfinance_provider_empty_data(mock_ticker):
    """Test YFinanceProvider behavior when ticker returns empty DataFrame."""
    mock_instance = MagicMock()
    mock_instance.history.return_value = pd.DataFrame()
    mock_ticker.return_value = mock_instance

    provider = YFinanceProvider()
    df = provider.fetch_market_data("INVALID_TICKER")

    assert df.empty
    assert "timestamp" in df.columns
    assert "close" in df.columns


def test_market_service_fetching():
    """Test MarketService end-to-end orchestration with mocked provider."""
    service = MarketService(provider_name="yfinance")

    mock_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=2, tz="UTC"),
            "open": [10.0, 11.0],
            "high": [12.0, 13.0],
            "low": [9.0, 10.0],
            "close": [11.0, 12.0],
            "volume": [100.0, 200.0],
            "symbol": ["MSFT", "MSFT"],
            "data_source": ["yfinance", "yfinance"],
        }
    )

    with patch.object(YFinanceProvider, "fetch_market_data", return_value=mock_df):
        response = service.get_market_data("MSFT")

        assert response.symbol == "MSFT"
        assert response.provider == "yfinance"
        assert response.row_count == 2
        assert response.validation_result.is_valid is True
