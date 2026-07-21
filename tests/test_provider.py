"""Unit tests for hardened market data provider and exception handling."""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

from data.providers.yfinance_provider import YFinanceProvider
from utils.exceptions import SymbolNotFoundError, NetworkError
from data.schemas.market_info import get_market_metadata


def test_market_metadata_resolution():
    """Verify market metadata resolution across multi-market symbols."""
    us_meta = get_market_metadata("AAPL")
    assert us_meta.exchange == "NASDAQ/NYSE"
    assert us_meta.currency == "USD"
    assert us_meta.country == "United States"

    in_meta = get_market_metadata("RELIANCE.NS")
    assert in_meta.exchange == "NSE"
    assert in_meta.currency == "INR"
    assert in_meta.country == "India"

    uae_meta = get_market_metadata("EMAAR.AE")
    assert uae_meta.exchange == "DFM"
    assert uae_meta.currency == "AED"
    assert uae_meta.country == "United Arab Emirates"


def test_yfinance_provider_empty_symbol():
    """Verify YFinanceProvider raises SymbolNotFoundError for empty symbol input."""
    provider = YFinanceProvider()
    with pytest.raises(SymbolNotFoundError):
        provider.fetch_market_data("")


@patch("yfinance.Ticker")
def test_yfinance_provider_not_found(mock_ticker):
    """Verify YFinanceProvider raises SymbolNotFoundError when ticker is delisted/not found."""
    mock_instance = MagicMock()
    mock_instance.history.side_effect = Exception("Symbol AAPL_INVALID not found or delisted")
    mock_ticker.return_value = mock_instance

    provider = YFinanceProvider()
    with pytest.raises(SymbolNotFoundError):
        provider.fetch_market_data("AAPL_INVALID")


@patch("yfinance.Ticker")
def test_yfinance_provider_network_error(mock_ticker):
    """Verify YFinanceProvider raises NetworkError on connection failure."""
    mock_instance = MagicMock()
    mock_instance.history.side_effect = Exception("Connection timeout while reaching endpoint")
    mock_ticker.return_value = mock_instance

    provider = YFinanceProvider()
    with pytest.raises(NetworkError):
        provider.fetch_market_data("AAPL")
