"""Abstract base class definition for market data providers."""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from data.schemas.market_data import ProviderMetadata


class BaseMarketDataProvider(ABC):
    """Abstract interface defining required behaviors for all market data providers."""

    @abstractmethod
    def fetch_market_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch and normalize historical market data for a symbol.

        Required normalized output columns:
        - timestamp (datetime64[ns, UTC] or DatetimeIndex)
        - open (float)
        - high (float)
        - low (float)
        - close (float)
        - volume (float)
        - symbol (str)
        - data_source (str)
        """
        pass

    @abstractmethod
    def fetch_latest_data(self, symbol: str) -> pd.DataFrame:
        """Fetch the most recent available market data bar for a symbol."""
        pass

    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is recognized and supported by the provider."""
        pass

    @abstractmethod
    def get_provider_metadata(self) -> ProviderMetadata:
        """Return provider capability metadata."""
        pass
