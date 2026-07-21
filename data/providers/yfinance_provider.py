"""Hardened YFinance market data provider implementation."""

from typing import Optional
import pandas as pd
import yfinance as yf
from datetime import datetime

from data.providers.base_provider import BaseMarketDataProvider
from data.schemas.market_data import ProviderMetadata, MarketMetadata
from data.schemas.market_info import get_market_metadata
from utils.exceptions import SymbolNotFoundError, NetworkError, InvalidDateRangeError
from utils.logging_config import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "data_source",
]


class YFinanceProvider(BaseMarketDataProvider):
    """Hardened market data provider utilizing yfinance API."""

    PROVIDER_NAME = "yfinance"

    def fetch_market_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch market data from Yahoo Finance and normalize into standard schema."""
        if not symbol or not symbol.strip():
            raise SymbolNotFoundError(symbol or "EMPTY", self.PROVIDER_NAME)

        clean_symbol = symbol.strip().upper()

        # Validate Date Range if specified
        if start_date and end_date:
            try:
                s_dt = datetime.strptime(start_date, "%Y-%m-%d")
                e_dt = datetime.strptime(end_date, "%Y-%m-%d")
                if s_dt > e_dt:
                    raise InvalidDateRangeError(start_date, end_date)
            except ValueError:
                pass  # Allow string dates if yfinance parses them natively

        logger.info(
            f"Fetching yfinance data for {clean_symbol} (period={period}, interval={interval}, start={start_date}, end={end_date})"
        )

        try:
            ticker = yf.Ticker(clean_symbol)
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                df = ticker.history(period=period or "1y", interval=interval)

            if df.empty:
                logger.warning(f"No market data returned from yfinance for symbol: {clean_symbol}")
                return pd.DataFrame(columns=REQUIRED_COLUMNS)

            df = self._normalize_dataframe(df, clean_symbol)
            logger.info(f"Successfully fetched {len(df)} rows for {clean_symbol}")
            return df

        except InvalidDateRangeError:
            raise
        except Exception as e:
            err_str = str(e).lower()
            if "not found" in err_str or "delisted" in err_str or "404" in err_str:
                logger.warning(f"Symbol {clean_symbol} not found or delisted: {str(e)}")
                raise SymbolNotFoundError(clean_symbol, self.PROVIDER_NAME) from e
            elif "connection" in err_str or "timeout" in err_str or "network" in err_str:
                logger.error(f"Network error while reaching yfinance for {clean_symbol}: {str(e)}")
                raise NetworkError(self.PROVIDER_NAME, e) from e
            else:
                logger.error(f"Unexpected error in yfinance provider for {clean_symbol}: {str(e)}")
                return pd.DataFrame(columns=REQUIRED_COLUMNS)

    def fetch_latest_data(self, symbol: str) -> pd.DataFrame:
        """Fetch the most recent bar of data available."""
        return self.fetch_market_data(symbol, period="5d", interval="1d").tail(1)

    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol exists on yfinance."""
        if not symbol or not symbol.strip():
            return False
        clean_symbol = symbol.strip().upper()
        try:
            ticker = yf.Ticker(clean_symbol)
            history = ticker.history(period="1d")
            return not history.empty
        except Exception:
            return False

    def get_symbol_metadata(self, symbol: str) -> MarketMetadata:
        """Return market metadata for a given symbol."""
        return get_market_metadata(symbol, self.PROVIDER_NAME)

    def get_provider_metadata(self) -> ProviderMetadata:
        """Return yfinance provider capabilities."""
        return ProviderMetadata(
            provider_name=self.PROVIDER_NAME,
            description="Yahoo Finance data provider via yfinance Python package",
            supports_realtime=False,
            supported_intervals=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
        )

    def _normalize_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize raw yfinance DataFrame column names, timestamps, and structure."""
        df = df.reset_index()

        date_col = None
        for col in ["Date", "Datetime", "index", "date"]:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            date_col = df.columns[0]

        df = df.rename(columns={date_col: "timestamp"})
        df.columns = [str(col).lower() for col in df.columns]

        column_mapping = {
            "adj close": "adj_close",
            "stock splits": "splits",
            "dividends": "dividends",
        }
        df = df.rename(columns=column_mapping)

        for req in ["open", "high", "low", "close", "volume"]:
            if req not in df.columns:
                df[req] = 0.0

        # Standardize timestamp to UTC timezone-aware datetime64[ns, UTC]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        df["symbol"] = symbol
        df["data_source"] = self.PROVIDER_NAME

        df = df[REQUIRED_COLUMNS].sort_values("timestamp").reset_index(drop=True)
        return df
