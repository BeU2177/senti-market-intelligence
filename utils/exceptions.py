"""Domain-specific application exceptions for Senti Market Intelligence."""


class MarketDataError(Exception):
    """Base exception for all market data domain errors."""
    pass


class SymbolNotFoundError(MarketDataError):
    """Raised when a requested symbol is invalid, unlisted, or not found by provider."""

    def __init__(self, symbol: str, provider: str = "yfinance"):
        self.symbol = symbol
        self.provider = provider
        super().__init__(f"Symbol '{symbol}' was not found or has no price data on provider '{provider}'.")


class NetworkError(MarketDataError):
    """Raised when network, DNS, or HTTP connection fails while calling provider."""

    def __init__(self, provider: str, original_error: Exception):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"Network communication failed with provider '{provider}': {str(original_error)}")


class RateLimitError(MarketDataError):
    """Raised when provider API rate limit is exceeded."""

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"Rate limit exceeded for provider '{provider}'. Please wait before retrying.")


class InvalidDateRangeError(MarketDataError):
    """Raised when requested start_date is after end_date or in the future."""

    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date
        super().__init__(f"Invalid date range specified: start_date '{start_date}' is after end_date '{end_date}'.")
