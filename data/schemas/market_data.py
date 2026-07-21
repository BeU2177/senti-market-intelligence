"""Market data schemas, enums, and data provenance structures."""

from enum import Enum
from datetime import datetime, timezone
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict


class FreshnessStatus(str, Enum):
    """Data freshness classification statuses."""
    REAL_TIME = "REAL_TIME"
    NEAR_REAL_TIME = "NEAR_REAL_TIME"
    DELAYED = "DELAYED"
    HISTORICAL = "HISTORICAL"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class MarketMetadata(BaseModel):
    """Metadata describing exchange, country, currency, and session rules for a symbol."""
    symbol: str
    exchange: str = "UNKNOWN"
    country: str = "Global"
    currency: str = "USD"
    timezone: str = "UTC"
    trading_session: str = "09:30-16:00"
    data_source: str = "yfinance"


class DatasetProvenance(BaseModel):
    """Provenance metadata tracking data lineage and reproducible parameters."""
    symbol: str
    data_source: str
    interval: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    row_count: int
    earliest_timestamp: Optional[datetime] = None
    latest_timestamp: Optional[datetime] = None
    downloaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_version: str = "v1.0"
    raw_file_path: Optional[str] = None
    processed_file_path: Optional[str] = None


class MarketRecord(BaseModel):
    """Normalized schema for a single financial market OHLCV record."""
    timestamp: datetime = Field(description="Normalized UTC datetime of the record")
    open: float = Field(description="Opening price")
    high: float = Field(description="Highest price")
    low: float = Field(description="Lowest price")
    close: float = Field(description="Closing price")
    volume: float = Field(description="Trading volume")
    symbol: str = Field(description="Ticker symbol")
    data_source: str = Field(description="Source provider name")


class ProviderMetadata(BaseModel):
    """Metadata describing a market data provider."""
    provider_name: str
    description: str
    supports_realtime: bool = False
    supported_intervals: List[str]


class DataValidationResult(BaseModel):
    """Structured report of data quality validation execution."""
    is_valid: bool
    symbol: str = "UNKNOWN"
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    row_count: int = 0
    missing_values: int = 0
    duplicate_timestamps: int = 0
    negative_prices: int = 0
    negative_volume: int = 0
    ohlc_errors: int = 0
    data_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    earliest_timestamp: Optional[datetime] = None
    latest_timestamp: Optional[datetime] = None
    freshness_status: FreshnessStatus = FreshnessStatus.UNKNOWN
    details: Dict[str, Any] = Field(default_factory=dict)


class MarketDataResponse(BaseModel):
    """Structured wrapper for market data requests."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbol: str
    provider: str
    row_count: int
    latest_timestamp: Optional[datetime] = None
    data_frame: Any = Field(default=None, description="Pandas DataFrame of normalized market records")
    validation_result: Optional[DataValidationResult] = None
    market_metadata: Optional[MarketMetadata] = None
    provenance: Optional[DatasetProvenance] = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
