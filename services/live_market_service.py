"""Live market ingestion service with rate-limit awareness, latency metrics, and data freshness validation."""

import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple
import pandas as pd

from config.settings import get_settings
from data.providers.base_provider import BaseMarketDataProvider
from data.providers.yfinance_provider import YFinanceProvider
from data.schemas.market_data import MarketDataResponse, FreshnessStatus
from utils.logging_config import get_logger

logger = get_logger(__name__)


class LiveMarketService:
    """Polling-based near-real-time market data ingestion service with freshness and latency metrics."""

    def __init__(self, provider: Optional[BaseMarketDataProvider] = None, max_allowed_freshness_sec: int = 300):
        self.provider = provider or YFinanceProvider()
        self.max_allowed_freshness_sec = max_allowed_freshness_sec
        self.last_fetch_time: Dict[str, float] = {}

    def fetch_live_quote(self, symbol: str, period: str = "1y", interval: str = "1d") -> Tuple[MarketDataResponse, Dict[str, Any]]:
        """Fetch latest market quote with ingestion timestamp, provider timestamp, latency, and freshness check."""
        clean_symbol = symbol.strip().upper()
        start_time = time.time()
        ingestion_dt = datetime.now(timezone.utc)

        # Rate-limiting guard: enforce minimum 2 seconds between live quote polls per symbol
        last = self.last_fetch_time.get(clean_symbol, 0.0)
        if start_time - last < 2.0:
            time.sleep(2.0 - (start_time - last))

        self.last_fetch_time[clean_symbol] = time.time()

        try:
            from services.market_service import MarketService
            m_service = MarketService()
            m_res = m_service.get_market_data(symbol=clean_symbol, period=period, interval=interval)
            latency_ms = round((time.time() - start_time) * 1000, 2)

            data_dt = m_res.latest_timestamp or ingestion_dt
            age_sec = (ingestion_dt - data_dt).total_seconds()

            if age_sec > self.max_allowed_freshness_sec:
                freshness = "STALE"
            else:
                freshness = "NEAR_REAL_TIME"

            metrics = {
                "symbol": clean_symbol,
                "data_timestamp": data_dt.isoformat(),
                "provider_timestamp": data_dt.isoformat(),
                "ingestion_timestamp": ingestion_dt.isoformat(),
                "latency_ms": latency_ms,
                "age_seconds": round(age_sec, 1),
                "freshness_status": freshness,
                "max_allowed_seconds": self.max_allowed_freshness_sec,
            }

            logger.info(
                f"LiveMarketService fetched quote for {clean_symbol}: latency={latency_ms}ms, "
                f"age={age_sec:.1f}s, status={freshness}"
            )
            return m_res, metrics

        except Exception as e:
            logger.error(f"LiveMarketService failed to fetch quote for {clean_symbol}: {str(e)}")
            latency_ms = round((time.time() - start_time) * 1000, 2)
            metrics = {
                "symbol": clean_symbol,
                "data_timestamp": None,
                "provider_timestamp": None,
                "ingestion_timestamp": ingestion_dt.isoformat(),
                "latency_ms": latency_ms,
                "age_seconds": 999999.0,
                "freshness_status": "STALE",
                "max_allowed_seconds": self.max_allowed_freshness_sec,
                "error": str(e),
            }
            raise RuntimeError(f"Live market quote fetch failed for {clean_symbol}: {str(e)}")
