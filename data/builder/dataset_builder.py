"""Reproducible Historical Dataset Builder orchestrating provider fetching, validation, storage, and provenance generation."""

from datetime import datetime, timezone
from typing import Optional
import pandas as pd

from data.providers.base_provider import BaseMarketDataProvider
from data.providers.yfinance_provider import YFinanceProvider
from data.validation.data_quality import DataQualityValidator
from data.storage.storage_manager import StorageManager
from data.schemas.market_data import (
    MarketDataResponse,
    DatasetProvenance,
    DataValidationResult,
)
from data.schemas.market_info import get_market_metadata
from utils.logging_config import get_logger

logger = get_logger(__name__)


class HistoricalDatasetBuilder:
    """Builder pipeline constructing reproducible market datasets."""

    def __init__(
        self,
        provider: Optional[BaseMarketDataProvider] = None,
        validator: Optional[DataQualityValidator] = None,
        storage_manager: Optional[StorageManager] = None,
    ):
        self.provider = provider or YFinanceProvider()
        self.validator = validator or DataQualityValidator()
        self.storage = storage_manager or StorageManager()

    def build_dataset(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = "1y",
        interval: str = "1d",
        save_to_disk: bool = True,
    ) -> MarketDataResponse:
        """Executes full ingestion, validation, normalization, storage, and provenance generation."""
        clean_symbol = symbol.strip().upper()
        logger.info(f"HistoricalDatasetBuilder starting dataset build for: {clean_symbol}")

        # 1. Resolve Multi-Market Metadata
        market_metadata = get_market_metadata(clean_symbol, self.provider.get_provider_metadata().provider_name)

        # 2. Fetch Raw Data via Provider
        raw_df = self.provider.fetch_market_data(
            symbol=clean_symbol,
            start_date=start_date,
            end_date=end_date,
            period=period,
            interval=interval,
        )

        if raw_df.empty:
            logger.warning(f"HistoricalDatasetBuilder: No data fetched for {clean_symbol}")
            validation_res = self.validator.validate(raw_df, symbol=clean_symbol, interval=interval)
            return MarketDataResponse(
                symbol=clean_symbol,
                provider=self.provider.get_provider_metadata().provider_name,
                row_count=0,
                data_frame=raw_df,
                validation_result=validation_res,
                market_metadata=market_metadata,
            )

        raw_path = None
        processed_path = None

        # 3. Save Raw Dataset (if requested)
        if save_to_disk:
            raw_path = str(self.storage.save_raw_dataset(raw_df, clean_symbol, interval))

        # 4. Process & Normalize DataFrame
        processed_df = self._process_dataframe(raw_df)

        # 5. Execute Data Quality Validation
        validation_res = self.validator.validate(processed_df, symbol=clean_symbol, interval=interval)

        # 6. Save Processed Dataset (if requested)
        if save_to_disk:
            processed_path = str(self.storage.save_processed_dataset(processed_df, clean_symbol, interval))

        # 7. Generate Data Provenance Metadata
        earliest_ts = validation_res.earliest_timestamp
        latest_ts = validation_res.latest_timestamp

        provenance = DatasetProvenance(
            symbol=clean_symbol,
            data_source=self.provider.get_provider_metadata().provider_name,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            row_count=len(processed_df),
            earliest_timestamp=earliest_ts,
            latest_timestamp=latest_ts,
            downloaded_at=datetime.now(timezone.utc),
            processed_at=datetime.now(timezone.utc),
            dataset_version="v1.0",
            raw_file_path=raw_path,
            processed_file_path=processed_path,
        )

        if save_to_disk:
            self.storage.save_provenance_metadata(provenance)

        logger.info(
            f"HistoricalDatasetBuilder completed for {clean_symbol}: rows={len(processed_df)}, "
            f"valid={validation_res.is_valid}, freshness={validation_res.freshness_status.value}"
        )

        return MarketDataResponse(
            symbol=clean_symbol,
            provider=self.provider.get_provider_metadata().provider_name,
            row_count=len(processed_df),
            latest_timestamp=latest_ts,
            data_frame=processed_df,
            validation_result=validation_res,
            market_metadata=market_metadata,
            provenance=provenance,
        )

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply strict timestamp normalization to UTC and chronological sorting."""
        df = df.copy()

        # Ensure UTC timezone awareness for timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)

        return df
