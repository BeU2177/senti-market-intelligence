"""Unit tests for HistoricalDatasetBuilder, Parquet persistence, and provenance generation."""

import pytest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from datetime import datetime, timezone

from data.builder.dataset_builder import HistoricalDatasetBuilder
from data.storage.storage_manager import StorageManager
from data.providers.yfinance_provider import YFinanceProvider


@pytest.fixture
def tmp_storage(tmp_path):
    """Fixture initializing StorageManager with temporary directory path."""
    return StorageManager(base_dir=tmp_path)


@pytest.fixture
def mock_df():
    """Fixture returning clean test market DataFrame."""
    dates = pd.date_range("2023-01-01", periods=5, tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [102.0, 103.0, 104.0, 105.0, 106.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            "symbol": ["AAPL"] * 5,
            "data_source": ["yfinance"] * 5,
        }
    )


def test_dataset_builder_pipeline(tmp_storage, mock_df):
    """Test HistoricalDatasetBuilder end-to-end execution and Parquet/JSON file outputs."""
    provider = YFinanceProvider()

    with patch.object(YFinanceProvider, "fetch_market_data", return_value=mock_df):
        builder = HistoricalDatasetBuilder(provider=provider, storage_manager=tmp_storage)
        response = builder.build_dataset(symbol="AAPL", period="1y", interval="1d", save_to_disk=True)

        assert response.symbol == "AAPL"
        assert response.row_count == 5
        assert response.validation_result.is_valid is True
        assert response.provenance is not None

        # Verify Parquet files and JSON metadata exist on disk
        raw_path = Path(response.provenance.raw_file_path)
        proc_path = Path(response.provenance.processed_file_path)
        assert raw_path.exists()
        assert proc_path.exists()

        # Load Parquet file back to verify Parquet reader/writer
        loaded_df = pd.read_parquet(proc_path)
        assert len(loaded_df) == 5
        assert "close" in loaded_df.columns
