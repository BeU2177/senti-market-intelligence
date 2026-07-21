"""Unit tests for feature engineering generators."""

import pytest
import pandas as pd
import numpy as np

from features.price_features import compute_price_features
from features.trend_features import compute_trend_features
from features.momentum_features import compute_momentum_features
from features.volatility_features import compute_volatility_features
from features.volume_features import compute_volume_features
from features.regime_features import compute_regime_features
from features.feature_pipeline import FeaturePipeline


@pytest.fixture
def sample_market_df():
    """Generates 100 bars of synthetic market data."""
    dates = pd.date_range("2023-01-01", periods=100, tz="UTC")
    np.random.seed(42)
    close_prices = 100.0 + np.cumsum(np.random.normal(0, 1, 100))
    open_prices = close_prices + np.random.normal(0, 0.5, 100)
    high_prices = np.maximum(close_prices, open_prices) + np.abs(np.random.normal(0, 0.5, 100))
    low_prices = np.minimum(close_prices, open_prices) - np.abs(np.random.normal(0, 0.5, 100))
    volumes = np.random.randint(1000, 10000, 100).astype(float)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
            "symbol": ["TEST"] * 100,
            "data_source": ["unit_test"] * 100,
        }
    )


def test_price_features(sample_market_df):
    """Test price action and return calculations."""
    res = compute_price_features(sample_market_df)
    assert "return_1d" in res.columns
    assert "return_5d" in res.columns
    assert "log_return_1d" in res.columns
    assert "high_low_range" in res.columns
    assert "close_pos_in_range" in res.columns

    # Verify return_1d at row 1 matches manual calculation
    manual_ret_1d = (sample_market_df["close"].iloc[1] / sample_market_df["close"].iloc[0]) - 1.0
    assert np.isclose(res["return_1d"].iloc[1], manual_ret_1d)


def test_trend_features(sample_market_df):
    """Test moving average indicators."""
    res = compute_trend_features(sample_market_df)
    assert "sma_5" in res.columns
    assert "sma_20" in res.columns
    assert "ema_12" in res.columns
    assert "sma_20_50_ratio" in res.columns

    # Verify SMA 5 calculation
    manual_sma_5 = sample_market_df["close"].iloc[0:5].mean()
    assert np.isclose(res["sma_5"].iloc[4], manual_sma_5)


def test_momentum_features(sample_market_df):
    """Test momentum indicators (RSI, MACD, Stochastic)."""
    res = compute_momentum_features(sample_market_df)
    assert "rsi_14" in res.columns
    assert "macd" in res.columns
    assert "stoch_k" in res.columns
    assert "roc_10" in res.columns


def test_volatility_features(sample_market_df):
    """Test volatility indicators (ATR, Bollinger Bands)."""
    res = compute_volatility_features(sample_market_df)
    assert "atr_14" in res.columns
    assert "volatility_20d" in res.columns
    assert "bb_upper" in res.columns
    assert "bb_lower" in res.columns


def test_volume_features(sample_market_df):
    """Test volume indicators and OBV."""
    res = compute_volume_features(sample_market_df)
    assert "volume_sma_20" in res.columns
    assert "obv" in res.columns
    assert "price_volume_trend" in res.columns


def test_regime_features(sample_market_df):
    """Test market regime classification."""
    trend_df = compute_trend_features(sample_market_df)
    vol_df = compute_volatility_features(sample_market_df)
    reg_df = compute_regime_features(sample_market_df, trend_df, vol_df)

    assert "market_regime" in reg_df.columns
    valid_regimes = {"BULLISH", "BEARISH", "SIDEWAYS", "HIGH_VOLATILITY", "UNKNOWN"}
    assert set(reg_df["market_regime"].unique()).issubset(valid_regimes)


def test_feature_pipeline_integration(sample_market_df):
    """Test end-to-end FeaturePipeline execution."""
    pipeline = FeaturePipeline()
    feat_dataset = pipeline.build_features(sample_market_df, symbol="TEST")

    assert feat_dataset.symbol == "TEST"
    assert feat_dataset.feature_version == "v1.0.0"
    assert len(feat_dataset.feature_columns) > 30
    assert len(feat_dataset.target_columns) == 8
    assert feat_dataset.leakage_report.is_clean is True
