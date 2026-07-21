"""Metadata models and feature dictionary registry for feature engineering."""

from enum import Enum
from typing import Dict, Optional, List
from pydantic import BaseModel


class FeatureCategory(str, Enum):
    """Categories of engineered features."""
    PRICE = "PRICE"
    TREND = "TREND"
    MOMENTUM = "MOMENTUM"
    VOLATILITY = "VOLATILITY"
    VOLUME = "VOLUME"
    REGIME = "REGIME"
    TARGET = "TARGET"


class FeatureMetadata(BaseModel):
    """Metadata describing individual engineered features and targets."""
    feature_name: str
    category: FeatureCategory
    window: Optional[int] = None
    uses_future_data: bool = False
    description: str


# Feature Metadata Registry
FEATURE_REGISTRY: Dict[str, FeatureMetadata] = {
    # Returns & Price Features
    "return_1d": FeatureMetadata(feature_name="return_1d", category=FeatureCategory.PRICE, window=1, description="1-day percentage price return"),
    "return_3d": FeatureMetadata(feature_name="return_3d", category=FeatureCategory.PRICE, window=3, description="3-day percentage price return"),
    "return_5d": FeatureMetadata(feature_name="return_5d", category=FeatureCategory.PRICE, window=5, description="5-day percentage price return"),
    "return_7d": FeatureMetadata(feature_name="return_7d", category=FeatureCategory.PRICE, window=7, description="7-day percentage price return"),
    "return_14d": FeatureMetadata(feature_name="return_14d", category=FeatureCategory.PRICE, window=14, description="14-day percentage price return"),
    "return_30d": FeatureMetadata(feature_name="return_30d", category=FeatureCategory.PRICE, window=30, description="30-day percentage price return"),
    "log_return_1d": FeatureMetadata(feature_name="log_return_1d", category=FeatureCategory.PRICE, window=1, description="1-day log price return"),
    "log_return_3d": FeatureMetadata(feature_name="log_return_3d", category=FeatureCategory.PRICE, window=3, description="3-day log price return"),
    "log_return_5d": FeatureMetadata(feature_name="log_return_5d", category=FeatureCategory.PRICE, window=5, description="5-day log price return"),
    "high_low_range": FeatureMetadata(feature_name="high_low_range", category=FeatureCategory.PRICE, window=1, description="High minus Low price range"),
    "open_close_range": FeatureMetadata(feature_name="open_close_range", category=FeatureCategory.PRICE, window=1, description="Close minus Open price range"),
    "high_low_pct_range": FeatureMetadata(feature_name="high_low_pct_range", category=FeatureCategory.PRICE, window=1, description="High-Low range relative to Open price"),
    "close_pos_in_range": FeatureMetadata(feature_name="close_pos_in_range", category=FeatureCategory.PRICE, window=1, description="Close relative position within daily High-Low range [0, 1]"),
    "gap_prev_close": FeatureMetadata(feature_name="gap_prev_close", category=FeatureCategory.PRICE, window=1, description="Open minus previous day Close gap"),
    "gap_prev_close_pct": FeatureMetadata(feature_name="gap_prev_close_pct", category=FeatureCategory.PRICE, window=1, description="Open minus previous day Close gap percentage"),

    # Trend Features
    "sma_5": FeatureMetadata(feature_name="sma_5", category=FeatureCategory.TREND, window=5, description="5-period Simple Moving Average"),
    "sma_10": FeatureMetadata(feature_name="sma_10", category=FeatureCategory.TREND, window=10, description="10-period Simple Moving Average"),
    "sma_20": FeatureMetadata(feature_name="sma_20", category=FeatureCategory.TREND, window=20, description="20-period Simple Moving Average"),
    "sma_50": FeatureMetadata(feature_name="sma_50", category=FeatureCategory.TREND, window=50, description="50-period Simple Moving Average"),
    "sma_100": FeatureMetadata(feature_name="sma_100", category=FeatureCategory.TREND, window=100, description="100-period Simple Moving Average"),
    "sma_200": FeatureMetadata(feature_name="sma_200", category=FeatureCategory.TREND, window=200, description="200-period Simple Moving Average"),
    "ema_12": FeatureMetadata(feature_name="ema_12", category=FeatureCategory.TREND, window=12, description="12-period Exponential Moving Average"),
    "ema_26": FeatureMetadata(feature_name="ema_26", category=FeatureCategory.TREND, window=26, description="26-period Exponential Moving Average"),
    "ema_50": FeatureMetadata(feature_name="ema_50", category=FeatureCategory.TREND, window=50, description="50-period Exponential Moving Average"),
    "close_rel_sma_20": FeatureMetadata(feature_name="close_rel_sma_20", category=FeatureCategory.TREND, window=20, description="Close price relative to SMA 20 ratio"),
    "close_rel_sma_50": FeatureMetadata(feature_name="close_rel_sma_50", category=FeatureCategory.TREND, window=50, description="Close price relative to SMA 50 ratio"),
    "close_rel_sma_200": FeatureMetadata(feature_name="close_rel_sma_200", category=FeatureCategory.TREND, window=200, description="Close price relative to SMA 200 ratio"),
    "sma_20_50_ratio": FeatureMetadata(feature_name="sma_20_50_ratio", category=FeatureCategory.TREND, window=50, description="SMA 20 divided by SMA 50 ratio"),
    "sma_50_200_ratio": FeatureMetadata(feature_name="sma_50_200_ratio", category=FeatureCategory.TREND, window=200, description="SMA 50 divided by SMA 200 ratio"),
    "ema_12_26_ratio": FeatureMetadata(feature_name="ema_12_26_ratio", category=FeatureCategory.TREND, window=26, description="EMA 12 divided by EMA 26 ratio"),
    "trend_slope_20": FeatureMetadata(feature_name="trend_slope_20", category=FeatureCategory.TREND, window=20, description="20-period SMA slope"),

    # Momentum Features
    "rsi_7": FeatureMetadata(feature_name="rsi_7", category=FeatureCategory.MOMENTUM, window=7, description="7-period Relative Strength Index"),
    "rsi_14": FeatureMetadata(feature_name="rsi_14", category=FeatureCategory.MOMENTUM, window=14, description="14-period Relative Strength Index"),
    "macd": FeatureMetadata(feature_name="macd", category=FeatureCategory.MOMENTUM, window=26, description="MACD line (EMA 12 - EMA 26)"),
    "macd_signal": FeatureMetadata(feature_name="macd_signal", category=FeatureCategory.MOMENTUM, window=9, description="MACD signal line (9-period EMA of MACD)"),
    "macd_hist": FeatureMetadata(feature_name="macd_hist", category=FeatureCategory.MOMENTUM, window=26, description="MACD histogram (MACD line - Signal line)"),
    "stoch_k": FeatureMetadata(feature_name="stoch_k", category=FeatureCategory.MOMENTUM, window=14, description="Stochastic Oscillator %K"),
    "stoch_d": FeatureMetadata(feature_name="stoch_d", category=FeatureCategory.MOMENTUM, window=3, description="Stochastic Oscillator %D"),
    "roc_10": FeatureMetadata(feature_name="roc_10", category=FeatureCategory.MOMENTUM, window=10, description="10-period Rate of Change"),
    "momentum_10": FeatureMetadata(feature_name="momentum_10", category=FeatureCategory.MOMENTUM, window=10, description="10-period Momentum"),

    # Volatility Features
    "atr_14": FeatureMetadata(feature_name="atr_14", category=FeatureCategory.VOLATILITY, window=14, description="14-period Average True Range"),
    "volatility_5d": FeatureMetadata(feature_name="volatility_5d", category=FeatureCategory.VOLATILITY, window=5, description="5-day rolling standard deviation of returns"),
    "volatility_10d": FeatureMetadata(feature_name="volatility_10d", category=FeatureCategory.VOLATILITY, window=10, description="10-day rolling standard deviation of returns"),
    "volatility_20d": FeatureMetadata(feature_name="volatility_20d", category=FeatureCategory.VOLATILITY, window=20, description="20-day rolling standard deviation of returns"),
    "volatility_30d": FeatureMetadata(feature_name="volatility_30d", category=FeatureCategory.VOLATILITY, window=30, description="30-day rolling standard deviation of returns"),
    "bb_upper": FeatureMetadata(feature_name="bb_upper", category=FeatureCategory.VOLATILITY, window=20, description="20-period Bollinger Upper Band"),
    "bb_middle": FeatureMetadata(feature_name="bb_middle", category=FeatureCategory.VOLATILITY, window=20, description="20-period Bollinger Middle Band"),
    "bb_lower": FeatureMetadata(feature_name="bb_lower", category=FeatureCategory.VOLATILITY, window=20, description="20-period Bollinger Lower Band"),
    "bb_bandwidth": FeatureMetadata(feature_name="bb_bandwidth", category=FeatureCategory.VOLATILITY, window=20, description="Bollinger Bandwidth ratio"),
    "bb_pct_b": FeatureMetadata(feature_name="bb_pct_b", category=FeatureCategory.VOLATILITY, window=20, description="Bollinger %B position within bands"),

    # Volume Features
    "volume_sma_5": FeatureMetadata(feature_name="volume_sma_5", category=FeatureCategory.VOLUME, window=5, description="5-period Volume SMA"),
    "volume_sma_20": FeatureMetadata(feature_name="volume_sma_20", category=FeatureCategory.VOLUME, window=20, description="20-period Volume SMA"),
    "volume_ratio_20": FeatureMetadata(feature_name="volume_ratio_20", category=FeatureCategory.VOLUME, window=20, description="Current Volume divided by 20-period Volume SMA"),
    "volume_change_pct": FeatureMetadata(feature_name="volume_change_pct", category=FeatureCategory.VOLUME, window=1, description="1-day Volume percentage change"),
    "obv": FeatureMetadata(feature_name="obv", category=FeatureCategory.VOLUME, window=1, description="On-Balance Volume"),
    "price_volume_trend": FeatureMetadata(feature_name="price_volume_trend", category=FeatureCategory.VOLUME, window=1, description="Price-Volume interaction (Return * Volume)"),

    # Regime Features
    "market_regime": FeatureMetadata(feature_name="market_regime", category=FeatureCategory.REGIME, window=50, description="Rule-based classification: BULLISH, BEARISH, SIDEWAYS, HIGH_VOLATILITY"),

    # Targets
    "future_return_1d": FeatureMetadata(feature_name="future_return_1d", category=FeatureCategory.TARGET, window=1, uses_future_data=True, description="1-day future price return (Close[t+1]/Close[t] - 1)"),
    "future_return_3d": FeatureMetadata(feature_name="future_return_3d", category=FeatureCategory.TARGET, window=3, uses_future_data=True, description="3-day future price return (Close[t+3]/Close[t] - 1)"),
    "future_return_5d": FeatureMetadata(feature_name="future_return_5d", category=FeatureCategory.TARGET, window=5, uses_future_data=True, description="5-day future price return (Close[t+5]/Close[t] - 1)"),
    "future_return_7d": FeatureMetadata(feature_name="future_return_7d", category=FeatureCategory.TARGET, window=7, uses_future_data=True, description="7-day future price return (Close[t+7]/Close[t] - 1)"),
    "direction_1d": FeatureMetadata(feature_name="direction_1d", category=FeatureCategory.TARGET, window=1, uses_future_data=True, description="1-day future direction (1 if return > 0 else 0)"),
    "direction_3d": FeatureMetadata(feature_name="direction_3d", category=FeatureCategory.TARGET, window=3, uses_future_data=True, description="3-day future direction (1 if return > 0 else 0)"),
    "direction_5d": FeatureMetadata(feature_name="direction_5d", category=FeatureCategory.TARGET, window=5, uses_future_data=True, description="5-day future direction (1 if return > 0 else 0)"),
    "direction_7d": FeatureMetadata(feature_name="direction_7d", category=FeatureCategory.TARGET, window=7, uses_future_data=True, description="7-day future direction (1 if return > 0 else 0)"),
}
