"""Volatility indicators and Bollinger Bands feature generator."""

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange, BollingerBands


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes ATR, rolling return volatility, and Bollinger Band features."""
    df_feat = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Average True Range (14)
    atr_ind = AverageTrueRange(high=high, low=low, close=close, window=14)
    df_feat["atr_14"] = atr_ind.average_true_range()

    # Rolling Return Volatility (standard deviation of 1-day returns)
    returns = close.pct_change()
    for w in [5, 10, 20, 30]:
        df_feat[f"volatility_{w}d"] = returns.rolling(window=w, min_periods=w).std()

    # Bollinger Bands (20 periods, 2 std)
    bb_ind = BollingerBands(close=close, window=20, window_dev=2)
    df_feat["bb_upper"] = bb_ind.bollinger_hband()
    df_feat["bb_middle"] = bb_ind.bollinger_mavg()
    df_feat["bb_lower"] = bb_ind.bollinger_lband()
    df_feat["bb_bandwidth"] = bb_ind.bollinger_wband()
    df_feat["bb_pct_b"] = bb_ind.bollinger_pband()

    return df_feat
