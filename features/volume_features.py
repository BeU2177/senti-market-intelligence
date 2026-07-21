"""Volume indicators and price-volume interaction feature generator."""

import numpy as np
import pandas as pd
from ta.volume import OnBalanceVolumeIndicator


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Volume SMAs, volume ratios, OBV, and price-volume interactions."""
    df_feat = pd.DataFrame(index=df.index)
    close = df["close"]
    volume = df["volume"]

    # Volume Moving Averages
    df_feat["volume_sma_5"] = volume.rolling(window=5, min_periods=5).mean()
    df_feat["volume_sma_20"] = volume.rolling(window=20, min_periods=20).mean()

    # Volume Ratio relative to 20-period SMA
    vol_sma_20 = df_feat["volume_sma_20"]
    df_feat["volume_ratio_20"] = np.where(vol_sma_20 > 0, volume / vol_sma_20, 1.0)

    # Volume Change %
    df_feat["volume_change_pct"] = volume.pct_change()

    # On-Balance Volume (OBV)
    obv_ind = OnBalanceVolumeIndicator(close=close, volume=volume)
    df_feat["obv"] = obv_ind.on_balance_volume()

    # Price-Volume Trend Interaction
    returns = close.pct_change()
    df_feat["price_volume_trend"] = returns * volume

    return df_feat
