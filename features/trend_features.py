"""Trend indicators and moving averages feature generator."""

import numpy as np
import pandas as pd


def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes backward-looking Simple & Exponential Moving Averages and trend ratios."""
    df_feat = pd.DataFrame(index=df.index)
    close = df["close"]

    # Simple Moving Averages
    for w in [5, 10, 20, 50, 100, 200]:
        df_feat[f"sma_{w}"] = close.rolling(window=w, min_periods=w).mean()

    # Exponential Moving Averages
    for w in [12, 26, 50]:
        df_feat[f"ema_{w}"] = close.ewm(span=w, adjust=False, min_periods=w).mean()

    # Close Relative to SMAs
    for w in [20, 50, 200]:
        sma = df_feat[f"sma_{w}"]
        df_feat[f"close_rel_sma_{w}"] = np.where(sma != 0, close / sma, 1.0)

    # Moving Average Ratios / Crossovers
    df_feat["sma_20_50_ratio"] = np.where(df_feat["sma_50"] != 0, df_feat["sma_20"] / df_feat["sma_50"], 1.0)
    df_feat["sma_50_200_ratio"] = np.where(df_feat["sma_200"] != 0, df_feat["sma_50"] / df_feat["sma_200"], 1.0)
    df_feat["ema_12_26_ratio"] = np.where(df_feat["ema_26"] != 0, df_feat["ema_12"] / df_feat["ema_26"], 1.0)

    # 20-period SMA Slope
    df_feat["trend_slope_20"] = df_feat["sma_20"].pct_change(5)

    return df_feat
