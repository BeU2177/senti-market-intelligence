"""Rule-based market regime classification features."""

import numpy as np
import pandas as pd


def compute_regime_features(df: pd.DataFrame, df_trend: pd.DataFrame, df_vol: pd.DataFrame) -> pd.DataFrame:
    """Classifies market regime into BULLISH, BEARISH, SIDEWAYS, or HIGH_VOLATILITY using backward-looking indicators."""
    df_feat = pd.DataFrame(index=df.index)
    close = df["close"]

    sma_20 = df_trend.get("sma_20", close.rolling(20).mean())
    sma_50 = df_trend.get("sma_50", close.rolling(50).mean())
    vol_20 = df_vol.get("volatility_20d", close.pct_change().rolling(20).std())

    # Rolling volatility threshold (80th percentile over 60-day window)
    high_vol_thresh = vol_20.rolling(window=60, min_periods=20).quantile(0.80)

    regimes = []
    for c, s20, s50, v20, hvt in zip(close, sma_20, sma_50, vol_20, high_vol_thresh):
        if pd.isna(s20) or pd.isna(s50):
            regimes.append("UNKNOWN")
        elif pd.notna(v20) and pd.notna(hvt) and v20 > hvt and v20 > 0.025:
            regimes.append("HIGH_VOLATILITY")
        elif s20 > s50 and c >= s20:
            regimes.append("BULLISH")
        elif s20 < s50 and c <= s20:
            regimes.append("BEARISH")
        else:
            regimes.append("SIDEWAYS")

    df_feat["market_regime"] = regimes
    return df_feat
