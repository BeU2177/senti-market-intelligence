"""Price action and return features generator."""

import numpy as np
import pandas as pd


def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes backward-looking price action and return features."""
    df_feat = pd.DataFrame(index=df.index)
    close = df["close"]
    open_p = df["open"]
    high = df["high"]
    low = df["low"]

    # Percentage Returns (backward-looking)
    for period in [1, 3, 5, 7, 14, 30]:
        df_feat[f"return_{period}d"] = close.pct_change(period)

    # Log Returns (backward-looking)
    for period in [1, 3, 5]:
        df_feat[f"log_return_{period}d"] = np.log(close / close.shift(period))

    # Price Ranges
    df_feat["high_low_range"] = high - low
    df_feat["open_close_range"] = close - open_p
    df_feat["high_low_pct_range"] = np.where(open_p != 0, (high - low) / open_p, 0.0)

    # Position within daily range [0, 1]
    hl_diff = high - low
    df_feat["close_pos_in_range"] = np.where(hl_diff > 0, (close - low) / hl_diff, 0.5)

    # Gaps relative to previous day close
    prev_close = close.shift(1)
    df_feat["gap_prev_close"] = open_p - prev_close
    df_feat["gap_prev_close_pct"] = np.where(prev_close != 0, (open_p - prev_close) / prev_close, 0.0)

    return df_feat
