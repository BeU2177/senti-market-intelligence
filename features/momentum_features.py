"""Momentum indicators feature generator."""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import MACD


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes RSI, MACD, Stochastic Oscillator, ROC, and Momentum indicators."""
    df_feat = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # RSI (7 & 14)
    rsi_7_ind = RSIIndicator(close=close, window=7)
    df_feat["rsi_7"] = rsi_7_ind.rsi()

    rsi_14_ind = RSIIndicator(close=close, window=14)
    df_feat["rsi_14"] = rsi_14_ind.rsi()

    # MACD (12, 26, 9)
    macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df_feat["macd"] = macd_ind.macd()
    df_feat["macd_signal"] = macd_ind.macd_signal()
    df_feat["macd_hist"] = macd_ind.macd_diff()

    # Stochastic Oscillator (14, 3)
    stoch_ind = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df_feat["stoch_k"] = stoch_ind.stoch()
    df_feat["stoch_d"] = stoch_ind.stoch_signal()

    # Rate of Change & Momentum
    roc_ind = ROCIndicator(close=close, window=10)
    df_feat["roc_10"] = roc_ind.roc()
    df_feat["momentum_10"] = close - close.shift(10)

    return df_feat
