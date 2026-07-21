"""Target generation module constructing future prediction labels."""

import numpy as np
import pandas as pd


class TargetBuilder:
    """Constructs forward-looking return targets and directional classification labels."""

    def __init__(self, horizons: list[int] = None):
        self.horizons = horizons or [1, 3, 5, 7]

    def build_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes forward-looking return and directional targets."""
        targets = pd.DataFrame(index=df.index)
        close = df["close"]

        for h in self.horizons:
            # Future return at t+h relative to t
            ret_col = f"future_return_{h}d"
            dir_col = f"direction_{h}d"

            future_close = close.shift(-h)
            targets[ret_col] = (future_close / close) - 1.0

            # Directional target (1 for positive future return, 0 otherwise, NaN if future_close is NaN)
            targets[dir_col] = np.where(
                targets[ret_col].isna(),
                np.nan,
                np.where(targets[ret_col] > 0, 1, 0)
            )

        return targets
