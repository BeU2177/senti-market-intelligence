"""PyTorch Dataset wrapper for sequential time-series tensors."""

import torch
from torch.utils.data import Dataset
import numpy as np


class FinancialTimeSeriesDataset(Dataset):
    """PyTorch Dataset wrapping 3D feature sequences and 2D multi-horizon targets."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y), "Feature sequences X and target array y must have identical length!"
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
