"""PyTorch MLP Baseline model for multi-horizon time-series forecasting."""

import torch
import torch.nn as nn


class PyTorchMLP(nn.Module):
    """Deep Multi-Layer Perceptron processing sequential feature representations."""

    def __init__(
        self,
        sequence_length: int,
        feature_count: int,
        output_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.feature_count = feature_count
        self.input_dim = sequence_length * feature_count

        layers = []
        in_dim = self.input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, sequence_length, feature_count]
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        feat = self.backbone(x_flat)
        return self.head(feat)
