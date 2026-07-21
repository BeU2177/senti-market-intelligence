"""PyTorch LSTM model preserving temporal causality for multi-horizon return predictions."""

import torch
import torch.nn as nn


class PyTorchLSTM(nn.Module):
    """Multi-layer unidirectional LSTM model for sequential forecasting."""

    def __init__(
        self,
        feature_count: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_count,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  # Unidirectional to preserve strict temporal causality
        )
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, sequence_length, feature_count]
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Take hidden representation at final timestep (t)
        last_timestep = lstm_out[:, -1, :]
        return self.fc_head(last_timestep)
