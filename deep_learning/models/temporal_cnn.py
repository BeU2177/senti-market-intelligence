"""PyTorch Temporal CNN model using 1D Causal Dilated Convolutions."""

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """1D Dilated Causal Convolutional Block with Residual Connection."""

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # Enforce exact sequence length matching after padding
        if out.size(2) != res.size(2):
            out = out[:, :, : res.size(2)]
        return self.relu(out + res)


class PyTorchTemporalCNN(nn.Module):
    """Temporal Convolutional Network processing [batch, channels, sequence_length]."""

    def __init__(
        self,
        feature_count: int,
        output_dim: int = 4,
        num_channels: list[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        num_channels = num_channels or [32, 64]
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = feature_count if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, sequence_length, feature_count]
        # PyTorch Conv1d expects: [batch_size, channels/feature_count, sequence_length]
        x_trans = x.transpose(1, 2)
        tcn_out = self.tcn(x_trans)
        # Pool across sequence dimension (take last timestep representation)
        last_step = tcn_out[:, :, -1]
        return self.head(last_step)
