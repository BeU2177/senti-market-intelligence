"""Unit tests for PyTorch deep learning model forward passes."""

import pytest
import torch

from deep_learning.models import PyTorchMLP, PyTorchLSTM, PyTorchTemporalCNN, PyTorchTemporalTransformer


def test_pytorch_models_forward_pass():
    """Verify PyTorch MLP, LSTM, Temporal CNN, and Transformer accept 3D tensors and output [batch_size, 4]."""
    batch_size = 8
    seq_len = 20
    feat_count = 10
    output_dim = 4

    x = torch.randn(batch_size, seq_len, feat_count)

    # 1. MLP
    mlp = PyTorchMLP(sequence_length=seq_len, feature_count=feat_count, output_dim=output_dim)
    out_mlp = mlp(x)
    assert out_mlp.shape == (batch_size, output_dim)

    # 2. LSTM
    lstm = PyTorchLSTM(feature_count=feat_count, hidden_dim=32, num_layers=2, output_dim=output_dim)
    out_lstm = lstm(x)
    assert out_lstm.shape == (batch_size, output_dim)

    # 3. Temporal CNN
    tcnn = PyTorchTemporalCNN(feature_count=feat_count, output_dim=output_dim)
    out_tcnn = tcnn(x)
    assert out_tcnn.shape == (batch_size, output_dim)

    # 4. Temporal Transformer
    tf = PyTorchTemporalTransformer(feature_count=feat_count, output_dim=output_dim, d_model=32, nhead=2)
    out_tf = tf(x)
    assert out_tf.shape == (batch_size, output_dim)
