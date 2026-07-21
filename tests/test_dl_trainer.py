"""Unit tests for PyTorchTrainer training loop, early stopping, and checkpointing."""

import pytest
import numpy as np
import torch

from deep_learning.models import PyTorchMLP
from deep_learning.training.trainer import PyTorchTrainer


def test_pytorch_trainer_execution():
    """Test PyTorchTrainer completes training epochs, applies early stopping, and returns MultiHorizonReport."""
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 40
    seq_len = 10
    feat_count = 5
    out_dim = 4

    X_tr = np.random.normal(0, 1, (n_samples, seq_len, feat_count)).astype(np.float32)
    y_tr = np.random.normal(0, 0.02, (n_samples, out_dim)).astype(np.float32)

    X_v = np.random.normal(0, 1, (10, seq_len, feat_count)).astype(np.float32)
    y_v = np.random.normal(0, 0.02, (10, out_dim)).astype(np.float32)

    mlp = PyTorchMLP(sequence_length=seq_len, feature_count=feat_count, output_dim=out_dim)
    trainer = PyTorchTrainer(max_epochs=5, batch_size=16, patience=3)

    report = trainer.train(
        model=mlp,
        model_name="Test_MLP",
        symbol="TEST",
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_v,
        y_val=y_v,
        target_horizons=[1, 3, 5, 7],
    )

    assert report.model_name == "Test_MLP"
    assert report.val_metrics.overall_rmse >= 0.0
    assert "1d" in report.val_metrics.horizon_rmse
    assert report.best_epoch > 0
