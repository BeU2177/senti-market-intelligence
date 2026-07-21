"""PyTorch deep learning model trainer with early stopping, multi-horizon metrics, and checkpointing."""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from deep_learning.datasets import FinancialTimeSeriesDataset
from deep_learning.training.early_stopping import EarlyStopping
from deep_learning.training.checkpointing import CheckpointManager
from deep_learning.schemas import DeepLearningMetrics, MultiHorizonReport, CheckpointMetadata
from models.preprocessing import FeaturePreprocessor
from utils.logging_config import get_logger

logger = get_logger(__name__)


class PyTorchTrainer:
    """Orchestrates PyTorch training loops, validation evaluation, early stopping, and checkpointing."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 100,
        patience: int = 15,
        loss_function: str = "huber",  # 'huber' or 'mse'
        gradient_clip: float = 1.0,
        device: Optional[str] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.loss_function_name = loss_function
        self.gradient_clip = gradient_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_manager = checkpoint_manager or CheckpointManager()

    def train(
        self,
        model: nn.Module,
        model_name: str,
        symbol: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        preprocessor: Optional[FeaturePreprocessor] = None,
        feature_columns: Optional[List[str]] = None,
        target_horizons: Optional[List[int]] = None,
    ) -> MultiHorizonReport:
        """Executes full PyTorch training loop across epochs with early stopping."""
        start_time = time.time()
        horizons = target_horizons or [1, 3, 5, 7]

        model.to(self.device)
        logger.info(f"PyTorchTrainer starting training for '{model_name}' on device '{self.device}'")

        # DataLoader setup
        train_dataset = FinancialTimeSeriesDataset(X_train, y_train)
        val_dataset = FinancialTimeSeriesDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=min(self.batch_size, len(val_dataset)), shuffle=False)

        # Loss Function
        if self.loss_function_name.lower() == "huber":
            criterion = nn.HuberLoss(delta=0.02)
        else:
            criterion = nn.MSELoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        early_stopping = EarlyStopping(patience=self.patience)

        for epoch in range(1, self.max_epochs + 1):
            model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()

                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)

                optimizer.step()
                train_loss += loss.item() * len(X_batch)

            train_loss /= len(train_dataset)

            # Validation Loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                    val_loss += loss.item() * len(X_batch)

            val_loss /= len(val_dataset)

            # Check Early Stopping
            if early_stopping(val_loss, epoch, model):
                break

        # Best Weights Restored by EarlyStopping
        model.eval()
        duration = round(time.time() - start_time, 2)

        # Compute Metrics
        tr_preds = self._predict(model, X_train)
        tr_metrics = self.evaluate_multi_horizon(y_train, tr_preds, horizons)

        val_preds = self._predict(model, X_val)
        val_metrics = self.evaluate_multi_horizon(y_val, val_preds, horizons)

        test_metrics = None
        if X_test is not None and y_test is not None and len(X_test) > 0:
            te_preds = self._predict(model, X_test)
            test_metrics = self.evaluate_multi_horizon(y_test, te_preds, horizons)

        # Save Checkpoint
        if preprocessor and feature_columns:
            ckpt_meta = CheckpointMetadata(
                symbol=symbol,
                model_name=model_name,
                sequence_length=X_train.shape[1],
                target_horizons=horizons,
                feature_columns=feature_columns,
                best_val_loss=early_stopping.best_loss,
                best_epoch=early_stopping.best_epoch,
            )
            self.ckpt_manager.save_checkpoint(model, optimizer, preprocessor, ckpt_meta)

        logger.info(
            f"PyTorchTrainer finished '{model_name}' for {symbol}: Best Epoch={early_stopping.best_epoch}, "
            f"Val Overall RMSE={val_metrics.overall_rmse:.6f}, Duration={duration}s"
        )

        return MultiHorizonReport(
            model_name=model_name,
            symbol=symbol,
            sequence_length=X_train.shape[1],
            train_metrics=tr_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            best_epoch=early_stopping.best_epoch,
            training_duration_sec=duration,
            device_used=str(self.device),
        )

    def _predict(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """Inference wrapper generating numpy predictions."""
        model.eval()
        ds = FinancialTimeSeriesDataset(X, np.zeros((len(X), 4)))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                preds = model(X_batch)
                all_preds.append(preds.cpu().numpy())

        return np.vstack(all_preds) if all_preds else np.empty((0, 4))

    def evaluate_multi_horizon(
        self, y_true: np.ndarray, y_pred: np.ndarray, horizons: List[int]
    ) -> DeepLearningMetrics:
        """Calculates per-horizon MAE, RMSE, Directional Accuracy, and overall RMSE."""
        h_mae = {}
        h_rmse = {}
        h_dir_acc = {}

        all_sq_errs = []

        for idx, h in enumerate(horizons):
            if idx >= y_true.shape[1] or idx >= y_pred.shape[1]:
                continue
            t_col = y_true[:, idx]
            p_col = y_pred[:, idx]

            mae = float(np.mean(np.abs(t_col - p_col)))
            sq_err = (t_col - p_col) ** 2
            rmse = float(np.sqrt(np.mean(sq_err)))
            all_sq_errs.extend(sq_err)

            same_sign = (t_col > 0) == (p_col > 0)
            dir_acc = float(np.mean(same_sign)) if len(same_sign) > 0 else 0.0

            key = f"{h}d"
            h_mae[key] = round(mae, 6)
            h_rmse[key] = round(rmse, 6)
            h_dir_acc[key] = round(dir_acc, 4)

        overall_rmse = float(np.sqrt(np.mean(all_sq_errs))) if all_sq_errs else 0.0

        return DeepLearningMetrics(
            horizon_mae=h_mae,
            horizon_rmse=h_rmse,
            horizon_directional_acc=h_dir_acc,
            overall_rmse=round(overall_rmse, 6),
        )
