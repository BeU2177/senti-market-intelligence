"""Early stopping callback monitoring validation loss."""

import copy
import torch
from utils.logging_config import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Early stopping callback restoring best model weights when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_weights = None
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float, epoch: int, model: torch.nn.Module) -> bool:
        """Evaluate validation loss and update best weights."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"EarlyStopping triggered at epoch {epoch}. Best validation loss: {self.best_loss:.6f} at epoch {self.best_epoch}.")
                model.load_state_dict(self.best_weights)

        return self.should_stop
