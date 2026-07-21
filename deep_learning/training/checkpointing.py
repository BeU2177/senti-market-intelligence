"""Checkpoint manager serializing PyTorch state_dict, optimizer, scaler, and metadata."""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
import joblib

from deep_learning.schemas import CheckpointMetadata
from models.preprocessing import FeaturePreprocessor
from utils.logging_config import get_logger

logger = get_logger(__name__)

BASE_DL_ARTIFACTS_DIR = Path("artifacts") / "deep_learning"


class CheckpointManager:
    """Manages saving and restoring PyTorch checkpoints and feature preprocessors."""

    def __init__(self, base_dir: Path = BASE_DL_ARTIFACTS_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        preprocessor: FeaturePreprocessor,
        metadata: CheckpointMetadata,
    ) -> Path:
        """Saves PyTorch state_dict, optimizer state, preprocessor, and metadata."""
        clean_symbol = metadata.symbol.strip().upper().replace("/", "_").replace("^", "")
        target_dir = self.base_dir / clean_symbol / metadata.model_name
        target_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = target_dir / "model_checkpoint.pt"
        prep_path = target_dir / "preprocessor.joblib"

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata.model_dump(mode="json"),
        }

        torch.save(checkpoint_payload, ckpt_path)
        joblib.dump(preprocessor, prep_path)

        logger.info(f"Saved PyTorch checkpoint and preprocessor to: {target_dir}")
        return ckpt_path
