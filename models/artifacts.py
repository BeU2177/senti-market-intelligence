"""Artifact serialization manager saving model bundles and logging experiment runs."""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import joblib

from models.schemas import ModelArtifactMetadata, ExperimentRecord
from models.preprocessing import FeaturePreprocessor
from utils.logging_config import get_logger

logger = get_logger(__name__)

BASE_ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = BASE_ARTIFACTS_DIR / "models"
EXPERIMENTS_DIR = BASE_ARTIFACTS_DIR / "experiments"


class ArtifactManager:
    """Manages saving/loading estimator models, preprocessors, metadata, and experiment logs."""

    def __init__(self, base_dir: Path = BASE_ARTIFACTS_DIR):
        self.models_dir = base_dir / "models"
        self.experiments_dir = base_dir / "experiments"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create artifact directories if they do not exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def save_model_artifact(
        self,
        estimator: Any,
        preprocessor: FeaturePreprocessor,
        metadata: ModelArtifactMetadata,
    ) -> Path:
        """Serialize estimator, preprocessor, and metadata into artifacts/models/<symbol>/<version>/."""
        clean_symbol = metadata.symbol.strip().upper().replace("/", "_").replace("^", "")
        model_version = metadata.model_version.strip()
        
        target_dir = self.models_dir / clean_symbol / f"{metadata.model_name}_{model_version}"
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save Estimator
        model_file = target_dir / "model.joblib"
        joblib.dump(estimator, model_file)

        # 2. Save Preprocessor
        prep_file = target_dir / "preprocessor.joblib"
        joblib.dump(preprocessor, prep_file)

        # 3. Save Metadata JSON
        meta_file = target_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata.model_dump(mode="json"), f, indent=2, default=str)

        logger.info(f"Saved complete model artifact bundle to: {target_dir}")
        return target_dir

    def log_experiment(self, experiment: ExperimentRecord) -> Path:
        """Append an experiment run record to artifacts/experiments/experiments.jsonl."""
        log_file = self.experiments_dir / "experiments.jsonl"
        logger.info(f"Logging experiment '{experiment.experiment_id}' to {log_file}")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(experiment.model_dump(mode="json"), default=str) + "\n")

        return log_file

    def load_experiment_history(self) -> List[Dict[str, Any]]:
        """Load past experiment runs from experiments.jsonl."""
        log_file = self.experiments_dir / "experiments.jsonl"
        if not log_file.exists():
            return []

        records = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
        return records
