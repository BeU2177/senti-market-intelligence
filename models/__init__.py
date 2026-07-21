"""Machine learning training, evaluation, model registry, and experiment tracking package."""

from models.schemas import (
    RegressionMetrics,
    FinancialMetrics,
    EvaluationReport,
    ExperimentRecord,
    ModelArtifactMetadata,
)
from models.data_splitter import TimeSplitter, WalkForwardSplitter
from models.preprocessing import FeaturePreprocessor
from models.baselines import PersistenceBaseline, HistoricalMeanBaseline
from models.model_registry import ModelRegistry
from models.evaluate import ModelEvaluator
from models.artifacts import ArtifactManager
from models.train import ModelTrainer

__all__ = [
    "RegressionMetrics",
    "FinancialMetrics",
    "EvaluationReport",
    "ExperimentRecord",
    "ModelArtifactMetadata",
    "TimeSplitter",
    "WalkForwardSplitter",
    "FeaturePreprocessor",
    "PersistenceBaseline",
    "HistoricalMeanBaseline",
    "ModelRegistry",
    "ModelEvaluator",
    "ArtifactManager",
    "ModelTrainer",
]
