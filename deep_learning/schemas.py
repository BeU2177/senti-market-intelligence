"""Schemas for PyTorch deep learning datasets, training configurations, multi-horizon predictions, and confidence assessments."""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class SequenceConfig(BaseModel):
    """Configuration for time-series lookback windows and target horizons."""
    sequence_length: int = Field(default=30, description="Number of historical trading days in lookback window")
    target_horizons: List[int] = Field(default_factory=lambda: [1, 3, 5, 7], description="Prediction target horizons in days")
    feature_count: int = Field(default=0, description="Total number of input features per timestep")


class DeepLearningMetrics(BaseModel):
    """Regression metrics calculated across multi-horizon outputs."""
    horizon_mae: Dict[str, float] = Field(default_factory=dict, description="MAE per target horizon (1d, 3d, 5d, 7d)")
    horizon_rmse: Dict[str, float] = Field(default_factory=dict, description="RMSE per target horizon (1d, 3d, 5d, 7d)")
    horizon_directional_acc: Dict[str, float] = Field(default_factory=dict, description="Directional Accuracy per target horizon [0, 1]")
    overall_rmse: float = Field(default=0.0, description="Average RMSE across all target horizons")


class MultiHorizonReport(BaseModel):
    """Evaluation report for PyTorch deep learning models."""
    model_name: str
    symbol: str
    sequence_length: int
    train_metrics: DeepLearningMetrics
    val_metrics: DeepLearningMetrics
    test_metrics: Optional[DeepLearningMetrics] = None
    best_epoch: int = 0
    training_duration_sec: float = 0.0
    device_used: str = "cpu"


class CheckpointMetadata(BaseModel):
    """Metadata saved along with serialized PyTorch model state_dict checkpoints."""
    symbol: str
    model_name: str
    sequence_length: int
    target_horizons: List[int]
    feature_columns: List[str]
    best_val_loss: float
    best_epoch: int
    dataset_version: str = "v1.0.0"
    feature_version: str = "v1.0.0"
    sentiment_version: Optional[str] = None
    saved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EnsemblePrediction(BaseModel):
    """Ensemble combined multi-horizon prediction outputs."""
    symbol: str
    current_price: float
    predicted_returns: Dict[str, float] = Field(description="Predicted returns for 1d, 3d, 5d, 7d horizons")
    predicted_prices: Dict[str, float] = Field(description="Converted predicted prices: P_t * (1 + R_t)")
    member_weights: Dict[str, float] = Field(description="Performance weights assigned to individual ensemble models")
    directional_signals: Dict[str, str] = Field(description="Directional signal per horizon: BULLISH, BEARISH, NEUTRAL")


class ConfidenceAssessment(BaseModel):
    """Non-fabricated confidence score derived from ensemble sign agreement and prediction dispersion."""
    confidence_level: str = Field(description="LOW, MEDIUM, or HIGH")
    ensemble_agreement_score: float = Field(description="Percentage sign agreement across ensemble members [0, 1]")
    prediction_std_dev: float = Field(description="Standard deviation of member predictions")
    validation_stability_score: float = Field(description="Validation stability metric")
    rationale: str = Field(description="Human-readable explanation of confidence scoring factors")
