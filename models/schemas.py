"""Schemas for machine learning model evaluation, experiments, and artifact serialization."""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class RegressionMetrics(BaseModel):
    """Standard statistical regression evaluation metrics."""
    mae: float = Field(description="Mean Absolute Error")
    rmse: float = Field(description="Root Mean Squared Error")
    mse: float = Field(description="Mean Squared Error")
    r2: float = Field(description="Coefficient of Determination R^2")


class FinancialMetrics(BaseModel):
    """Financial and directional evaluation metrics."""
    directional_accuracy: float = Field(description="Percentage of predictions with correct directional sign [0, 1]")
    positive_precision: float = Field(description="Precision of positive return predictions")
    negative_precision: float = Field(description="Precision of negative return predictions")
    mean_predicted_return: float = Field(description="Average predicted return")
    mean_actual_return: float = Field(description="Average actual target return")
    prediction_correlation: float = Field(description="Pearson correlation between predicted and actual returns (Information Coefficient)")


class EvaluationReport(BaseModel):
    """Evaluation report containing metrics for Train, Validation, and Test splits."""
    model_name: str
    target_name: str
    feature_set_type: str = "market_plus_sentiment"
    train_metrics: RegressionMetrics
    train_financial: FinancialMetrics
    val_metrics: RegressionMetrics
    val_financial: FinancialMetrics
    test_metrics: Optional[RegressionMetrics] = None
    test_financial: Optional[FinancialMetrics] = None
    val_fold_stability_std: Optional[float] = None


class ExperimentRecord(BaseModel):
    """Record of a single model training experiment run."""
    experiment_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str
    market: str = "US"
    target: str
    feature_set_type: str
    feature_version: str
    sentiment_version: Optional[str] = None
    model_name: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    train_row_count: int
    val_row_count: int
    test_row_count: int
    validation_strategy: str = "chronological_70_15_15"
    metrics: Dict[str, float] = Field(default_factory=dict)
    execution_duration_sec: float = 0.0
    random_seed: int = 42


class ModelArtifactMetadata(BaseModel):
    """Metadata saved along with serialized trained model artifacts."""
    symbol: str
    model_name: str
    model_version: str = "v1.0.0"
    target_name: str
    feature_set_type: str
    feature_columns: List[str]
    dataset_version: str = "v1.0.0"
    feature_version: str = "v1.0.0"
    sentiment_version: Optional[str] = None
    trained_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    val_rmse: float
    val_directional_accuracy: float
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
