"""Pydantic DTO schemas for production FastAPI endpoints."""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response DTO."""
    status: str = "ok"
    app_name: str = "Senti Market Intelligence API"
    version: str = "v1.0.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReadinessResponse(BaseModel):
    """Readiness probe response verifying component availability."""
    status: str = "ready"
    market_provider: str = "online"
    models_available: bool = True
    vector_db_available: bool = True
    llm_available: bool = True


class MarketDataDTO(BaseModel):
    """Market data DTO."""
    symbol: str
    price: float
    high: float
    low: float
    open: float
    volume: float
    market_regime: str
    data_freshness_status: str
    data_timestamp: str
    latency_ms: float


class PredictionDTO(BaseModel):
    """Multi-horizon prediction DTO."""
    prediction_id: str
    symbol: str
    prediction_timestamp: str
    data_as_of: str
    current_price: float
    predicted_returns: Dict[str, float]
    predicted_prices: Dict[str, float]
    directional_signals: Dict[str, str]
    ensemble_weights: Dict[str, float]
    confidence_level: str
    agreement_score: float
    model_metadata: Dict[str, Any]
    risk_disclaimer: str


class SentimentDTO(BaseModel):
    """Sentiment DTO."""
    symbol: str
    news_count: int
    latest_articles: List[Dict[str, Any]]


class AnalysisRequestDTO(BaseModel):
    """Market analysis request body."""
    symbol: str = "AAPL"
    question: str = "Why is the model currently predicting this directional move?"
    period: str = "1y"
    interval: str = "1d"


class ErrorDTO(BaseModel):
    """Structured error DTO."""
    error: str
    detail: Optional[str] = None
