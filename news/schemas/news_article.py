"""Schemas for news articles, sentiment outputs, and aggregated sentiment features."""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class NewsArticle(BaseModel):
    """Normalized schema for a single financial news article."""
    article_id: str = Field(description="Unique hash or ID of the article")
    title: str = Field(description="Article headline title")
    description: Optional[str] = Field(default="", description="Article short summary or description")
    content: Optional[str] = Field(default="", description="Full or snippet content")
    source: str = Field(description="News publisher source name")
    url: str = Field(description="URL link to full article")
    published_at: datetime = Field(description="UTC publication datetime")
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC ingestion datetime")
    symbol: str = Field(description="Associated stock ticker symbol")
    market: str = Field(default="US", description="Associated market country or region")
    language: str = Field(default="en", description="ISO 639-1 language code")
    entity_match_confidence: float = Field(default=1.0, description="Match confidence score [0, 1]")


class ArticleSentiment(BaseModel):
    """Sentiment classification and probability distribution for an article."""
    article_id: str
    published_at: datetime
    symbol: str
    positive_probability: float
    negative_probability: float
    neutral_probability: float
    predicted_label: str
    sentiment_score: float = Field(description="Calculated as Positive Prob - Negative Prob in [-1, 1]")
    confidence: float


class AggregatedSentimentFeatures(BaseModel):
    """Aggregated sentiment features at a specific market timestamp T."""
    timestamp: datetime
    symbol: str
    sentiment_score: float = 0.0
    weighted_sentiment_score: float = 0.0
    positive_probability: float = 0.333
    negative_probability: float = 0.333
    neutral_probability: float = 0.334
    weighted_positive_prob: float = 0.333
    weighted_negative_prob: float = 0.333
    sentiment_confidence: float = 0.0
    news_count: int = 0
    positive_news_ratio: float = 0.0
    negative_news_ratio: float = 0.0
    neutral_news_ratio: float = 0.0
    sentiment_1d_change: float = 0.0
    sentiment_3d_change: float = 0.0
    sentiment_7d_change: float = 0.0
    sentiment_rolling_mean: float = 0.0
    sentiment_rolling_std: float = 0.0
    news_volume_change: float = 0.0
