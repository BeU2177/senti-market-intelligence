"""Sentiment analysis package for FinBERT NLP scoring and time-decay pipeline."""

from news.sentiment.sentiment_model import FinBERTSentimentModel, get_sentiment_model
from news.sentiment.sentiment_pipeline import SentimentPipeline, SENTIMENT_FEATURE_VERSION

__all__ = [
    "FinBERTSentimentModel",
    "get_sentiment_model",
    "SentimentPipeline",
    "SENTIMENT_FEATURE_VERSION",
]
