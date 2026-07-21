"""Financial news ingestion, NLP sentiment, and temporal feature pipeline package."""

from news.schemas.news_article import NewsArticle, ArticleSentiment, AggregatedSentimentFeatures
from news.providers.base_news_provider import BaseNewsProvider
from news.providers.news_api_provider import NewsAPIProvider
from news.ingestion.news_ingestion import NewsIngestionService
from news.sentiment.sentiment_model import FinBERTSentimentModel, get_sentiment_model
from news.sentiment.sentiment_pipeline import SentimentPipeline, SENTIMENT_FEATURE_VERSION

__all__ = [
    "NewsArticle",
    "ArticleSentiment",
    "AggregatedSentimentFeatures",
    "BaseNewsProvider",
    "NewsAPIProvider",
    "NewsIngestionService",
    "FinBERTSentimentModel",
    "get_sentiment_model",
    "SentimentPipeline",
    "SENTIMENT_FEATURE_VERSION",
]
