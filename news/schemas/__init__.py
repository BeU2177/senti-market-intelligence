"""Schemas package for news articles and sentiment structures."""

from news.schemas.news_article import (
    NewsArticle,
    ArticleSentiment,
    AggregatedSentimentFeatures,
)

__all__ = [
    "NewsArticle",
    "ArticleSentiment",
    "AggregatedSentimentFeatures",
]
