"""Providers package for financial news ingestion."""

from news.providers.base_news_provider import BaseNewsProvider
from news.providers.news_api_provider import NewsAPIProvider

__all__ = ["BaseNewsProvider", "NewsAPIProvider"]
