"""Unit tests for news ingestion, article deduplication, and timestamp normalization."""

import pytest
from datetime import datetime, timezone, timedelta

from news.schemas.news_article import NewsArticle
from news.processing.deduplication import ArticleDeduplicator
from news.processing.normalization import ArticleNormalizer
from news.providers.news_api_provider import NewsAPIProvider


@pytest.fixture
def sample_raw_articles():
    """Generates sample raw news articles containing duplicates and unnormalized dates."""
    now = datetime.now(timezone.utc)
    return [
        NewsArticle(
            article_id="1",
            title="Apple Reports Record Quarterly Revenue",
            description="Apple Inc announced financial results for its fiscal quarter.",
            source="Reuters",
            url="https://example.com/news/1",
            published_at=now - timedelta(hours=2),
            symbol="AAPL",
        ),
        NewsArticle(
            article_id="1",  # Duplicate ID & URL
            title="Apple Reports Record Quarterly Revenue",
            description="Apple Inc announced financial results for its fiscal quarter.",
            source="Reuters",
            url="https://example.com/news/1",
            published_at=now - timedelta(hours=2),
            symbol="AAPL",
        ),
        NewsArticle(
            article_id="2",
            title="Apple Reports Record Quarterly Revenue!",  # Duplicate normalized title
            description="Duplicate title test.",
            source="Bloomberg",
            url="https://example.com/news/2",
            published_at=now - timedelta(hours=1),
            symbol="AAPL",
        ),
        NewsArticle(
            article_id="3",
            title="Tech Stocks Fall Ahead of Fed Meeting",
            description="Markets slip amidst interest rate concerns.",
            source="CNBC",
            url="https://example.com/news/3",
            published_at=now - timedelta(hours=5),
            symbol="AAPL",
        ),
    ]


def test_article_deduplication(sample_raw_articles):
    """Test ArticleDeduplicator filters duplicate IDs, URLs, and titles."""
    dedup = ArticleDeduplicator()
    unique = dedup.deduplicate(sample_raw_articles)

    assert len(unique) == 2
    titles = [a.title for a in unique]
    assert "Apple Reports Record Quarterly Revenue" in titles
    assert "Tech Stocks Fall Ahead of Fed Meeting" in titles


def test_article_normalization(sample_raw_articles):
    """Test ArticleNormalizer enforces UTC datetimes and calculates entity confidence."""
    norm = ArticleNormalizer()
    normalized = norm.normalize(sample_raw_articles)

    for art in normalized:
        assert art.published_at.tzinfo == timezone.utc
        assert art.fetched_at.tzinfo == timezone.utc
        assert 0.0 <= art.entity_match_confidence <= 1.0


def test_news_api_provider_fallback():
    """Test NewsAPIProvider fallback mechanism when API key is missing."""
    provider = NewsAPIProvider(api_key=None)
    articles = provider.fetch_news("AAPL")

    # Yahoo Finance fallback should return valid articles or empty list without raising
    assert isinstance(articles, list)
    if articles:
        assert articles[0].symbol == "AAPL"
        assert articles[0].published_at.tzinfo == timezone.utc
