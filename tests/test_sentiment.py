"""Unit tests for FinBERT sentiment model scoring and exponential time-decay weighting."""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta

from news.schemas.news_article import NewsArticle
from news.sentiment.sentiment_model import get_sentiment_model
from news.sentiment.sentiment_pipeline import SentimentPipeline


@pytest.fixture
def sample_sentiment_articles():
    """Generates articles with positive, negative, and neutral headlines."""
    now = datetime.now(timezone.utc)
    return [
        NewsArticle(
            article_id="pos1",
            title="Company Earnings Surge 50% Record High Revenue Profit Beat",
            description="Massive quarterly growth driven by strong sales.",
            source="Test",
            url="https://example.com/pos1",
            published_at=now - timedelta(hours=1),
            symbol="AAPL",
        ),
        NewsArticle(
            article_id="neg1",
            title="Company Faces Massive Loss Debt Crash Plunge Warning",
            description="Severe financial distress and declining sales reported.",
            source="Test",
            url="https://example.com/neg1",
            published_at=now - timedelta(hours=3),
            symbol="AAPL",
        ),
    ]


def test_sentiment_model_probabilities(sample_sentiment_articles):
    """Test sentiment probabilities sum to 1 and sentiment_score is bounded in [-1, 1]."""
    model = get_sentiment_model()
    sentiments = model.predict_sentiment_batch(sample_sentiment_articles)

    assert len(sentiments) == 2
    for s in sentiments:
        # Probabilities sum to ~1.0
        prob_sum = s.positive_probability + s.negative_probability + s.neutral_probability
        assert np.isclose(prob_sum, 1.0, atol=0.01)

        # Score bounded between -1.0 and 1.0
        assert -1.0 <= s.sentiment_score <= 1.0
        # Formula check: score = positive_prob - negative_prob
        expected_score = round(s.positive_probability - s.negative_probability, 4)
        assert np.isclose(s.sentiment_score, expected_score, atol=0.001)

    # First article (positive headline) should have higher positive probability
    assert sentiments[0].sentiment_score > sentiments[1].sentiment_score


def test_time_decay_weighting():
    """Test recent news receives higher exponential decay weight than old news."""
    now = datetime.now(timezone.utc)
    pipeline = SentimentPipeline(decay_lambda=0.5)

    recent_art = NewsArticle(
        article_id="recent",
        title="Bullish Surge Beat Profit",
        source="Test",
        url="https://example.com/recent",
        published_at=now - timedelta(hours=2),
        symbol="AAPL",
    )

    old_art = NewsArticle(
        article_id="old",
        title="Bearish Loss Collapse",
        source="Test",
        url="https://example.com/old",
        published_at=now - timedelta(days=10),
        symbol="AAPL",
    )

    sent_map = {
        "recent": pipeline.model.predict_sentiment_batch([recent_art])[0],
        "old": pipeline.model.predict_sentiment_batch([old_art])[0],
    }

    # Aggregate at timestamp now
    agg = pipeline._aggregate_at_timestamp(now, "AAPL", [recent_art, old_art], sent_map)

    # Because recent_art is positive and heavily weighted compared to 10-day old article,
    # weighted_sentiment_score should be closer to recent_art's score than unweighted mean
    assert agg.weighted_sentiment_score > agg.sentiment_score
