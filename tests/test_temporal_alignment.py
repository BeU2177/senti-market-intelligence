"""Unit tests for temporal alignment and future news leakage prevention."""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta

from news.schemas.news_article import NewsArticle
from news.sentiment.sentiment_pipeline import SentimentPipeline


def test_temporal_alignment_leakage_prevention():
    """Verify that news published after market timestamp T (published_at > T) is excluded from features at T."""
    # Market Bar timestamp T = 2025-01-10 10:00 UTC
    T = datetime(2025, 1, 10, 10, 0, 0, tzinfo=timezone.utc)
    
    market_df = pd.DataFrame({
        "timestamp": [T],
        "close": [150.0],
    })

    # News A: Published BEFORE T (09:00 UTC) -> Positive
    news_a = NewsArticle(
        article_id="news_a",
        title="Company Profits Rise Record High Growth",
        source="Test",
        url="https://example.com/news_a",
        published_at=datetime(2025, 1, 10, 9, 0, 0, tzinfo=timezone.utc),
        symbol="AAPL",
    )

    # News B: Published AFTER T (11:00 UTC) -> Negative
    news_b = NewsArticle(
        article_id="news_b",
        title="Company Bankruptcy Crash Plunge Loss",
        source="Test",
        url="https://example.com/news_b",
        published_at=datetime(2025, 1, 10, 11, 0, 0, tzinfo=timezone.utc),
        symbol="AAPL",
    )

    pipeline = SentimentPipeline()
    
    # 1. Run pipeline with both News A and News B
    sent_df = pipeline.generate_sentiment_features(market_df, [news_a, news_b], symbol="AAPL")

    # The features at timestamp T must see ONLY News A (news_count = 1)
    assert len(sent_df) == 1
    assert sent_df["news_count"].iloc[0] == 1
    # Weighted sentiment score at T must be positive (derived exclusively from News A)
    assert sent_df["weighted_sentiment_score"].iloc[0] > 0.0

    # 2. Add market bar T2 = 2025-01-10 12:00 UTC (after News B)
    T2 = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
    market_df_multi = pd.DataFrame({"timestamp": [T, T2], "close": [150.0, 148.0]})

    sent_df_multi = pipeline.generate_sentiment_features(market_df_multi, [news_a, news_b], symbol="AAPL")

    # At timestamp T: news_count is 1 (News A only)
    assert sent_df_multi["news_count"].iloc[0] == 1
    # At timestamp T2: news_count is 2 (Both News A and News B included)
    assert sent_df_multi["news_count"].iloc[1] == 2
