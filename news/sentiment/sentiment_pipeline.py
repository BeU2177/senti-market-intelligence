"""Temporal alignment, time-decay sentiment aggregation, and momentum features pipeline."""

import math
from datetime import datetime, timezone, timedelta
from typing import List, Optional
import numpy as np
import pandas as pd

from news.schemas.news_article import NewsArticle, ArticleSentiment, AggregatedSentimentFeatures
from news.sentiment.sentiment_model import get_sentiment_model, FinBERTSentimentModel
from utils.logging_config import get_logger

logger = get_logger(__name__)

SENTIMENT_FEATURE_VERSION = "v1.0.0"


class SentimentPipeline:
    """Orchestrates batch NLP inference, exponential time-decay weighting, and leakage-free temporal alignment."""

    def __init__(self, decay_lambda: float = 0.5, sentiment_model: Optional[FinBERTSentimentModel] = None):
        self.decay_lambda = decay_lambda
        self.model = sentiment_model or get_sentiment_model()

    def generate_sentiment_features(
        self,
        market_df: pd.DataFrame,
        articles: List[NewsArticle],
        symbol: str = "UNKNOWN",
    ) -> pd.DataFrame:
        """Aligns sentiment features temporally with market timestamps T without future news leakage."""
        clean_symbol = symbol.strip().upper()
        logger.info(f"SentimentPipeline starting feature generation for {clean_symbol} (articles={len(articles)})")

        if "timestamp" not in market_df.columns or market_df.empty:
            return pd.DataFrame(index=market_df.index)

        # 1. Run Batch NLP Sentiment Inference on Articles
        article_sentiments: List[ArticleSentiment] = self.model.predict_sentiment_batch(articles)

        # 2. Map Article ID to Sentiment Object
        sentiment_map = {s.article_id: s for s in article_sentiments}

        # 3. Iterate over Market Timestamps T and aggregate strictly published_at <= T
        aggregated_rows = []
        market_ts_list = pd.to_datetime(market_df["timestamp"], utc=True)

        for T in market_ts_list:
            T_dt = T.to_pydatetime() if hasattr(T, "to_pydatetime") else T
            if T_dt.tzinfo is None:
                T_dt = T_dt.replace(tzinfo=timezone.utc)

            # Strictly filter out future articles: published_at <= T
            valid_articles = [art for art in articles if art.published_at <= T_dt]

            row_feat = self._aggregate_at_timestamp(T_dt, clean_symbol, valid_articles, sentiment_map)
            aggregated_rows.append(row_feat.model_dump())

        sent_df = pd.DataFrame(aggregated_rows)

        # 4. Compute Rolling Momentum & Sentiment Trend Features
        sent_df = self._compute_sentiment_momentum(sent_df)

        logger.info(f"SentimentPipeline completed for {clean_symbol}: generated {len(sent_df.columns)} sentiment features across {len(sent_df)} rows")
        return sent_df

    def _aggregate_at_timestamp(
        self,
        T: datetime,
        symbol: str,
        articles: List[NewsArticle],
        sentiment_map: dict[str, ArticleSentiment],
    ) -> AggregatedSentimentFeatures:
        """Aggregate sentiment scores for articles published at or before timestamp T using time-decay weighting."""
        if not articles:
            return AggregatedSentimentFeatures(
                timestamp=T,
                symbol=symbol,
                sentiment_score=0.0,
                weighted_sentiment_score=0.0,
                positive_probability=0.333,
                negative_probability=0.333,
                neutral_probability=0.334,
                weighted_positive_prob=0.333,
                weighted_negative_prob=0.333,
                sentiment_confidence=0.0,
                news_count=0,
                positive_news_ratio=0.0,
                negative_news_ratio=0.0,
                neutral_news_ratio=0.0,
            )

        weights = []
        scores = []
        pos_probs = []
        neg_probs = []
        neu_probs = []
        confidences = []
        labels = []

        # Lookback window: consider news up to 30 days old relative to T
        max_lookback_days = 30.0

        for art in articles:
            age_days = max(0.0, (T - art.published_at).total_seconds() / 86400.0)
            if age_days > max_lookback_days:
                continue

            # Exponential decay weight: w = exp(-lambda * age_in_days)
            weight = math.exp(-self.decay_lambda * age_days)
            weights.append(weight)

            sent = sentiment_map.get(art.article_id)
            if sent:
                scores.append(sent.sentiment_score)
                pos_probs.append(sent.positive_probability)
                neg_probs.append(sent.negative_probability)
                neu_probs.append(sent.neutral_probability)
                confidences.append(sent.confidence)
                labels.append(sent.predicted_label)
            else:
                scores.append(0.0)
                pos_probs.append(0.333)
                neg_probs.append(0.333)
                neu_probs.append(0.334)
                confidences.append(0.5)
                labels.append("neutral")

        total_articles = len(scores)
        if total_articles == 0 or sum(weights) == 0:
            return AggregatedSentimentFeatures(
                timestamp=T,
                symbol=symbol,
                sentiment_score=0.0,
                weighted_sentiment_score=0.0,
                positive_probability=0.333,
                negative_probability=0.333,
                neutral_probability=0.334,
                weighted_positive_prob=0.333,
                weighted_negative_prob=0.333,
                sentiment_confidence=0.0,
                news_count=0,
                positive_news_ratio=0.0,
                negative_news_ratio=0.0,
                neutral_news_ratio=0.0,
            )

        sum_w = sum(weights)
        norm_w = [w / sum_w for w in weights]

        raw_score = sum(s for s in scores) / total_articles
        weighted_score = sum(s * w for s, w in zip(scores, norm_w))

        pos_p = sum(p for p in pos_probs) / total_articles
        neg_p = sum(p for p in neg_probs) / total_articles
        neu_p = sum(p for p in neu_probs) / total_articles

        w_pos_p = sum(p * w for p, w in zip(pos_probs, norm_w))
        w_neg_p = sum(p * w for p, w in zip(neg_probs, norm_w))

        avg_conf = sum(c for c in confidences) / total_articles

        pos_cnt = labels.count("positive")
        neg_cnt = labels.count("negative")
        neu_cnt = labels.count("neutral")

        return AggregatedSentimentFeatures(
            timestamp=T,
            symbol=symbol,
            sentiment_score=round(raw_score, 4),
            weighted_sentiment_score=round(weighted_score, 4),
            positive_probability=round(pos_p, 4),
            negative_probability=round(neg_p, 4),
            neutral_probability=round(neu_p, 4),
            weighted_positive_prob=round(w_pos_p, 4),
            weighted_negative_prob=round(w_neg_p, 4),
            sentiment_confidence=round(avg_conf, 4),
            news_count=total_articles,
            positive_news_ratio=round(pos_cnt / total_articles, 4),
            negative_news_ratio=round(neg_cnt / total_articles, 4),
            neutral_news_ratio=round(neu_cnt / total_articles, 4),
        )

    def _compute_sentiment_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates backward-looking sentiment momentum and rolling metrics."""
        if df.empty or "weighted_sentiment_score" not in df.columns:
            return df

        w_score = df["weighted_sentiment_score"]
        news_cnt = df["news_count"]

        df["sentiment_1d_change"] = w_score.diff(1).fillna(0.0)
        df["sentiment_3d_change"] = w_score.diff(3).fillna(0.0)
        df["sentiment_7d_change"] = w_score.diff(7).fillna(0.0)

        df["sentiment_rolling_mean"] = w_score.rolling(window=7, min_periods=1).mean()
        df["sentiment_rolling_std"] = w_score.rolling(window=7, min_periods=1).std().fillna(0.0)

        df["news_volume_change"] = news_cnt.pct_change().fillna(0.0)

        return df
