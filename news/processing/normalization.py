"""News article timestamp normalization and entity match confidence module."""

from datetime import datetime, timezone
from typing import List
from news.schemas.news_article import NewsArticle
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ArticleNormalizer:
    """Normalizes publication timestamps and scores symbol entity match confidence."""

    def normalize(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Normalize article timestamps and calculate entity match scores."""
        normalized: List[NewsArticle] = []

        for art in articles:
            # Enforce UTC timezone awareness for published_at and fetched_at
            pub_dt = art.published_at
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            else:
                pub_dt = pub_dt.astimezone(timezone.utc)

            fetch_dt = art.fetched_at
            if fetch_dt.tzinfo is None:
                fetch_dt = fetch_dt.replace(tzinfo=timezone.utc)
            else:
                fetch_dt = fetch_dt.astimezone(timezone.utc)

            # Entity Match Confidence calculation
            confidence = self._calculate_entity_confidence(art.symbol, art.title, art.description or "")

            normalized.append(
                NewsArticle(
                    article_id=art.article_id,
                    title=art.title,
                    description=art.description,
                    content=art.content,
                    source=art.source,
                    url=art.url,
                    published_at=pub_dt,
                    fetched_at=fetch_dt,
                    symbol=art.symbol,
                    market=art.market,
                    language=art.language,
                    entity_match_confidence=confidence,
                )
            )

        return normalized

    def _calculate_entity_confidence(self, symbol: str, title: str, description: str) -> float:
        """Score how strongly an article is linked to the requested symbol."""
        clean_sym = symbol.strip().upper().replace(".NS", "").replace(".AE", "")
        text = f"{title} {description}".upper()

        if clean_sym in title.upper():
            return 1.0
        elif clean_sym in text:
            return 0.85
        else:
            return 0.60
