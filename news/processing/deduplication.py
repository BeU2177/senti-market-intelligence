"""Deterministic news article deduplication module."""

import re
from typing import List
from news.schemas.news_article import NewsArticle
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ArticleDeduplicator:
    """Deduplicates news articles using URL, ID, and normalized title matching."""

    def deduplicate(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Filter out duplicate news articles deterministically."""
        if not articles:
            return []

        seen_urls = set()
        seen_ids = set()
        seen_titles = set()
        unique_articles: List[NewsArticle] = []

        for art in articles:
            # 1. URL Check
            url_clean = art.url.strip().lower()
            if url_clean and url_clean in seen_urls:
                continue

            # 2. Article ID Check
            if art.article_id in seen_ids:
                continue

            # 3. Normalized Title Check
            norm_title = self._normalize_title(art.title)
            if norm_title and norm_title in seen_titles:
                continue

            # Mark as seen and retain article
            if url_clean:
                seen_urls.add(url_clean)
            seen_ids.add(art.article_id)
            if norm_title:
                seen_titles.add(norm_title)

            unique_articles.append(art)

        logger.info(f"Deduplicated {len(articles)} articles down to {len(unique_articles)} unique articles")
        return unique_articles

    def _normalize_title(self, title: str) -> str:
        """Strip punctuation and convert to lowercase for duplicate title comparison."""
        clean = re.sub(r"[^\w\s]", "", title.lower())
        return " ".join(clean.split())
