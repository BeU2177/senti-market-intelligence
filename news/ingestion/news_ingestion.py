"""News ingestion service coordinating retrieval, deduplication, and normalization."""

from typing import List, Optional
from news.providers.base_news_provider import BaseNewsProvider
from news.providers.news_api_provider import NewsAPIProvider
from news.processing.deduplication import ArticleDeduplicator
from news.processing.normalization import ArticleNormalizer
from news.schemas.news_article import NewsArticle
from data.storage.storage_manager import StorageManager
from utils.logging_config import get_logger

logger = get_logger(__name__)


class NewsIngestionService:
    """Service handling news retrieval, deduplication, normalization, and raw/processed storage."""

    def __init__(
        self,
        provider: Optional[BaseNewsProvider] = None,
        storage_manager: Optional[StorageManager] = None,
    ):
        self.provider = provider or NewsAPIProvider()
        self.deduplicator = ArticleDeduplicator()
        self.normalizer = ArticleNormalizer()
        self.storage = storage_manager or StorageManager()

    def ingest_news(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        query: Optional[str] = None,
        save_to_disk: bool = True,
    ) -> List[NewsArticle]:
        """Ingests, deduplicates, normalizes, and stores news articles for a symbol."""
        clean_symbol = symbol.strip().upper()
        logger.info(f"NewsIngestionService starting ingestion for {clean_symbol}")

        # 1. Fetch Raw Articles
        raw_articles = self.provider.fetch_news(
            symbol=clean_symbol,
            start_date=start_date,
            end_date=end_date,
            query=query,
        )

        if not raw_articles:
            logger.warning(f"NewsIngestionService: No news returned for {clean_symbol}")
            return []

        # 2. Deduplicate
        dedup_articles = self.deduplicator.deduplicate(raw_articles)

        # 3. Normalize Timestamps & Entity Match Confidence
        norm_articles = self.normalizer.normalize(dedup_articles)

        # 4. Save to Disk
        if save_to_disk:
            self.storage.save_news_articles(norm_articles, clean_symbol)

        logger.info(f"NewsIngestionService completed for {clean_symbol}: ingested={len(norm_articles)} articles")
        return norm_articles
