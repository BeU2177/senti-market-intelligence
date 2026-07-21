"""Incremental news update service performing deduplication and sentiment aggregation."""

from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any

from news.ingestion.news_ingestion import NewsIngestionService
from news.schemas.news_article import NewsArticle
from news.sentiment.sentiment_pipeline import SentimentPipeline
from data.storage.storage_manager import StorageManager
from utils.logging_config import get_logger

logger = get_logger(__name__)


class NewsUpdateService:
    """Incremental news ingestion and FinBERT sentiment update pipeline."""

    def __init__(self, storage_manager: Optional[StorageManager] = None):
        self.storage = storage_manager or StorageManager()
        self.ingestion_service = NewsIngestionService(storage_manager=self.storage)
        self.sentiment_pipeline = SentimentPipeline()

    def update_news_and_sentiment(self, symbol: str, lookback_days: int = 7) -> List[NewsArticle]:
        """Fetch news since last timestamp, deduplicate against archive, and update sentiment."""
        clean_symbol = symbol.strip().upper()
        start_dt = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        start_str = start_dt.strftime("%Y-%m-%d")

        logger.info(f"NewsUpdateService starting incremental news update for {clean_symbol}")

        articles = self.ingestion_service.ingest_news(
            symbol=clean_symbol,
            start_date=start_str,
            save_to_disk=True,
        )

        logger.info(f"NewsUpdateService updated {len(articles)} articles for {clean_symbol}")
        return articles
