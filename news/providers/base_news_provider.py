"""Abstract base class interface for news data providers."""

from abc import ABC, abstractmethod
from typing import List, Optional
from news.schemas.news_article import NewsArticle


class BaseNewsProvider(ABC):
    """Abstract interface defining required methods for all news data providers."""

    @abstractmethod
    def fetch_news(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        query: Optional[str] = None,
        page_size: int = 50,
    ) -> List[NewsArticle]:
        """Fetch news articles for a target symbol or search query."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider identification name."""
        pass
