"""NewsAPI and Yahoo Finance financial news provider implementation."""

import hashlib
import requests
from datetime import datetime, timezone, timedelta
from typing import List, Optional
import yfinance as yf

from config.settings import get_settings
from news.providers.base_news_provider import BaseNewsProvider
from news.schemas.news_article import NewsArticle
from utils.logging_config import get_logger

logger = get_logger(__name__)


class NewsAPIProvider(BaseNewsProvider):
    """Financial news provider integrating NewsAPI with fallback to Yahoo Finance ticker news."""

    PROVIDER_NAME = "news_api"

    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2/everything"

    def get_provider_name(self) -> str:
        return self.PROVIDER_NAME

    def fetch_news(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        query: Optional[str] = None,
        page_size: int = 50,
    ) -> List[NewsArticle]:
        """Fetch news articles using NewsAPI if credentials exist, else fallback to ticker news."""
        clean_symbol = symbol.strip().upper()
        logger.info(f"Fetching news for {clean_symbol} (query={query}, start={start_date}, end={end_date})")

        if self.api_key:
            try:
                articles = self._fetch_from_news_api(clean_symbol, start_date, end_date, query, page_size)
                if articles:
                    return articles
            except Exception as e:
                logger.warning(f"NewsAPI query failed for {clean_symbol}: {str(e)}. Falling back to Yahoo Finance news.")

        # Fallback to Yahoo Finance News feed
        return self._fetch_from_yfinance_news(clean_symbol)

    def _fetch_from_news_api(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        query: Optional[str],
        page_size: int,
    ) -> List[NewsArticle]:
        """Execute HTTP request to NewsAPI REST endpoint."""
        search_term = query or f"{symbol} stock market OR earnings"
        params = {
            "q": search_term,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": self.api_key,
        }
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date

        response = requests.get(self.base_url, params=params, timeout=10)
        if response.status_code != 200:
            logger.warning(f"NewsAPI HTTP Error {response.status_code}: {response.text}")
            return []

        data = response.json()
        raw_articles = data.get("articles", [])
        
        articles: List[NewsArticle] = []
        for item in raw_articles:
            pub_date_str = item.get("publishedAt")
            try:
                pub_dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            except Exception:
                pub_dt = datetime.now(timezone.utc)

            url = item.get("url", "")
            title = item.get("title", "")
            if not title or not url:
                continue

            art_id = hashlib.sha256(f"{url}_{pub_date_str}".encode()).hexdigest()[:16]

            articles.append(
                NewsArticle(
                    article_id=art_id,
                    title=title,
                    description=item.get("description", "") or "",
                    content=item.get("content", "") or "",
                    source=item.get("source", {}).get("name", "NewsAPI"),
                    url=url,
                    published_at=pub_dt,
                    fetched_at=datetime.now(timezone.utc),
                    symbol=symbol,
                    language="en",
                )
            )

        logger.info(f"NewsAPI returned {len(articles)} articles for {symbol}")
        return articles

    def _fetch_from_yfinance_news(self, symbol: str) -> List[NewsArticle]:
        """Fallback news ingestion from Yahoo Finance ticker news feed."""
        try:
            ticker = yf.Ticker(symbol)
            news_items = ticker.news or []
            articles: List[NewsArticle] = []

            for idx, item in enumerate(news_items):
                # Handle nested content dict structure in yfinance news
                content_dict = item.get("content", item)
                title = content_dict.get("title", "")
                summary = content_dict.get("summary", "") or content_dict.get("description", "")
                provider = content_dict.get("provider", {}).get("displayName", "Yahoo Finance") if isinstance(content_dict.get("provider"), dict) else "Yahoo Finance"
                
                canonical = content_dict.get("canonicalUrl", {})
                url = canonical.get("url", "") if isinstance(canonical, dict) else content_dict.get("link", f"https://finance.yahoo.com/quote/{symbol}")

                pub_str = content_dict.get("pubDate")
                if pub_str:
                    try:
                        pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                    except Exception:
                        pub_dt = datetime.now(timezone.utc) - timedelta(hours=idx)
                else:
                    pub_dt = datetime.now(timezone.utc) - timedelta(hours=idx)

                if not title:
                    continue

                art_id = hashlib.sha256(f"{title}_{symbol}_{pub_dt.isoformat()}".encode()).hexdigest()[:16]

                articles.append(
                    NewsArticle(
                        article_id=art_id,
                        title=title,
                        description=summary or title,
                        content=summary or title,
                        source=provider,
                        url=url,
                        published_at=pub_dt,
                        fetched_at=datetime.now(timezone.utc),
                        symbol=symbol,
                        language="en",
                    )
                )

            logger.info(f"Yahoo Finance news returned {len(articles)} articles for {symbol}")
            return articles

        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance news for {symbol}: {str(e)}")
            return []
