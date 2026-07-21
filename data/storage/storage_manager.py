"""Storage manager handling Parquet persistence for raw and processed datasets and JSON provenance metadata."""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from data.schemas.market_data import DatasetProvenance
from utils.logging_config import get_logger

logger = get_logger(__name__)

BASE_DIR = Path("data")
RAW_MARKET_DIR = BASE_DIR / "raw" / "market"
PROCESSED_MARKET_DIR = BASE_DIR / "processed" / "market"
RAW_NEWS_DIR = BASE_DIR / "raw" / "news"
PROCESSED_NEWS_DIR = BASE_DIR / "processed" / "news"
FEATURES_MARKET_DIR = BASE_DIR / "features" / "market"
METADATA_DIR = BASE_DIR / "metadata"


class StorageManager:
    """Manages reading and writing raw/processed Parquet files for market data, news articles, and JSON metadata."""

    def __init__(self, base_dir: Path = BASE_DIR):
        self.raw_dir = base_dir / "raw" / "market"
        self.processed_dir = base_dir / "processed" / "market"
        self.raw_news_dir = base_dir / "raw" / "news"
        self.processed_news_dir = base_dir / "processed" / "news"
        self.features_dir = base_dir / "features" / "market"
        self.metadata_dir = base_dir / "metadata"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create storage directories if they do not exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.raw_news_dir.mkdir(parents=True, exist_ok=True)
        self.processed_news_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def save_raw_dataset(self, df: pd.DataFrame, symbol: str, interval: str) -> Path:
        """Save un-transformed raw DataFrame to Parquet format."""
        clean_symbol = symbol.strip().upper().replace("/", "_").replace("^", "")
        file_name = f"{clean_symbol}_{interval}_raw.parquet"
        file_path = self.raw_dir / file_name

        logger.info(f"Saving raw Parquet dataset to: {file_path}")
        df.to_parquet(file_path, engine="pyarrow", index=False)
        return file_path

    def save_processed_dataset(self, df: pd.DataFrame, symbol: str, interval: str) -> Path:
        """Save normalized processed DataFrame to Parquet format."""
        clean_symbol = symbol.strip().upper().replace("/", "_").replace("^", "")
        file_name = f"{clean_symbol}_{interval}_processed.parquet"
        file_path = self.processed_dir / file_name

        logger.info(f"Saving processed Parquet dataset to: {file_path}")
        df.to_parquet(file_path, engine="pyarrow", index=False)
        return file_path

    def save_provenance_metadata(self, provenance: DatasetProvenance) -> Path:
        """Save dataset provenance metadata to JSON file."""
        clean_symbol = provenance.symbol.strip().upper().replace("/", "_").replace("^", "")
        file_name = f"{clean_symbol}_{provenance.interval}_metadata.json"
        file_path = self.metadata_dir / file_name

        logger.info(f"Saving provenance metadata JSON to: {file_path}")
        metadata_dict = provenance.model_dump(mode="json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        return file_path

    def load_processed_dataset(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Load processed dataset Parquet file if exists."""
        clean_symbol = symbol.strip().upper().replace("/", "_").replace("^", "")
        file_path = self.processed_dir / f"{clean_symbol}_{interval}_processed.parquet"
        if not file_path.exists():
            return None
        logger.info(f"Loading processed Parquet dataset from: {file_path}")
        return pd.read_parquet(file_path, engine="pyarrow")

    def save_feature_dataset(self, df: pd.DataFrame, symbol: str, interval: str) -> Path:
        """Save feature dataset DataFrame to Parquet format."""
        clean_symbol = symbol.strip().upper().replace("/", "_").replace("^", "")
        file_name = f"{clean_symbol}_{interval}_features.parquet"
        file_path = self.features_dir / file_name

        logger.info(f"Saving feature Parquet dataset to: {file_path}")
        df.to_parquet(file_path, engine="pyarrow", index=False)
        return file_path

    def load_feature_dataset(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Load feature dataset Parquet file if exists."""
        clean_symbol = symbol.strip().upper().replace("/", "_").replace("^", "")
        file_path = self.features_dir / f"{clean_symbol}_{interval}_features.parquet"
        if not file_path.exists():
            return None
        logger.info(f"Loading feature Parquet dataset from: {file_path}")
        return pd.read_parquet(file_path, engine="pyarrow")

    def save_news_articles(self, articles: list, symbol: str) -> Path:
        """Save processed news articles to JSON and Parquet formats."""
        clean_symbol = symbol.strip().upper().replace("/", "_").replace("^", "")
        file_path_json = self.processed_news_dir / f"{clean_symbol}_news.json"
        
        logger.info(f"Saving news articles to: {file_path_json}")
        data = [a.model_dump(mode="json") if hasattr(a, "model_dump") else a for a in articles]
        import json
        with open(file_path_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
            
        if data:
            df_news = pd.DataFrame(data)
            file_path_pq = self.processed_news_dir / f"{clean_symbol}_news.parquet"
            df_news.to_parquet(file_path_pq, engine="pyarrow", index=False)

        return file_path_json

    def load_news_articles(self, symbol: str) -> list:
        """Load news articles from processed JSON or Parquet if exists."""
        clean_symbol = symbol.strip().upper().replace("/", "_").replace("^", "")
        file_path_json = self.processed_news_dir / f"{clean_symbol}_news.json"
        if not file_path_json.exists():
            return []
            
        import json
        with open(file_path_json, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        from news.schemas.news_article import NewsArticle
        articles = []
        for item in raw_data:
            if isinstance(item.get("published_at"), str):
                item["published_at"] = datetime.fromisoformat(item["published_at"])
            if isinstance(item.get("fetched_at"), str):
                item["fetched_at"] = datetime.fromisoformat(item["fetched_at"])
            articles.append(NewsArticle(**item))
            
        return articles
