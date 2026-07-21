"""News processing package for deduplication and normalization."""

from news.processing.deduplication import ArticleDeduplicator
from news.processing.normalization import ArticleNormalizer

__all__ = ["ArticleDeduplicator", "ArticleNormalizer"]
