"""Semantic search retriever returning text chunks with source provenance citations."""

from typing import List, Dict, Any, Optional
from rag.vector_store.chroma_store import ChromaStore
from utils.logging_config import get_logger

logger = get_logger(__name__)


class Retriever:
    """Semantic search retriever for financial knowledge context."""

    def __init__(self, vector_store: Optional[ChromaStore] = None):
        self.store = vector_store or ChromaStore()

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve top-K relevant knowledge chunks with source citations."""
        logger.info(f"Retriever querying ChromaStore for '{query}' (top_k={top_k})")
        return self.store.query_similarity(query=query, top_k=top_k)
