"""Persistent vector store using SentenceTransformers embeddings and NumPy cosine similarity search."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from rag.chunking.splitter import DocumentChunk
from rag.embeddings.embedding_model import get_embedding_model, LocalEmbeddingModel
from utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_VECTOR_DIR = Path("artifacts") / "vector_db"
VECTOR_FILE_NAME = "market_knowledge.json"


class CollectionWrapper:
    """Wrapper providing count() property for vector collection compatibility."""

    def __init__(self, get_count_fn):
        self._get_count_fn = get_count_fn

    def count(self) -> int:
        return self._get_count_fn()


class ChromaStore:
    """Persistent vector store performing cosine similarity search on local embeddings."""

    def __init__(
        self,
        persist_dir: Path = DEFAULT_VECTOR_DIR,
        collection_name: str = "market_knowledge",
        embedding_model: Optional[LocalEmbeddingModel] = None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model or get_embedding_model()
        self.store_file = self.persist_dir / VECTOR_FILE_NAME

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_data: List[Dict[str, Any]] = []
        self._load_store()

        self.collection = CollectionWrapper(lambda: len(self.chunks_data))
        logger.info(f"ChromaStore (LocalVectorStore) initialized at '{self.store_file}' (items={len(self.chunks_data)})")

    def _load_store(self) -> None:
        """Load stored chunks and embeddings from disk if file exists."""
        if self.store_file.exists():
            try:
                with open(self.store_file, "r", encoding="utf-8") as f:
                    self.chunks_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load vector store from {self.store_file}: {str(e)}")
                self.chunks_data = []

    def _save_store(self) -> None:
        """Persist chunks and embeddings to disk."""
        try:
            with open(self.store_file, "w", encoding="utf-8") as f:
                json.dump(self.chunks_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save vector store to {self.store_file}: {str(e)}")

    def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Upsert document text chunks and embeddings into store."""
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        embeddings = self.embedding_model.embed_texts(texts)

        # Existing chunk map to avoid duplicates
        existing_map = {item["chunk_id"]: idx for idx, item in enumerate(self.chunks_data)}

        for c, emb in zip(chunks, embeddings):
            item = {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "title": c.title,
                "source": c.source,
                "category": c.category,
                "chunk_index": c.chunk_index,
                "content": c.content,
                "embedding": emb.tolist(),
            }
            if c.chunk_id in existing_map:
                self.chunks_data[existing_map[c.chunk_id]] = item
            else:
                self.chunks_data.append(item)

        self._save_store()
        logger.info(f"ChromaStore upserted {len(chunks)} chunks into store. Total items: {len(self.chunks_data)}")
        return len(chunks)

    def query_similarity(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Perform cosine similarity vector search for query string."""
        if not self.chunks_data:
            return []

        q_vec = np.array(self.embedding_model.embed_query(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        q_unit = q_vec / q_norm

        scores = []
        for idx, item in enumerate(self.chunks_data):
            emb_vec = np.array(item["embedding"], dtype=np.float32)
            emb_norm = np.linalg.norm(emb_vec)
            if emb_norm == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(q_unit, emb_vec / emb_norm))
            scores.append((similarity, idx))

        # Sort descending by similarity score
        scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = scores[: min(top_k, len(scores))]

        hits = []
        for sim, idx in top_indices:
            item = self.chunks_data[idx]
            hits.append({
                "content": item["content"],
                "title": item["title"],
                "source": item["source"],
                "category": item["category"],
                "similarity_score": round(sim, 4),
            })

        return hits
