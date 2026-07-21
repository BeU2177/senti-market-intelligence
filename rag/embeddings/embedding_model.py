"""Local embedding model interface using SentenceTransformers."""

from functools import lru_cache
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbeddingModel:
    """Generates 384-dim dense vector embeddings using local SentenceTransformer model."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        logger.info(f"Loading local embedding model '{self.model_name}'...")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Local embedding model loaded (dimension={self.dimension}).")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Encode list of strings into 2D numpy embedding matrix [N, D]."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> List[float]:
        """Encode single text query string into embedding list."""
        vec = self.model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]
        return vec.astype(np.float32).tolist()


@lru_cache(maxsize=1)
def get_embedding_model() -> LocalEmbeddingModel:
    """Returns singleton cached instance of LocalEmbeddingModel."""
    return LocalEmbeddingModel()
