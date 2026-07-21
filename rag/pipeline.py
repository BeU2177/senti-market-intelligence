"""RAG pipeline orchestrating knowledge base ingestion and semantic retrieval."""

from typing import List, Dict, Any, Optional
from rag.documents.loader import DocumentLoader
from rag.chunking.splitter import TextSplitter
from rag.vector_store.chroma_store import ChromaStore
from rag.retrieval.retriever import Retriever
from utils.logging_config import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """Orchestrates knowledge ingestion, vector indexing, and grounded context retrieval."""

    def __init__(self, vector_store: Optional[ChromaStore] = None):
        self.loader = DocumentLoader()
        self.splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
        self.store = vector_store or ChromaStore()
        self.retriever = Retriever(vector_store=self.store)

    def ingest_knowledge_base(self) -> int:
        """Loads documents from data/knowledge_base/, chunks text, and indexes in ChromaDB."""
        docs = self.loader.load_documents()
        if not docs:
            logger.warning("RAGPipeline: No documents found to ingest.")
            return 0

        chunks = self.splitter.split_documents(docs)
        count = self.store.upsert_chunks(chunks)
        logger.info(f"RAGPipeline ingested {count} chunks from {len(docs)} knowledge base documents.")
        return count

    def retrieve_context(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge context for a query, auto-ingesting knowledge base if store is empty."""
        if self.store.collection.count() == 0:
            logger.info("ChromaStore collection is empty. Auto-ingesting knowledge base...")
            self.ingest_knowledge_base()

        return self.retriever.retrieve(query=query, top_k=top_k)
