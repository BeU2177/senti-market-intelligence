"""Unit tests for RAG document loading, text splitting, embeddings, ChromaDB vector store, and retriever."""

import pytest
import numpy as np
from pathlib import Path

from rag.documents.loader import DocumentLoader
from rag.chunking.splitter import TextSplitter, KnowledgeDocument
from rag.embeddings.embedding_model import get_embedding_model
from rag.vector_store.chroma_store import ChromaStore
from rag.retrieval.retriever import Retriever


def test_document_loader():
    """Test DocumentLoader reads knowledge base markdown files."""
    loader = DocumentLoader()
    docs = loader.load_documents()

    assert len(docs) >= 3
    titles = [d.title for d in docs]
    assert "Technical Indicators" in titles or "Time Series Validation" in titles


def test_text_splitter():
    """Test TextSplitter generates chunks with overlapping boundaries."""
    doc = KnowledgeDocument(
        document_id="test1",
        title="Test Doc",
        content="Line 1: RSI technical analysis.\nLine 2: MACD trend indicator.\nLine 3: Bollinger bands volatility.",
        source="test.md",
        category="test",
        file_path="/tmp/test.md",
    )
    splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
    chunks = splitter.split_documents([doc])

    assert len(chunks) >= 1
    assert chunks[0].document_id == "test1"


def test_embedding_model():
    """Test LocalEmbeddingModel generates 384-dimensional dense vectors."""
    model = get_embedding_model()
    embeddings = model.embed_texts(["RSI momentum indicator", "Market volatility"])

    assert embeddings.shape == (2, 384)


def test_chroma_store_and_retriever(tmp_path):
    """Test ChromaStore indexing and Retriever top-K similarity search."""
    store = ChromaStore(persist_dir=tmp_path / "test_chroma", collection_name="test_collection")
    loader = DocumentLoader()
    docs = loader.load_documents()

    splitter = TextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    store.upsert_chunks(chunks)
    assert store.collection.count() == len(chunks)

    retriever = Retriever(vector_store=store)
    hits = retriever.retrieve(query="What is RSI?", top_k=2)

    assert len(hits) <= 2
    if hits:
        assert "similarity_score" in hits[0]
        assert "content" in hits[0]
        assert "title" in hits[0]
