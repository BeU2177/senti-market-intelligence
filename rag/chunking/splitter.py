"""Text splitter module with configurable chunk size, overlap, and provenance metadata."""

from typing import List
from pydantic import BaseModel
from rag.documents.loader import KnowledgeDocument
from utils.logging_config import get_logger

logger = get_logger(__name__)


class DocumentChunk(BaseModel):
    """Text chunk with parent document provenance metadata."""
    chunk_id: str
    document_id: str
    title: str
    source: str
    category: str
    chunk_index: int
    content: str


class TextSplitter:
    """Splits documents into overlapping text chunks while preserving provenance metadata."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[KnowledgeDocument]) -> List[DocumentChunk]:
        """Splits a list of KnowledgeDocuments into DocumentChunks."""
        chunks: List[DocumentChunk] = []

        for doc in documents:
            doc_chunks = self._split_single_document(doc)
            chunks.extend(doc_chunks)

        logger.info(f"TextSplitter generated {len(chunks)} chunks from {len(documents)} documents (size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks

    def _split_single_document(self, doc: KnowledgeDocument) -> List[DocumentChunk]:
        """Splits single document text by paragraphs/lines while keeping chunk size and overlap."""
        text = doc.content
        lines = text.split("\n")

        chunks: List[DocumentChunk] = []
        current_chunk = []
        current_len = 0
        chunk_idx = 0

        for line in lines:
            line_len = len(line)
            if current_len + line_len > self.chunk_size and current_chunk:
                chunk_text = "\n".join(current_chunk).strip()
                if chunk_text:
                    c_id = f"{doc.document_id}_c{chunk_idx}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=c_id,
                            document_id=doc.document_id,
                            title=doc.title,
                            source=doc.source,
                            category=doc.category,
                            chunk_index=chunk_idx,
                            content=chunk_text,
                        )
                    )
                    chunk_idx += 1

                # Retain overlap lines
                overlap_lines = []
                overlap_len = 0
                for prev_line in reversed(current_chunk):
                    if overlap_len + len(prev_line) <= self.chunk_overlap:
                        overlap_lines.insert(0, prev_line)
                        overlap_len += len(prev_line)
                    else:
                        break

                current_chunk = overlap_lines
                current_len = overlap_len

            current_chunk.append(line)
            current_len += line_len

        if current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                c_id = f"{doc.document_id}_c{chunk_idx}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=c_id,
                        document_id=doc.document_id,
                        title=doc.title,
                        source=doc.source,
                        category=doc.category,
                        chunk_index=chunk_idx,
                        content=chunk_text,
                    )
                )

        return chunks
