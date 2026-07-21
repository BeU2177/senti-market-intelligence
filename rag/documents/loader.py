"""Document loader for financial knowledge base files."""

import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from utils.logging_config import get_logger

logger = get_logger(__name__)


class KnowledgeDocument(BaseModel):
    """Container for loaded knowledge document text and provenance metadata."""
    document_id: str
    title: str
    content: str
    source: str
    category: str = "financial_education"
    file_path: str
    version: str = "v1.0.0"


class DocumentLoader:
    """Loads Markdown and text documents from knowledge base directory."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path("data") / "knowledge_base" / "financial_docs"

    def load_documents(self) -> List[KnowledgeDocument]:
        """Reads all markdown files from knowledge base folder."""
        if not self.base_dir.exists():
            logger.warning(f"Knowledge base directory '{self.base_dir}' does not exist.")
            return []

        docs: List[KnowledgeDocument] = []
        for file_path in self.base_dir.glob("*.md"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                if not content:
                    continue

                title = file_path.stem.replace("_", " ").title()
                doc_id = hashlib.sha256(f"{file_path.name}_{len(content)}".encode()).hexdigest()[:12]

                docs.append(
                    KnowledgeDocument(
                        document_id=doc_id,
                        title=title,
                        content=content,
                        source=file_path.name,
                        category="financial_education",
                        file_path=str(file_path.absolute()),
                    )
                )
            except Exception as e:
                logger.error(f"Failed to load document '{file_path}': {str(e)}")

        logger.info(f"DocumentLoader loaded {len(docs)} knowledge documents from '{self.base_dir}'")
        return docs
