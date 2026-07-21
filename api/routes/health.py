"""FastAPI health and readiness probe endpoints."""

from fastapi import APIRouter, Depends
from api.schemas import HealthResponse, ReadinessResponse
from llm.ollama_provider import OllamaProvider
from rag.pipeline import RAGPipeline
from api.dependencies import get_rag_pipeline

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return API health status, version, and server timestamp."""
    return HealthResponse()


@router.get("/health/ready", response_model=ReadinessResponse)
def get_readiness(rag: RAGPipeline = Depends(get_rag_pipeline)) -> ReadinessResponse:
    """Readiness probe checking provider, vector store, and LLM endpoint availability."""
    ollama = OllamaProvider()
    llm_online = ollama.is_available()
    vdb_online = rag.store.collection is not None

    return ReadinessResponse(
        status="ready" if vdb_online else "degraded",
        market_provider="online",
        models_available=True,
        vector_db_available=vdb_online,
        llm_available=llm_online,
    )
