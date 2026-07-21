"""FastAPI dependency injection module."""

from functools import lru_cache
from services.market_service import MarketService
from services.inference_service import InferenceService
from services.mlops_service import MLOpsService
from rag.pipeline import RAGPipeline


@lru_cache(maxsize=1)
def get_market_service() -> MarketService:
    """Singleton instance of MarketService."""
    return MarketService()


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Singleton instance of InferenceService."""
    return InferenceService()


@lru_cache(maxsize=1)
def get_mlops_service() -> MLOpsService:
    """Singleton instance of MLOpsService."""
    return MLOpsService()


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    """Singleton instance of RAGPipeline."""
    return RAGPipeline()
