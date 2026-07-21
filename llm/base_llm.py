"""Abstract interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from llm.response_schema import MarketAnalysisResponse


class BaseLLMProvider(ABC):
    """Abstract base class for local and remote LLM providers."""

    @abstractmethod
    def generate_analysis(
        self,
        prompt: str,
        market_context_str: str,
        rag_context_str: str,
        user_query: str,
    ) -> MarketAnalysisResponse:
        """Generate structured MarketAnalysisResponse from prompt and context."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the LLM endpoint is online and responsive."""
        pass
