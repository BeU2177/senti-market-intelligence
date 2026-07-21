"""Unit tests for OllamaProvider and MarketAnalysisResponse schema."""

import pytest

from llm.ollama_provider import OllamaProvider
from llm.response_schema import MarketAnalysisResponse


def test_ollama_provider_fallback():
    """Test OllamaProvider generates structured fallback analysis when Ollama is offline."""
    provider = OllamaProvider(base_url="http://localhost:99999", timeout=1)  # Invalid URL
    assert not provider.is_available()

    ctx_str = '{"symbol": "AAPL", "current_price": 150.0, "market_regime": "BULLISH", "ensemble_prediction": {"predicted_returns": {"1d": 0.01}, "predicted_prices": {"1d": 151.5}}, "confidence": {"confidence_level": "HIGH"}}'
    rag_str = "[1] Title: RSI Indicator\nContent: RSI over 70 is overbought."

    resp = provider.generate_analysis(
        prompt="Context: {market_context}\nRAG: {rag_context}\nQuery: {user_query}",
        market_context_str=ctx_str,
        rag_context_str=rag_str,
        user_query="Why is AAPL positive?",
    )

    assert isinstance(resp, MarketAnalysisResponse)
    assert resp.confidence_level == "HIGH"
    assert "AAPL" in resp.summary
    assert len(resp.supporting_evidence) > 0
