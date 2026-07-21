"""Unit tests for MarketAnalystAgent intent routing, context assembly, and grounding."""

import pytest

from llm.agent import MarketAnalystAgent
from services.market_service import MarketService


def test_market_analyst_agent_intent_detection():
    """Test MarketAnalystAgent intent routing logic."""
    agent = MarketAnalystAgent()

    assert agent._detect_intent("Why is AAPL predicted to rise?") == "prediction_explanation"
    assert agent._detect_intent("What are the main risks for TSLA?") == "risk_analysis"
    assert agent._detect_intent("What is RSI indicator?") == "rag_qa"
    assert agent._detect_intent("Give me an overall analysis of MSFT") == "market_analysis"


def test_market_analyst_agent_grounded_analysis():
    """Test MarketAnalystAgent generates grounded response with citations."""
    service = MarketService()
    agent = MarketAnalystAgent()

    response = agent.analyze(
        user_query="Why is AAPL forecast positive for the next 5 trading days?",
        market_service_ref=service,
        symbol="AAPL",
        period="1mo",
    )

    assert response is not None
    assert "AAPL" in response.summary or "AAPL" in response.market_context_overview
    assert len(response.sources) > 0
    assert response.confidence_level in ["HIGH", "MEDIUM", "LOW"]
