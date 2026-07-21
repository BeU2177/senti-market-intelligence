"""Structured Pydantic schemas for LLM analytical outputs."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class MarketAnalysisResponse(BaseModel):
    """Structured response object returned by MarketAnalystAgent."""
    summary: str = Field(description="Executive summary of market status and prediction analysis")
    market_context_overview: str = Field(description="Overview of price action, regime classification, and technical indicators")
    model_prediction_explanation: str = Field(description="Explanation of classical ML and PyTorch multi-horizon return predictions")
    supporting_evidence: List[str] = Field(default_factory=list, description="Key supporting evidence from indicators, news, or model consensus")
    risk_factors: List[str] = Field(default_factory=list, description="Primary downside/upside risks and market caveats")
    uncertainty_assessment: str = Field(description="Evaluation of model disagreement, volatility, and data freshness limitations")
    confidence_level: str = Field(default="MEDIUM", description="HIGH, MEDIUM, or LOW confidence score")
    sources: List[str] = Field(default_factory=list, description="Source citations from news articles or knowledge base documents")
