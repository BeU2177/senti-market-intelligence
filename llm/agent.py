"""Grounded Market Analyst Agent coordinating RAG retrieval, structured context assembly, and LLM reasoning."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from llm.base_llm import BaseLLMProvider
from llm.ollama_provider import OllamaProvider
from llm.response_schema import MarketAnalysisResponse
from rag.pipeline import RAGPipeline
from utils.logging_config import get_logger

logger = get_logger(__name__)

PROMPT_DIR = Path("prompts")


class MarketAnalystAgent:
    """Deterministic intelligence agent routing user queries to RAG knowledge, Market Context, and LLM explanation layer."""

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
    ):
        self.llm = llm_provider or OllamaProvider()
        self.rag = rag_pipeline or RAGPipeline()

    def analyze(
        self,
        user_query: str,
        market_service_ref: Any,
        symbol: str = "AAPL",
        period: str = "1y",
        interval: str = "1d",
    ) -> MarketAnalysisResponse:
        """Executes grounded analysis workflow matching user intent."""
        clean_symbol = symbol.strip().upper()
        logger.info(f"MarketAnalystAgent processing query for {clean_symbol}: '{user_query}'")

        # 1. Detect Intent
        intent = self._detect_intent(user_query)

        # 2. Assemble Structured Market Context
        market_context_dict = self._build_market_context(
            market_service=market_service_ref, symbol=clean_symbol, period=period, interval=interval
        )

        # 3. Retrieve RAG Knowledge Context
        rag_hits = self.rag.retrieve_context(query=user_query, top_k=4)
        rag_context_str = self._format_rag_hits(rag_hits)

        # 4. Select Versioned Prompt Template
        prompt_template = self._load_prompt_template(intent)

        # 5. Execute LLM Reasoning / Fallback
        response = self.llm.generate_analysis(
            prompt=prompt_template,
            market_context_str=json.dumps(market_context_dict, default=str, indent=2),
            rag_context_str=rag_context_str,
            user_query=user_query,
        )

        # Append retrieved RAG citations if missing
        rag_sources = [hit["title"] for hit in rag_hits]
        combined_sources = list(set(response.sources + rag_sources))
        response.sources = combined_sources

        logger.info(f"MarketAnalystAgent generated grounded response for {clean_symbol} (Confidence={response.confidence_level})")
        return response

    def _detect_intent(self, query: str) -> str:
        """Detect query intent: market_analysis, prediction_explanation, rag_qa, risk_analysis."""
        q = query.lower()
        if "predict" in q or "forecast" in q or "why" in q or "model" in q:
            return "prediction_explanation"
        elif "risk" in q or "downside" in q or "loss" in q or "threat" in q:
            return "risk_analysis"
        elif "what is" in q or "how does" in q or "explain" in q or "rsi" in q or "macd" in q:
            return "rag_qa"
        else:
            return "market_analysis"

    def _build_market_context(
        self, market_service: Any, symbol: str, period: str, interval: str
    ) -> Dict[str, Any]:
        """Fetch live ticker data, indicators, regime, news, FinBERT sentiment, classical ML, and PyTorch predictions."""
        try:
            m_res = market_service.get_market_data(symbol=symbol, period=period, interval=interval)
            df = m_res.data_frame

            current_price = float(df["close"].iloc[-1]) if df is not None and not df.empty else 0.0
            latest_row = df.iloc[-1].to_dict() if df is not None and not df.empty else {}

            articles = market_service.get_news_articles(symbol=symbol)
            news_summary = [
                {"title": a.title, "source": a.source, "published_at": a.published_at.isoformat()}
                for a in articles[:5]
            ]

            # Run Full Ensemble Benchmark to get classical ML + PyTorch predictions
            ens_res = market_service.run_full_ensemble_benchmark(
                symbol=symbol, sequence_length=30, period=period, interval=interval
            )
            ep = ens_res["ensemble_prediction"]
            ca = ens_res["confidence_assessment"]

            return {
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "market_regime": latest_row.get("market_regime", "BULLISH"),
                "rsi_14": round(float(latest_row.get("rsi_14", 50.0)), 2),
                "macd": round(float(latest_row.get("macd", 0.0)), 4),
                "volatility_20d": round(float(latest_row.get("volatility_20d", 0.015)), 4),
                "recent_news": news_summary,
                "ensemble_prediction": {
                    "predicted_returns": ep.predicted_returns,
                    "predicted_prices": ep.predicted_prices,
                    "member_weights": ep.member_weights,
                    "directional_signals": ep.directional_signals,
                },
                "confidence": {
                    "confidence_level": ca.confidence_level,
                    "ensemble_agreement_score": ca.ensemble_agreement_score,
                    "prediction_std_dev": ca.prediction_std_dev,
                },
            }
        except Exception as e:
            logger.error(f"Error assembling MarketContext for {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "current_price": 0.0,
                "market_regime": "UNKNOWN",
                "status": f"Error loading context: {str(e)}",
            }

    def _format_rag_hits(self, hits: list) -> str:
        """Format retrieved vector search hits into string for prompt context."""
        if not hits:
            return "No specific RAG knowledge base hits found."

        formatted = []
        for idx, h in enumerate(hits, 1):
            formatted.append(f"[{idx}] Title: {h['title']} (Source: {h['source']}, Score: {h['similarity_score']:.2f})\nContent: {h['content']}")
        return "\n\n".join(formatted)

    def _load_prompt_template(self, intent: str) -> str:
        """Load versioned prompt text file from prompts/ directory."""
        file_map = {
            "prediction_explanation": PROMPT_DIR / "prediction_explanation.txt",
            "risk_analysis": PROMPT_DIR / "risk_analysis.txt",
            "rag_qa": PROMPT_DIR / "rag_qa.txt",
            "market_analysis": PROMPT_DIR / "market_analysis.txt",
        }
        path = file_map.get(intent, PROMPT_DIR / "market_analysis.txt")

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()

        # Fallback inline template
        return "Market Context: {market_context}\nRAG Context: {rag_context}\nUser Query: {user_query}"
