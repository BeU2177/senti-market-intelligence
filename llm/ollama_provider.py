"""Local Ollama LLM provider implementation with graceful offline fallback."""

import json
import requests
from typing import Optional, Dict, Any
from llm.base_llm import BaseLLMProvider
from llm.response_schema import MarketAnalysisResponse
from utils.logging_config import get_logger

logger = get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Integrates local Ollama models (e.g. llama3.2, mistral, qwen2.5) via HTTP REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.2",
        temperature: float = 0.2,
        timeout: int = 15,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if local Ollama server is running and responsive."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def generate_analysis(
        self,
        prompt: str,
        market_context_str: str,
        rag_context_str: str,
        user_query: str,
    ) -> MarketAnalysisResponse:
        """Query local Ollama model if available, else fallback to deterministic rule-based analysis."""
        formatted_prompt = prompt.format(
            market_context=market_context_str,
            rag_context=rag_context_str,
            user_query=user_query,
        )

        if self.is_available():
            try:
                return self._call_ollama(formatted_prompt)
            except Exception as e:
                logger.warning(f"Ollama API query failed ({str(e)}). Falling back to rule-based analysis.")

        logger.info("Ollama unavailable or offline. Executing grounded rule-based analytical fallback.")
        return self._generate_fallback(market_context_str, rag_context_str, user_query)

    def _call_ollama(self, prompt: str) -> MarketAnalysisResponse:
        """Execute HTTP POST request to Ollama /api/generate."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        if resp.status_code == 200:
            data = resp.json()
            raw_response = data.get("response", "{}")
            
            # Parse JSON into MarketAnalysisResponse Pydantic model
            parsed = json.loads(raw_response)
            return MarketAnalysisResponse(**parsed)
        else:
            raise RuntimeError(f"Ollama returned HTTP Status {resp.status_code}: {resp.text}")

    def _generate_fallback(
        self, market_context_str: str, rag_context_str: str, user_query: str
    ) -> MarketAnalysisResponse:
        """Grounded rule-based fallback analytical generator when local LLM is offline."""
        try:
            ctx = json.loads(market_context_str)
        except Exception:
            ctx = {}

        symbol = ctx.get("symbol", "TARGET")
        price = ctx.get("current_price", 0.0)
        regime = ctx.get("market_regime", "UNKNOWN")
        ens_pred = ctx.get("ensemble_prediction", {})
        conf = ctx.get("confidence", {})
        news = ctx.get("recent_news", [])

        ret_1d = ens_pred.get("predicted_returns", {}).get("1d", 0.0)
        price_1d = ens_pred.get("predicted_prices", {}).get("1d", price)

        direction_str = "BULLISH" if ret_1d > 0 else ("BEARISH" if ret_1d < 0 else "NEUTRAL")

        summary = (
            f"Grounded analysis for {symbol} (Current Price: ${price:.2f}). The ensemble models "
            f"predict a {direction_str} 1-day return of {ret_1d*100:+.2f}% (Target Price: ${price_1d:.2f})."
        )

        overview = f"Symbol {symbol} is currently classified under a {regime} market regime with price ${price:.2f}."
        exp = (
            f"The ensemble combines predictions across Classical ML models and PyTorch deep learning models (LSTM, MLP, Temporal CNN). "
            f"The 1-day target is predicted at ${price_1d:.2f} based on inverse validation RMSE weighting."
        )

        evidence = [
            f"Market Regime: {regime}",
            f"1-Day Predicted Return: {ret_1d*100:+.2f}%",
            f"Recent News Count: {len(news)} articles analyzed via FinBERT",
        ]

        risks = [
            "Market volatility or macroeconomic announcements may trigger price breakouts.",
            "Short-term directional signals carry inherent model variance.",
        ]

        sources = [n.get("source", "News") for n in news[:3]] + ["Knowledge Base"]

        return MarketAnalysisResponse(
            summary=summary,
            market_context_overview=overview,
            model_prediction_explanation=exp,
            supporting_evidence=evidence,
            risk_factors=risks,
            uncertainty_assessment=f"Confidence level rated as {conf.get('confidence_level', 'MEDIUM')} based on ensemble sign agreement.",
            confidence_level=conf.get("confidence_level", "MEDIUM"),
            sources=list(set(sources)),
        )
