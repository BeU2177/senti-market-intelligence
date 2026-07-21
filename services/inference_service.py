"""Production inference service orchestrating model loading, ensemble prediction, and confidence scoring."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

from services.live_market_service import LiveMarketService
from services.news_update_service import NewsUpdateService
from services.feature_update_service import FeatureUpdateService
from deep_learning.ensemble import EnsembleForecaster
from data.storage.storage_manager import StorageManager
from utils.logging_config import get_logger

logger = get_logger(__name__)


class InferenceService:
    """Production prediction service generating multi-horizon ensemble forecasts."""

    def __init__(self, storage_manager: Optional[StorageManager] = None):
        self.storage = storage_manager or StorageManager()
        self.live_market = LiveMarketService()
        self.news_service = NewsUpdateService(storage_manager=self.storage)
        self.feature_service = FeatureUpdateService()
        self.ensemble_forecaster = EnsembleForecaster()

    def predict_multi_horizon(
        self,
        symbol: str,
        sequence_length: int = 30,
        include_sentiment: bool = True,
    ) -> Dict[str, Any]:
        """Generates real-time/near-real-time multi-horizon forecasts with confidence scores and freshness metrics."""
        clean_symbol = symbol.strip().upper()
        logger.info(f"InferenceService starting multi-horizon prediction pipeline for {clean_symbol}")

        # 1. Live Market Quote & Freshness Metrics
        market_res, freshness_metrics = self.live_market.fetch_live_quote(clean_symbol)
        df_market = market_res.data_frame

        if df_market is None or df_market.empty:
            raise ValueError(f"InferenceService failed: Empty market data returned for {clean_symbol}")

        current_price = float(df_market["close"].iloc[-1])

        # 2. News Articles & Sentiment Updates
        articles = []
        if include_sentiment:
            try:
                articles = self.news_service.update_news_and_sentiment(clean_symbol)
            except Exception as e:
                logger.warning(f"InferenceService news update failed for {clean_symbol}: {str(e)}")

        # 3. Fresh Feature Generation & Validation
        feat_dataset, val_status = self.feature_service.generate_and_validate_features(
            market_df=df_market,
            symbol=clean_symbol,
            articles=articles,
            include_sentiment=include_sentiment,
        )

        if not val_status["is_valid"]:
            raise RuntimeError(f"Feature schema validation failed for {clean_symbol}: {val_status['issues']}")

        # 4. Multi-Horizon Predictions via MarketService Pipeline
        from services.market_service import MarketService
        m_service = MarketService()

        benchmark_res = m_service.run_full_ensemble_benchmark(
            symbol=clean_symbol,
            sequence_length=sequence_length,
            period="1y",
            interval="1d",
        )

        ep = benchmark_res["ensemble_prediction"]
        ca = benchmark_res["confidence_assessment"]

        prediction_payload = {
            "prediction_id": f"pred_{clean_symbol}_{int(datetime.now(timezone.utc).timestamp())}",
            "symbol": clean_symbol,
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
            "data_as_of": freshness_metrics["data_timestamp"],
            "current_price": current_price,
            "data_freshness": freshness_metrics,
            "market_regime": str(df_market["market_regime"].iloc[-1]) if "market_regime" in df_market.columns else "BULLISH",
            "predicted_returns": ep.predicted_returns,
            "predicted_prices": ep.predicted_prices,
            "directional_signals": ep.directional_signals,
            "ensemble_weights": ep.member_weights,
            "confidence_assessment": {
                "level": ca.confidence_level,
                "agreement_score": ca.ensemble_agreement_score,
                "std_dev": ca.prediction_std_dev,
                "rationale": ca.rationale,
            },
            "model_metadata": {
                "classical_best_model": benchmark_res["classical_ml_best"],
                "feature_version": feat_dataset.feature_version,
                "sentiment_version": feat_dataset.sentiment_version,
                "sequence_length": sequence_length,
            },
            "risk_disclaimer": "Analytical prediction platform only. Not financial advice.",
        }

        logger.info(
            f"InferenceService completed for {clean_symbol}: 1d Price=${ep.predicted_prices['1d']:.2f}, "
            f"Confidence={ca.confidence_level}"
        )
        return prediction_payload
