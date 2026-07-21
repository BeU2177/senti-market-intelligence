"""Composable Feature Pipeline orchestrating feature generation, target building, and leakage validation."""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from features.feature_metadata import FEATURE_REGISTRY, FeatureMetadata, FeatureCategory
from features.price_features import compute_price_features
from features.trend_features import compute_trend_features
from features.momentum_features import compute_momentum_features
from features.volatility_features import compute_volatility_features
from features.volume_features import compute_volume_features
from features.regime_features import compute_regime_features
from features.target_builder import TargetBuilder
from features.leakage_validator import LeakageValidator, LeakageValidationReport
from news.schemas.news_article import NewsArticle
from news.sentiment.sentiment_pipeline import SentimentPipeline, SENTIMENT_FEATURE_VERSION
from utils.logging_config import get_logger

logger = get_logger(__name__)

FEATURE_VERSION = "v1.0.0"


class FeatureDataset(BaseModel):
    """Container for processed feature dataset and execution metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbol: str
    feature_version: str = FEATURE_VERSION
    sentiment_version: Optional[str] = None
    row_count: int
    data_frame: Any = Field(default=None, description="Pandas DataFrame containing market data, technical features, sentiment features, and targets")
    feature_columns: List[str] = Field(default_factory=list)
    sentiment_columns: List[str] = Field(default_factory=list)
    target_columns: List[str] = Field(default_factory=list)
    metadata: Dict[str, FeatureMetadata] = Field(default_factory=dict)
    leakage_report: Optional[LeakageValidationReport] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FeaturePipeline:
    """Pipeline generating leakage-free technical features, NLP sentiment features, and target labels."""

    def __init__(self, target_horizons: List[int] = None, trim_warmup: bool = False):
        self.target_builder = TargetBuilder(horizons=target_horizons or [1, 3, 5, 7])
        self.leakage_validator = LeakageValidator()
        self.sentiment_pipeline = SentimentPipeline()
        self.trim_warmup = trim_warmup

    def build_features(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        articles: Optional[List[NewsArticle]] = None,
        include_sentiment: bool = True,
    ) -> FeatureDataset:
        """Executes full technical feature engineering, sentiment alignment, target building, and leakage validation."""
        clean_symbol = symbol.strip().upper()
        logger.info(f"FeaturePipeline: Building features for {clean_symbol} (include_sentiment={include_sentiment})")

        if df is None or df.empty:
            logger.warning(f"FeaturePipeline: Empty DataFrame passed for {clean_symbol}")
            return FeatureDataset(
                symbol=clean_symbol,
                row_count=0,
                data_frame=pd.DataFrame(),
                leakage_report=LeakageValidationReport(is_clean=False, leakage_errors=["Empty DataFrame"]),
            )

        # Ensure sorted by timestamp
        df_base = df.sort_values("timestamp").reset_index(drop=True).copy()

        # 1. Generate Technical Feature Subsets
        price_df = compute_price_features(df_base)
        trend_df = compute_trend_features(df_base)
        mom_df = compute_momentum_features(df_base)
        vol_df = compute_volatility_features(df_base)
        volu_df = compute_volume_features(df_base)
        regime_df = compute_regime_features(df_base, trend_df, vol_df)

        tech_features = [df_base, price_df, trend_df, mom_df, vol_df, volu_df, regime_df]
        
        # 2. Generate Sentiment Features (if requested and articles provided)
        sentiment_df = pd.DataFrame()
        sentiment_cols = []
        sent_ver = None

        if include_sentiment and articles is not None and len(articles) > 0:
            sent_df_raw = self.sentiment_pipeline.generate_sentiment_features(df_base, articles, symbol=clean_symbol)
            if not sent_df_raw.empty:
                sent_cols_to_drop = [c for c in ["timestamp", "symbol"] if c in sent_df_raw.columns]
                sentiment_df = sent_df_raw.drop(columns=sent_cols_to_drop)
                sentiment_cols = list(sentiment_df.columns)
                tech_features.append(sentiment_df)
                sent_ver = SENTIMENT_FEATURE_VERSION

        # Combine all features
        features_combined = pd.concat(tech_features, axis=1)

        # 3. Build Targets
        targets_df = self.target_builder.build_targets(df_base)
        full_df = pd.concat([features_combined, targets_df], axis=1)

        # Identify Feature & Target Columns
        raw_cols = set(df_base.columns)
        target_cols = list(targets_df.columns)
        feature_cols = [c for c in full_df.columns if c not in raw_cols and c not in target_cols]

        # 4. Clean Infinite Values
        num_cols = [c for c in feature_cols if c in full_df.columns and pd.api.types.is_numeric_dtype(full_df[c])]
        full_df[num_cols] = full_df[num_cols].replace([np.inf, -np.inf], np.nan)

        # 5. Optional Warm-up Row Trimming
        if self.trim_warmup and len(full_df) > 50:
            full_df = full_df.iloc[20:].reset_index(drop=True)

        # 6. Execute Leakage Validation
        leakage_report = self.leakage_validator.validate_dataset(
            df=full_df,
            feature_cols=feature_cols,
            target_cols=target_cols,
        )

        # 7. Extract Metadata Registry subset
        feature_meta = {col: FEATURE_REGISTRY.get(col, FeatureMetadata(
            feature_name=col,
            category=FeatureCategory.MOMENTUM if col in sentiment_cols else FeatureCategory.PRICE,
            description="Engineered feature"
        )) for col in feature_cols + target_cols}

        logger.info(
            f"FeaturePipeline completed for {clean_symbol}: total_cols={len(full_df.columns)}, "
            f"features={len(feature_cols)}, sentiment_features={len(sentiment_cols)}, "
            f"targets={len(target_cols)}, leakage_clean={leakage_report.is_clean}"
        )

        return FeatureDataset(
            symbol=clean_symbol,
            feature_version=FEATURE_VERSION,
            sentiment_version=sent_ver,
            row_count=len(full_df),
            data_frame=full_df,
            feature_columns=feature_cols,
            sentiment_columns=sentiment_cols,
            target_columns=target_cols,
            metadata=feature_meta,
            leakage_report=leakage_report,
        )
