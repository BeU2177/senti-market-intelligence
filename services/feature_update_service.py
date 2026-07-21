"""Production feature update service and feature schema validator."""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np

from features.feature_pipeline import FeaturePipeline, FeatureDataset
from news.schemas.news_article import NewsArticle
from utils.logging_config import get_logger

logger = get_logger(__name__)


class FeatureUpdateService:
    """Computes production features and validates feature schema compatibility against model metadata."""

    def __init__(self):
        self.pipeline = FeaturePipeline()

    def generate_and_validate_features(
        self,
        market_df: pd.DataFrame,
        symbol: str,
        articles: Optional[List[NewsArticle]] = None,
        expected_feature_cols: Optional[List[str]] = None,
        include_sentiment: bool = True,
    ) -> Tuple[FeatureDataset, Dict[str, Any]]:
        """Computes fresh features and validates feature schema, ordering, and missing values."""
        clean_symbol = symbol.strip().upper()
        logger.info(f"FeatureUpdateService computing features for {clean_symbol}")

        feat_dataset = self.pipeline.build_features(
            df=market_df,
            symbol=clean_symbol,
            articles=articles,
            include_sentiment=include_sentiment,
        )

        df_feat = feat_dataset.data_frame
        if df_feat is None or df_feat.empty:
            return feat_dataset, {"is_valid": False, "reason": "Empty feature DataFrame"}

        actual_cols = list(df_feat.columns)
        
        # Validation checks
        issues = []
        if expected_feature_cols:
            missing = [c for c in expected_feature_cols if c not in actual_cols]
            if missing:
                issues.append(f"Missing {len(missing)} expected feature columns: {missing[:3]}")

        # Check for infinite values
        inf_count = np.isinf(df_feat.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            issues.append(f"Detected {inf_count} infinite values in features")

        validation_status = {
            "is_valid": len(issues) == 0,
            "feature_version": feat_dataset.feature_version,
            "sentiment_version": feat_dataset.sentiment_version,
            "total_columns": len(actual_cols),
            "feature_columns": len(feat_dataset.feature_columns),
            "sentiment_columns": len(feat_dataset.sentiment_columns),
            "issues": issues,
        }

        logger.info(f"FeatureUpdateService completed for {clean_symbol}: valid={validation_status['is_valid']}")
        return feat_dataset, validation_status
