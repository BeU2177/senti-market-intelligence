"""Unit tests for ModelTrainer ablation experiments."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from features.feature_pipeline import FeatureDataset
from models.train import ModelTrainer


@pytest.fixture
def sample_feature_dataset(tmp_path):
    """Generates a sample FeatureDataset for ablation testing."""
    dates = pd.date_range("2023-01-01", periods=60, tz="UTC")
    np.random.seed(42)

    df = pd.DataFrame({
        "timestamp": dates,
        "close": 100.0 + np.cumsum(np.random.normal(0, 1, 60)),
        "return_1d": np.random.normal(0, 0.02, 60),
        "rsi_14": np.random.uniform(30, 70, 60),
        "sentiment_score": np.random.uniform(-0.5, 0.5, 60),
        "future_return_1d": np.random.normal(0, 0.02, 60),
    })

    return FeatureDataset(
        symbol="TEST",
        row_count=60,
        data_frame=df,
        feature_columns=["return_1d", "rsi_14", "sentiment_score"],
        sentiment_columns=["sentiment_score"],
        target_columns=["future_return_1d"],
    )


def test_ablation_experiment_execution(sample_feature_dataset):
    """Test ModelTrainer.run_ablation_experiment compares Market-Only vs Market-Plus-Sentiment."""
    trainer = ModelTrainer(random_seed=42)

    ablation_res = trainer.run_ablation_experiment(
        feature_dataset=sample_feature_dataset,
        target_column="future_return_1d",
        model_names=["ridge", "random_forest"],
    )

    assert ablation_res["symbol"] == "TEST"
    assert "market_only" in ablation_res
    assert "market_plus_sentiment" in ablation_res

    mo_best = ablation_res["market_only"]["best_report"]
    mps_best = ablation_res["market_plus_sentiment"]["best_report"]

    assert mo_best.feature_set_type == "market_only"
    assert mps_best.feature_set_type == "market_plus_sentiment"
    assert mo_best.test_metrics is not None
    assert mps_best.test_metrics is not None
