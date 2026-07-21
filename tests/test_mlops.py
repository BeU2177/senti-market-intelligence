"""Unit tests for MLOps prediction logging, outcome tracking, feature drift detection, and model promotion."""

import pytest
import pandas as pd
from pathlib import Path

from services.mlops_service import MLOpsService


def test_mlops_prediction_logging_and_outcome_tracking(tmp_path):
    """Test MLOpsService logging prediction and recording realized outcome."""
    mlops = MLOpsService(base_dir=tmp_path / "test_predictions")

    pred_payload = {
        "prediction_id": "pred_test_123",
        "symbol": "AAPL",
        "predicted_returns": {"1d": 0.015},
    }

    pred_id = mlops.log_prediction(pred_payload)
    assert pred_id == "pred_test_123"
    assert (tmp_path / "test_predictions" / "predictions.jsonl").exists()

    outcome = mlops.track_realized_outcome(
        prediction_id="pred_test_123",
        symbol="AAPL",
        predicted_return_1d=0.015,
        actual_realized_return_1d=0.012,
    )

    assert outcome["is_direction_correct"] is True
    assert outcome["absolute_error_1d"] == round(0.003, 6)


def test_mlops_feature_drift_calculation(tmp_path):
    """Test MLOpsService feature drift detection between baseline and current dataframes."""
    mlops = MLOpsService(base_dir=tmp_path / "test_predictions")

    baseline_df = pd.DataFrame({"rsi_14": [50.0, 52.0, 48.0, 51.0], "macd": [0.1, 0.2, 0.1, 0.15]})
    current_df = pd.DataFrame({"rsi_14": [80.0, 85.0, 82.0, 84.0], "macd": [0.11, 0.19, 0.12, 0.14]})

    drift_report = mlops.calculate_feature_drift(baseline_df, current_df, ["rsi_14", "macd"])

    assert drift_report["rsi_14"]["is_drift_detected"] is True
    assert drift_report["macd"]["is_drift_detected"] is False


def test_mlops_model_promotion_lifecycle(tmp_path):
    """Test MLOpsService candidate model promotion decision."""
    mlops = MLOpsService(base_dir=tmp_path / "test_predictions")

    # Candidate has lower (better) RMSE -> PROMOTED_TO_PRODUCTION
    promo1 = mlops.promote_model_lifecycle(
        symbol="AAPL",
        model_name="PyTorchLSTM",
        candidate_val_rmse=0.012,
        production_val_rmse=0.018,
    )
    assert promo1["promotion_stage"] == "PRODUCTION"
    assert promo1["action"] == "PROMOTED_TO_PRODUCTION"

    # Candidate has higher (worse) RMSE -> REJECTED
    promo2 = mlops.promote_model_lifecycle(
        symbol="AAPL",
        model_name="RandomForest",
        candidate_val_rmse=0.025,
        production_val_rmse=0.018,
    )
    assert promo2["promotion_stage"] == "REJECTED"
