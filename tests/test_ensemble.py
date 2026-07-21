"""Unit tests for EnsembleForecaster performance weighting and calibrated confidence scoring."""

import pytest
import numpy as np

from deep_learning.ensemble import EnsembleForecaster


def test_ensemble_forecaster_weighting_and_confidence():
    """Test inverse RMSE model weighting, sign agreement, and non-fabricated confidence assessment."""
    forecaster = EnsembleForecaster()

    preds = {
        "Classical_Ridge": {"1d": 0.015, "3d": 0.035, "5d": 0.050, "7d": 0.070},
        "PyTorch_LSTM": {"1d": 0.012, "3d": 0.030, "5d": 0.045, "7d": 0.065},
        "PyTorch_TemporalCNN": {"1d": 0.018, "3d": 0.038, "5d": 0.055, "7d": 0.075},
    }
    val_rmse = {
        "Classical_Ridge": 0.010,
        "PyTorch_LSTM": 0.008,  # Lowest RMSE -> Highest weight
        "PyTorch_TemporalCNN": 0.012,
    }

    ens_pred, conf = forecaster.combine_predictions(
        symbol="AAPL",
        current_price=150.0,
        member_predictions=preds,
        member_val_rmse=val_rmse,
        horizons=[1, 3, 5, 7],
    )

    # PyTorch_LSTM should have highest performance weight
    weights = ens_pred.member_weights
    assert weights["PyTorch_LSTM"] > weights["Classical_Ridge"]
    assert weights["PyTorch_LSTM"] > weights["PyTorch_TemporalCNN"]

    # All 3 models predict positive return -> 100% agreement -> HIGH confidence
    assert np.isclose(conf.ensemble_agreement_score, 1.0)
    assert conf.confidence_level == "HIGH"
    assert ens_pred.predicted_prices["1d"] > 150.0
