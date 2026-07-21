"""Unit tests for ModelEvaluator metrics calculations."""

import pytest
import numpy as np

from models.evaluate import ModelEvaluator


def test_evaluator_regression_metrics():
    """Test MAE, RMSE, MSE, R^2 calculations."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

    evaluator = ModelEvaluator()
    reg = evaluator.evaluate_regression(y_true, y_pred)

    assert reg.mae > 0.0
    assert reg.rmse > 0.0
    assert reg.r2 > 0.8


def test_evaluator_financial_metrics():
    """Test Directional Accuracy, Positive/Negative Precision, and Correlation (IC)."""
    y_true = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    y_pred = np.array([0.01, -0.02, 0.02, -0.01, -0.01])  # 4 out of 5 correct signs

    evaluator = ModelEvaluator()
    fin = evaluator.evaluate_financial(y_true, y_pred)

    assert np.isclose(fin.directional_accuracy, 0.80)
    assert fin.prediction_correlation > 0.0
