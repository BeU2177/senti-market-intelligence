"""Evaluation metrics calculator for regression and directional financial signals."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.schemas import RegressionMetrics, FinancialMetrics


class ModelEvaluator:
    """Calculates statistical regression metrics and financial directional metrics."""

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
        """Calculate MAE, RMSE, MSE, R^2."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 and np.var(y_true) > 0 else 0.0

        return RegressionMetrics(
            mae=round(mae, 6),
            rmse=round(rmse, 6),
            mse=round(mse, 6),
            r2=round(r2, 6),
        )

    def evaluate_financial(self, y_true: np.ndarray, y_pred: np.ndarray) -> FinancialMetrics:
        """Calculate Directional Accuracy, Positive/Negative Precision, Mean Returns, and Correlation (IC)."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Directional Sign Match (True when signs agree)
        same_sign = (y_true > 0) == (y_pred > 0)
        dir_acc = float(np.mean(same_sign)) if len(same_sign) > 0 else 0.0

        # Positive Return Precision (when y_pred > 0, how often is y_true > 0?)
        pred_pos_mask = y_pred > 0
        if pred_pos_mask.sum() > 0:
            pos_prec = float(np.mean(y_true[pred_pos_mask] > 0))
        else:
            pos_prec = 0.0

        # Negative Return Precision (when y_pred <= 0, how often is y_true <= 0?)
        pred_neg_mask = y_pred <= 0
        if pred_neg_mask.sum() > 0:
            neg_prec = float(np.mean(y_true[pred_neg_mask] <= 0))
        else:
            neg_prec = 0.0

        mean_pred = float(np.mean(y_pred))
        mean_actual = float(np.mean(y_true))

        # Prediction Correlation (Information Coefficient)
        if np.std(y_pred) > 1e-8 and np.std(y_true) > 1e-8:
            ic = float(np.corrcoef(y_pred, y_true)[0, 1])
        else:
            ic = 0.0

        return FinancialMetrics(
            directional_accuracy=round(dir_acc, 4),
            positive_precision=round(pos_prec, 4),
            negative_precision=round(neg_prec, 4),
            mean_predicted_return=round(mean_pred, 6),
            mean_actual_return=round(mean_actual, 6),
            prediction_correlation=round(ic, 4),
        )
