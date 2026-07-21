"""Production MLOps service for prediction logging, realized outcome tracking, feature drift monitoring, and model promotion."""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from utils.logging_config import get_logger

logger = get_logger(__name__)

PREDICTIONS_DIR = Path("artifacts") / "predictions"
PREDICTIONS_LOG_FILE = PREDICTIONS_DIR / "predictions.jsonl"
OUTCOMES_LOG_FILE = PREDICTIONS_DIR / "outcomes.jsonl"
PROMOTIONS_LOG_FILE = PREDICTIONS_DIR / "promotions.jsonl"


class MLOpsService:
    """Manages prediction logging, realized outcome tracking, feature drift monitoring, and candidate model promotions."""

    def __init__(self, base_dir: Path = PREDICTIONS_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.pred_file = self.base_dir / "predictions.jsonl"
        self.outcomes_file = self.base_dir / "outcomes.jsonl"
        self.promotions_file = self.base_dir / "promotions.jsonl"

    def log_prediction(self, prediction_payload: Dict[str, Any]) -> str:
        """Log production prediction payload to JSON lines archive."""
        logger.info(f"MLOpsService logging prediction ID '{prediction_payload.get('prediction_id')}'")
        with open(self.pred_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(prediction_payload, default=str) + "\n")
        return prediction_payload.get("prediction_id", "")

    def track_realized_outcome(
        self,
        prediction_id: str,
        symbol: str,
        predicted_return_1d: float,
        actual_realized_return_1d: float,
    ) -> Dict[str, Any]:
        """Record realized outcome error and directional correctness when target horizon expires."""
        error_1d = abs(predicted_return_1d - actual_realized_return_1d)
        correct_direction = (predicted_return_1d > 0) == (actual_realized_return_1d > 0)

        outcome_payload = {
            "prediction_id": prediction_id,
            "symbol": symbol,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "predicted_return_1d": round(predicted_return_1d, 6),
            "actual_realized_return_1d": round(actual_realized_return_1d, 6),
            "absolute_error_1d": round(error_1d, 6),
            "is_direction_correct": bool(correct_direction),
        }

        with open(self.outcomes_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(outcome_payload, default=str) + "\n")

        logger.info(f"MLOpsService tracked outcome for '{prediction_id}': AE={error_1d:.6f}, DirCorrect={correct_direction}")
        return outcome_payload

    def calculate_feature_drift(
        self, baseline_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: List[str]
    ) -> Dict[str, Any]:
        """Calculate statistical mean and std dev drift between training baseline and production features."""
        drift_report = {}
        for col in feature_cols:
            if col in baseline_df.columns and col in current_df.columns:
                b_mean, b_std = float(baseline_df[col].mean()), float(baseline_df[col].std())
                c_mean, c_std = float(current_df[col].mean()), float(current_df[col].std())

                mean_diff = abs(c_mean - b_mean)
                drift_detected = mean_diff > (2.0 * b_std) if b_std > 0 else False

                drift_report[col] = {
                    "baseline_mean": round(b_mean, 4),
                    "current_mean": round(c_mean, 4),
                    "mean_drift_abs": round(mean_diff, 4),
                    "is_drift_detected": bool(drift_detected),
                }

        return drift_report

    def promote_model_lifecycle(
        self,
        symbol: str,
        model_name: str,
        candidate_val_rmse: float,
        production_val_rmse: float,
    ) -> Dict[str, Any]:
        """Manages candidate model promotion: CANDIDATE -> VALIDATED -> APPROVED -> PRODUCTION."""
        if candidate_val_rmse < production_val_rmse:
            stage = "PRODUCTION"
            action = "PROMOTED_TO_PRODUCTION"
        else:
            stage = "REJECTED"
            action = "RETAINED_CURRENT_PRODUCTION_MODEL"

        promotion_record = {
            "symbol": symbol,
            "model_name": model_name,
            "candidate_val_rmse": round(candidate_val_rmse, 6),
            "production_val_rmse": round(production_val_rmse, 6),
            "promotion_stage": stage,
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.promotions_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(promotion_record, default=str) + "\n")

        logger.info(f"MLOpsService model promotion decision for {symbol}/{model_name}: {stage} ({action})")
        return promotion_record
