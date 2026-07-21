"""Model ensembling and non-fabricated confidence scoring based on performance weights and agreement."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from deep_learning.schemas import EnsemblePrediction, ConfidenceAssessment
from utils.logging_config import get_logger

logger = get_logger(__name__)


class EnsembleForecaster:
    """Performance-weighted ensemble forecaster combining Classical ML and PyTorch Deep Learning predictions."""

    def combine_predictions(
        self,
        symbol: str,
        current_price: float,
        member_predictions: Dict[str, Dict[str, float]],  # {model_name: {1d: ret, 3d: ret, ...}}
        member_val_rmse: Dict[str, float],               # {model_name: val_rmse}
        horizons: List[int] = None,
    ) -> Tuple[EnsemblePrediction, ConfidenceAssessment]:
        """Combines member predictions using inverse validation error weights and assesses calibrated confidence."""
        horizons = horizons or [1, 3, 5, 7]
        model_names = list(member_predictions.keys())

        if not model_names:
            raise ValueError("Ensemble combination failed: member_predictions is empty.")

        # 1. Compute Performance Weights using Inverse Validation RMSE
        inv_rmse = {}
        for m in model_names:
            rmse = member_val_rmse.get(m, 1.0)
            inv_rmse[m] = 1.0 / (max(rmse, 1e-6))

        sum_inv = sum(inv_rmse.values())
        weights = {m: round(inv_rmse[m] / sum_inv, 4) for m in model_names}

        # 2. Weighted Blend of Predictions per Horizon
        weighted_returns: Dict[str, float] = {}
        predicted_prices: Dict[str, float] = {}
        directional_signals: Dict[str, str] = {}
        per_horizon_preds: Dict[str, List[float]] = {f"{h}d": [] for h in horizons}

        for h in horizons:
            h_key = f"{h}d"
            blend_ret = 0.0

            for m in model_names:
                ret = member_predictions[m].get(h_key, 0.0)
                blend_ret += ret * weights[m]
                per_horizon_preds[h_key].append(ret)

            weighted_returns[h_key] = round(blend_ret, 6)
            predicted_prices[h_key] = round(current_price * (1.0 + blend_ret), 2)

            if blend_ret > 0.002:
                directional_signals[h_key] = "BULLISH 🟢"
            elif blend_ret < -0.002:
                directional_signals[h_key] = "BEARISH 🔴"
            else:
                directional_signals[h_key] = "SIDEWAYS / NEUTRAL ⚪"

        # 3. Calculate Sign Agreement and Prediction Dispersion across 1d horizon
        h1_preds = per_horizon_preds.get("1d", [0.0])
        pos_count = sum(1 for r in h1_preds if r > 0)
        neg_count = sum(1 for r in h1_preds if r < 0)
        max_consensus = max(pos_count, neg_count)
        agreement_ratio = max_consensus / len(h1_preds) if h1_preds else 1.0
        dispersion_std = float(np.std(h1_preds)) if len(h1_preds) > 1 else 0.0

        # 4. Calibrated Confidence Scoring (Non-fabricated)
        if agreement_ratio >= 0.75 and dispersion_std < 0.015:
            conf_level = "HIGH"
            rationale = f"High ensemble consensus ({agreement_ratio*100:.0f}% sign agreement) and low prediction dispersion (std={dispersion_std:.4f})."
        elif agreement_ratio >= 0.60:
            conf_level = "MEDIUM"
            rationale = f"Moderate ensemble consensus ({agreement_ratio*100:.0f}% sign agreement) across member models."
        else:
            conf_level = "LOW"
            rationale = f"Low ensemble consensus ({agreement_ratio*100:.0f}% sign agreement) with conflicting directional signals."

        ensemble_pred = EnsemblePrediction(
            symbol=symbol,
            current_price=current_price,
            predicted_returns=weighted_returns,
            predicted_prices=predicted_prices,
            member_weights=weights,
            directional_signals=directional_signals,
        )

        confidence_eval = ConfidenceAssessment(
            confidence_level=conf_level,
            ensemble_agreement_score=round(agreement_ratio, 4),
            prediction_std_dev=round(dispersion_std, 6),
            validation_stability_score=round(1.0 - dispersion_std, 4),
            rationale=rationale,
        )

        logger.info(
            f"EnsembleForecaster for {symbol}: 1d Ret={weighted_returns.get('1d')}, 1d Price={predicted_prices.get('1d')}, "
            f"Confidence={conf_level} (Agreement={agreement_ratio*100:.0f}%)"
        )

        return ensemble_pred, confidence_eval
