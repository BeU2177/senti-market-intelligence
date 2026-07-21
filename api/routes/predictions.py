"""FastAPI multi-horizon prediction endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from api.schemas import PredictionDTO
from services.inference_service import InferenceService
from services.mlops_service import MLOpsService
from api.dependencies import get_inference_service, get_mlops_service

router = APIRouter(prefix="/prediction", tags=["Predictions"])


@router.get("/{symbol}", response_model=PredictionDTO)
def get_prediction(
    symbol: str,
    sequence_length: int = 30,
    inference: InferenceService = Depends(get_inference_service),
    mlops: MLOpsService = Depends(get_mlops_service),
) -> PredictionDTO:
    """Generate real-time/near-real-time multi-horizon forecasts, ensemble weights, and log prediction."""
    try:
        payload = inference.predict_multi_horizon(symbol=symbol, sequence_length=sequence_length)
        
        # Log prediction to MLOps audit file
        mlops.log_prediction(payload)

        ca = payload["confidence_assessment"]

        return PredictionDTO(
            prediction_id=payload["prediction_id"],
            symbol=payload["symbol"],
            prediction_timestamp=payload["prediction_timestamp"],
            data_as_of=payload["data_as_of"],
            current_price=payload["current_price"],
            predicted_returns=payload["predicted_returns"],
            predicted_prices=payload["predicted_prices"],
            directional_signals=payload["directional_signals"],
            ensemble_weights=payload["ensemble_weights"],
            confidence_level=ca["level"],
            agreement_score=ca["agreement_score"],
            model_metadata=payload["model_metadata"],
            risk_disclaimer=payload["risk_disclaimer"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error for {symbol}: {str(e)}")
