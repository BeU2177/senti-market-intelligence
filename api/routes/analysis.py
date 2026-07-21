"""FastAPI grounded AI market analysis endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from api.schemas import AnalysisRequestDTO
from services.market_service import MarketService
from llm.response_schema import MarketAnalysisResponse
from api.dependencies import get_market_service

router = APIRouter(prefix="/analysis", tags=["AI Intelligence"])


@router.post("", response_model=MarketAnalysisResponse)
def post_analysis(
    req: AnalysisRequestDTO,
    service: MarketService = Depends(get_market_service),
) -> MarketAnalysisResponse:
    """Execute grounded MarketAnalystAgent analytical pipeline returning structured JSON response."""
    try:
        response = service.analyze_market_with_agent(
            user_query=req.question,
            symbol=req.symbol,
            period=req.period,
            interval=req.interval,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error for {req.symbol}: {str(e)}")
