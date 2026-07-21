"""FastAPI market data endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from api.schemas import MarketDataDTO
from services.market_service import MarketService
from services.live_market_service import LiveMarketService
from api.dependencies import get_market_service

router = APIRouter(prefix="/market", tags=["Market Data"])


@router.get("/{symbol}", response_model=MarketDataDTO)
def get_market_data(
    symbol: str,
    service: MarketService = Depends(get_market_service),
) -> MarketDataDTO:
    """Fetch latest market quote, freshness status, and latency metrics."""
    try:
        live_service = LiveMarketService()
        m_res, metrics = live_service.fetch_live_quote(symbol)

        df = m_res.data_frame
        latest_row = df.iloc[-1].to_dict() if df is not None and not df.empty else {}

        return MarketDataDTO(
            symbol=m_res.symbol,
            price=float(latest_row.get("close", 0.0)),
            high=float(latest_row.get("high", 0.0)),
            low=float(latest_row.get("low", 0.0)),
            open=float(latest_row.get("open", 0.0)),
            volume=float(latest_row.get("volume", 0.0)),
            market_regime=str(latest_row.get("market_regime", "BULLISH")),
            data_freshness_status=metrics["freshness_status"],
            data_timestamp=metrics["data_timestamp"],
            latency_ms=metrics["latency_ms"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market data error for {symbol}: {str(e)}")
