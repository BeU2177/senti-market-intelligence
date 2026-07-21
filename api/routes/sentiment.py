"""FastAPI sentiment endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from api.schemas import SentimentDTO
from services.market_service import MarketService
from api.dependencies import get_market_service

router = APIRouter(prefix="/sentiment", tags=["Sentiment"])


@router.get("/{symbol}", response_model=SentimentDTO)
def get_sentiment(
    symbol: str,
    service: MarketService = Depends(get_market_service),
) -> SentimentDTO:
    """Fetch latest financial news articles and sentiment probabilities for symbol."""
    try:
        articles = service.get_news_articles(symbol=symbol)
        
        art_dicts = []
        for a in articles[:10]:
            art_dicts.append({
                "article_id": a.article_id,
                "title": a.title,
                "source": a.source,
                "published_at": a.published_at.isoformat(),
                "url": a.url,
                "confidence": a.entity_match_confidence,
            })

        return SentimentDTO(
            symbol=symbol.upper(),
            news_count=len(articles),
            latest_articles=art_dicts,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment error for {symbol}: {str(e)}")
