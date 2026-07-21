# Production API Specification — Senti Market Intelligence

The platform exposes a production-ready FastAPI application (`api/main.py`) serving multi-horizon forecasts, market data, FinBERT sentiment, and grounded AI analysis.

## Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | API health check, application version, and server timestamp |
| `GET` | `/health/ready` | Readiness probe checking vector database and LLM endpoint availability |
| `GET` | `/market/{symbol}` | Current market bar, freshness status, and ingestion latency metrics |
| `GET` | `/prediction/{symbol}` | Multi-horizon forecasts ($1d, 3d, 5d, 7d$), ensemble weights, and confidence |
| `GET` | `/sentiment/{symbol}` | FinBERT sentiment probabilities, news article count, and headline feed |
| `POST` | `/analysis` | Grounded AI market analyst response matching query intent |

## Sample Requests & Responses

### 1. `GET /health`
```json
{
  "status": "ok",
  "app_name": "Senti Market Intelligence API",
  "version": "v1.0.0",
  "timestamp": "2026-07-21T20:13:46Z"
}
```

### 2. `GET /prediction/AAPL`
```json
{
  "prediction_id": "pred_AAPL_1784674400",
  "symbol": "AAPL",
  "prediction_timestamp": "2026-07-21T20:13:46Z",
  "data_as_of": "2026-07-20T04:00:00+00:00",
  "current_price": 327.95,
  "predicted_returns": {
    "1d": 0.002454,
    "3d": 0.006939,
    "5d": 0.01026,
    "7d": 0.014298
  },
  "predicted_prices": {
    "1d": 328.544,
    "3d": 330.014,
    "5d": 331.103,
    "7d": 332.427
  },
  "directional_signals": {
    "1d": "BULLISH",
    "3d": "BULLISH",
    "5d": "BULLISH",
    "7d": "BULLISH"
  },
  "ensemble_weights": {
    "PyTorch_LSTM": 0.3547,
    "PyTorch_Transformer": 0.2301,
    "PyTorch_TemporalCNN": 0.1695,
    "XGBoost": 0.1412,
    "PyTorch_MLP": 0.1045
  },
  "confidence_level": "HIGH",
  "agreement_score": 1.0,
  "risk_disclaimer": "Analytical prediction platform only. Not financial advice."
}
```

### 3. `POST /analysis`
**Request Body**:
```json
{
  "symbol": "AAPL",
  "question": "Why is the model forecast positive over the next 5 trading days?"
}
```
