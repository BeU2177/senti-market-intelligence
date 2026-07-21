# Microservice Deployment Guide — Senti Market Intelligence

## Docker & Docker Compose Setup

The platform is containerized using multi-service Docker configuration (`docker-compose.yml`):

- **`market-intelligence-api`**: FastAPI production application running on port `8000`.
- **`streamlit-dashboard`**: Streamlit interactive user interface running on port `8501`.

## Local Docker Deployment Commands

```bash
# 1. Build and start services in background
docker-compose up --build -d

# 2. Check container status
docker-compose ps

# 3. View application logs
docker-compose logs -f

# 4. Stop microservices
docker-compose down
```

## Environment Variables

Configured via `.env` (template provided in `.env.example`):
- `DEFAULT_SYMBOL`: Default ticker (e.g. `AAPL`).
- `MARKET_DATA_PROVIDER`: Provider name (`yfinance`).
- `NEWS_API_KEY`: NewsAPI key for financial news ingestion.
- `OLLAMA_BASE_URL`: Local Ollama REST URL (`http://localhost:11434`).
- `MAX_ALLOWED_FRESHNESS_SEC`: Max data age in seconds (`300`).
- `LOG_LEVEL`: Logging level (`INFO`).
