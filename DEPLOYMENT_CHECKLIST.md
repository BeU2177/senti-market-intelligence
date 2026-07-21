# Senti Market Intelligence — Production Deployment Checklist

This document details the step-by-step procedure for deploying **Senti Market Intelligence** into production environments.

---

## 1. Environment & Prerequisites

- [ ] **Python Version**: Ensure Python 3.13+ is installed (`python --version`).
- [ ] **Docker & Docker Compose**: Ensure Docker Engine 24+ and Docker Compose v2+ are installed (`docker --version`, `docker compose version`).
- [ ] **Hardware Allocation**: Recommended minimum 4 CPU cores, 8 GB RAM, 10 GB disk storage.

---

## 2. Secrets & Configuration Setup

- [ ] Copy template environment configuration:
  ```bash
  cp .env.example .env
  ```
- [ ] Set production variables in `.env`:
  - `NEWS_API_KEY`: Set your valid NewsAPI.org API key.
  - `LOG_LEVEL`: Set to `INFO` or `WARNING`.
  - `MAX_ALLOWED_FRESHNESS_SEC`: Set to `300` (5 minutes).
  - `OLLAMA_BASE_URL`: Set to `http://localhost:11434` (or remote LLM endpoint).

---

## 3. Storage & Vector Database Initialization

- [ ] Create persistent storage directories:
  ```bash
  mkdir -p data/raw/market data/processed/market data/processed/news data/metadata artifacts/models artifacts/experiments artifacts/predictions artifacts/vector_db
  ```
- [ ] Initialize financial knowledge base embeddings:
  ```bash
  python -c "from rag.pipeline import RAGPipeline; RAGPipeline().ingest_knowledge_base()"
  ```

---

## 4. Model Artifact Setup & Verification

- [ ] Verify trained PyTorch model checkpoints and preprocessor scalers exist under `artifacts/models/<symbol>/`.
- [ ] Run feature schema validation check:
  ```bash
  python -c "from services.feature_update_service import FeatureUpdateService; print(FeatureUpdateService())"
  ```

---

## 5. Deployment Execution (Docker Compose)

- [ ] Build and launch microservices:
  ```bash
  docker-compose up --build -d
  ```
- [ ] Verify container status:
  ```bash
  docker-compose ps
  ```
  Expected services:
  - `market-intelligence-api` (Port 8000)
  - `streamlit-dashboard` (Port 8501)

---

## 6. Health & Readiness Verification

- [ ] Probe API health endpoint:
  ```bash
  curl http://localhost:8000/health
  ```
  Expected output: `{"status":"ok","app_name":"Senti Market Intelligence API","version":"v1.0.0",...}`

- [ ] Probe readiness endpoint:
  ```bash
  curl http://localhost:8000/health/ready
  ```
  Expected output: `{"status":"ready","market_provider":"online","models_available":true,...}`

- [ ] Probe prediction endpoint for test symbol (`AAPL`):
  ```bash
  curl http://localhost:8000/prediction/AAPL
  ```

---

## 7. Monitoring & Audit Verification

- [ ] Check prediction logging: `artifacts/predictions/predictions.jsonl` receives structured JSON payloads.
- [ ] Check outcome tracking: `artifacts/predictions/outcomes.jsonl` tracks realized target return errors.
- [ ] Check model promotions: `artifacts/predictions/promotions.jsonl` logs `CANDIDATE` $\to$ `PRODUCTION` decisions.

---

## 8. Rollback Procedure

- [ ] In case of critical failure, stop services:
  ```bash
  docker-compose down
  ```
- [ ] Roll back to previous approved model checkpoint under `artifacts/models/<symbol>/`.
- [ ] Restart microservices:
  ```bash
  docker-compose up -d
  ```
