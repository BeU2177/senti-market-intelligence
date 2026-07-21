# RAG Knowledge System & Local LLM Integration

## Financial Knowledge Base

Stored under `data/knowledge_base/financial_docs/`:
- `technical_indicators.md`: RSI, MACD, Bollinger Bands, ATR calculations and interpretation rules.
- `time_series_validation.md`: Chronological splitting, purging, embargoing, and directional accuracy vs profitability.
- `market_microstructure.md`: Market regimes (Bullish, Bearish, Sideways, High Volatility) and news sentiment time decay.

## Local Vector Store & Retrieval

- **Embedding Model**: Local 384-dimensional `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Database**: `ChromaStore` persistent vector store at `artifacts/vector_db/market_knowledge.json`.
- **Search Method**: Dense vector cosine similarity search returning top-K hits with source citations.

## Grounded LLM Analyst Agent (`MarketAnalystAgent`)

- **Role**: Explains multi-horizon predictions, market regime, news sentiment, technical indicators, and downside risks.
- **Provider**: Connects to local Ollama HTTP REST API (`http://localhost:11434/api/generate`) running `llama3.2`, `mistral`, or `qwen2.5`.
- **Offline Fallback**: Rule-based analytical fallback generator executes when local Ollama is offline or uninstalled.
- **Grounding Rules**: The LLM acts strictly as an analytical explanation layer. It never fabricates prices, news, or model metrics, and preserves document citations.
