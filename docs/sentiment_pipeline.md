# Financial News Ingestion & FinBERT NLP Sentiment Pipeline

## Financial News Ingestion & Deduplication

1. **News Providers**: Ingests articles from NewsAPI.org with fallback to Yahoo Finance news RSS feeds.
2. **Deduplication**: Hash deduplication on article URLs and titles prevents double-counting duplicate headlines across sources.

## FinBERT NLP Model

- **Model**: `ProsusAI/finbert` (Hugging Face Transformers).
- **Probabilities**: Computes Softmax probabilities for $P_{\text{positive}}$, $P_{\text{negative}}$, and $P_{\text{neutral}}$ satisfying $P_{\text{pos}} + P_{\text{neg}} + P_{\text{neu}} = 1.0$.
- **Bounded Score**: $Score = P_{\text{positive}} - P_{\text{negative}} \in [-1, 1]$.

## Temporal Alignment & Time-Decay

1. **Zero Lookahead Leakage**: For any historical market bar at timestamp $T_{\text{bar}}$, only news published at $T_{\text{published}} \le T_{\text{bar}}$ are included.
2. **Exponential Time-Decay Weighting**: News impact decays exponentially over time:
   $$w_i = \exp(-\lambda \cdot age\_in\_days)$$
3. **18 Aggregate Features**: Computes decay-weighted mean sentiment score, positive ratio, negative ratio, 3-day momentum, 7-day velocity, and article volume metrics.
