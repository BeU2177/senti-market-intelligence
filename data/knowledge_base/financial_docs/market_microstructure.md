# Market Microstructure & Regime Dynamics

## Market Regimes
Financial markets transition through distinct behavioral regimes characterized by trending or mean-reverting price action:
- **Bullish Trend**: Strong positive price momentum with price above moving averages (SMA 50 > SMA 200).
- **Bearish Trend**: Strong negative price momentum with price below moving averages (SMA 50 < SMA 200).
- **Sideways / Range-Bound**: Low directional momentum where prices fluctuate within horizontal support and resistance channels.
- **High Volatility**: Market regimes marked by elevated Average True Range (ATR) or wide Bollinger Bands, often driven by macroeconomic announcements or market shocks.

## Financial News Sentiment Integration
News sentiment acts as an exogenous signal influencing short-term market participant expectations.
Key principles for NLP sentiment features:
- **Exponential Time-Decay**: News impact degrades over time according to $w_i = \exp(-\lambda \cdot age\_in\_days)$. Recent news carries significantly higher weight.
- **Temporal Alignment**: News published at $T_{published} > T_{bar}$ must NEVER be used in feature calculations at $T_{bar}$ to prevent lookahead leakage.
- **Non-Price Modification**: Sentiment features are passed as model input signals; sentiment scores must NEVER be directly added to or multiplied by raw stock prices ($Price \neq Price + Sentiment$).
