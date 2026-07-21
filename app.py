"""Streamlit web application entry point for Senti Market Intelligence."""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

from config.settings import get_settings
from services.market_service import MarketService
from utils.logging_config import setup_logging

# Initialize Logging and Settings
setup_logging()
settings = get_settings()
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title=f"{settings.APP_NAME} - Phase 8",
    page_icon="📈",
    layout="wide",
)


def check_api_health() -> bool:
    """Check if local FastAPI backend server is online."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        return resp.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=settings.CACHE_TTL_SECONDS)
def load_market_dataset(
    symbol: str,
    period: str,
    interval: str,
    start_date: str = None,
    end_date: str = None,
):
    """Cached helper wrapper to fetch market dataset via MarketService."""
    service = MarketService()
    return service.get_market_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        period=period,
        interval=interval,
        save_to_disk=True,
    )


@st.cache_data(ttl=settings.CACHE_TTL_SECONDS)
def load_feature_dataset(
    symbol: str,
    period: str,
    interval: str,
    start_date: str = None,
    end_date: str = None,
    include_sentiment: bool = True,
):
    """Cached helper wrapper to fetch feature dataset via MarketService."""
    service = MarketService()
    return service.get_feature_dataset(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        period=period,
        interval=interval,
        include_sentiment=include_sentiment,
        save_to_disk=True,
    )


@st.cache_data(ttl=settings.CACHE_TTL_SECONDS)
def load_news_articles(symbol: str, start_date: str = None, end_date: str = None):
    """Cached helper wrapper to fetch news articles via MarketService."""
    service = MarketService()
    return service.get_news_articles(symbol=symbol, start_date=start_date, end_date=end_date, save_to_disk=True)


def main():
    st.title("📈 SENTI MARKET INTELLIGENCE")
    st.caption("Phase 8 — Production Inference API, Real-Time Data Pipeline & MLOps Platform")

    st.markdown("---")

    # Sidebar Controls & API Status Indicator
    st.sidebar.header("System Status & Controls")
    
    api_online = check_api_health()
    if api_online:
        st.sidebar.success("🟢 FastAPI Backend: CONNECTED (http://localhost:8000)")
    else:
        st.sidebar.warning("🟡 FastAPI Backend: OFFLINE (Direct Python Service Mode)")

    st.sidebar.markdown("**Sample Multi-Market Symbols:**")
    st.sidebar.caption("• US: `AAPL`, `MSFT`, `TSLA`\n• India: `RELIANCE.NS`, `TCS.NS`\n• UAE: `EMAAR.AE`, `SALIK.AE`")

    symbol_input = st.sidebar.text_input("Ticker Symbol", value=settings.DEFAULT_SYMBOL)
    
    range_mode = st.sidebar.radio("Selection Mode", options=["Period", "Custom Date Range"])
    
    period = "1y"
    start_date_str = None
    end_date_str = None
    
    if range_mode == "Period":
        period = st.sidebar.selectbox(
            "Period",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            index=3,
        )
    else:
        col_start, col_end = st.sidebar.columns(2)
        default_start = datetime.now() - timedelta(days=365)
        start_dt = col_start.date_input("Start Date", value=default_start)
        end_dt = col_end.date_input("End Date", value=datetime.now())
        start_date_str = start_dt.strftime("%Y-%m-%d")
        end_date_str = end_dt.strftime("%Y-%m-%d")

    interval = st.sidebar.selectbox(
        "Interval",
        options=["1d", "1wk", "1mo"],
        index=0,
    )

    include_sentiment_input = st.sidebar.checkbox("Include FinBERT Sentiment Features", value=True)

    if not symbol_input or not symbol_input.strip():
        st.warning("Please enter a valid ticker symbol in the sidebar.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Market Data & Quality",
        "⚙️ Feature Engineering",
        "📰 Financial News & Sentiment",
        "🤖 Classical ML Training",
        "🧠 PyTorch Deep Learning",
        "💬 AI Analyst & Grounded RAG",
    ])

    # --- TAB 1: Market Data & Validation ---
    with tab1:
        with st.spinner(f"Fetching market data for {symbol_input.upper()}..."):
            data_response = load_market_dataset(
                symbol=symbol_input,
                period=period,
                interval=interval,
                start_date=start_date_str,
                end_date=end_date_str,
            )

        df = data_response.data_frame
        val_result = data_response.validation_result
        meta = data_response.market_metadata

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Symbol", data_response.symbol)
        col2.metric("Exchange", meta.exchange if meta else "UNKNOWN")
        col3.metric("Country", meta.country if meta else "Global")
        col4.metric("Currency", meta.currency if meta else "USD")
        col5.metric("Timezone", meta.timezone if meta else "UTC")

        st.markdown("---")

        col_fresh, col_rows, col_source, col_ts = st.columns(4)
        freshness_str = val_result.freshness_status.value if val_result else "UNKNOWN"
        freshness_badge = {
            "REAL_TIME": "🟢 REAL TIME",
            "NEAR_REAL_TIME": "🟢 NEAR REAL TIME",
            "DELAYED": "🟡 DELAYED",
            "HISTORICAL": "🔵 HISTORICAL",
            "STALE": "🔴 STALE",
            "UNKNOWN": "⚪ UNKNOWN",
        }.get(freshness_str, freshness_str)

        col_fresh.metric("Data Freshness Status", freshness_badge)
        col_rows.metric("Record Count", f"{data_response.row_count:,}")
        col_source.metric("Data Source", data_response.provider)

        latest_ts_str = (
            data_response.latest_timestamp.strftime("%Y-%m-%d %H:%M UTC")
            if data_response.latest_timestamp
            else "N/A"
        )
        col_ts.metric("Latest Timestamp", latest_ts_str)

        st.markdown("### Data Quality & Gap Analysis")
        if val_result:
            if val_result.is_valid:
                st.success("✅ **Data Validation Passed** — Schema, price bounds, OHLC rules, and volume checks passed.")
            else:
                st.error("❌ **Data Validation Failed** — Critical data quality errors detected.")

        if df is not None and not df.empty:
            st.markdown("### Market Price History")
            st.line_chart(df.set_index("timestamp")[["close", "open", "high", "low"]])
            st.dataframe(df, use_container_width=True)

    # --- TAB 2: Feature Engineering ---
    with tab2:
        with st.spinner(f"Building feature dataset for {symbol_input.upper()}..."):
            feat_dataset = load_feature_dataset(
                symbol=symbol_input,
                period=period,
                interval=interval,
                start_date=start_date_str,
                end_date=end_date_str,
                include_sentiment=include_sentiment_input,
            )

        f_df = feat_dataset.data_frame
        l_report = feat_dataset.leakage_report

        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        f_col1.metric("Feature Version", feat_dataset.feature_version)
        f_col2.metric("Technical Features", len(feat_dataset.feature_columns) - len(feat_dataset.sentiment_columns))
        f_col3.metric("Sentiment Features", len(feat_dataset.sentiment_columns))
        f_col4.metric("Target Labels", len(feat_dataset.target_columns))

        clean_badge = "✅ CLEAN (No Leakage)" if (l_report and l_report.is_clean) else "❌ LEAKAGE DETECTED"
        st.info(f"Leakage Validation Status: **{clean_badge}** (Temporal Alignment: `published_at <= T` enforced)")

        if f_df is not None and not f_df.empty:
            st.markdown("### Engineered Dataset Preview")
            st.dataframe(f_df, use_container_width=True)

    # --- TAB 3: Financial News & Sentiment ---
    with tab3:
        with st.spinner(f"Ingesting news for {symbol_input.upper()}..."):
            articles = load_news_articles(
                symbol=symbol_input,
                start_date=start_date_str,
                end_date=end_date_str,
            )

        n_col1, n_col2, n_col3 = st.columns(3)
        n_col1.metric("Article Count", len(articles))
        n_col2.metric("NLP Model", "FinBERT (ProsusAI)")
        n_col3.metric("Score Formula", "P(Pos) - P(Neg) ∈ [-1, 1]")

        st.markdown("---")

        if articles:
            st.markdown("### Ingested Financial News & Sentiment Probabilities")
            from news.sentiment.sentiment_model import get_sentiment_model
            model = get_sentiment_model()
            sentiments = model.predict_sentiment_batch(articles)

            art_rows = []
            for art, s in zip(articles, sentiments):
                art_rows.append({
                    "Published (UTC)": art.published_at.strftime("%Y-%m-%d %H:%M"),
                    "Source": art.source,
                    "Title": art.title,
                    "Label": s.predicted_label,
                    "Score": s.sentiment_score,
                    "P(Pos)": s.positive_probability,
                    "P(Neg)": s.negative_probability,
                    "P(Neu)": s.neutral_probability,
                    "Match Confidence": art.entity_match_confidence,
                })

            st.dataframe(pd.DataFrame(art_rows), use_container_width=True)
        else:
            st.info(f"No news articles found for symbol '{symbol_input.upper()}'.")

    # --- TAB 4: Classical ML Training & Ablation ---
    with tab4:
        st.markdown("### Classical ML Time-Series Pipeline")
        target_selected = st.selectbox(
            "Prediction Target Column",
            options=["future_return_1d", "future_return_3d", "future_return_5d", "future_return_7d"],
            index=0,
        )

        if st.button("🚀 Train Classical Models & Run Ablation Experiment"):
            with st.spinner(f"Executing ML training and ablation experiment for {symbol_input.upper()}..."):
                service = MarketService()
                ablation_res = service.run_ablation_experiment(
                    symbol=symbol_input,
                    target_column=target_selected,
                    period=period,
                    interval=interval,
                )

            st.success("✅ Training & Ablation Experiment Completed Successfully!")
            mps = ablation_res["market_plus_sentiment"]["best_report"]
            st.write(f"**Best Selected Model:** `{ablation_res['market_plus_sentiment']['best_model']}` (Val RMSE: `{mps.val_metrics.rmse:.6f}`)")

    # --- TAB 5: PyTorch Deep Learning ---
    with tab5:
        st.markdown("### PyTorch Deep Learning & Production Inference Pipeline")
        seq_len_selected = st.selectbox(
            "Lookback Window (Sequence Length)",
            options=[10, 20, 30, 60],
            index=2,
            format_func=lambda x: f"{x} Trading Days",
        )

        if st.button("🚀 Run PyTorch Deep Learning & Model Ensemble Benchmark"):
            with st.spinner(f"Generating multi-horizon predictions for {symbol_input.upper()}..."):
                if api_online:
                    resp = requests.get(f"{API_BASE_URL}/prediction/{symbol_input}?sequence_length={seq_len_selected}")
                    if resp.status_code == 200:
                        p_data = resp.json()
                        st.success("✅ Multi-Horizon Production Predictions Fetched via FastAPI!")
                        st.write(
                            f"**Ensemble 1-Day Forecast:** `{p_data['predicted_returns']['1d']*100:+.2f}%` "
                            f"(Target Price: `${p_data['predicted_prices']['1d']:.2f}`) | "
                            f"**Confidence:** `{p_data['confidence_level']}`"
                        )
                    else:
                        st.error(f"FastAPI error: {resp.text}")
                else:
                    service = MarketService()
                    ens_res = service.run_full_ensemble_benchmark(
                        symbol=symbol_input,
                        sequence_length=seq_len_selected,
                        period=period,
                        interval=interval,
                    )
                    ep = ens_res["ensemble_prediction"]
                    st.success("✅ PyTorch Deep Learning Benchmark Completed!")
                    st.write(
                        f"**Ensemble 1-Day Forecast:** `{ep.predicted_returns['1d']*100:+.2f}%` "
                        f"(Target Price: `${ep.predicted_prices['1d']:.2f}`) | "
                        f"**Confidence:** `{ens_res['confidence_assessment'].confidence_level}`"
                    )

    # --- TAB 6: AI Analyst & Grounded RAG Agent ---
    with tab6:
        st.markdown("### Grounded AI Market Intelligence Analyst & RAG Agent")
        st.caption("Combines Vector Search (ChromaDB + SentenceTransformers) with Ollama LLM Reasoning & Grounded Context")

        st.markdown("**Quick Preset Queries:**")
        preset = st.radio(
            "Select Preset Question or Type Below:",
            options=[
                f"Why is {symbol_input.upper()} forecast to move over the next 5 trading days?",
                "Explain RSI and Bollinger Bands technical indicators",
                f"What are the primary risk factors for {symbol_input.upper()}?",
                "How does financial time-series validation prevent data leakage?",
                "Custom Query",
            ],
            index=0,
        )

        default_text = preset if preset != "Custom Query" else f"Provide a complete market analysis for {symbol_input.upper()}."
        query_input = st.text_area("Your Query to the AI Analyst:", value=default_text, height=100)

        if st.button("💡 Ask AI Market Analyst Agent"):
            with st.spinner(f"Agent synthesizing market context, news, ML forecasts, and RAG knowledge for {symbol_input.upper()}..."):
                if api_online:
                    post_payload = {"symbol": symbol_input, "question": query_input, "period": period, "interval": interval}
                    resp = requests.post(f"{API_BASE_URL}/analysis", json=post_payload)
                    if resp.status_code == 200:
                        a_data = resp.json()
                        st.markdown("### 📋 Executive Summary")
                        st.info(f"**Summary:** {a_data['summary']}")
                        st.write(f"**Assessed Confidence Level:** {a_data['confidence_level']}")

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.subheader("📊 Market Overview")
                            st.write(a_data['market_context_overview'])
                            st.subheader("🧠 Model Prediction Explanation")
                            st.write(a_data['model_prediction_explanation'])
                            st.subheader("✅ Supporting Evidence")
                            for ev in a_data['supporting_evidence']:
                                st.write(f"• {ev}")
                        with col_b:
                            st.subheader("⚠️ Downside & Market Risk Factors")
                            for rf in a_data['risk_factors']:
                                st.write(f"• {rf}")
                            st.subheader("🔍 Uncertainty Assessment")
                            st.write(a_data['uncertainty_assessment'])
                            st.subheader("📚 Source Citations & Provenance")
                            for src in a_data['sources']:
                                st.write(f"• `{src}`")
                    else:
                        st.error(f"FastAPI Error: {resp.text}")
                else:
                    service = MarketService()
                    analysis_resp = service.analyze_market_with_agent(
                        user_query=query_input,
                        symbol=symbol_input,
                        period=period,
                        interval=interval,
                    )

                    st.markdown("---")
                    st.markdown("### 📋 Executive Summary")
                    st.info(f"**Summary:** {analysis_resp.summary}")

                    c_level = analysis_resp.confidence_level
                    c_badge = {"HIGH": "🟢 HIGH CONFIDENCE", "MEDIUM": "🟡 MEDIUM CONFIDENCE", "LOW": "🔴 LOW CONFIDENCE"}.get(c_level, c_level)
                    st.write(f"**Assessed Confidence Level:** {c_badge}")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.subheader("📊 Market & Regime Overview")
                        st.write(analysis_resp.market_context_overview)
                        st.subheader("🧠 Model Prediction Explanation")
                        st.write(analysis_resp.model_prediction_explanation)
                        st.subheader("✅ Supporting Evidence")
                        for ev in analysis_resp.supporting_evidence:
                            st.write(f"• {ev}")
                    with col_b:
                        st.subheader("⚠️ Downside & Market Risk Factors")
                        for rf in analysis_resp.risk_factors:
                            st.write(f"• {rf}")
                        st.subheader("🔍 Uncertainty Assessment")
                        st.write(analysis_resp.uncertainty_assessment)
                        st.subheader("📚 Source Citations & Provenance")
                        for src in analysis_resp.sources:
                            st.write(f"• `{src}`")


if __name__ == "__main__":
    main()
