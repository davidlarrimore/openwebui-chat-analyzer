from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

st.set_page_config(
    page_title="Sentiment Analysis | Open WebUI Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend.core.processing import perform_sentiment_analysis
from frontend.ui import components
from frontend.ui.charts import (
    create_sentiment_distribution_chart,
    create_sentiment_time_series,
)
from frontend.ui.page_state import ensure_data_ready


def render_page() -> None:
    _, processed = ensure_data_ready()

    st.subheader("Sentiment Analysis")
    _, filtered_messages, _, filter_key = components.render_filters(
        processed.chats,
        processed.messages,
        filter_prefix="sentiment",
    )

    sentiment_df = perform_sentiment_analysis(filtered_messages)
    if sentiment_df.empty:
        st.warning("⚠️ Sentiment analysis unavailable - No user messages found for analysis.")
        return

    col1, col2 = st.columns(2)
    with col1:
        dist_fig = create_sentiment_distribution_chart(sentiment_df)
        st.plotly_chart(dist_fig, use_container_width=True, key=f"sentiment_distribution_chart_{filter_key}")
    with col2:
        time_fig = create_sentiment_time_series(sentiment_df)
        st.plotly_chart(time_fig, use_container_width=True, key=f"sentiment_time_chart_{filter_key}")

    st.markdown("### Sentiment Breakdown")
    total = len(sentiment_df)
    sentiment_counts = sentiment_df["sentiment_category"].value_counts()
    col1, col2, col3 = st.columns(3)
    with col1:
        positive_pct = (sentiment_counts.get("Positive", 0) / total) * 100 if total else 0
        st.metric(label="Positive Messages", value=f"{positive_pct:.1f}%")
    with col2:
        neutral_pct = (sentiment_counts.get("Neutral", 0) / total) * 100 if total else 0
        st.metric(label="Neutral Messages", value=f"{neutral_pct:.1f}%")
    with col3:
        negative_pct = (sentiment_counts.get("Negative", 0) / total) * 100 if total else 0
        st.metric(label="Negative Messages", value=f"{negative_pct:.1f}%")


if __name__ == "__main__":
    render_page()
