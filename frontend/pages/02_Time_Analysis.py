from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

st.set_page_config(
    page_title="Time Analysis | Open WebUI Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend.ui import components
from frontend.ui.charts import (
    create_conversation_length_distribution,
    create_time_series_chart,
    create_user_activity_chart,
)
from frontend.ui.page_state import ensure_data_ready


def render_page() -> None:
    _, processed = ensure_data_ready()

    st.subheader("Time-based Analysis")
    filtered_chats, filtered_messages, _, filter_key = components.render_filters(
        processed.chats,
        processed.messages,
        filter_prefix="time",
    )

    col1, col2 = st.columns(2)
    with col1:
        time_fig = create_time_series_chart(filtered_messages)
        st.plotly_chart(time_fig, use_container_width=True, key=f"time_series_chart_{filter_key}")
    with col2:
        length_fig = create_conversation_length_distribution(filtered_messages)
        st.plotly_chart(length_fig, use_container_width=True, key=f"conversation_length_chart_{filter_key}")

    activity_fig = create_user_activity_chart(filtered_messages)
    st.plotly_chart(activity_fig, use_container_width=True, key=f"activity_heatmap_{filter_key}")


if __name__ == "__main__":
    render_page()
