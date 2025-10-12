from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

st.set_page_config(
    page_title="Content Analysis | Open WebUI Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend.ui import components
from frontend.ui.charts import (
    create_average_message_length_chart,
    create_message_length_histogram,
    generate_word_cloud,
)
from frontend.ui.page_state import ensure_data_ready


def render_page() -> None:
    _, processed = ensure_data_ready()

    st.subheader("Content Analysis")
    filtered_chats, filtered_messages, _, filter_key = components.render_filters(
        processed.chats,
        processed.messages,
        filter_prefix="content",
    )

    wordcloud = generate_word_cloud(filtered_messages)
    if wordcloud:
        col_left, col_wordcloud, col_right = st.columns([1, 6, 1])
        with col_wordcloud:
            st.markdown("### ☁️ Word Cloud (User Messages)")
            st.image(wordcloud.to_image(), use_container_width=True)
        col_left.empty()
        col_right.empty()
    else:
        if filtered_messages.empty:
            st.warning("⚠️ Word cloud unavailable - No messages available. Upload data or adjust filters.")
        else:
            roles = filtered_messages.get("role")
            contents = filtered_messages.get("content")
            has_user = False
            has_text = False
            if roles is not None:
                try:
                    has_user = any(roles.fillna("").astype(str).str.lower() == "user")
                except Exception:
                    has_user = False
            if contents is not None:
                try:
                    has_text = any(contents.astype(str).str.strip() != "")
                except Exception:
                    has_text = False

            if not has_text:
                st.warning("⚠️ Word cloud unavailable - Messages are present but no textual content was found.")
            elif not has_user:
                st.warning("⚠️ Word cloud unavailable - No messages labeled with role 'user' were found.")
            else:
                st.warning("⚠️ Word cloud unavailable - Text processing failed or no significant terms were extracted.")

    if not filtered_messages.empty:
        col1, col2 = st.columns(2)
        with col1:
            avg_fig = create_average_message_length_chart(filtered_messages)
            st.plotly_chart(avg_fig, use_container_width=True, key=f"avg_message_length_chart_{filter_key}")
        with col2:
            hist_fig = create_message_length_histogram(filtered_messages)
            st.plotly_chart(hist_fig, use_container_width=True, key=f"message_length_hist_{filter_key}")


if __name__ == "__main__":
    render_page()
