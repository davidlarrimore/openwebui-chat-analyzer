from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

st.set_page_config(
    page_title="Browse Chats | Open WebUI Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend.ui import components
from frontend.ui.page_state import ensure_data_ready


def render_page() -> None:
    _, processed = ensure_data_ready()

    st.subheader("Browse Chats")
    filtered_chats, filtered_messages, _, filter_key = components.render_filters(
        processed.chats,
        processed.messages,
        filter_prefix="browse",
    )

    chat_user_map = filtered_chats.set_index("chat_id")["user_display"].to_dict() if not filtered_chats.empty else {}
    components.render_browse_data(
        filtered_chats,
        filtered_messages,
        chat_user_map,
        filter_key=filter_key,
    )


if __name__ == "__main__":
    render_page()
