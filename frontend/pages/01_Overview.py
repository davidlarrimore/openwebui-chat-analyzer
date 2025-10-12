from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

st.set_page_config(
    page_title="Overview | Open WebUI Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend.core.api import BackendError, get_summary_status  # noqa: E402
from frontend.core.processing import (  # noqa: E402
    calculate_engagement_metrics,
    compute_date_summary,
    summarize_summary_status,
)
from frontend.ui import components
from frontend.ui.page_state import ensure_data_ready
from frontend.ui.charts import (  # noqa: E402
    create_model_usage_chart,
    create_model_usage_pie,
    create_token_consumption_chart,
    create_user_adoption_chart,
)
from frontend.utils.logging import get_logger  # noqa: E402


LOGGER = get_logger("frontend.overview")


def _maybe_show_summary_status_toast() -> None:
    try:
        summary_status = get_summary_status()
    except BackendError as exc:
        LOGGER.warning("Unable to retrieve summary status: %s", exc)
        st.warning(f"Unable to retrieve summary status: {exc}")
        return

    message_and_icon = summarize_summary_status(summary_status)
    if not message_and_icon:
        st.session_state.pop("_summary_state_last_toast", None)
        return

    message, icon = message_and_icon
    last_message = st.session_state.get("_summary_state_last_toast")
    if message != last_message:
        st.toast(message, icon=icon)
        st.session_state["_summary_state_last_toast"] = message


def render_page() -> None:
    _, processed = ensure_data_ready()

    _maybe_show_summary_status_toast()

    metrics = calculate_engagement_metrics(processed.chats, processed.messages)
    if not metrics:
        st.info("No messages available in the dataset yet. Upload data to get started.")
        return

    date_summary = compute_date_summary(processed.messages)
    components.render_overview_metrics(metrics, date_summary)

    token_fig = create_token_consumption_chart(processed.messages)
    if token_fig and len(token_fig.data) > 0:
        st.plotly_chart(token_fig, use_container_width=True, key="token_consumption_chart_overview")
    else:
        st.info("Token consumption chart unavailable - insufficient timestamped data.")

    st.markdown("---")

    st.subheader("Model Usage Analysis")
    assistant_msgs = processed.messages[
        (processed.messages["role"] == "assistant")
        & (processed.messages["model"] != "")
    ]
    if assistant_msgs.empty:
        st.warning("ℹ️ No model information available in the dataset.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                create_model_usage_chart(processed.messages),
                use_container_width=True,
                key="model_usage_bar_overview",
            )
        with col2:
            st.plotly_chart(
                create_model_usage_pie(processed.messages),
                use_container_width=True,
                key="model_usage_pie_overview",
            )

    st.markdown("---")

    st.subheader("User Adoption Overview")
    date_min = date_summary.get("date_min_ts")
    date_max = date_summary.get("date_max_ts")
    adoption_fig = create_user_adoption_chart(
        processed.chats,
        processed.messages,
        date_min,
        date_max,
    )
    if adoption_fig is None:
        st.info("Need user messages with timestamps to plot adoption.")
    else:
        st.plotly_chart(adoption_fig, use_container_width=True, key="user_adoption_chart_overview")

    st.info(
        "Navigate using the sidebar tabs to explore detailed analyses:\n"
        "📈 Time Analysis, 💭 Content Analysis, 😊 Sentiment, 🔍 Search Chats, and 🗂 Browse Chats."
    )


if __name__ == "__main__":
    render_page()
