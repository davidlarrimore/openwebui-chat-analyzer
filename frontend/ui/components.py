from __future__ import annotations

import html
import json
import math
import re
import uuid
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from frontend.core.constants import ALL_MODELS_OPTION, ALL_USERS_OPTION
from frontend.core.models import DatasetMeta, DatasetPanel
from frontend.core.processing import (
    build_model_options,
    build_user_options,
    determine_dataset_source,
    filter_dataframes_by_user_model,
    format_day,
    format_relative_time,
    format_timestamp,
)
from frontend.utils import state as app_state


def render_sidebar_branding(container: Optional[DeltaGenerator] = None) -> None:
    """Render a compact app title in the sidebar."""
    target = container or st.sidebar
    target.markdown(
        (
            "<div style=\"font-size:0.85rem;font-weight:600;color:#1f2937;"
            "letter-spacing:0.015em;margin:0.25rem 0 0.75rem 0;\">"
            "💬 Open WebUI Chat Analyzer"
            "</div>" \
            "<hr/>"
        ),
        unsafe_allow_html=True,
    )


def render_sidebar_status(meta: Optional[DatasetMeta], *, container: Optional[DeltaGenerator] = None) -> None:
    """Render a small status card indicating dataset load state."""
    target = container or st.sidebar
    metadata = (meta.app_metadata if meta else {}) or {}
    dataset_ready = bool(
        meta
        and (
            meta.chat_count > 0
            or metadata.get("chat_count", 0) > 0
            or metadata.get("message_count", 0) > 0
        )
    )

    status_icon = "🟢" if dataset_ready else "🔴"
    status_text = "Loaded" if dataset_ready else "Not Loaded"

    source_info = determine_dataset_source(meta) if meta else None
    source_label = source_info.label if source_info else "Unknown"
    detail_value = (source_info.detail if source_info else "") or ""
    safe_source_label = html.escape(source_label or "Unknown")

    last_loaded_raw = metadata.get("dataset_pulled_at") or metadata.get("chat_uploaded_at") or metadata.get("users_uploaded_at")
    last_loaded_display = format_timestamp(last_loaded_raw)
    last_loaded_relative = format_relative_time(last_loaded_raw)
    last_loaded_line = (
        f"{last_loaded_display}"
        if last_loaded_display and last_loaded_display not in {"", "Unknown", "N/A"}
        else "Not available"
    )
    if dataset_ready and last_loaded_relative and last_loaded_relative not in {"Unknown", ""}:
        last_loaded_line = f"{last_loaded_line} ({last_loaded_relative})"
    safe_last_loaded = html.escape(str(last_loaded_line))

    chat_count = metadata.get("chat_count", meta.chat_count if meta else 0)
    try:
        chat_count_display = f"{int(chat_count):,}"
    except (TypeError, ValueError):
        chat_count_display = str(chat_count or 0)

    detail_line = ""
    if detail_value:
        detail_line = (
            "<div style='color:#4b5563;font-size:0.78rem;margin-top:0.15rem;'>Detail: "
            f"{html.escape(detail_value)}</div>"
        )

    card_html = (
        "<div style='border:1px solid #dbeafe;border-radius:10px;padding:0.75rem 0.9rem;"
        "background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 100%);margin-bottom:0.9rem;'>"
        f"<div style='font-weight:600;color:#1e3a8a;font-size:0.85rem;'>{status_icon} Data Status: {status_text}</div>"
        f"<div style='color:#374151;font-size:0.78rem;margin-top:0.4rem;'>Source: {safe_source_label}</div>"
        f"{detail_line}"
        f"<div style='color:#374151;font-size:0.78rem;margin-top:0.35rem;'>Date Loaded: {safe_last_loaded}</div>"
        "</div>"
    )
    target.markdown(card_html, unsafe_allow_html=True)


def render_chat_summary(
    summary: str,
    *,
    user: Optional[str] = None,
    model: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Render chat metadata and summary within a styled container."""
    summary = (summary or "").strip()
    safe_summary = html.escape(summary) if summary else "<em>No summary available.</em>"

    meta_items = []
    if user:
        meta_items.append(
            f"<span style='white-space:nowrap;'>👤 <strong>User:</strong> {html.escape(str(user))}</span>"
        )
    if model:
        meta_items.append(
            f"<span style='white-space:nowrap;'>🤖 <strong>Model:</strong> {html.escape(str(model))}</span>"
        )
    metadata_html = ""
    if meta_items:
        metadata_html = (
            "<div style='display:flex;flex-wrap:wrap;gap:0.75rem;font-size:0.75rem;"
            "margin-bottom:0.4rem;color:#1f2937;'>"
            + "".join(meta_items)
            + "</div>"
        )

    st.markdown(
        (
            "<div style='margin:0.5rem 0 1rem 0;"
            "padding:0.6rem 0.75rem;"
            "background-color:#eef2ff;"
            "border-radius:10px;"
            "color:#1f2937;"
            "white-space:normal;"
            "word-break:break-word;'>"
            f"{metadata_html}"
            "<div style='font-size:0.82rem;'>"
            "<span style=\"font-weight:600;\">📝 Summary:</span> "
            f"{safe_summary}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_backend_wait_splash(base_url: str, error: Optional[Exception] = None) -> None:
    """Render a friendly splash screen while the backend service spins up."""
    base_url_display = html.escape(base_url)

    st.markdown(
        """
        <style>
            .owui-backend-wait-wrapper {
                padding: 3rem 0 4rem 0;
            }
            .owui-backend-wait-card {
                margin: 0 auto;
                max-width: 520px;
                padding: 2.5rem 2.75rem;
                border-radius: 18px;
                background: linear-gradient(135deg, #eef2ff 0%, #e0f2fe 100%);
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
                text-align: center;
                color: #1f2937;
                border: 1px solid rgba(99, 102, 241, 0.12);
            }
            .owui-backend-wait-icon {
                font-size: 3rem;
                margin-bottom: 1.25rem;
            }
            .owui-backend-wait-card h2 {
                margin-bottom: 0.75rem;
                font-weight: 700;
                font-size: 1.75rem;
            }
            .owui-backend-wait-card p {
                font-size: 1rem;
                line-height: 1.6;
                margin-bottom: 1.5rem;
            }
            .owui-backend-wait-baseurl {
                display: inline-block;
                padding: 0.35rem 0.6rem;
                background: rgba(15, 23, 42, 0.08);
                border-radius: 999px;
                font-family: "Fira Code", "Source Code Pro", monospace;
                font-size: 0.85rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown(
            f"""
            <div class="owui-backend-wait-wrapper">
                <div class="owui-backend-wait-card">
                    <div class="owui-backend-wait-icon">🚀</div>
                    <h2>Warming up the backend</h2>
                    <p>We're standing by until the API responds. Once the service at
                    <span class="owui-backend-wait-baseurl">{base_url_display}</span>
                    is reachable, click "Retry now" or refresh the page to continue.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        retry_col, hint_col = st.columns([1, 1])
        with retry_col:
            if st.button("Retry now"):
                app_state.trigger_rerun()
        with hint_col:
            st.caption("Start the FastAPI backend service, then retry once it's ready.")

        if error is not None:
            with st.expander("Technical details"):
                st.code(str(error), language="text")
                status_code = getattr(error, "status_code", None)
                if status_code is not None:
                    st.caption(f"HTTP status: {status_code}")


def render_dataset_panel(panel: DatasetPanel, *, container: Optional[DeltaGenerator] = None) -> None:
    """Render the dataset summary panel."""
    target = container or st
    panel_wrapper_style = "border:1px solid #bfdbfe;border-radius:12px;overflow:hidden;margin-bottom:1rem;"
    panel_header_style = (
        "background-color:#f3f4f6;color:#374151;font-weight:600;font-size:1.0rem;"
        "padding:0.65rem 1.1rem;border-bottom:1px solid #bfdbfe;"
    )
    panel_body_style = "background-color:#eaf2ff;padding:0.9rem 1rem;"
    columns_wrapper_style = (
        "display:grid;"
        "grid-template-columns:repeat(12,minmax(0,1fr));"
        "gap:0.75rem;"
        "align-items:start;"
    )
    base_column_style = (
        "background-color:#f9fbff;"
        "border-radius:10px;"
        "padding:0.75rem 0.85rem;"
        "box-shadow:0 0 0 1px rgba(191,219,254,0.6);"
        "min-width:0;"
    )
    source_column_style = base_column_style + "grid-column:1 / span 12;"
    third_column_style = base_column_style + "grid-column:span 4;"

    panel_html = (
        f"<div style=\"{panel_wrapper_style}\">"
        f"<div style=\"{panel_header_style}\">Loaded Data</div>"
        f"<div style=\"{panel_body_style}\">"
        f"<div style=\"{columns_wrapper_style}\">"
        f"<div style=\"{source_column_style}\">{panel.source_html}</div>"
        f"<div style=\"{third_column_style}\">{panel.chat_html}</div>"
        f"<div style=\"{third_column_style}\">{panel.user_html}</div>"
        f"<div style=\"{third_column_style}\">{panel.model_html}</div>"
        f"</div></div></div>"
    )
    target.markdown(panel_html, unsafe_allow_html=True)
    target.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)


def render_processing_log_panel(
    *,
    container: Optional[DeltaGenerator] = None,
    description: str = "This panel will show data processing logs",
) -> Callable[[], None]:
    """Render the primary processing log panel and return a refresh callback."""
    target = container or st
    placeholder = target.empty()

    panel_wrapper_style = "border:1px solid #bfdbfe;border-radius:12px;overflow:hidden;margin-bottom:1rem;"
    panel_header_style = (
        "background-color:#f3f4f6;color:#374151;font-weight:600;font-size:1.0rem;"
        "padding:0.65rem 1.1rem;border-bottom:1px solid #bfdbfe;"
    )
    panel_body_style = "background-color:#f9fafb;padding:0.9rem 1rem;height:300px;"
    log_line_style = "font-family:'Fira Code','Source Code Pro',monospace;font-size:0.85rem;color:#1f2937;"
    container_id = st.session_state.setdefault("processing_log_container_id", f"processing-log-{uuid.uuid4().hex}")

    def _render() -> None:
        entries = app_state.get_processing_log_entries()
        if entries:
            line_blocks = []
            for entry in entries:
                timestamp = html.escape(entry.get("timestamp", "--:--:--"))
                emoji = html.escape(entry.get("emoji", "ℹ️ "))
                message = html.escape(entry.get("message", ""))
                line_blocks.append(
                    f"<div style=\"{log_line_style}white-space:pre-wrap;line-height:1.4;margin-bottom:0.3rem;\">"
                    f"{timestamp} {emoji} {message}"
                    "</div>"
                )
            content_html = "".join(line_blocks)
        else:
            content_html = (
                "<div style='color:#6b7280;font-size:0.9rem;line-height:1.5;'>"
                f"{html.escape(description)}"
                "</div>"
            )

        scroll_container = (
            f"<div id=\"{container_id}\" style='height:100%;overflow-y:auto;padding-right:0.25rem;'>"
            f"{content_html}"
            "</div>"
        )

        panel_html = (
            f"<div style=\"{panel_wrapper_style}\">"
            f"<div style=\"{panel_header_style}\">Processing Log</div>"
            f"<div style=\"{panel_body_style}\">{scroll_container}</div>"
            f"</div>"
            f"<script>(function() {{"
            f"const el = window.document.getElementById('{container_id}');"
            f"if (el) {{ el.scrollTop = el.scrollHeight; }}"
            f"}})();</script>"
        )
        placeholder.markdown(panel_html, unsafe_allow_html=True)

    _render()
    return _render


def render_overview_metrics(metrics: Dict[str, float], date_summary: Dict[str, Optional[str]]) -> None:
    """Render top-level metric cards."""
    st.header("Overview")
    st.caption(f"Date Range: {date_summary.get('date_min', 'N/A')} - {date_summary.get('date_max', 'N/A')}")
    total_days = date_summary.get("total_days", 0) or 0
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Days", value=total_days)
    with col2:
        st.metric(label="Total Chats", value=f"{metrics.get('total_chats', 0):,}")
    with col3:
        st.metric(label="Unique Users", value=f"{metrics.get('unique_users', 0):,}")
    with col4:
        st.metric(label="User Files Uploaded", value=f"{metrics.get('files_uploaded', 0):,}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Avg Msgs/Chat", value=f"{metrics.get('avg_messages_per_chat', 0):,.1f}")
    with col2:
        st.metric(label="Avg Input Tokens/Chat", value=f"{metrics.get('avg_input_tokens_per_chat', 0):,.1f}")
    with col3:
        st.metric(label="Avg Output Tokens/Chat", value=f"{metrics.get('avg_output_tokens_per_chat', 0):,.1f}")
    with col4:
        st.metric(label="Total Tokens", value=f"{metrics.get('total_tokens', 0):,}")


def render_filters(
    chats_df: pd.DataFrame,
    messages_df: pd.DataFrame,
    *,
    filter_prefix: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str], str]:
    """Render user/model filter controls and return filtered dataframes."""
    app_state.ensure_filter_defaults()
    user_options, user_display_lookup = build_user_options(chats_df)

    if st.session_state.user_filter not in user_options:
        st.session_state.user_filter = ALL_USERS_OPTION

    model_options = list(build_model_options(messages_df))
    if st.session_state.model_filter not in model_options:
        st.session_state.model_filter = ALL_MODELS_OPTION

    col_user, col_model = st.columns(2)
    with col_user:
        user_index = user_options.index(st.session_state.user_filter)
        st.selectbox(
            "Select user",
            options=user_options,
            index=user_index,
            format_func=lambda uid: "All Users" if uid == ALL_USERS_OPTION else str(user_display_lookup.get(uid, uid)),
            key="user_filter",
            on_change=app_state.reset_model_filter,
        )
    with col_model:
        model_index = model_options.index(st.session_state.model_filter)
        st.selectbox(
            "Select model",
            options=model_options,
            index=model_index,
            key="model_filter",
            on_change=app_state.reset_browse_page,
        )

    filtered_chats, filtered_messages = filter_dataframes_by_user_model(
        chats_df,
        messages_df,
        user_id=st.session_state.user_filter,
        model_name=st.session_state.model_filter,
    )

    model_key = st.session_state.model_filter or ALL_MODELS_OPTION
    user_key = "AllUsers" if st.session_state.user_filter == ALL_USERS_OPTION else st.session_state.user_filter
    raw_filter_key = f"{filter_prefix}_{model_key}_{user_key}"
    filter_key = re.sub(r"\W+", "_", str(raw_filter_key))
    app_state.update_filter_key(filter_key)

    return filtered_chats, filtered_messages, user_display_lookup, filter_key


def render_search_interface(
    chats_df: pd.DataFrame,
    messages_df: pd.DataFrame,
    chat_user_map: Dict[str, str],
    *,
    widget_prefix: str,
) -> None:
    """Render the search interface for conversations."""
    if messages_df.empty:
        st.info("No messages available to search.")
        return

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_query = st.text_input(
            "Search in conversations:",
            placeholder="Enter search terms...",
            key=f"search_input_{widget_prefix}",
        )
    with col2:
        role_filter = st.selectbox(
            "Filter by role:",
            ["All", "user", "assistant"],
            key=f"role_filter_{widget_prefix}",
        )
    with col3:
        max_conversations = st.selectbox(
            "Max conversations:",
            [5, 10, 15, 20],
            key=f"max_conversations_{widget_prefix}",
        )

    if not search_query:
        st.caption("Enter a search term to explore matching conversations.")
        return

    filtered = messages_df[messages_df["content"].str.contains(search_query, case=False, na=False)]
    if role_filter != "All":
        filtered = filtered[filtered["role"] == role_filter]

    conversation_ids = filtered["chat_id"].unique()
    if len(conversation_ids) > 0:
        ordered_conversations = (
            messages_df[messages_df["chat_id"].isin(conversation_ids)]
            .groupby("chat_id", as_index=False)["timestamp"]
            .min()
            .sort_values("timestamp", ascending=True, na_position="last")
        )
        ordered_conversation_ids = ordered_conversations["chat_id"].tolist()
    else:
        ordered_conversation_ids = []

    st.success(f"✅ Found {len(conversation_ids)} conversations containing '{search_query}'")
    pattern = re.compile(re.escape(search_query), re.IGNORECASE)

    for idx, cid in enumerate(ordered_conversation_ids[:max_conversations]):
        conv_msgs = messages_df[messages_df["chat_id"] == cid].copy()
        conv_msgs["role_priority"] = conv_msgs["role"].map({"user": 0, "assistant": 1}).fillna(2)
        conv_msgs = conv_msgs.sort_values(["timestamp", "role_priority"], ascending=[True, True])
        chat_info_rows = chats_df[chats_df["chat_id"] == cid]
        if chat_info_rows.empty:
            continue
        chat_info = chat_info_rows.iloc[0]
        subject = chat_info["title"] or cid
        date_display = format_day(chat_info["created_at"])
        date = "Unknown" if date_display in {"N/A", ""} else date_display
        models = [m for m in conv_msgs["model"].unique() if m]
        model_name = models[0] if models else "Unknown"
        user_display = chat_user_map.get(cid, chat_info.get("user_display", chat_info.get("user_id", "User")))
        file_upload_flag = "📎" if chat_info.get("files_uploaded", 0) > 0 else ""
        with st.expander(f"Thread #{idx + 1}: {subject} | Date: {date} | Model: {model_name} | {file_upload_flag}", expanded=False):
            render_chat_summary(
                str(chat_info.get("summary_128", "")),
                user=user_display,
                model=model_name,
                title=subject,
            )
            for _, msg in conv_msgs.iterrows():
                highlighted = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", str(msg["content"]))
                timestamp_display = format_timestamp(msg["timestamp"])
                timestamp = "--:--" if timestamp_display in {"N/A", ""} else timestamp_display
                if msg["role"] == "user":
                    st.markdown(
                        f"<div style='background-color:#e6f7ff; padding:8px; border-radius:5px; margin-bottom:4px;'>"
                        f"<strong>{user_display}</strong> <span style='float:right;color:#555;'>{timestamp}</span><br>{highlighted}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='background-color:#f0f0f0; padding:8px; border-radius:5px; margin-bottom:4px;'>"
                        f"<strong>🤖 Assistant ({msg['model']})</strong> <span style='float:right;color:#555;'>{timestamp}</span><br>{highlighted}</div>",
                        unsafe_allow_html=True,
                    )
            full_thread = {
                "chat_id": chat_info["chat_id"],
                "user_id": chat_info["user_id"],
                "user_name": user_display,
                "title": chat_info["title"],
                "created_at": chat_info["created_at"].isoformat() if pd.notna(chat_info["created_at"]) else None,
                "updated_at": chat_info["updated_at"].isoformat() if pd.notna(chat_info["updated_at"]) else None,
                "archived": bool(chat_info.get("archived")),
                "pinned": bool(chat_info.get("pinned")),
                "tags": chat_info.get("tags"),
                "files": chat_info.get("files", []),
                "messages": json.loads(
                    conv_msgs.drop(columns=["role_priority"], errors="ignore").to_json(
                        orient="records",
                        date_format="iso",
                    )
                ),
            }
            thread_json = json.dumps(full_thread, indent=2)
            st.download_button(
                label="📥 Download Thread (JSON)",
                data=thread_json,
                file_name=f"thread_{cid}.json",
                mime="application/json",
                key=f"download_thread_{widget_prefix}_{cid}",
            )
            for file_item in chat_info.get("files", []):
                file_name = file_item.get("filename") or file_item.get("name")
                st.markdown(f"📎 {file_name}")


def render_instructions() -> None:
    """Render instructions when no dataset is available."""
    st.warning("Use Direct Connect to link the analyzer with your Open WebUI instance and pull in chats.")
    st.header("How to enable Direct Connect")
    st.write(
        """
        1. **Open Open WebUI** and sign in with an account that can access the Admin panel.
        2. **Generate an API token** from Settings → Personal Access Tokens.
        3. **Confirm the base URL** where Open WebUI is running (for example `http://localhost:3000`).
        4. **Enter the URL and token** in the Direct Connect panel and press **Load Data** to import chats, users, and models.
        """
    )
    st.header("What you'll get from this analysis")
    st.write(
        """
        - **Overview Metrics**: Total chats, messages, unique users, average messages per conversation, user vs assistant message ratios
        - **Time Analysis**: Daily message activity trends, activity heatmaps by hour and day, conversation length distributions
        - **Model Usage**: Distribution of different AI models used, usage statistics and percentages, model performance insights
        - **Content Analysis**: Word clouds from user messages, message length analysis, content pattern identification
        - **Sentiment Analysis**: Positive, neutral, negative message ratios, sentiment trends over time, user satisfaction indicators
        - **Search & Filter**: Full-text search across all messages, filter by user role or time period, export filtered results
        """
    )


def render_browse_data(
    filtered_chats: pd.DataFrame,
    filtered_messages: pd.DataFrame,
    chat_user_map: Dict[str, str],
    *,
    filter_key: str,
) -> None:
    """Render browse chats pane with pagination."""
    chat_table = filtered_chats.copy()
    if chat_table.empty:
        st.info("No chats found for the current filters.")
        return
    if filtered_messages.empty:
        st.info("No messages to display.")
        return

    chat_user_lookup = filtered_chats.set_index("chat_id")["user_display"].to_dict()
    first_prompts = (
        filtered_messages[filtered_messages["role"] == "user"]
        .groupby("chat_id")["timestamp"]
        .min()
        .reset_index()
    )
    first_prompts = first_prompts.merge(
        filtered_chats[["chat_id", "title"]],
        on="chat_id",
        how="left",
    )
    first_prompts = first_prompts.sort_values("timestamp", ascending=False, na_position="last")
    total_threads = len(first_prompts)
    threads_options = [5, 10, 20, 50]
    current_threads = app_state.get_threads_per_page()
    if current_threads not in threads_options:
        current_threads = 10
    threads_per_page = st.selectbox(
        "Threads per page",
        threads_options,
        index=threads_options.index(current_threads),
        key=f"threads_per_page_{filter_key}",
    )
    app_state.set_threads_per_page(threads_per_page)
    total_pages = math.ceil(total_threads / threads_per_page) if total_threads else 1
    if st.session_state.page > total_pages:
        st.session_state.page = total_pages
    if st.session_state.page < 1:
        st.session_state.page = 1
    page = st.session_state.page
    start_idx = (page - 1) * threads_per_page
    end_idx = start_idx + threads_per_page
    first_prompts = first_prompts.iloc[start_idx:end_idx]
    for _, row in first_prompts.iterrows():
        thread_id = row["chat_id"]
        title = row["title"] or thread_id
        timestamp_value = row["timestamp"]
        started_display = format_timestamp(timestamp_value)
        date = "Unknown" if started_display in {"N/A", ""} else started_display
        chat_rows = filtered_chats[filtered_chats["chat_id"] == thread_id]
        if chat_rows.empty:
            continue
        chat_info = chat_rows.iloc[0]
        file_upload_flag = "📎" if chat_info.get("files_uploaded", 0) > 0 else ""
        user_display = chat_user_lookup.get(thread_id, chat_info.get("user_id", "User"))
        header = f"{title} (Started: {date})"
        header = f"{header} | {user_display}"
        if file_upload_flag:
            header = f"{header} | {file_upload_flag}"
        with st.expander(header, expanded=False):
            thread_msgs = filtered_messages[filtered_messages["chat_id"] == thread_id].copy()
            thread_msgs["role_priority"] = thread_msgs["role"].map({"user": 0, "assistant": 1}).fillna(2)
            thread_msgs = thread_msgs.sort_values(["timestamp", "role_priority"], ascending=[True, True])
            models = [m for m in thread_msgs["model"].unique() if m]
            model_name = models[0] if models else "Unknown"
            render_chat_summary(
                str(chat_info.get("summary_128", "")),
                user=user_display,
                model=model_name,
                title=title,
            )
            for _, msg in thread_msgs.iterrows():
                timestamp_display = format_timestamp(msg["timestamp"])
                timestamp = "--:--" if timestamp_display in {"N/A", ""} else timestamp_display
                content = str(msg["content"]).replace("\n", "<br>")
                if msg["role"] == "user":
                    st.markdown(
                        f"<div style='background-color:#e6f7ff; padding:8px; border-radius:5px; margin-bottom:4px;'>"
                        f"<strong>{user_display}</strong> <span style='color:#555;'>[{timestamp}]</span><br>{content}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='background-color:#f0f0f0; padding:8px; border-radius:5px; margin-bottom:4px;'>"
                        f"<strong>🤖 Assistant ({msg['model']})</strong> <span style='color:#555;'>[{timestamp}]</span><br>{content}</div>",
                        unsafe_allow_html=True,
                    )
            full_thread = {
                "chat_id": chat_info["chat_id"],
                "user_id": chat_info["user_id"],
                "user_name": user_display,
                "title": chat_info["title"],
                "created_at": chat_info["created_at"].isoformat() if pd.notna(chat_info["created_at"]) else None,
                "updated_at": chat_info["updated_at"].isoformat() if pd.notna(chat_info["updated_at"]) else None,
                "archived": bool(chat_info.get("archived")),
                "pinned": bool(chat_info.get("pinned")),
                "tags": chat_info.get("tags"),
                "files": chat_info.get("files", []),
                "messages": json.loads(
                    thread_msgs.drop(columns=["role_priority"], errors="ignore").to_json(
                        orient="records",
                        date_format="iso",
                    )
                ),
            }
            thread_json = json.dumps(full_thread, indent=2)
            st.download_button(
                label="📥 Download Thread (JSON)",
                data=thread_json,
                file_name=f"thread_{filter_key}_{thread_id}.json",
                mime="application/json",
                key=f"browse_download_{filter_key}_{thread_id}",
            )
            for file_item in chat_info.get("files", []):
                file_name = file_item.get("filename") or file_item.get("name")
                st.markdown(f"📎 {file_name}")
    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("Previous", key=f"prev_page_{filter_key}") and st.session_state.page > 1:
            st.session_state.page -= 1
    with col_info:
        st.write(f"Page {st.session_state.page} of {total_pages}")
    with col_next:
        if st.button("Next", key=f"next_page_{filter_key}") and st.session_state.page < total_pages:
            st.session_state.page += 1
