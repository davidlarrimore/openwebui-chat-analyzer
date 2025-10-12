#!/usr/bin/env python3
"""
Open WebUI Chat Data Analyzer
A modern Streamlit dashboard for analyzing Open WebUI chat export

Installation:
pip install streamlit pandas plotly wordcloud textblob networkx

Usage:
streamlit run frontend/app.py
"""

import base64
from datetime import datetime
import html
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timedelta
from urllib.parse import urlparse
from uuid import uuid4

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from textblob import TextBlob
from wordcloud import WordCloud
from dotenv import load_dotenv

# Load environment variables from a `.env` file if present.
load_dotenv()

OPENWEBUI_DEFAULT_HOST = os.getenv("OWUI_DIRECT_HOST", "http://localhost:3000").strip() or "http://localhost:3000"
OPENWEBUI_DEFAULT_API_KEY = os.getenv("OWUI_DIRECT_API_KEY", "").strip()

# Configure page
st.set_page_config(
    page_title="Open WebUI Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)




def create_header():
    st.title("💬 Open WebUI Chat Analyzer")
    st.subheader("Transform your conversation data into actionable insights with beautiful visualizations")


def trigger_rerun():
    """Trigger a Streamlit rerun compatible across versions."""
    rerun_fn = getattr(st, "experimental_rerun", None) or getattr(st, "rerun", None)
    if rerun_fn is not None:
        rerun_fn()


def render_chat_summary(summary: str) -> None:
    """Render a chat summary below a card header."""
    summary = (summary or "").strip()
    if not summary:
        return
    safe_text = html.escape(summary)
    st.markdown(
        (
            "<div style='margin:0.5rem 0 1rem 0;"
            "padding:0.6rem 0.75rem;"
            "background-color:#eef2ff;"
            "border-radius:10px;"
            "color:#1f2937;"
            "white-space:normal;"
            "word-break:break-word;'>"
            "<span style=\"font-weight:600;\">📝 Summary:</span> "
            f"{safe_text}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

ALL_USERS_OPTION = "__ALL_USERS__"
ALL_MODELS_OPTION = "All Models"

class BackendError(Exception):
    """Custom exception raised when the backend API is unreachable or returns an error."""


def get_api_base_url_candidates():
    """Return backend API base URLs to try in priority order."""
    base_url = "http://backend:8502"
    return [base_url]


def get_api_base_url() -> str:
    """Resolve the configured API base URL."""
    return get_api_base_url_candidates()[0]


def _request_backend(path: str, method: str = "GET", *, timeout: float = 30.0, **kwargs):
    """Execute a request against the backend API."""
    response = None
    last_exc = None
    for base_url in get_api_base_url_candidates():
        url = f"{base_url}{path}"
        try:
            response = requests.request(
                method=method.upper(),
                url=url,
                timeout=timeout,
                **kwargs,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            detail = ""
            if exc.response is not None:
                try:
                    payload = exc.response.json()
                except ValueError:
                    payload = None
                if isinstance(payload, dict):
                    detail_value = payload.get("detail")
                    if detail_value:
                        detail = str(detail_value)
                if not detail:
                    text = exc.response.text.strip()
                    if text:
                        detail = text
            message = f"Backend request failed: {exc}"
            if detail:
                message = f"{message} - {detail}"
            raise BackendError(message) from exc
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            continue
        else:
            break
    else:
        raise BackendError(f"Backend request failed: {last_exc}") from last_exc

    try:
        assert response is not None  # for type checkers
    except AssertionError:
        raise BackendError("Backend request failed due to an unknown error.")

    if response.status_code == 204 or method.upper() == "HEAD":
        return {}

    try:
        return response.json()
    except ValueError as exc:
        raise BackendError("Backend returned an invalid JSON response.") from exc


def get_summary_status():
    """Fetch the status of the background summarizer job."""
    return _request_backend("/api/v1/summaries/status")


def wait_for_summary_completion(
    message: str,
    timeout: int = 180,
    poll_interval: float = 2.0,
    on_update=None,
    show_spinner: bool = True,
) -> dict:
    """Poll the backend until the summarizer job completes or times out."""
    start_time = time.time()
    last_state = None
    status: dict = {}

    spinner_ctx = st.spinner(message) if show_spinner and message else nullcontext()
    with spinner_ctx:
        while True:
            try:
                status = get_summary_status()
            except BackendError as exc:
                st.warning(f"Unable to check summary status: {exc}")
                break

            state = status.get("state", "idle")
            if state != last_state:
                last_state = state

            if on_update is not None:
                try:
                    on_update(status)
                except Exception:
                    pass

            if state in (None, "idle", "completed", "failed", "cancelled"):
                break

            if time.time() - start_time >= timeout:
                st.warning("Summary job is still running; results will appear once complete.")
                break

            time.sleep(poll_interval)

    return status


def _init_direct_connect_log() -> None:
    st.session_state.setdefault("show_direct_connect_log", False)
    st.session_state.setdefault("direct_connect_log", [])
    st.session_state.setdefault("direct_connect_progress_index", None)
    st.session_state.setdefault("direct_connect_log_expanded", True)


def _append_direct_connect_log(message: str, emoji: str = "ℹ️ ") -> None:
    _init_direct_connect_log()
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state["direct_connect_log"].append({
        "timestamp": timestamp,
        "emoji": emoji,
        "message": message,
    })


def _set_direct_connect_progress(completed: int, total: int) -> None:
    _init_direct_connect_log()
    idx = st.session_state.get("direct_connect_progress_index")
    percent = 0.0
    if total > 0:
        percent = (completed / total) * 100
    entry_message = f"Building chat summaries... {completed}/{total} ({percent:.1f}%)"
    timestamp = datetime.now().strftime("%H:%M:%S")
    if idx is None:
        _append_direct_connect_log(entry_message, "🧠")
        st.session_state["direct_connect_progress_index"] = len(st.session_state["direct_connect_log"]) - 1
    else:
        try:
            st.session_state["direct_connect_log"][idx]["timestamp"] = timestamp
            st.session_state["direct_connect_log"][idx]["message"] = entry_message
        except (IndexError, KeyError):
            st.session_state["direct_connect_progress_index"] = None


def _reset_direct_connect_progress() -> None:
    st.session_state["direct_connect_progress_index"] = None


def _reset_direct_connect_log(show: bool = True) -> None:
    st.session_state["direct_connect_log"] = []
    st.session_state["show_direct_connect_log"] = show
    st.session_state["direct_connect_progress_index"] = None
    st.session_state["direct_connect_log_expanded"] = True


def render_direct_connect_log(container=None, *, context: str = None, key_prefix: str = "direct_connect") -> None:
    _init_direct_connect_log()
    show = st.session_state.get("show_direct_connect_log")
    if container is None:
        container = st.container()
    if not show:
        try:
            container.empty()
        except AttributeError:
            pass
        return

    try:
        container.empty()
        target = container.container()
    except AttributeError:
        target = container

    log_entries = st.session_state.get("direct_connect_log", [])
    with target:
        accordion_fn = getattr(st, "accordion", None) or st.expander
        expanded = st.session_state.get("direct_connect_log_expanded", True)
        with accordion_fn("Processing Log", expanded=expanded):
            if log_entries:
                lines = [
                    f"{entry.get('timestamp', '--:--:--')} {entry.get('emoji', 'ℹ️ ')} {entry.get('message', '')}"
                    for entry in log_entries
                ]
                log_text = "\n".join(lines)
            else:
                log_text = "(no log entries yet)"

            st.code(log_text, language="text")


def fetch_dataset_metadata():
    """Fetch dataset metadata from the backend."""
    return _request_backend("/api/v1/datasets/meta")


def sync_openwebui_dataset(hostname: str, api_key: str):
    """Trigger a direct sync from an Open WebUI instance."""
    payload = {"hostname": hostname}
    token = (api_key or "").strip()
    if token:
        payload["api_key"] = token
    return _request_backend("/api/v1/openwebui/sync", method="POST", json=payload)


@st.cache_data(show_spinner=False)
def load_and_process_data(dataset_id: str):
    """Fetch chats and messages from the backend and convert them into DataFrames."""
    chats_payload = _request_backend("/api/v1/chats")
    messages_payload = _request_backend("/api/v1/messages")

    chats_df = pd.DataFrame(chats_payload)
    messages_df = pd.DataFrame(messages_payload)

    if not chats_df.empty:
        if "summary_128" not in chats_df.columns:
            chats_df["summary_128"] = ""
        else:
            chats_df["summary_128"] = chats_df["summary_128"].fillna("")
        for column in ("created_at", "updated_at"):
            if column in chats_df.columns:
                chats_df[column] = pd.to_datetime(chats_df[column])

    if not messages_df.empty and "timestamp" in messages_df.columns:
        messages_df["timestamp"] = pd.to_datetime(messages_df["timestamp"])

    return chats_df, messages_df


@st.cache_data(show_spinner=False)
def load_user_data(dataset_id: str):
    """Fetch user metadata from the backend."""
    users_payload = _request_backend("/api/v1/users")
    users_df = pd.DataFrame(users_payload)
    if users_df.empty:
        return pd.DataFrame(columns=["user_id", "name"])

    users_df["user_id"] = users_df["user_id"].astype(str)
    users_df["name"] = users_df["name"].astype(str)
    return users_df


def upload_chat_export(file_obj):
    """Upload a new chat export file to the backend."""
    filename = getattr(file_obj, "name", "chat_export.json")
    filetype = getattr(file_obj, "type", None) or "application/json"
    file_bytes = file_obj.getvalue() if hasattr(file_obj, "getvalue") else file_obj.read()
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    files = {"file": (filename, file_bytes, filetype)}
    return _request_backend("/api/v1/uploads/chat-export", method="POST", files=files)


def upload_users_csv(file_obj):
    """Upload a users CSV file to the backend."""
    filename = getattr(file_obj, "name", "users.csv")
    filetype = getattr(file_obj, "type", None) or "text/csv"
    file_bytes = file_obj.getvalue() if hasattr(file_obj, "getvalue") else file_obj.read()
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    files = {"file": (filename, file_bytes, filetype)}
    return _request_backend("/api/v1/uploads/users", method="POST", files=files)


def reset_model_filter():
    st.session_state["model_filter"] = ALL_MODELS_OPTION
    st.session_state["page"] = 1


def reset_browse_page():
    st.session_state["page"] = 1

def calculate_engagement_metrics(chats_df, messages_df):
    """Calculate user engagement metrics"""
    if messages_df.empty:
        return {}
    
    total_chats = len(chats_df)
    total_messages = len(messages_df)
    unique_users = chats_df['user_id'].nunique()
    
    # Messages per chat
    msg_per_chat = messages_df.groupby('chat_id').size().mean()
    
    # User messages vs assistant messages
    user_messages = len(messages_df[messages_df['role'] == 'user'])
    assistant_messages = len(messages_df[messages_df['role'] == 'assistant'])
    
    # Average conversation length
    conversations = messages_df.groupby('chat_id').size()
    avg_conversation_length = conversations.mean()
    
    files_uploaded = chats_df['files_uploaded'].sum()

    # Average output tokens per chat (assistant messages) based on character length
    assistant_msgs = messages_df[messages_df['role'] == 'assistant'].copy()
    assistant_msgs['token_count'] = assistant_msgs['content'].str.len()
    output_tokens_per_chat = assistant_msgs.groupby('chat_id')['token_count'].sum()
    avg_output_tokens_per_chat = output_tokens_per_chat.mean() if not output_tokens_per_chat.empty else 0

    # Average input tokens per chat (user messages) based on character length
    user_msgs = messages_df[messages_df['role'] == 'user'].copy()
    user_msgs['token_count'] = user_msgs['content'].str.len()
    input_tokens_per_chat = user_msgs.groupby('chat_id')['token_count'].sum()
    avg_input_tokens_per_chat = input_tokens_per_chat.mean() if not input_tokens_per_chat.empty else 0

    # Total tokens across all messages
    total_input_tokens = input_tokens_per_chat.sum() if not input_tokens_per_chat.empty else 0
    total_output_tokens = output_tokens_per_chat.sum() if not output_tokens_per_chat.empty else 0
    total_tokens = total_input_tokens + total_output_tokens

    return {
        'total_chats': total_chats,
        'total_messages': total_messages,
        'unique_users': unique_users,
        'avg_messages_per_chat': msg_per_chat,
        'user_messages': user_messages,
        'assistant_messages': assistant_messages,
        'avg_conversation_length': avg_conversation_length,
        'avg_input_tokens_per_chat': avg_input_tokens_per_chat,
        'avg_output_tokens_per_chat': avg_output_tokens_per_chat,
        'total_tokens': total_tokens,
        'files_uploaded': files_uploaded,
    }

def create_time_series_chart(messages_df):
    """Create time series visualization with modern styling"""
    if messages_df.empty:
        return go.Figure()

    timestamp_series = messages_df["timestamp"].dropna()
    if timestamp_series.empty:
        return go.Figure()

    tz = timestamp_series.dt.tz
    now = pd.Timestamp.now(tz=tz) if tz is not None else pd.Timestamp.now()
    end_date = now.normalize()
    start_date = end_date - pd.Timedelta(days=29)

    normalized_dates = messages_df["timestamp"].dt.normalize()
    recent_dates = normalized_dates[
        (normalized_dates >= start_date) & (normalized_dates <= end_date)
    ]
    date_index = pd.date_range(start=start_date, end=end_date, freq="D", tz=tz)
    daily_counts = recent_dates.value_counts().reindex(date_index, fill_value=0).sort_index()

    daily_messages = daily_counts.reset_index()
    daily_messages.columns = ["date", "message_count"]
    
    fig = px.line(daily_messages, x='date', y='message_count',
                  title='Daily Message Activity')
    
    fig.update_traces(
        line_color='#3b82f6', 
        line_width=3,
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.1)'
    )
    
    fig.update_layout(
        hovermode='x unified',
        showlegend=False,
        template='plotly_white',
        title_font_size=18,
        title_font_color='#1f2937',
        title_font_weight=600,
        xaxis_title="Date",
        yaxis_title="Number of Messages",
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, l=50, r=50, b=50)
    )
    
    return fig

def create_user_activity_chart(messages_df):
    """Create user activity heatmap with modern styling"""
    if messages_df.empty:
        return go.Figure()
    
    # Extract hour and day of week
    messages_df['hour'] = messages_df['timestamp'].dt.hour
    messages_df['day_of_week'] = messages_df['timestamp'].dt.day_name()
    
    # Create activity matrix
    activity_matrix = messages_df.groupby(['day_of_week', 'hour']).size().reset_index()
    activity_matrix.columns = ['day_of_week', 'hour', 'count']
    
    # Pivot for heatmap
    heatmap_data = activity_matrix.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=list(range(24)),
        y=heatmap_data.index,
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>Hour: %{x}:00<br>Messages: %{z}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title="Messages",
            tickmode="linear",
            tick0=0,
            len=0.7,
            thickness=15,
            x=1.02,
            xanchor='left',
            y=0.5,
            yanchor='middle'
        )
    ))
    
    fig.update_layout(
        title='Message Activity Heatmap',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        template='plotly_white',
        title_font_size=18,
        title_font_color='#1f2937',
        title_font_weight=600,
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, l=50, r=50, b=50)
    )
    
    return fig

def create_model_usage_chart(messages_df):
    """Create model usage trend chart showing chats by model over time."""
    if messages_df.empty:
        return go.Figure()

    assistant_msgs = messages_df[
        (messages_df["role"] == "assistant") & (messages_df["model"] != "")
    ].dropna(subset=["timestamp"]).copy()
    if assistant_msgs.empty:
        return go.Figure()

    assistant_msgs["timestamp"] = pd.to_datetime(assistant_msgs["timestamp"])
    first_model_use = (
        assistant_msgs.groupby(["chat_id", "model"])["timestamp"]
        .min()
        .reset_index()
    )
    if first_model_use.empty:
        return go.Figure()

    first_model_use["date"] = first_model_use["timestamp"].dt.normalize()
    daily_counts = (
        first_model_use.groupby(["date", "model"])["chat_id"]
        .nunique()
        .rename("chat_count")
        .reset_index()
    )

    if daily_counts.empty:
        return go.Figure()

    tz_info = daily_counts["date"].dt.tz
    date_min = daily_counts["date"].min()
    date_max = daily_counts["date"].max()
    date_range = pd.date_range(date_min, date_max, freq="D", tz=tz_info)
    models = sorted(daily_counts["model"].unique())
    complete_index = pd.MultiIndex.from_product(
        [date_range, models], names=["date", "model"]
    )
    daily_counts = (
        daily_counts.set_index(["date", "model"])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )
    daily_counts["date"] = daily_counts["date"].dt.normalize()

    fig = px.line(
        daily_counts,
        x="date",
        y="chat_count",
        color="model",
        title="Chats by Model Over Time",
        markers=True
    )
    fig.update_traces(line_width=3)
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        title_font_size=18,
        title_font_color="#1f2937",
        title_font_weight=600,
        xaxis_title="Date",
        yaxis_title="Chats",
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, l=50, r=50, b=50),
        legend_title_text="Model",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )
    fig.update_xaxes(tickformat="%Y-%m-%d", dtick=86400000.0)

    return fig

def create_model_usage_pie(messages_df):
    """Create a pie chart of overall model usage."""
    if messages_df.empty:
        return go.Figure()

    assistant_msgs = messages_df[
        (messages_df["role"] == "assistant") & (messages_df["model"] != "")
    ]
    if assistant_msgs.empty:
        return go.Figure()

    model_counts = assistant_msgs["model"].value_counts()
    fig = px.pie(
        names=model_counts.index,
        values=model_counts.values,
        title="Model Usage Share"
    )
    fig.update_layout(
        template="plotly_white",
        title_font_size=18,
        title_font_color="#1f2937",
        title_font_weight=600,
        margin=dict(t=60, l=20, r=20, b=20),
        showlegend=True,
        font=dict(family="Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )
    fig.update_traces(textinfo="percent+label")
    return fig

def create_token_consumption_chart(messages_df):
    """Create a slim line chart showing total token consumption per day."""
    if messages_df.empty:
        return go.Figure()

    token_df = messages_df.dropna(subset=["timestamp"]).copy()
    if token_df.empty:
        return go.Figure()

    token_df["token_count"] = token_df["content"].astype(str).str.len()
    token_df["date"] = token_df["timestamp"].dt.normalize()
    daily_tokens = (
        token_df.groupby("date")["token_count"]
        .sum()
        .reset_index()
        .sort_values("date")
    )
    if daily_tokens.empty:
        return go.Figure()

    tz_info = daily_tokens["date"].dt.tz
    full_range = pd.date_range(
        daily_tokens["date"].min(),
        daily_tokens["date"].max(),
        freq="D",
        tz=tz_info
    )
    daily_tokens = (
        daily_tokens.set_index("date")
        .reindex(full_range, fill_value=0)
        .rename_axis("date")
        .reset_index()
        .rename(columns={"index": "date"})
    )

    fig = px.line(
        daily_tokens,
        x="date",
        y="token_count",
        title="Daily Token Consumption Over Time",
        markers=True
    )
    fig.update_traces(line_color="#6366f1", line_width=2)
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        title_font_size=16,
        title_font_color="#1f2937",
        title_font_weight=600,
        xaxis_title="Date",
        yaxis_title="Tokens",
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, l=40, r=40, b=40),
        height=260
    )
    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        tickmode="linear",
        dtick=86400000.0
    )

    return fig

def compute_top_users(chats_df, messages_df, top_n=10):
    """Return aggregated metrics for the most active users."""
    if chats_df.empty or messages_df.empty:
        return pd.DataFrame(columns=["user", "chat_count", "message_count", "token_percentage"])

    chat_lookup = (
        chats_df[["chat_id", "user_display"]]
        .dropna(subset=["chat_id"])
        .drop_duplicates(subset=["chat_id"])
    )
    messages_with_users = messages_df.merge(chat_lookup, on="chat_id", how="left")
    messages_with_users["user_display"] = messages_with_users["user_display"].fillna("Unknown User")

    # Approximate tokens by character count for all messages in the user's chats
    messages_with_users["token_count"] = messages_with_users["content"].astype(str).str.len()
    tokens_per_user = messages_with_users.groupby("user_display")["token_count"].sum()
    total_tokens = tokens_per_user.sum()

    chat_counts = (
        chats_df.groupby("user_display")["chat_id"]
        .nunique()
        .reindex(tokens_per_user.index, fill_value=0)
    )
    message_counts = messages_with_users.groupby("user_display").size()

    summary = pd.DataFrame({
        "user": tokens_per_user.index,
        "chat_count": chat_counts.values,
        "message_count": message_counts.values,
        "token_percentage": np.where(
            total_tokens > 0,
            (tokens_per_user.values / total_tokens) * 100,
            0.0
        ),
        "token_count": tokens_per_user.values,
    })

    summary = summary.sort_values(
        by=["token_count", "chat_count", "message_count"],
        ascending=False
    ).drop(columns="token_count")

    summary = summary.head(top_n).reset_index(drop=True)

    return summary

def create_user_adoption_chart(chats_df, messages_df, date_min=None, date_max=None):
    """Create a cumulative user adoption chart showing daily growth."""
    if chats_df.empty or messages_df.empty:
        return None

    chat_lookup = (
        chats_df[["chat_id", "user_display"]]
        .dropna(subset=["chat_id"])
        .drop_duplicates(subset=["chat_id"])
    )
    user_messages = (
        messages_df[messages_df["role"] == "user"]
        .merge(chat_lookup, on="chat_id", how="left")
    )
    user_messages = user_messages.dropna(subset=["timestamp"])
    if user_messages.empty:
        return None

    user_messages["timestamp"] = pd.to_datetime(
        user_messages["timestamp"], errors="coerce"
    )
    user_messages = user_messages.dropna(subset=["timestamp"])
    if user_messages.empty:
        return None

    first_message_dates = (
        user_messages.groupby("user_display")["timestamp"]
        .min()
        .dropna()
        .reset_index()
    )
    if first_message_dates.empty:
        return None

    first_message_dates["first_date"] = first_message_dates["timestamp"].dt.normalize()
    daily_new_users = (
        first_message_dates.groupby("first_date")
        .size()
        .rename("new_users")
        .reset_index()
        .sort_values("first_date")
    )
    daily_new_users["cumulative_users"] = daily_new_users["new_users"].cumsum()

    if daily_new_users.empty:
        return None

    first_activity_date = daily_new_users["first_date"].min()
    start_date = first_activity_date - pd.Timedelta(days=1)

    if date_max is not None:
        end_date = pd.to_datetime(date_max).normalize()
    else:
        end_date = daily_new_users["first_date"].max()

    if start_date is None or end_date is None:
        return None

    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    if all_dates.empty:
        return None

    daily_adoption = pd.DataFrame({"date": all_dates})
    daily_adoption = daily_adoption.merge(
        daily_new_users.rename(columns={"first_date": "date"})[["date", "cumulative_users"]],
        on="date",
        how="left"
    )
    daily_adoption["cumulative_users"] = daily_adoption["cumulative_users"].ffill().fillna(0)
    daily_adoption["cumulative_users"] = daily_adoption["cumulative_users"].astype(int)
    daily_adoption["new_users"] = daily_adoption["cumulative_users"].diff().fillna(daily_adoption["cumulative_users"])
    daily_adoption["new_users"] = daily_adoption["new_users"].astype(int)

    fig = px.line(
        daily_adoption,
        x="date",
        y="cumulative_users",
        markers=True,
        custom_data=["new_users"]
    )
    fig.update_traces(line_color="#10b981", line_width=3)
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        xaxis_title="Date",
        yaxis_title="Cumulative Users",
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, l=50, r=50, b=50),
        hoverlabel=dict(bgcolor="rgba(17,24,39,0.9)", font=dict(color="white"))
    )
    fig.update_xaxes(
        tickmode="linear",
        dtick=24 * 60 * 60 * 1000,
        tickformat="%b %d"
    )
    fig.update_traces(
        hovertemplate=(
            "%{x|%b %d, %Y}<br>"
            "Cumulative Users: %{y}<br>"
            "New Users: %{customdata[0]}"
        )
    )

    fig.update_xaxes(range=[start_date, end_date])

    return fig

def create_conversation_length_distribution(messages_df):
    """Create conversation length distribution with modern styling"""
    if messages_df.empty:
        return go.Figure()
    
    # Calculate conversation lengths
    conv_lengths = messages_df.groupby('chat_id').size()
    
    fig = px.histogram(
        x=conv_lengths.values,
        nbins=20,
        title='Conversation Length Distribution',
        color_discrete_sequence=['#3b82f6']
    )
    
    fig.update_layout(
        template='plotly_white',
        title_font_size=18,
        title_font_color='#1f2937',
        title_font_weight=600,
        xaxis_title="Messages per Conversation",
        yaxis_title="Number of Conversations",
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, l=50, r=50, b=50)
    )
    
    return fig

def generate_word_cloud(messages_df):
    """Generate word cloud from user messages"""
    if messages_df is None or messages_df.empty:
        return None

    # Normalize role column to string and lowercase for robust matching
    if 'role' in messages_df.columns:
        roles = messages_df['role'].fillna('').astype(str).str.lower()
    else:
        roles = pd.Series([''] * len(messages_df))

    # Get user messages only (case-insensitive match for 'user')
    user_mask = roles == 'user'
    user_messages = messages_df.loc[user_mask, 'content'] if user_mask.any() else pd.Series([], dtype=object)

    if user_messages.empty:
        return None
    
    # Combine all text
    text = ' '.join(user_messages.astype(str))
    
    # Clean text
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text)
    
    if len(text.strip()) == 0:
        return None
    
    # Generate word cloud with modern colors
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            relative_scaling=0.5,
            colormap='viridis',
            font_path=None,
            prefer_horizontal=0.9
        ).generate(text)

        return wordcloud
    except Exception:
        # If anything fails, return None so the caller can show a friendly warning
        return None

def perform_sentiment_analysis(messages_df):
    """Perform sentiment analysis on messages"""
    if messages_df.empty:
        return pd.DataFrame()
    
    # Get user messages for sentiment analysis
    user_messages = messages_df[messages_df['role'] == 'user'].copy()
    
    if len(user_messages) == 0:
        return pd.DataFrame()
    
    # Calculate sentiment scores
    sentiments = []
    for content in user_messages['content']:
        try:
            blob = TextBlob(str(content))
            sentiment = blob.sentiment.polarity
            sentiments.append(sentiment)
        except:
            sentiments.append(0)
    
    user_messages['sentiment'] = sentiments
    
    # Categorize sentiments
    user_messages['sentiment_category'] = pd.cut(
        user_messages['sentiment'],
        bins=[-1, -0.1, 0.1, 1],
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    return user_messages

def create_search_interface(chats_df, messages_df, chat_user_map=None, widget_prefix="default"):
    """Create enhanced search interface returning threads"""
    if messages_df.empty:
        return

    chat_user_map = chat_user_map or {}

    st.header("Search Conversations")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_query = st.text_input(
            "Search in conversations:",
            placeholder="Enter search terms...",
            key=f"search_input_{widget_prefix}"
        )
    with col2:
        role_filter = st.selectbox(
            "Filter by role:",
            ["All", "user", "assistant"],
            key=f"role_filter_{widget_prefix}"
        )
    with col3:
        max_conversations = st.selectbox(
            "Max conversations:",
            [5, 10, 15, 20],
            key=f"max_conversations_{widget_prefix}"
        )

    if search_query:
        # Filter messages by query and optional role
        filtered = messages_df[messages_df['content'].str.contains(search_query, case=False, na=False)]
        if role_filter != "All":
            filtered = filtered[filtered['role'] == role_filter]
        # Retrieve subject, date, and model for each conversation
        # Identify matching conversations
        conversation_ids = filtered['chat_id'].unique()
        st.success(f"✅ Found {len(conversation_ids)} conversations containing '{search_query}'")
        # Compile regex for highlighting
        pattern = re.compile(re.escape(search_query), re.IGNORECASE)
        # Display each conversation thread
        for idx, cid in enumerate(conversation_ids[:max_conversations]):
            conv_msgs = messages_df[messages_df['chat_id'] == cid].sort_values('timestamp')
            # Gather chat metadata
            chat_info_rows = chats_df[chats_df['chat_id'] == cid]
            if chat_info_rows.empty:
                continue
            chat_info = chat_info_rows.iloc[0]
            subject = chat_info['title'] or cid
            date = chat_info['created_at'].strftime('%Y-%m-%d')
            models = [m for m in conv_msgs['model'].unique() if m]
            model_name = models[0] if models else 'Unknown'
            user_display = chat_user_map.get(cid, chat_info.get('user_display', chat_info.get('user_id', 'User')))
            # Determine icon for attachments
            file_upload_flag = "📎" if chat_info.get('files_uploaded', 0) > 0 else ""
            # Display expander with enriched title
            with st.expander(f"Thread #{idx+1}: {subject} | Date: {date} | Model: {model_name} | {file_upload_flag} ", expanded=False):
                render_chat_summary(str(chat_info.get('summary_128', "")))
                for _, msg in conv_msgs.iterrows():
                    # Highlight search terms
                    highlighted = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", str(msg['content']))
                    timestamp = msg['timestamp'].strftime('%Y-%m-%d %H:%M')
                    # Differentiate user vs assistant
                    if msg['role'] == 'user':
                        user_label = chat_user_map.get(msg['chat_id'], user_display)
                        st.markdown(
                            f"<div style='background-color:#e6f7ff; padding:8px; border-radius:5px; margin-bottom:4px;'>"
                            f"<strong>{user_label}</strong> <span style='float:right;color:#555;'>{timestamp}</span><br>{highlighted}</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div style='background-color:#f0f0f0; padding:8px; border-radius:5px; margin-bottom:4px;'>"
                            f"<strong>🤖 Assistant ({msg['model']})</strong> <span style='float:right;color:#555;'>{timestamp}</span><br>{highlighted}</div>",
                            unsafe_allow_html=True
                        )
                # Export full thread JSON
                full_thread = {
                    'chat_id': chat_info['chat_id'],
                    'user_id': chat_info['user_id'],
                    'user_name': user_display,
                    'title': chat_info['title'],
                    'created_at': chat_info['created_at'].isoformat(),
                    'updated_at': chat_info['updated_at'].isoformat(),
                    'archived': bool(chat_info['archived']),
                    'pinned': bool(chat_info['pinned']),
                    'tags': chat_info['tags'],
                    'files': chat_info.get('files', []),
                    'messages': json.loads(conv_msgs.to_json(orient='records', date_format='iso'))
                }
                thread_json = json.dumps(full_thread, indent=2)
                st.download_button(
                    label="📥 Download Thread (JSON)",
                    data=thread_json,
                    file_name=f"thread_{cid}.json",
                    mime="application/json",
                    key=f"download_thread_{cid}"
                )
                # List attachments
                for file_item in chat_info.get('files', []):
                    file_name = file_item.get('filename') or file_item.get('name')
                    st.markdown(f"📎 {file_name}")

def create_export_section(chats_df, messages_df):
    """Create modern export section"""
    # Visual divider above export section
    st.markdown('---')
    st.header("Export Data")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Chat Data")
        csv_chats = chats_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Chat Data CSV",
            data=csv_chats,
            file_name="openwebui_chat_data.csv",
            mime="text/csv",
            key="download_chats"
        )
    with col2:
        st.subheader("Message Data")
        csv_messages = messages_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Message Data CSV",
            data=csv_messages,
            file_name="openwebui_message_data.csv",
            mime="text/csv",
            key="download_messages"
        )

def create_instructions():
    """Create modern instructions section"""
    st.header("How to export data from Open WebUI")
    st.write(
        """
        1. **Access your Open WebUI admin panel**
        2. **Go to Settings → Data & Privacy**
        3. **Click "Export All Chats"** to download your chat data as JSON
        4. **Upload the downloaded JSON file** using the file uploader above
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

def main():
    # Create modern header
    create_header()

    try:
        dataset_meta = fetch_dataset_metadata()
    except BackendError as exc:
        st.error(
            "Unable to connect to the backend API. "
            "Ensure the FastAPI service is running and reachable at "
            f"`{get_api_base_url()}`.\n\nDetails: {exc}"
        )
        st.stop()

    try:
        summary_status_initial = get_summary_status()
    except BackendError as exc:
        st.warning(f"Unable to retrieve summary status: {exc}")
        summary_status_initial = {}

    dataset_id = dataset_meta.get("dataset_id", "unknown")
    app_metadata = dataset_meta.get("app_metadata") or {}
    raw_dataset_source = dataset_meta.get("source", "")
    dataset_source_label = app_metadata.get("dataset_source") or raw_dataset_source or "Unknown source"

    def _format_timestamp(value) -> str:
        if not value:
            return "N/A"
        try:
            ts_value = pd.to_datetime(value)
        except Exception:
            return str(value)
        if pd.isna(ts_value):
            return "N/A"
        if getattr(ts_value, "tzinfo", None) is None:
            return ts_value.strftime("%Y-%m-%d %H:%M")
        return ts_value.tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")

    def _format_relative_time(value) -> str:
        if not value:
            return ""
        try:
            ts_value = pd.to_datetime(value)
        except Exception:
            return ""
        if pd.isna(ts_value):
            return ""
        try:
            if getattr(ts_value, "tzinfo", None) is None:
                ts_value = ts_value.tz_localize("UTC")
            else:
                ts_value = ts_value.tz_convert("UTC")
        except Exception:
            try:
                ts_value = pd.Timestamp(ts_value).tz_localize("UTC")
            except Exception:
                return ""
        now = pd.Timestamp.now(tz="UTC")
        delta = now - ts_value
        seconds = max(delta.total_seconds(), 0.0)
        if seconds < 60:
            return "1 minute ago"
        minutes = int(seconds // 60)
        if minutes < 60:
            minutes = max(1, minutes)
            unit = "minute" if minutes == 1 else "minutes"
            return f"{minutes} {unit} ago"
        hours = int(seconds // 3600)
        if hours < 24:
            hours = max(1, hours)
            unit = "hour" if hours == 1 else "hours"
            return f"{hours} {unit} ago"
        days = max(1, int(seconds // 86400))
        unit = "day" if days == 1 else "days"
        return f"{days} {unit} ago"

    def _format_day(value) -> str:
        if not value:
            return "N/A"
        try:
            dt_value = pd.to_datetime(value)
        except Exception:
            return str(value)
        if pd.isna(dt_value):
            return "N/A"
        return dt_value.strftime("%Y-%m-%d")

    summary_state = summary_status_initial.get("state")
    summary_state_message = None
    summary_state_icon = None
    if summary_state == "running":
        completed = summary_status_initial.get("completed", 0)
        total = summary_status_initial.get("total", 0)
        summary_state_message = f"Summaries in progress ({completed}/{total})"
        summary_state_icon = "🧠"
    elif summary_state == "failed":
        detail = summary_status_initial.get("message", "Unknown error")
        summary_state_message = f"Summary job failed: {detail}"
        summary_state_icon = "⚠️"
    elif summary_state == "cancelled":
        summary_state_message = "Summary job was cancelled because the dataset changed before completion."
        summary_state_icon = "⚠️"
    elif summary_state == "completed" and summary_status_initial.get("total", 0):
        summary_state_message = "Chat summaries are up to date."
        summary_state_icon = "🧠"

    last_summary_toast = st.session_state.get("_summary_state_last_toast")
    if summary_state_message and summary_state_message != last_summary_toast:
        st.toast(summary_state_message, icon=summary_state_icon)
    st.session_state["_summary_state_last_toast"] = summary_state_message

    chat_uploaded_display = _format_timestamp(app_metadata.get("chat_uploaded_at"))
    users_uploaded_display = _format_timestamp(app_metadata.get("users_uploaded_at"))
    dataset_pulled_value = app_metadata.get("dataset_pulled_at")
    dataset_pulled_display = _format_timestamp(dataset_pulled_value)
    dataset_pulled_relative = _format_relative_time(dataset_pulled_value)
    first_day_display = _format_day(app_metadata.get("first_chat_day"))
    last_day_display = _format_day(app_metadata.get("last_chat_day"))

    if first_day_display == "N/A" and last_day_display == "N/A":
        range_display = "N/A"
    elif first_day_display != "N/A" and last_day_display != "N/A":
        range_display = f"{first_day_display} - {last_day_display}"
    else:
        range_display = first_day_display if last_day_display == "N/A" else last_day_display

    chat_count_meta = app_metadata.get("chat_count", dataset_meta.get("chat_count", 0))
    user_count_meta = app_metadata.get("user_count", dataset_meta.get("user_count", 0))
    try:
        chat_count_display = f"{int(chat_count_meta):,}"
    except (TypeError, ValueError):
        chat_count_display = str(chat_count_meta)
    try:
        user_count_display = f"{int(user_count_meta):,}"
    except (TypeError, ValueError):
        user_count_display = str(user_count_meta)
    dataset_ready = chat_count_meta > 0

    normalized_source_label = (dataset_source_label or "").strip()
    normalized_source_lower = normalized_source_label.lower()
    normalized_raw_source = (raw_dataset_source or "").strip()
    is_url_source = normalized_source_label.startswith(("http://", "https://"))
    is_openwebui_source = normalized_raw_source.startswith("openwebui:")
    is_file_upload_source = (
        normalized_source_lower == "local upload"
        or normalized_raw_source.startswith(("upload:", "default:", "json:", "chat export"))
    )
    if is_url_source or is_openwebui_source:
        dataset_source_type = "Direct Connect"
        dataset_source_detail = (
            normalized_source_label
            if is_url_source
            else normalized_raw_source.split(":", 1)[1].strip() if ":" in normalized_raw_source else normalized_source_label
        )
    elif is_file_upload_source and dataset_ready:
        dataset_source_type = "File Upload"
        if ":" in normalized_raw_source:
            dataset_source_detail = normalized_raw_source.split(":", 1)[1].strip()
        else:
            dataset_source_detail = normalized_source_label if normalized_source_lower not in {"local upload"} else ""
    elif dataset_ready and normalized_source_label and normalized_source_lower not in {"unknown source", ""}:
        dataset_source_type = normalized_source_label
        dataset_source_detail = ""
    else:
        dataset_source_type = "Not Loaded"
        dataset_source_detail = ""

    default_hostname = OPENWEBUI_DEFAULT_HOST
    prev_default_hostname = st.session_state.get("_openwebui_hostname_default")
    current_hostname = st.session_state.get("openwebui_hostname")
    if prev_default_hostname != default_hostname:
        if current_hostname in (None, "", prev_default_hostname):
            st.session_state["openwebui_hostname"] = default_hostname
        st.session_state["_openwebui_hostname_default"] = default_hostname
    else:
        st.session_state.setdefault("openwebui_hostname", default_hostname)

    default_api_key = OPENWEBUI_DEFAULT_API_KEY
    prev_default_api_key = st.session_state.get("_openwebui_api_key_default")
    current_api_key = st.session_state.get("openwebui_api_key")
    if prev_default_api_key != default_api_key:
        if current_api_key in (None, "", prev_default_api_key):
            st.session_state["openwebui_api_key"] = default_api_key
        st.session_state["_openwebui_api_key_default"] = default_api_key
    else:
        st.session_state.setdefault("openwebui_api_key", default_api_key)

    direct_connect_active = dataset_ready and dataset_source_type == "Direct Connect"
    direct_connect_label = "🔌 Direct Connect to Open WebUI"
    if direct_connect_active:
        direct_connect_label += " ✅"

    with st.expander(direct_connect_label, expanded=not dataset_ready):
        st.write(
            "Pull the latest chats and user records directly from an Open WebUI instance. "
            "Provide the base URL and an API key with sufficient permissions."
        )
        hostname = st.text_input(
            "Hostname",
            key="openwebui_hostname",
            help="Example: http://localhost:3000"
        )
        api_key = st.text_input(
            "API Key (Bearer token)",
            key="openwebui_api_key",
            type="password",
            help="Generate this token from Open WebUI settings."
        )
        go_clicked = st.button("Go", key="openwebui_sync_button")
        log_placeholder = st.empty()
        render_direct_connect_log(log_placeholder, context="directconnect")

        if go_clicked:
            trimmed_host = hostname.strip()
            if not trimmed_host:
                st.error("Hostname is required to connect to Open WebUI.")
            else:
                _reset_direct_connect_log(show=True)
                _append_direct_connect_log(f"Connecting to Open WebUI at {trimmed_host}...", "🔌")
                render_direct_connect_log(log_placeholder, context="directconnect")
                try:
                    sync_result = sync_openwebui_dataset(trimmed_host, api_key)
                except BackendError as exc:
                    _append_direct_connect_log(f"Failed to connect: {exc}", "❌")
                    render_direct_connect_log(log_placeholder, context="directconnect")
                    st.error(f"Failed to sync data: {exc}")
                else:
                    dataset_info = (sync_result or {}).get("dataset", {})
                    chat_count = dataset_info.get("chat_count", 0)
                    user_count = dataset_info.get("user_count", 0)
                    _append_direct_connect_log(f"Retrieved {chat_count} chats", "📥")
                    _append_direct_connect_log(f"Retrieved {user_count} users", "🙋")
                    _append_direct_connect_log("Summarizer job queued", "🤖")
                    render_direct_connect_log(log_placeholder, context="directconnect")

                    def _summary_callback(status: dict) -> None:
                        state = status.get("state")
                        total = status.get("total", 0)
                        completed = status.get("completed", 0)
                        if state == "running":
                            _set_direct_connect_progress(completed, total)
                        elif state in {"completed", "failed", "cancelled"}:
                            if total:
                                _set_direct_connect_progress(completed, total)
                            _reset_direct_connect_progress()
                        render_direct_connect_log(log_placeholder, context="directconnect")

                    load_and_process_data.clear()
                    load_user_data.clear()
                    _append_direct_connect_log("🧠 Building chat summaries...")
                    render_direct_connect_log(log_placeholder, context="directconnect")
                    status = wait_for_summary_completion(
                        "🧠 Building chat summaries...",
                        on_update=_summary_callback,
                        show_spinner=False,
                    )
                    state = status.get("state")
                    if state == "completed":
                        _append_direct_connect_log("Chat summaries rebuilt", "✅")
                        st.toast("Chat summaries rebuilt.", icon="🧠")
                    elif state == "failed":
                        message = status.get("message", "Unknown error")
                        _append_direct_connect_log(f"Summary job failed: {message}", "⚠️")
                        st.error(f"Summary job failed: {message}")
                    elif state == "cancelled":
                        _append_direct_connect_log("Summary job cancelled (dataset changed)", "⚠️")
                        st.warning("Summary job was cancelled because the dataset changed.")
                    elif state == "running":
                        _append_direct_connect_log("Summaries still running in the background", "ℹ️ ")
                        st.info("Summaries are still running in the background; new headlines will appear once complete.")

                    _append_direct_connect_log("Done", "✅")
                    st.session_state["direct_connect_log_expanded"] = False
                    render_direct_connect_log(log_placeholder, context="directconnect")
                    st.toast("Open WebUI data synced successfully!", icon="✅")
                    trigger_rerun()

    chat_upload_state = st.session_state.get("chat_file_uploader")
    chat_loaded = dataset_ready or (chat_upload_state is not None)
    file_upload_active = dataset_ready and dataset_source_type == "File Upload"
    expander_label = "📁 Upload Files"
    if file_upload_active:
        expander_label += " ✅"

    with st.expander(expander_label, expanded=not chat_loaded):
        st.write(
            "Load chat exports here to explore your conversations.\n\n"
            "**Download instructions**:\n"
            "- **Chats**: Admin Panel → Settings → Database → Export All Chats (All Users)\n"
            "- **Users**: Admin Panel → Settings → Database → Export Users\n\n"
            "Upload files directly below. Backend defaults can still be placed in the `/data` directory."
        )

        upload_col, users_col = st.columns([3, 2])
        with upload_col:
            st.subheader("Upload Chats")
            uploaded_file = st.file_uploader(
                "Select JSON file",
                type=['json'],
                help="Upload the JSON file exported from your Open WebUI instance",
                label_visibility="collapsed",
                key="chat_file_uploader"
            )
            if uploaded_file is not None:
                with st.spinner("📤 Uploading chat export..."):
                    try:
                        upload_chat_export(uploaded_file)
                    except BackendError as exc:
                        st.error(f"Failed to upload chat export: {exc}")
                    else:
                        load_user_data.clear()
                        status = wait_for_summary_completion("🧠 Building chat summaries...")
                        state = status.get("state")
                        if state == "completed":
                            st.toast("Chat summaries rebuilt.", icon="🧠")
                        elif state == "failed":
                            st.error(f"Summary job failed: {status.get('message', 'Unknown error')}")
                        elif state == "cancelled":
                            st.warning("Summary job was cancelled because the dataset changed.")
                        load_and_process_data.clear()
                        st.session_state.pop("chat_file_uploader", None)
                        st.toast("Chat export uploaded successfully!", icon="✅")
                        trigger_rerun()
        with users_col:
            st.subheader("Upload Users")
            uploaded_users_file = st.file_uploader(
                "Optional users CSV",
                type=['csv'],
                help="Upload a CSV with `user_id` and `name` columns to map chats to user names",
                label_visibility="collapsed",
                key="users_csv_uploader"
            )
            if uploaded_users_file is not None:
                with st.spinner("📤 Uploading users CSV..."):
                    try:
                        upload_users_csv(uploaded_users_file)
                    except BackendError as exc:
                        st.error(f"Failed to upload users CSV: {exc}")
                    else:
                        load_user_data.clear()
                        st.session_state.pop("users_csv_uploader", None)
                        st.toast("User data uploaded successfully!", icon="🧑")
                        trigger_rerun()
    if dataset_ready:
        with st.spinner("🔄 Processing chat data..."):
            chats_df, messages_df = load_and_process_data(dataset_id)

        if chats_df is not None and messages_df is not None:
            users_df = load_user_data(dataset_id)

            chats_df['user_id'] = chats_df['user_id'].fillna('').astype(str)
            fallback_ids = chats_df['user_id'].replace({'nan': '', 'None': ''})
            user_display_map = {}
            if not users_df.empty:
                users_df['user_id'] = users_df['user_id'].astype(str)
                user_display_map = dict(zip(users_df['user_id'], users_df['name']))
            chats_df['user_display'] = chats_df['user_id'].map(user_display_map)
            chats_df['user_display'] = chats_df['user_display'].fillna(fallback_ids)
            chats_df.loc[chats_df['user_display'].isin(['', 'nan', 'None']), 'user_display'] = "User"
            chat_user_display_map_all = chats_df.set_index('chat_id')['user_display'].to_dict()
            messages_df['chat_user_display'] = messages_df['chat_id'].map(chat_user_display_map_all)
            messages_df['chat_user_display'] = messages_df['chat_user_display'].fillna("User")
            messages_df['model'] = messages_df['model'].fillna('')

            dataset_signature = (
                dataset_id,
                len(chats_df),
                len(messages_df),
            )
            if st.session_state.get("dataset_signature") != dataset_signature:
                st.session_state.user_filter = ALL_USERS_OPTION
                st.session_state.model_filter = ALL_MODELS_OPTION
                st.session_state.page = 1
            st.session_state.dataset_signature = dataset_signature

            metrics = calculate_engagement_metrics(chats_df, messages_df)
            # Compute date range and total days from messages
            date_min_dt = messages_df['timestamp'].min() if not messages_df.empty else None
            date_max_dt = messages_df['timestamp'].max() if not messages_df.empty else None
            if date_min_dt is not None and date_max_dt is not None:
                date_min = date_min_dt.strftime('%m/%d')
                date_max = date_max_dt.strftime('%m/%d')
                total_days = (date_max_dt.date() - date_min_dt.date()).days + 1
                date_range = f"{date_min} - {date_max} ({total_days} days)"
            else:
                date_min = date_max = "N/A"
                total_days = 0
                date_range = "N/A"
            section_text_style = "color:#6b7280;font-size:0.85rem;line-height:1.35;"
            header_style = "font-weight:600;color:#1d4ed8;margin-bottom:0.25rem;"
            panel_wrapper_style = (
                "border:1px solid #bfdbfe;border-radius:12px;overflow:hidden;margin-bottom:1rem;"
            )
            panel_header_style = (
                "background-color:#f3f4f6;color:#374151;font-weight:600;font-size:1.05rem;"
                "padding:0.75rem 1.25rem;border-bottom:1px solid #bfdbfe;"
            )
            panel_body_style = (
                "background-color:#eaf2ff;padding:1rem 1.25rem;"
            )
            columns_wrapper_style = "display:flex;flex-wrap:wrap;gap:1.5rem;"
            column_style = f"flex:1;min-width:220px;{section_text_style}"

            detail_value = (dataset_source_detail or "").strip()
            connection_line = f"Connection Type: {html.escape(dataset_source_type)}"
            if dataset_source_type == "Direct Connect" and detail_value:
                link_target = (
                    detail_value
                    if detail_value.startswith(("http://", "https://"))
                    else f"https://{detail_value.lstrip('/')}"
                )
                safe_label = html.escape(detail_value)
                safe_href = html.escape(link_target, quote=True)
                connection_line = (
                    "Connection Type: Direct Connect "
                    f"(<a href=\"{safe_href}\" target=\"_blank\" rel=\"noopener noreferrer\">{safe_label}</a>)"
                )
                detail_value = ""
            source_lines = [
                f"<span style=\"{header_style}\">Data Source</span>",
                connection_line,
            ]
            if detail_value:
                escaped_detail = html.escape(detail_value)
                if dataset_source_type == "Direct Connect":
                    source_lines.append(escaped_detail)
                else:
                    source_lines.append(f"Source: {escaped_detail}")
            pulled_value = dataset_pulled_display
            if pulled_value and pulled_value not in {"N/A", "Unknown"}:
                relative_value = dataset_pulled_relative
                relative_suffix = f" ({html.escape(relative_value)})" if relative_value else ""
                source_lines.append(f"Last Pulled: {html.escape(pulled_value)}{relative_suffix}")
            source_content = "<br>".join(line for line in source_lines if line)

            chat_lines = [
                f"<span style=\"{header_style}\">Chats</span>",
                f"Uploaded: {html.escape(chat_uploaded_display)}",
                f"Count: {html.escape(chat_count_display)}",
            ]
            if range_display and range_display != "N/A":
                chat_lines.append(f"Range: {html.escape(range_display)}")
            chat_content = "<br>".join(line for line in chat_lines if line)

            user_lines = [
                f"<span style=\"{header_style}\">Users</span>",
                f"Uploaded: {html.escape(users_uploaded_display)}",
                f"Count: {html.escape(user_count_display)}",
            ]
            user_content = "<br>".join(line for line in user_lines if line)

            panel_html = (
                f"<div style=\"{panel_wrapper_style}\">"
                f"<div style=\"{panel_header_style}\">Loaded Data</div>"
                f"<div style=\"{panel_body_style}\">"
                f"<div style=\"{columns_wrapper_style}\">"
                f"<div style=\"{column_style}\">{source_content}</div>"
                f"<div style=\"{column_style}\">{chat_content}</div>"
                f"<div style=\"{column_style}\">{user_content}</div>"
                f"</div></div></div>"
            )
            st.markdown(panel_html, unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
            st.header("Overview")
            st.caption(f"Date Range: {date_min} - {date_max}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Days", value=total_days)
            with col2:
                st.metric(label="Total Chats", value=f"{metrics['total_chats']:,}")       
            with col3:
                st.metric(label="Unique Users", value=f"{metrics['unique_users']:,}")             
            with col4:
                files_uploaded = metrics.get('files_uploaded', 0)
                st.metric(label="User Files Uploaded", value=f"{files_uploaded:,}")  
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Avg Msgs/Chat", value=f"{metrics['avg_messages_per_chat']:,.1f}")          
            with col2:
                avg_input_tokens = metrics.get('avg_input_tokens_per_chat', 0)
                st.metric(label="Avg Input Tokens/Chat", value=f"{avg_input_tokens:,.1f}")
            with col3:
                st.metric(label="Avg Output Tokens/Chat", value=f"{metrics['avg_output_tokens_per_chat']:,.1f}")
            with col4:
                total_tokens = metrics.get('total_tokens', 0)
                st.metric(label="Total Tokens", value=f"{total_tokens:,}")                
            token_fig = create_token_consumption_chart(messages_df)
            if token_fig and len(token_fig.data) > 0:
                st.plotly_chart(token_fig, use_container_width=True, key="token_consumption_chart")
            else:
                st.info("Token consumption chart unavailable - insufficient timestamped data.")
            # Visual divider between overview and analysis
            st.markdown('---')

            st.subheader("Model Usage Analysis")
            model_data = messages_df[
                (messages_df['role'] == 'assistant') & (messages_df['model'] != '')
            ]
            if model_data.empty:
                st.warning("ℹ️ No model information available in the uploaded data")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    model_fig = create_model_usage_chart(messages_df)
                    st.plotly_chart(model_fig, use_container_width=True, key="model_usage_chart")
                with col2:
                    model_stats_fig = create_model_usage_pie(messages_df)
                    st.plotly_chart(model_stats_fig, use_container_width=True, key="model_usage_pie")

            st.markdown('---')

            st.subheader("User Analysis")
            top_users_df = compute_top_users(chats_df, messages_df, top_n=10)
            adoption_fig = create_user_adoption_chart(chats_df, messages_df, date_min_dt, date_max_dt)
            col_users, col_chart = st.columns([1.2, 1])
            with col_users:
                st.markdown("#### Top 10 Users")
                if top_users_df.empty:
                    st.info("No user activity found for analysis.")
                else:
                    display_df = top_users_df.rename(columns={
                        "user": "User",
                        "chat_count": "Chats",
                        "message_count": "Messages",
                        "token_percentage": "Token %"
                    })
                    display_df.index = display_df.index + 1
                    display_df.index.name = "Rank"
                    formatted_df = display_df.style.format({
                        "Chats": "{:,.0f}",
                        "Messages": "{:,.0f}",
                        "Token %": "{:.1f}%"
                    })
                    st.dataframe(formatted_df, use_container_width=True)
            with col_chart:
                st.markdown("#### User Adoption Over Time")
                if adoption_fig is None:
                    st.info("Need user messages with timestamps to plot adoption.")
                else:
                    st.plotly_chart(adoption_fig, use_container_width=True, key="user_adoption_chart")

            # Visual divider between model usage and model selection
            st.markdown('---')

            # Chat Analysis header + filter controls
            st.markdown('### Chat Analysis')
            user_display_lookup = (
                chats_df[['user_id', 'user_display']]
                .drop_duplicates(subset=['user_id'])
                .set_index('user_id')['user_display']
                .to_dict()
            )
            user_ids = [
                uid for uid in chats_df['user_id'].astype(str).unique().tolist()
                if uid not in ('', 'nan', 'None')
            ]
            user_ids_sorted = sorted(
                user_ids,
                key=lambda uid: str(user_display_lookup.get(uid) or uid or "User").lower()
            )
            user_options = [ALL_USERS_OPTION] + user_ids_sorted
            if "user_filter" not in st.session_state:
                st.session_state.user_filter = ALL_USERS_OPTION
            if st.session_state.user_filter not in user_options:
                st.session_state.user_filter = ALL_USERS_OPTION
            if "model_filter" not in st.session_state:
                st.session_state.model_filter = ALL_MODELS_OPTION

            col_user, col_model = st.columns(2)
            with col_user:
                user_index = user_options.index(st.session_state.user_filter)
                st.selectbox(
                    'Select user',
                    options=user_options,
                    index=user_index,
                    format_func=lambda uid: "All Users" if uid == ALL_USERS_OPTION else str(user_display_lookup.get(uid, uid)),
                    key='user_filter',
                    on_change=reset_model_filter
                )
            user_filter_value = st.session_state.user_filter
            user_filter = None if user_filter_value == ALL_USERS_OPTION else user_filter_value

            if user_filter is None:
                relevant_chats = chats_df.copy()
                relevant_messages = messages_df.copy()
            else:
                relevant_chats = chats_df[chats_df['user_id'] == user_filter].copy()
                relevant_chat_ids = relevant_chats['chat_id'].unique().tolist()
                relevant_messages = messages_df[messages_df['chat_id'].isin(relevant_chat_ids)].copy()

            model_options = [ALL_MODELS_OPTION] + sorted(
                [m for m in relevant_messages['model'].unique() if m]
            ) if not relevant_messages.empty else [ALL_MODELS_OPTION]

            if st.session_state.model_filter not in model_options:
                st.session_state.model_filter = ALL_MODELS_OPTION

            with col_model:
                model_index = model_options.index(st.session_state.model_filter)
                st.selectbox(
                    'Select model',
                    options=model_options,
                    index=model_index,
                    key='model_filter',
                    on_change=reset_browse_page
                )
            model_filter = st.session_state.model_filter

            filtered_messages = relevant_messages
            filtered_chats = relevant_chats

            if model_filter and model_filter != ALL_MODELS_OPTION:
                chat_ids_model = filtered_messages[filtered_messages['model'] == model_filter]['chat_id'].unique().tolist()
                if chat_ids_model:
                    filtered_messages = filtered_messages[filtered_messages['chat_id'].isin(chat_ids_model)].copy()
                    filtered_chats = filtered_chats[filtered_chats['chat_id'].isin(chat_ids_model)].copy()
                else:
                    filtered_messages = filtered_messages.iloc[0:0].copy()
                    filtered_chats = filtered_chats.iloc[0:0].copy()

            filtered_chat_user_map = filtered_chats.set_index('chat_id')['user_display'].to_dict() if not filtered_chats.empty else {}
            model_key = model_filter if model_filter else ALL_MODELS_OPTION
            user_key = "AllUsers" if user_filter_value == ALL_USERS_OPTION else user_filter
            raw_filter_key = f"{model_key}_{user_key}"
            filter_key = re.sub(r'\W+', '_', str(raw_filter_key))

            if "page" not in st.session_state:
                st.session_state.page = 1
            previous_filter_key = st.session_state.get("last_filter_key")
            if previous_filter_key != filter_key:
                st.session_state.page = 1
                st.session_state.last_filter_key = filter_key

            # Create tabs for different analyses (add a model-filtered Overview tab)
            tab_overview, tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🧾 Overview",
                "📈 Time Analysis",
                "💭 Content Analysis",
                "😊 Sentiment",
                "🔍 Search",
                "🗂 Browse Data"
            ])

            # Filtered Overview tab (uses the same metrics as the top overview but respects the model filter)
            with tab_overview:
                st.subheader("Overview (Filtered)")
                metrics_filtered = calculate_engagement_metrics(filtered_chats, filtered_messages)
                if not metrics_filtered:
                    st.info("No data available for the selected model / filter.")
                else:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(label="Total Messages", value=f"{metrics_filtered.get('total_messages', 0):,}")
                    with col2:
                        st.metric(label="Total Chats", value=f"{metrics_filtered.get('total_chats', 0):,}")
                    with col3:
                        st.metric(label="Unique Users", value=f"{metrics_filtered.get('unique_users', 0):,}")
                    with col4:
                        st.metric(label="User Files Uploaded", value=f"{metrics_filtered.get('files_uploaded', 0):,}")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(label="Avg Msgs/Chat", value=f"{metrics_filtered.get('avg_messages_per_chat', 0):,.1f}")
                    with col2:
                        st.metric(label="Avg Input Tokens/Chat", value=f"{metrics_filtered.get('avg_input_tokens_per_chat', 0):,.1f}")
                    with col3:
                        st.metric(label="Avg Output Tokens/Chat", value=f"{metrics_filtered.get('avg_output_tokens_per_chat', 0):,.1f}")
                    with col4:
                        st.metric(label="Total Tokens", value=f"{metrics_filtered.get('total_tokens', 0):,}")

            with tab1:
                st.subheader("Time-based Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    time_fig = create_time_series_chart(filtered_messages)
                    st.plotly_chart(time_fig, use_container_width=True, key=f"time_series_chart_{filter_key}")
                with col2:
                    length_fig = create_conversation_length_distribution(filtered_messages)
                    st.plotly_chart(length_fig, use_container_width=True, key=f"conversation_length_chart_{filter_key}")
                activity_fig = create_user_activity_chart(filtered_messages)
                st.plotly_chart(activity_fig, use_container_width=True, key=f"activity_heatmap_{filter_key}")
            with tab2:
                st.subheader("Content Analysis")
                wordcloud = generate_word_cloud(filtered_messages)
                if wordcloud:
                    col_left, col_wordcloud, col_right = st.columns([1, 6, 1])
                    with col_wordcloud:
                        st.markdown("### ☁️ Word Cloud (User Messages)")
                        img = wordcloud.to_image()
                        st.image(img, use_container_width=True)
                    col_left.empty()
                    col_right.empty()
                else:
                    # Provide a more actionable warning based on what data is present
                    if filtered_messages is None or filtered_messages.empty:
                        st.warning("⚠️ Word cloud unavailable - No messages available. Try uploading a JSON export or connect to an environment first.")
                    else:
                        # Check if there are any user messages with non-empty content
                        roles = filtered_messages.get('role')
                        contents = filtered_messages.get('content')
                        has_user = False
                        has_text = False
                        if roles is not None:
                            try:
                                has_user = any(roles.fillna('').astype(str).str.lower() == 'user')
                            except Exception:
                                has_user = False
                        if contents is not None:
                            try:
                                has_text = any(contents.astype(str).str.strip() != '')
                            except Exception:
                                has_text = False

                        if not has_text:
                            st.warning("⚠️ Word cloud unavailable - Messages are present but no textual content was found.")
                        elif not has_user:
                            st.warning("⚠️ Word cloud unavailable - No messages labeled with role 'user' were found. Check the data's 'role' field.")
                        else:
                            st.warning("⚠️ Word cloud unavailable - Text processing failed or no significant terms were extracted.")
                if not filtered_messages.empty:
                    filtered_messages['content_length'] = filtered_messages['content'].str.len()
                    col1, col2 = st.columns(2)
                    with col1:
                        # Use the filtered dataset here — content_length was added to filtered_messages
                        avg_length = filtered_messages.groupby('role')['content_length'].mean()
                        fig = px.bar(
                            x=avg_length.index,
                            y=avg_length.values,
                            title="Average Message Length by Role",
                            color=avg_length.index,
                            color_discrete_sequence=['#3b82f6', '#8b5cf6']
                        )
                        fig.update_layout(
                            template='plotly_white',
                            title_font_size=18,
                            title_font_color='#1f2937',
                            title_font_weight=600,
                            xaxis_title="Role",
                            yaxis_title="Average Characters",
                            font=dict(family="Inter, sans-serif"),
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=60, l=50, r=50, b=50)
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"avg_message_length_chart_{filter_key}")
                    with col2:
                        fig = px.histogram(
                            filtered_messages,
                            x='content_length',
                            title="Message Length Distribution",
                            color_discrete_sequence=['#3b82f6']
                        )
                        fig.update_layout(
                            template='plotly_white',
                            title_font_size=18,
                            title_font_color='#1f2937',
                            title_font_weight=600,
                            xaxis_title="Message Length (characters)",
                            yaxis_title="Number of Messages",
                            font=dict(family="Inter, sans-serif"),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=60, l=50, r=50, b=50)
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"message_length_hist_{filter_key}")
            with tab3:
                st.subheader("Sentiment Analysis")
                sentiment_df = perform_sentiment_analysis(filtered_messages)
                if not sentiment_df.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        sentiment_counts = sentiment_df['sentiment_category'].value_counts()
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution (User Messages)",
                            color_discrete_sequence=['#ef4444', '#6b7280', '#22c55e']
                        )
                        fig.update_layout(
                            template='plotly_white',
                            title_font_size=18,
                            title_font_color='#1f2937',
                            title_font_weight=600,
                            font=dict(family="Inter, sans-serif"),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=60, l=50, r=50, b=50)
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"sentiment_distribution_chart_{filter_key}")
                    with col2:
                        sentiment_time = sentiment_df.groupby(sentiment_df['timestamp'].dt.date)['sentiment'].mean().reset_index()
                        fig = px.line(
                            sentiment_time,
                            x='timestamp',
                            y='sentiment',
                            title="Average Sentiment Over Time"
                        )
                        fig.update_traces(line_color='#3b82f6', line_width=3)
                        fig.add_hline(y=0, line_dash="dash", line_color="#6b7280", annotation_text="Neutral")
                        fig.update_layout(
                            template='plotly_white',
                            title_font_size=18,
                            title_font_color='#1f2937',
                            title_font_weight=600,
                            xaxis_title="Date",
                            yaxis_title="Average Sentiment",
                            font=dict(family="Inter, sans-serif"),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=60, l=50, r=50, b=50)
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"sentiment_time_chart_{filter_key}")
                    st.markdown("### Sentiment Breakdown")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        positive_pct = (sentiment_counts.get('Positive', 0) / len(sentiment_df)) * 100
                        st.metric(label="Positive Messages", value=f"{positive_pct:.1f}%")
                    with col2:
                        neutral_pct = (sentiment_counts.get('Neutral', 0) / len(sentiment_df)) * 100
                        st.metric(label="Neutral Messages", value=f"{neutral_pct:.1f}%")
                    with col3:
                        negative_pct = (sentiment_counts.get('Negative', 0) / len(sentiment_df)) * 100
                        st.metric(label="Negative Messages", value=f"{negative_pct:.1f}%")
                else:
                    st.warning("⚠️ Sentiment analysis unavailable - No user messages found for analysis")
            with tab4:
                create_search_interface(
                    filtered_chats,
                    filtered_messages,
                    filtered_chat_user_map,
                    widget_prefix=filter_key
                )
            with tab5:
                st.subheader("Browse Data")
                chat_table = filtered_chats.copy()
                if chat_table.empty:
                    st.info("No chats found for the current filters.")
                else:
                    if filtered_messages.empty:
                        st.info("No messages to display.")
                    else:
                        chat_user_map = filtered_chats.set_index('chat_id')['user_display'].to_dict()
                        first_prompts = (
                            filtered_messages[filtered_messages['role'] == 'user']
                            .groupby('chat_id')['timestamp']
                            .min()
                            .reset_index()
                        )
                        first_prompts = first_prompts.merge(
                            filtered_chats[['chat_id', 'title']],
                            on='chat_id',
                            how='left'
                        )
                        first_prompts = first_prompts.sort_values('timestamp', ascending=False)
                        total_threads = len(first_prompts)
                        threads_per_page = st.selectbox(
                            "Threads per page",
                            [5, 10, 20, 50],
                            index=1,
                            key="threads_per_page"
                        )
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
                            thread_id = row['chat_id']
                            title = row['title'] or thread_id
                            date = row['timestamp'].strftime('%Y-%m-%d %H:%M')
                            chat_rows = filtered_chats[filtered_chats['chat_id'] == thread_id]
                            if chat_rows.empty:
                                continue
                            chat_info = chat_rows.iloc[0]
                            file_upload_flag = "📎" if chat_info.get('files_uploaded', 0) > 0 else ""
                            user_display = chat_user_map.get(thread_id, chat_info.get('user_id', 'User'))
                            with st.expander(f"{title} (Started: {date}) {file_upload_flag}", expanded=False):
                                render_chat_summary(str(chat_info.get('summary_128', "")))
                                thread_msgs = filtered_messages[filtered_messages['chat_id'] == thread_id].sort_values('timestamp')
                                for _, msg in thread_msgs.iterrows():
                                    timestamp = msg['timestamp'].strftime('%Y-%m-%d %H:%M')
                                    content = msg['content'].replace('\n', '<br>')
                                    if msg['role'] == 'user':
                                        st.markdown(
                                            f"<div style='background-color:#e6f7ff; padding:8px; border-radius:5px; margin-bottom:4px;'>"
                                            f"<strong>{user_display}</strong> <span style='color:#555;'>[{timestamp}]</span><br>{content}</div>",
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        st.markdown(
                                            f"<div style='background-color:#f0f0f0; padding:8px; border-radius:5px; margin-bottom:4px;'>"
                                            f"<strong>🤖 Assistant ({msg['model']})</strong> <span style='color:#555;'>[{timestamp}]</span><br>{content}</div>",
                                            unsafe_allow_html=True
                                        )
                                full_thread = {
                                    'chat_id': chat_info['chat_id'],
                                    'user_id': chat_info['user_id'],
                                    'user_name': user_display,
                                    'title': chat_info['title'],
                                    'created_at': chat_info['created_at'].isoformat(),
                                    'updated_at': chat_info['updated_at'].isoformat(),
                                    'archived': bool(chat_info['archived']),
                                    'pinned': bool(chat_info['pinned']),
                                    'tags': chat_info['tags'],
                                    'files': chat_info.get('files', []),
                                    'messages': json.loads(thread_msgs.to_json(orient='records', date_format='iso'))
                                }
                                thread_json = json.dumps(full_thread, indent=2)
                                st.download_button(
                                    label="📥 Download Thread (JSON)",
                                    data=thread_json,
                                    file_name=f"thread_{thread_id}.json",
                                    mime="application/json",
                                    key=f"browse_download_{thread_id}"
                                )
                                for file_item in chat_info.get('files', []):
                                    file_name = file_item.get('filename') or file_item.get('name')
                                    st.markdown(f"📎 {file_name}")
                        col_prev, col_info, col_next = st.columns([1, 2, 1])
                        with col_prev:
                            if st.button("Previous", key="prev_page") and st.session_state.page > 1:
                                st.session_state.page -= 1
                        with col_info:
                            st.write(f"Page {st.session_state.page} of {total_pages}")
                        with col_next:
                            if st.button("Next", key="next_page") and st.session_state.page < total_pages:
                                st.session_state.page += 1
            create_export_section(chats_df, messages_df)
    
    else:
        st.warning("Upload your Open WebUI JSON export file to begin analysis.")
        create_instructions()

if __name__ == "__main__":
    main()
