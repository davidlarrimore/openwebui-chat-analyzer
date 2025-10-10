#!/usr/bin/env python3
"""
Open WebUI Chat Data Analyzer
A modern Streamlit dashboard for analyzing Open WebUI chat export

Installation:
pip install streamlit pandas plotly wordcloud textblob networkx

Usage:
streamlit run openwebui_chat_analyzer.py
"""

import streamlit as st
import pandas as pd
import json
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import math
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import re
from textblob import TextBlob
from wordcloud import WordCloud
import networkx as nx
import base64
from io import BytesIO
from pathlib import Path

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


def _read_file_source_bytes(file_source):
    """Return raw bytes for multiple supported file input types."""
    if file_source is None:
        return None
    if isinstance(file_source, Path):
        return file_source.read_bytes()
    if isinstance(file_source, str):
        return Path(file_source).read_bytes()
    if isinstance(file_source, bytes):
        return file_source
    if hasattr(file_source, "getvalue"):
        raw = file_source.getvalue()
        return raw if isinstance(raw, bytes) else raw.encode("utf-8")
    if hasattr(file_source, "read"):
        raw = file_source.read()
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        return raw if isinstance(raw, bytes) else raw.encode("utf-8")
    raise ValueError("Unsupported file source type")


@st.cache_data
def _parse_chat_export(raw_bytes):
    """Parse raw chat export bytes into chat and message DataFrames."""
    if raw_bytes is None:
        raise ValueError("No data found in provided file source")

    text = raw_bytes.decode("utf-8")
    data = json.loads(text)

    # Initialize lists for different data types
    chats = []
    messages = []

    # Process each chat
    for item in data:
        chat_info = {
            'chat_id': item.get('id', ''),
            'user_id': item.get('user_id', ''),
            'title': item.get('title', ''),
            'created_at': pd.to_datetime(item.get('created_at', 0), unit='s'),
            'updated_at': pd.to_datetime(item.get('updated_at', 0), unit='s'),
            'archived': item.get('archived', False),
            'pinned': item.get('pinned', False),
            'tags': item.get('meta', {}).get('tags', []),
            'files_uploaded': len(item.get('chat', {}).get('files', [])),
            'files': item.get('chat', {}).get('files', []),
        }
        chats.append(chat_info)

        # Process messages in this chat
        chat_data = item.get('chat', {})
        chat_messages = chat_data.get('messages', [])

        for msg in chat_messages:
            message_info = {
                'chat_id': item.get('id', ''),
                'message_id': msg.get('id', ''),
                'parent_id': msg.get('parentId'),
                'role': msg.get('role', ''),
                'content': msg.get('content', ''),
                'timestamp': pd.to_datetime(msg.get('timestamp', 0), unit='s'),
                'model': msg.get('model', ''),
                'models': msg.get('models', [])
            }
            messages.append(message_info)

    # Create DataFrames
    chats_df = pd.DataFrame(chats)
    messages_df = pd.DataFrame(messages)

    return chats_df, messages_df


def load_and_process_data(file_source):
    """Load and process Open WebUI JSON data."""
    try:
        raw_bytes = _read_file_source_bytes(file_source)
        return _parse_chat_export(raw_bytes)
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None, None


ALL_USERS_OPTION = "__ALL_USERS__"
ALL_MODELS_OPTION = "All Models"

def find_default_chat_export():
    """Locate the most recent default chat export in the data directory."""
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        return None

    exports = sorted(
        data_dir.glob("all-chats-export*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return exports[0] if exports else None

@st.cache_data
def _parse_user_csv(raw_bytes):
    """Parse raw CSV bytes into a normalized users DataFrame."""
    if raw_bytes is None:
        raise ValueError("No data found in provided CSV source")

    csv_buffer = BytesIO(raw_bytes)
    users_df = pd.read_csv(csv_buffer)
    if users_df.empty:
        return pd.DataFrame(columns=["user_id", "name"])

    users_df.columns = [str(col).strip().lower() for col in users_df.columns]
    id_col = None
    for candidate in ("user_id", "id"):
        if candidate in users_df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError("CSV must contain a 'user_id' or 'id' column")

    name_col = None
    for candidate in ("name", "full_name", "display_name", "username"):
        if candidate in users_df.columns:
            name_col = candidate
            break

    if name_col is None:
        raise ValueError("CSV must contain a column with user names (e.g. 'name')")

    users_df = users_df[[id_col, name_col]].rename(columns={id_col: "user_id", name_col: "name"})
    users_df = users_df.dropna(subset=["user_id"]).drop_duplicates(subset=["user_id"], keep="first")
    users_df["user_id"] = users_df["user_id"].astype(str)
    users_df["name"] = users_df["name"].astype(str).str.strip()
    return users_df


def load_user_data(file_source):
    """Load optional user metadata from CSV."""
    if file_source is None:
        return pd.DataFrame(columns=["user_id", "name"])

    try:
        raw_bytes = _read_file_source_bytes(file_source)
        return _parse_user_csv(raw_bytes)
    except Exception as e:
        st.warning(f"Unable to load user CSV: {e}")
        return pd.DataFrame(columns=["user_id", "name"])


def find_default_users_file():
    """Locate a default users.csv file in the data directory if present."""
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        return None
    users_file = data_dir / "users.csv"
    return users_file if users_file.exists() else None


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

    return summary.head(top_n)

def create_user_adoption_chart(chats_df, messages_df, date_min=None, date_max=None):
    """Create a cumulative user adoption chart based on first user message."""
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

    first_message_dates = (
        user_messages.groupby("user_display")["timestamp"]
        .min()
        .dropna()
        .reset_index()
        .sort_values("timestamp")
    )
    if first_message_dates.empty:
        return None

    first_message_dates["cumulative_users"] = np.arange(1, len(first_message_dates) + 1)

    fig = px.line(
        first_message_dates,
        x="timestamp",
        y="cumulative_users",
        title="User Adoption Over Time",
        markers=True
    )
    fig.update_traces(line_color="#10b981", line_width=3)
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        title_font_size=18,
        title_font_color="#1f2937",
        title_font_weight=600,
        xaxis_title="First Message Date",
        yaxis_title="Cumulative Users",
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, l=50, r=50, b=50)
    )

    if date_min is not None and date_max is not None:
        fig.update_xaxes(range=[date_min, date_max])

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
    
    default_export_path = find_default_chat_export()
    default_users_path = find_default_users_file()

    chat_upload_state = st.session_state.get("chat_file_uploader")
    chat_loaded = (default_export_path is not None) or (chat_upload_state is not None)
    expander_label = "📁 Upload Files"
    if chat_loaded:
        expander_label = "📁 Upload Files ✅"

    with st.expander(expander_label, expanded=not chat_loaded):
        st.write(
            "Load chat exports here to explore your conversations.\n\n"
            "**Download instructions**:\n"
            "- **Chats**: Admin Panel → Settings → Database → Export All Chats (All Users)\n"
            "- **Users**: Admin Panel → Settings → Database → Export Users\n\n"
            "To auto-load on startup, place the downloaded files in the `/data` directory, "
            "or upload them directly using the inputs below."
        )

        upload_col, users_col = st.columns([3, 2])
        with upload_col:
            chat_has_file = (default_export_path is not None) or (st.session_state.get("chat_file_uploader") is not None)
            chat_subheader = "Upload Chats ✅" if chat_has_file else "Upload Chats"
            st.subheader(chat_subheader)
            uploaded_file = st.file_uploader(
                "Select JSON file",
                type=['json'],
                help="Upload the JSON file exported from your Open WebUI instance",
                label_visibility="collapsed",
                key="chat_file_uploader"
            )
            if uploaded_file is None and default_export_path is not None:
                st.info(f"Loaded default export from `data/{default_export_path.name}`")
        with users_col:
            users_has_file = (default_users_path is not None) or (st.session_state.get("users_csv_uploader") is not None)
            user_subheader = "Upload Users ✅" if users_has_file else "Upload Users"
            st.subheader(user_subheader)
            uploaded_users_file = st.file_uploader(
                "Optional users CSV",
                type=['csv'],
                help="Upload a CSV with `user_id` and `name` columns to map chats to user names",
                label_visibility="collapsed",
                key="users_csv_uploader"
            )
            if uploaded_users_file is None and default_users_path is not None:
                st.info(f"Loaded user directory from `data/{default_users_path.name}`")

    data_source = None
    data_label = ""
    loaded_from_default = False
    user_data_source = None
    user_data_label = ""
    user_loaded_from_default = False

    if uploaded_file is not None:
        data_source = uploaded_file
        data_label = getattr(uploaded_file, "name", "uploaded file")
    elif default_export_path is not None:
        data_source = default_export_path
        data_label = default_export_path.name
        loaded_from_default = True

    if uploaded_users_file is not None:
        user_data_source = uploaded_users_file
        user_data_label = getattr(uploaded_users_file, "name", "uploaded users file")
    elif default_users_path is not None:
        user_data_source = default_users_path
        user_data_label = default_users_path.name
        user_loaded_from_default = True
    
    if data_source is not None:
        # Load and process data with enhanced loading state
        with st.spinner("🔄 Processing chat data..."):
            chats_df, messages_df = load_and_process_data(data_source)
        
        if chats_df is not None and messages_df is not None:
            users_df = load_user_data(user_data_source)
            if user_data_source is not None and not users_df.empty:
                if not user_loaded_from_default:
                    st.toast("User data loaded successfully!", icon="🧑")
            elif user_data_source is not None and users_df.empty:
                st.warning("User CSV was provided but no usable records were found.")

            st.toast(f"Data loaded successfully!", icon="✅")

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
                data_label or "",
                len(chats_df),
                len(messages_df)
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

            model_data = messages_df[
                (messages_df['role'] == 'assistant') & (messages_df['model'] != '')
            ]
            if model_data.empty:
                st.warning("ℹ️ No model information available in the uploaded data")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Model Usage Analysis")
                    model_fig = create_model_usage_chart(messages_df)
                    st.plotly_chart(model_fig, use_container_width=True, key="model_usage_chart")
                with col2:
                    st.subheader("Model Statistics")
                    model_stats_fig = create_model_usage_pie(messages_df)
                    st.plotly_chart(model_stats_fig, use_container_width=True, key="model_usage_pie")

            st.subheader("User Analysis")
            top_users_df = compute_top_users(chats_df, messages_df, top_n=10)
            adoption_fig = create_user_adoption_chart(chats_df, messages_df, date_min_dt, date_max_dt)
            col_users, col_chart = st.columns([1.2, 1])
            with col_users:
                if top_users_df.empty:
                    st.info("No user activity found for analysis.")
                else:
                    display_df = top_users_df.rename(columns={
                        "user": "User",
                        "chat_count": "Chats",
                        "message_count": "Messages",
                        "token_percentage": "Token %"
                    })
                    formatted_df = display_df.style.format({
                        "Chats": "{:,.0f}",
                        "Messages": "{:,.0f}",
                        "Token %": "{:.1f}%"
                    })
                    st.dataframe(formatted_df, use_container_width=True)
            with col_chart:
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
                if not filtered_messages.empty:
                    # Compute the first user message timestamp per chat
                    chat_user_map = filtered_chats.set_index('chat_id')['user_display'].to_dict()
                    first_prompts = (
                        filtered_messages[filtered_messages['role'] == 'user']
                        .groupby('chat_id')['timestamp']
                        .min()
                        .reset_index()
                    )
                    # Merge with chat titles
                    first_prompts = first_prompts.merge(
                        filtered_chats[['chat_id', 'title']],
                        on='chat_id',
                        how='left'
                    )
                    # Sort descending by first prompt date
                    first_prompts = first_prompts.sort_values('timestamp', ascending=False)
                    # Pagination setup
                    total_threads = len(first_prompts)
                    # Threads per page selector
                    threads_per_page = st.selectbox(
                        "Threads per page",
                        [5, 10, 20, 50],
                        index=1,
                        key="threads_per_page"
                    )
                    # Calculate total pages and slice for current page
                    total_pages = math.ceil(total_threads / threads_per_page) if total_threads else 1
                    if st.session_state.page > total_pages:
                        st.session_state.page = total_pages
                    if st.session_state.page < 1:
                        st.session_state.page = 1
                    page = st.session_state.page
                    start_idx = (page - 1) * threads_per_page
                    end_idx = start_idx + threads_per_page
                    first_prompts = first_prompts.iloc[start_idx:end_idx]
                    # Display each thread in an expander
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
                            # List attachments
                            for file_item in chat_info.get('files', []):
                                file_name = file_item.get('filename') or file_item.get('name')
                                st.markdown(f"📎 {file_name}")
                    # Page navigation controls
                    col_prev, col_info, col_next = st.columns([1, 2, 1])
                    with col_prev:
                        if st.button("Previous", key="prev_page") and st.session_state.page > 1:
                            st.session_state.page -= 1
                    with col_info:
                        st.write(f"Page {st.session_state.page} of {total_pages}")
                    with col_next:
                        if st.button("Next", key="next_page") and st.session_state.page < total_pages:
                            st.session_state.page += 1
                else:
                    st.info("No messages to display.")
            create_export_section(chats_df, messages_df)
    
    else:
        if default_export_path is None:
            st.warning("Please upload your Open WebUI JSON export file to begin analysis")
        else:
            st.error("Unable to load chat data.")
        create_instructions()

if __name__ == "__main__":
    main()
