from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud


def create_time_series_chart(messages_df: pd.DataFrame) -> go.Figure:
    """Create a daily message volume chart."""
    if messages_df.empty or "timestamp" not in messages_df.columns:
        return go.Figure()

    daily_counts = (
        messages_df.dropna(subset=["timestamp"])
        .copy()
        .assign(date=lambda df: df["timestamp"].dt.normalize())
        .groupby("date")
        .size()
        .rename("message_count")
        .reset_index()
    )

    if daily_counts.empty:
        return go.Figure()

    tz_info = daily_counts["date"].dt.tz
    date_min = daily_counts["date"].min()
    date_max = daily_counts["date"].max()
    date_range = pd.date_range(date_min, date_max, freq="D", tz=tz_info)
    daily_counts = (
        daily_counts.set_index("date")
        .reindex(date_range, fill_value=0)
        .rename_axis("date")
        .reset_index()
        .rename(columns={"index": "date"})
    )

    fig = px.area(
        daily_counts,
        x="date",
        y="message_count",
        title="Messages Over Time",
    )
    fig.update_traces(line_color="#2563eb")
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        title_font_size=18,
        title_font_color="#1f2937",
        title_font_weight=600,
        xaxis_title="Date",
        yaxis_title="Messages",
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, l=50, r=50, b=50),
    )
    fig.update_xaxes(tickformat="%Y-%m-%d", dtick=86400000.0)
    return fig


def create_user_activity_chart(messages_df: pd.DataFrame) -> go.Figure:
    """Create heatmap of messages by weekday/hour."""
    if messages_df.empty or "timestamp" not in messages_df.columns:
        return go.Figure()

    active_messages = messages_df.dropna(subset=["timestamp"]).copy()
    if active_messages.empty:
        return go.Figure()

    active_messages["weekday"] = active_messages["timestamp"].dt.day_name()
    active_messages["hour"] = active_messages["timestamp"].dt.hour
    ordered_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    active_messages["weekday"] = pd.Categorical(active_messages["weekday"], categories=ordered_weekdays, ordered=True)

    heatmap_data = (
        active_messages.groupby(["weekday", "hour"])
        .size()
        .reset_index(name="message_count")
    )
    pivot = heatmap_data.pivot(index="weekday", columns="hour", values="message_count").fillna(0)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Blues",
            colorbar=dict(title="Messages"),
        )
    )
    fig.update_layout(
        title="Activity Heatmap (by Hour and Day)",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        template="plotly_white",
        title_font_size=18,
        title_font_color="#1f2937",
        title_font_weight=600,
        margin=dict(t=60, l=80, r=40, b=40),
    )
    return fig


def create_model_usage_chart(messages_df: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart of model usage."""
    if messages_df.empty:
        return go.Figure()

    assistant_msgs = messages_df[
        (messages_df["role"] == "assistant") & (messages_df["model"] != "")
    ]
    if assistant_msgs.empty:
        return go.Figure()

    model_counts = assistant_msgs["model"].value_counts().sort_values(ascending=True)
    fig = go.Figure(
        data=go.Bar(
            x=model_counts.values,
            y=model_counts.index,
            orientation="h",
            marker=dict(color="#6366f1"),
        )
    )
    fig.update_layout(
        title="Assistant Responses by Model",
        xaxis_title="Messages",
        yaxis_title="Model",
        template="plotly_white",
        title_font_size=18,
        title_font_color="#1f2937",
        title_font_weight=600,
        margin=dict(t=60, l=120, r=40, b=40),
    )
    return fig


def create_model_usage_pie(messages_df: pd.DataFrame) -> go.Figure:
    """Create pie chart of model usage."""
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
        title="Model Usage Share",
    )
    fig.update_layout(
        template="plotly_white",
        title_font_size=18,
        title_font_color="#1f2937",
        title_font_weight=600,
        margin=dict(t=60, l=20, r=20, b=20),
        showlegend=True,
        font=dict(family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
    )
    fig.update_traces(textinfo="percent+label")
    return fig


def create_token_consumption_chart(messages_df: pd.DataFrame) -> go.Figure:
    """Create total token consumption chart (approx by characters)."""
    if messages_df.empty or "timestamp" not in messages_df.columns:
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
        tz=tz_info,
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
        markers=True,
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
        height=260,
    )
    fig.update_xaxes(tickformat="%Y-%m-%d", tickmode="linear", dtick=86400000.0)
    return fig


def create_user_adoption_chart(
    chats_df: pd.DataFrame,
    messages_df: pd.DataFrame,
    date_min: Optional[pd.Timestamp] = None,
    date_max: Optional[pd.Timestamp] = None,
) -> Optional[go.Figure]:
    """Create a cumulative user adoption chart."""
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

    user_messages["timestamp"] = pd.to_datetime(user_messages["timestamp"], errors="coerce")
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
        date_max = pd.to_datetime(date_max)
    if date_min is not None:
        date_min = pd.to_datetime(date_min)

    full_range = pd.date_range(
        start=start_date if date_min is None else min(start_date, date_min),
        end=date_max or daily_new_users["first_date"].max(),
        freq="D",
    )
    daily_new_users = (
        daily_new_users.set_index("first_date")
        .reindex(full_range, fill_value=0)
        .rename_axis("date")
        .reset_index()
        .rename(columns={"index": "date"})
    )
    daily_new_users["cumulative_users"] = daily_new_users["new_users"].cumsum()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_new_users["date"],
            y=daily_new_users["cumulative_users"],
            mode="lines+markers",
            line=dict(color="#10b981", width=3),
            name="Cumulative Users",
        )
    )
    fig.update_layout(
        title="Cumulative User Adoption",
        xaxis_title="Date",
        yaxis_title="Cumulative Users",
        template="plotly_white",
        hovermode="x unified",
        title_font_size=18,
        title_font_color="#1f2937",
        title_font_weight=600,
        margin=dict(t=60, l=60, r=40, b=40),
    )
    fig.update_xaxes(tickformat="%Y-%m-%d")
    return fig


def create_conversation_length_distribution(messages_df: pd.DataFrame) -> go.Figure:
    """Create histogram of conversation lengths."""
    if messages_df.empty:
        return go.Figure()

    conversation_lengths = (
        messages_df.groupby("chat_id")
        .size()
        .reset_index(name="message_count")
        .sort_values("message_count", ascending=False)
    )
    if conversation_lengths.empty:
        return go.Figure()

    fig = px.histogram(
        conversation_lengths,
        x="message_count",
        nbins=20,
        title="Conversation Length Distribution",
    )
    fig.update_layout(
        template="plotly_white",
        title_font_size=18,
        title_font_color="#1f2937",
        title_font_weight=600,
        xaxis_title="Messages per Chat",
        yaxis_title="Chats",
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, l=50, r=50, b=50),
    )
    return fig


def generate_word_cloud(messages_df: pd.DataFrame) -> Optional[WordCloud]:
    """Generate a WordCloud for user messages."""
    if messages_df.empty:
        return None

    user_messages = messages_df[messages_df["role"] == "user"]
    if user_messages.empty or "content" not in user_messages.columns:
        return None

    text = " ".join(user_messages["content"].astype(str))
    if not text.strip():
        return None

    return WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
    ).generate(text)


def create_average_message_length_chart(messages_df: pd.DataFrame) -> go.Figure:
    """Average message length by role."""
    if messages_df.empty:
        return go.Figure()

    temp_df = messages_df.copy()
    temp_df["content_length"] = temp_df["content"].astype(str).str.len()
    avg_length = temp_df.groupby("role")["content_length"].mean()
    if avg_length.empty:
        return go.Figure()

    fig = px.bar(
        x=avg_length.index,
        y=avg_length.values,
        title="Average Message Length by Role",
        color=avg_length.index,
        color_discrete_sequence=["#3b82f6", "#8b5cf6"],
    )
    fig.update_layout(
        template="plotly_white",
        title_font_size=18,
        title_font_color="#1f2937",
        title_font_weight=600,
        xaxis_title="Role",
        yaxis_title="Average Characters",
        font=dict(family="Inter, sans-serif"),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, l=50, r=50, b=50),
    )
    return fig


def create_message_length_histogram(messages_df: pd.DataFrame) -> go.Figure:
    """Overall message length histogram."""
    if messages_df.empty:
        return go.Figure()

    temp_df = messages_df.copy()
    temp_df["content_length"] = temp_df["content"].astype(str).str.len()
    fig = px.histogram(
        temp_df,
        x="content_length",
        title="Message Length Distribution",
        color_discrete_sequence=["#3b82f6"],
    )
    fig.update_layout(
        template="plotly_white",
        title_font_size=18,
        title_font_color="#1f2937",
        title_font_weight=600,
        xaxis_title="Message Length (characters)",
        yaxis_title="Number of Messages",
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, l=50, r=50, b=50),
    )
    return fig


def create_sentiment_distribution_chart(sentiment_df: pd.DataFrame) -> go.Figure:
    """Pie chart for sentiment distribution."""
    if sentiment_df.empty or "sentiment_category" not in sentiment_df.columns:
        return go.Figure()

    sentiment_counts = sentiment_df["sentiment_category"].value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution (User Messages)",
        color_discrete_sequence=["#ef4444", "#6b7280", "#22c55e"],
    )
    fig.update_layout(
        template="plotly_white",
        title_font_size=18,
        title_font_color="#1f2937",
        title_font_weight=600,
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, l=50, r=50, b=50),
    )
    return fig


def create_sentiment_time_series(sentiment_df: pd.DataFrame) -> go.Figure:
    """Average sentiment over time."""
    if sentiment_df.empty or "timestamp" not in sentiment_df.columns:
        return go.Figure()

    sentiment_time = sentiment_df.groupby(sentiment_df["timestamp"].dt.date)["sentiment"].mean().reset_index()
    if sentiment_time.empty:
        return go.Figure()

    fig = px.line(
        sentiment_time,
        x="timestamp",
        y="sentiment",
        title="Average Sentiment Over Time",
    )
    fig.update_traces(line_color="#3b82f6", line_width=3)
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280", annotation_text="Neutral")
    fig.update_layout(
        template="plotly_white",
        title_font_size=18,
        title_font_color="#1f2937",
        title_font_weight=600,
        xaxis_title="Date",
        yaxis_title="Average Sentiment",
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, l=50, r=50, b=50),
    )
    return fig
