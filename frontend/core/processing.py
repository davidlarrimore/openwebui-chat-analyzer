from __future__ import annotations

import html
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from textblob import TextBlob

from .constants import ALL_MODELS_OPTION, ALL_USERS_OPTION
from .models import DatasetMeta, DatasetPanel, DatasetSourceInfo, SummaryStatus


def calculate_engagement_metrics(
    chats_df: pd.DataFrame,
    messages_df: pd.DataFrame,
) -> Dict[str, float]:
    """Calculate engagement metrics for the provided dataframes."""
    if chats_df is None or messages_df is None or messages_df.empty:
        return {}

    total_chats = len(chats_df)
    total_messages = len(messages_df)
    unique_users = chats_df["user_id"].nunique() if "user_id" in chats_df.columns else 0

    msg_per_chat = messages_df.groupby("chat_id").size().mean() if total_chats else 0

    user_messages = messages_df[messages_df["role"] == "user"]
    assistant_messages = messages_df[messages_df["role"] == "assistant"]

    conversations = messages_df.groupby("chat_id").size()
    avg_conversation_length = conversations.mean() if not conversations.empty else 0

    files_uploaded = chats_df.get("files_uploaded", pd.Series(dtype=float)).fillna(0).sum()

    assistant_msgs = assistant_messages.copy()
    assistant_msgs["token_count"] = assistant_msgs["content"].astype(str).str.len()
    output_tokens_per_chat = assistant_msgs.groupby("chat_id")["token_count"].sum()
    avg_output_tokens_per_chat = output_tokens_per_chat.mean() if not output_tokens_per_chat.empty else 0

    user_msgs = user_messages.copy()
    user_msgs["token_count"] = user_msgs["content"].astype(str).str.len()
    input_tokens_per_chat = user_msgs.groupby("chat_id")["token_count"].sum()
    avg_input_tokens_per_chat = input_tokens_per_chat.mean() if not input_tokens_per_chat.empty else 0

    total_input_tokens = input_tokens_per_chat.sum() if not input_tokens_per_chat.empty else 0
    total_output_tokens = output_tokens_per_chat.sum() if not output_tokens_per_chat.empty else 0

    return {
        "total_chats": total_chats,
        "total_messages": total_messages,
        "unique_users": unique_users,
        "avg_messages_per_chat": float(msg_per_chat or 0),
        "avg_conversation_length": float(avg_conversation_length or 0),
        "user_messages": len(user_messages),
        "assistant_messages": len(assistant_messages),
        "files_uploaded": float(files_uploaded or 0),
        "avg_input_tokens_per_chat": float(avg_input_tokens_per_chat or 0),
        "avg_output_tokens_per_chat": float(avg_output_tokens_per_chat or 0),
        "total_input_tokens": float(total_input_tokens or 0),
        "total_output_tokens": float(total_output_tokens or 0),
        "total_tokens": float(total_input_tokens + total_output_tokens),
    }


def compute_top_users(chats_df: pd.DataFrame, messages_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Aggregate metrics for the most active users."""
    if chats_df.empty or messages_df.empty:
        return pd.DataFrame(columns=["user", "chat_count", "message_count", "token_percentage"])

    chat_lookup = (
        chats_df[["chat_id", "user_display"]]
        .dropna(subset=["chat_id"])
        .drop_duplicates(subset=["chat_id"])
    )
    messages_with_users = messages_df.merge(chat_lookup, on="chat_id", how="left")
    messages_with_users["user_display"] = messages_with_users["user_display"].fillna("Unknown User")

    messages_with_users["token_count"] = messages_with_users["content"].astype(str).str.len()
    tokens_per_user = messages_with_users.groupby("user_display")["token_count"].sum()
    total_tokens = tokens_per_user.sum()

    chat_counts = (
        chats_df.groupby("user_display")["chat_id"]
        .nunique()
        .reindex(tokens_per_user.index, fill_value=0)
    )
    message_counts = messages_with_users.groupby("user_display").size()

    summary = pd.DataFrame(
        {
            "user": tokens_per_user.index,
            "chat_count": chat_counts.values,
            "message_count": message_counts.values,
            "token_percentage": np.where(
                total_tokens > 0,
                (tokens_per_user.values / total_tokens) * 100,
                0.0,
            ),
            "token_count": tokens_per_user.values,
        }
    )

    summary = (
        summary.sort_values(by=["token_count", "chat_count", "message_count"], ascending=False)
        .drop(columns="token_count")
        .head(top_n)
        .reset_index(drop=True)
    )

    return summary


def perform_sentiment_analysis(messages_df: pd.DataFrame) -> pd.DataFrame:
    """Compute TextBlob sentiment metrics for user messages."""
    if messages_df.empty:
        return pd.DataFrame()

    user_messages = messages_df[messages_df["role"] == "user"].copy()
    if user_messages.empty or "content" not in user_messages.columns:
        return pd.DataFrame()

    sentiments = []
    for content in user_messages["content"]:
        try:
            blob = TextBlob(str(content))
            sentiments.append(blob.sentiment.polarity)
        except Exception:
            sentiments.append(0.0)

    user_messages["sentiment"] = sentiments
    user_messages["sentiment_category"] = pd.cut(
        user_messages["sentiment"],
        bins=[-1, -0.1, 0.1, 1],
        labels=["Negative", "Neutral", "Positive"],
    )
    return user_messages


def format_timestamp(value: Optional[str]) -> str:
    """Format an ISO timestamp to a friendly string."""
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


def format_relative_time(value: Optional[str]) -> str:
    """Return a relative time delta such as '5 minutes ago'."""
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


def format_day(value: Optional[str]) -> str:
    """Return a YYYY-MM-DD representation of a date."""
    if not value:
        return "N/A"
    try:
        dt_value = pd.to_datetime(value)
    except Exception:
        return str(value)
    if pd.isna(dt_value):
        return "N/A"
    return dt_value.strftime("%Y-%m-%d")


def determine_dataset_source(meta: DatasetMeta) -> DatasetSourceInfo:
    """Determine dataset source label and detail string."""
    raw_source = meta.source or ""
    metadata = meta.app_metadata or {}
    dataset_source_label = metadata.get("dataset_source") or raw_source or "Unknown source"

    normalized_label = dataset_source_label.strip()
    normalized_lower = normalized_label.lower()
    normalized_raw_source = raw_source.strip()
    is_url_source = normalized_label.startswith(("http://", "https://"))
    is_openwebui_source = normalized_raw_source.startswith("openwebui:")
    is_file_upload_source = (
        normalized_lower == "local upload"
        or normalized_raw_source.startswith(("upload:", "default:", "json:", "chat export"))
    )

    if is_url_source or is_openwebui_source:
        detail = (
            normalized_label
            if is_url_source
            else normalized_raw_source.split(":", 1)[1].strip()
            if ":" in normalized_raw_source
            else normalized_label
        )
        return DatasetSourceInfo(label="Direct Connect", detail=detail)
    if is_file_upload_source and meta.chat_count > 0:
        if ":" in normalized_raw_source:
            detail = normalized_raw_source.split(":", 1)[1].strip()
        else:
            detail = "" if normalized_lower == "local upload" else normalized_label
        return DatasetSourceInfo(label="File Upload", detail=detail)
    if meta.chat_count > 0 and normalized_label and normalized_lower not in {"unknown source", ""}:
        return DatasetSourceInfo(label=normalized_label, detail="")
    return DatasetSourceInfo(label="Not Loaded", detail="")


def build_dataset_panel(meta: DatasetMeta) -> DatasetPanel:
    """Build HTML fragments for the dataset info panel."""
    metadata = meta.app_metadata or {}
    dataset_source_info = determine_dataset_source(meta)

    header_style = "font-weight:600;color:#1d4ed8;margin-bottom:0.25rem;"
    section_text_style = "color:#6b7280;font-size:0.85rem;line-height:1.35;"

    chat_uploaded_display = format_timestamp(metadata.get("chat_uploaded_at"))
    users_uploaded_display = format_timestamp(metadata.get("users_uploaded_at"))
    dataset_pulled_value = metadata.get("dataset_pulled_at")
    dataset_pulled_display = format_timestamp(dataset_pulled_value)
    dataset_pulled_relative = format_relative_time(dataset_pulled_value)
    first_day_display = format_day(metadata.get("first_chat_day"))
    last_day_display = format_day(metadata.get("last_chat_day"))

    if first_day_display == "N/A" and last_day_display == "N/A":
        range_display = "N/A"
    elif first_day_display != "N/A" and last_day_display != "N/A":
        range_display = f"{first_day_display} - {last_day_display}"
    else:
        range_display = first_day_display if last_day_display == "N/A" else last_day_display

    chat_count_meta = metadata.get("chat_count", meta.chat_count or 0)
    user_count_meta = metadata.get("user_count", meta.user_count or 0)
    try:
        chat_count_display = f"{int(chat_count_meta):,}"
    except (TypeError, ValueError):
        chat_count_display = str(chat_count_meta)
    try:
        user_count_display = f"{int(user_count_meta):,}"
    except (TypeError, ValueError):
        user_count_display = str(user_count_meta)

    connection_line = f"Connection Type: {html.escape(dataset_source_info.label)}"
    detail_value = (dataset_source_info.detail or "").strip()
    if dataset_source_info.label == "Direct Connect" and detail_value:
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
        if dataset_source_info.label == "Direct Connect":
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

    return DatasetPanel(
        source_html=f"<div style=\"{section_text_style}\">{source_content}</div>",
        chat_html=f"<div style=\"{section_text_style}\">{chat_content}</div>",
        user_html=f"<div style=\"{section_text_style}\">{user_content}</div>",
    )


def compute_date_summary(messages_df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Return min/max date strings and total days for the dataset."""
    if messages_df is None or messages_df.empty or "timestamp" not in messages_df.columns:
        return {"date_min": "N/A", "date_max": "N/A", "total_days": 0, "date_range": "N/A"}

    ts_series = messages_df["timestamp"].dropna()
    if ts_series.empty:
        return {"date_min": "N/A", "date_max": "N/A", "total_days": 0, "date_range": "N/A"}

    date_min_dt = ts_series.min()
    date_max_dt = ts_series.max()
    date_min = date_min_dt.strftime("%m/%d")
    date_max = date_max_dt.strftime("%m/%d")
    total_days = (date_max_dt.date() - date_min_dt.date()).days + 1
    date_range = f"{date_min} - {date_max} ({total_days} days)"
    return {
        "date_min": date_min,
        "date_max": date_max,
        "total_days": total_days,
        "date_range": date_range,
        "date_min_ts": date_min_dt,
        "date_max_ts": date_max_dt,
    }


def filter_dataframes_by_user_model(
    chats_df: pd.DataFrame,
    messages_df: pd.DataFrame,
    *,
    user_id: Optional[str],
    model_name: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return chats/messages filtered by the selected user and model."""
    relevant_chats = chats_df.copy()
    relevant_messages = messages_df.copy()

    if user_id and user_id != ALL_USERS_OPTION:
        relevant_chats = relevant_chats[relevant_chats["user_id"] == user_id].copy()
        relevant_chat_ids = relevant_chats["chat_id"].unique().tolist()
        relevant_messages = relevant_messages[relevant_messages["chat_id"].isin(relevant_chat_ids)].copy()

    if model_name and model_name != ALL_MODELS_OPTION:
        chat_ids_model = relevant_messages[relevant_messages["model"] == model_name]["chat_id"].unique().tolist()
        if chat_ids_model:
            relevant_messages = relevant_messages[relevant_messages["chat_id"].isin(chat_ids_model)].copy()
            relevant_chats = relevant_chats[relevant_chats["chat_id"].isin(chat_ids_model)].copy()
        else:
            relevant_messages = relevant_messages.iloc[0:0].copy()
            relevant_chats = relevant_chats.iloc[0:0].copy()

    return relevant_chats, relevant_messages


def build_user_options(chats_df: pd.DataFrame) -> Tuple[Iterable[str], Dict[str, str]]:
    """Return user options and lookup map for the filter control."""
    if chats_df.empty:
        return [ALL_USERS_OPTION], {}

    user_display_lookup = (
        chats_df[["user_id", "user_display"]]
        .drop_duplicates(subset=["user_id"])
        .set_index("user_id")["user_display"]
        .to_dict()
    )
    user_ids = [
        uid for uid in chats_df["user_id"].astype(str).unique().tolist() if uid not in ("", "nan", "None")
    ]
    user_ids_sorted = sorted(user_ids, key=lambda uid: str(user_display_lookup.get(uid) or uid or "User").lower())
    user_options = [ALL_USERS_OPTION] + user_ids_sorted
    return user_options, user_display_lookup


def build_model_options(messages_df: pd.DataFrame) -> Iterable[str]:
    """Return model options for the filter control."""
    if messages_df.empty or "model" not in messages_df.columns:
        return [ALL_MODELS_OPTION]
    models = sorted({m for m in messages_df["model"].unique() if m})
    return [ALL_MODELS_OPTION, *models]


def summarize_summary_status(status: SummaryStatus) -> Optional[Tuple[str, str]]:
    """Return a (message, icon) tuple for toast notifications."""
    if not status:
        return None

    state = status.state
    if state == "running":
        message = f"Summaries in progress ({status.completed}/{status.total})"
        return message, "🧠"
    if state == "failed":
        detail = status.message or "Unknown error"
        return f"Summary job failed: {detail}", "⚠️"
    if state == "cancelled":
        return "Summary job was cancelled because the dataset changed before completion.", "⚠️"
    if state == "completed" and status.total:
        return "Chat summaries are up to date.", "🧠"
    return None
