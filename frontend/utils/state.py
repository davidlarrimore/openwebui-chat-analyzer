from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from frontend.core.constants import ALL_MODELS_OPTION, ALL_USERS_OPTION
from frontend.core.models import DatasetMeta, ProcessedData
from frontend.core.timezone import now_in_display_timezone


DIRECT_CONNECT_LOG_KEY = "direct_connect_log"
DIRECT_CONNECT_PROGRESS_INDEX_KEY = "direct_connect_progress_index"
DIRECT_CONNECT_EVENT_IDS_KEY = "direct_connect_seen_events"
UPLOAD_LOG_KEY = "upload_log"
UPLOAD_PROGRESS_INDEX_KEY = "upload_progress_index"
UPLOAD_EVENT_IDS_KEY = "upload_seen_events"
ADMIN_LOG_KEY = "admin_action_log"
ADMIN_PROGRESS_INDEX_KEY = "admin_action_progress_index"
ADMIN_EVENT_IDS_KEY = "admin_action_seen_events"
PROCESSING_LOG_SOURCE_KEY = "processing_log_source"


def trigger_rerun() -> None:
    """Trigger a Streamlit rerun using the most compatible API."""
    rerun_fn = getattr(st, "experimental_rerun", None) or getattr(st, "rerun", None)
    if rerun_fn:
        rerun_fn()


def ensure_direct_connect_state() -> None:
    """Initialise direct connect log state."""
    st.session_state.setdefault("show_direct_connect_log", False)
    st.session_state.setdefault(DIRECT_CONNECT_LOG_KEY, [])
    st.session_state.setdefault(DIRECT_CONNECT_PROGRESS_INDEX_KEY, None)
    st.session_state.setdefault("direct_connect_log_expanded", True)
    st.session_state.setdefault(DIRECT_CONNECT_EVENT_IDS_KEY, [])
    st.session_state.setdefault(PROCESSING_LOG_SOURCE_KEY, None)


def reset_direct_connect_log(show: bool = True) -> None:
    ensure_direct_connect_state()
    st.session_state[DIRECT_CONNECT_LOG_KEY] = []
    st.session_state["show_direct_connect_log"] = show
    st.session_state[DIRECT_CONNECT_PROGRESS_INDEX_KEY] = None
    st.session_state["direct_connect_log_expanded"] = True
    st.session_state[DIRECT_CONNECT_EVENT_IDS_KEY] = []
    set_processing_log_source("direct_connect")


def append_direct_connect_log(message: str, emoji: str = "ℹ️ ") -> None:
    ensure_direct_connect_state()
    set_processing_log_source("direct_connect")
    timestamp = now_in_display_timezone().strftime("%H:%M:%S")
    st.session_state[DIRECT_CONNECT_LOG_KEY].append(
        {
            "timestamp": timestamp,
            "emoji": emoji,
            "message": message,
        }
    )


def set_direct_connect_progress(completed: int, total: int) -> None:
    ensure_direct_connect_state()
    idx = st.session_state.get(DIRECT_CONNECT_PROGRESS_INDEX_KEY)
    percent = (completed / total * 100) if total else 0.0
    entry_message = f"Building chat summaries... {completed}/{total} ({percent:.1f}%)"
    timestamp = now_in_display_timezone().strftime("%H:%M:%S")
    if idx is None:
        append_direct_connect_log(entry_message, "🧠")
        st.session_state[DIRECT_CONNECT_PROGRESS_INDEX_KEY] = len(st.session_state[DIRECT_CONNECT_LOG_KEY]) - 1
    else:
        try:
            st.session_state[DIRECT_CONNECT_LOG_KEY][idx]["timestamp"] = timestamp
            st.session_state[DIRECT_CONNECT_LOG_KEY][idx]["message"] = entry_message
        except (IndexError, KeyError):
            st.session_state[DIRECT_CONNECT_PROGRESS_INDEX_KEY] = None


def reset_direct_connect_progress() -> None:
    ensure_direct_connect_state()
    st.session_state[DIRECT_CONNECT_PROGRESS_INDEX_KEY] = None


def get_direct_connect_log() -> List[Dict[str, str]]:
    ensure_direct_connect_state()
    return list(st.session_state.get(DIRECT_CONNECT_LOG_KEY, []))


def register_direct_connect_event(event_id: str) -> bool:
    """Return True if the event id has not been seen before."""
    ensure_direct_connect_state()
    if not event_id:
        return False
    seen: List[str] = st.session_state.get(DIRECT_CONNECT_EVENT_IDS_KEY, [])
    if event_id in seen:
        return False
    seen.append(event_id)
    st.session_state[DIRECT_CONNECT_EVENT_IDS_KEY] = seen
    return True


def set_direct_connect_log_expanded(expanded: bool) -> None:
    ensure_direct_connect_state()
    st.session_state["direct_connect_log_expanded"] = expanded


def is_direct_connect_log_expanded() -> bool:
    ensure_direct_connect_state()
    return bool(st.session_state.get("direct_connect_log_expanded", True))


def ensure_admin_log_state() -> None:
    """Initialise admin action log state."""
    st.session_state.setdefault("show_admin_log", False)
    st.session_state.setdefault(ADMIN_LOG_KEY, [])
    st.session_state.setdefault(ADMIN_PROGRESS_INDEX_KEY, None)
    st.session_state.setdefault("admin_log_expanded", True)
    st.session_state.setdefault(ADMIN_EVENT_IDS_KEY, [])
    st.session_state.setdefault(PROCESSING_LOG_SOURCE_KEY, None)


def reset_admin_log(show: bool = True) -> None:
    ensure_admin_log_state()
    st.session_state[ADMIN_LOG_KEY] = []
    st.session_state["show_admin_log"] = show
    st.session_state[ADMIN_PROGRESS_INDEX_KEY] = None
    st.session_state["admin_log_expanded"] = True
    st.session_state[ADMIN_EVENT_IDS_KEY] = []
    set_processing_log_source("admin")


def append_admin_log(message: str, emoji: str = "ℹ️ ") -> None:
    ensure_admin_log_state()
    set_processing_log_source("admin")
    timestamp = now_in_display_timezone().strftime("%H:%M:%S")
    st.session_state[ADMIN_LOG_KEY].append(
        {
            "timestamp": timestamp,
            "emoji": emoji,
            "message": message,
        }
    )


def set_admin_progress(completed: int, total: int) -> None:
    ensure_admin_log_state()
    idx = st.session_state.get(ADMIN_PROGRESS_INDEX_KEY)
    percent = (completed / total * 100) if total else 0.0
    entry_message = f"Building chat summaries... {completed}/{total} ({percent:.1f}%)"
    timestamp = now_in_display_timezone().strftime("%H:%M:%S")
    if idx is None:
        append_admin_log(entry_message, "🧠")
        st.session_state[ADMIN_PROGRESS_INDEX_KEY] = len(st.session_state[ADMIN_LOG_KEY]) - 1
    else:
        try:
            st.session_state[ADMIN_LOG_KEY][idx]["timestamp"] = timestamp
            st.session_state[ADMIN_LOG_KEY][idx]["message"] = entry_message
        except (IndexError, KeyError):
            st.session_state[ADMIN_PROGRESS_INDEX_KEY] = None


def reset_admin_progress() -> None:
    ensure_admin_log_state()
    st.session_state[ADMIN_PROGRESS_INDEX_KEY] = None


def get_admin_log() -> List[Dict[str, str]]:
    ensure_admin_log_state()
    return list(st.session_state.get(ADMIN_LOG_KEY, []))


def register_admin_event(event_id: str) -> bool:
    """Return True if the admin log event id has not been seen before."""
    ensure_admin_log_state()
    if not event_id:
        return False
    seen: List[str] = st.session_state.get(ADMIN_EVENT_IDS_KEY, [])
    if event_id in seen:
        return False
    seen.append(event_id)
    st.session_state[ADMIN_EVENT_IDS_KEY] = seen
    return True


def set_admin_log_expanded(expanded: bool) -> None:
    ensure_admin_log_state()
    st.session_state["admin_log_expanded"] = expanded


def is_admin_log_expanded() -> bool:
    ensure_admin_log_state()
    return bool(st.session_state.get("admin_log_expanded", True))


def ensure_upload_log_state() -> None:
    """Initialise upload log state."""
    st.session_state.setdefault("show_upload_log", False)
    st.session_state.setdefault(UPLOAD_LOG_KEY, [])
    st.session_state.setdefault(UPLOAD_PROGRESS_INDEX_KEY, None)
    st.session_state.setdefault("upload_log_expanded", True)
    st.session_state.setdefault(UPLOAD_EVENT_IDS_KEY, [])
    st.session_state.setdefault(PROCESSING_LOG_SOURCE_KEY, None)


def reset_upload_log(show: bool = True) -> None:
    ensure_upload_log_state()
    st.session_state[UPLOAD_LOG_KEY] = []
    st.session_state["show_upload_log"] = show
    st.session_state[UPLOAD_PROGRESS_INDEX_KEY] = None
    st.session_state["upload_log_expanded"] = True
    st.session_state[UPLOAD_EVENT_IDS_KEY] = []
    set_processing_log_source("upload")


def append_upload_log(message: str, emoji: str = "ℹ️ ") -> None:
    ensure_upload_log_state()
    set_processing_log_source("upload")
    timestamp = now_in_display_timezone().strftime("%H:%M:%S")
    st.session_state[UPLOAD_LOG_KEY].append(
        {
            "timestamp": timestamp,
            "emoji": emoji,
            "message": message,
        }
    )


def set_upload_progress(completed: int, total: int) -> None:
    ensure_upload_log_state()
    idx = st.session_state.get(UPLOAD_PROGRESS_INDEX_KEY)
    percent = (completed / total * 100) if total else 0.0
    entry_message = f"Building chat summaries... {completed}/{total} ({percent:.1f}%)"
    timestamp = now_in_display_timezone().strftime("%H:%M:%S")
    if idx is None:
        append_upload_log(entry_message, "🧠")
        st.session_state[UPLOAD_PROGRESS_INDEX_KEY] = len(st.session_state[UPLOAD_LOG_KEY]) - 1
    else:
        try:
            st.session_state[UPLOAD_LOG_KEY][idx]["timestamp"] = timestamp
            st.session_state[UPLOAD_LOG_KEY][idx]["message"] = entry_message
        except (IndexError, KeyError):
            st.session_state[UPLOAD_PROGRESS_INDEX_KEY] = None


def reset_upload_progress() -> None:
    ensure_upload_log_state()
    st.session_state[UPLOAD_PROGRESS_INDEX_KEY] = None


def get_upload_log() -> List[Dict[str, str]]:
    ensure_upload_log_state()
    return list(st.session_state.get(UPLOAD_LOG_KEY, []))


def set_processing_log_source(source: Optional[str]) -> None:
    st.session_state[PROCESSING_LOG_SOURCE_KEY] = source


def get_processing_log_source() -> Optional[str]:
    return st.session_state.get(PROCESSING_LOG_SOURCE_KEY)


def get_processing_log_entries() -> List[Dict[str, str]]:
    source = get_processing_log_source()
    if source == "direct_connect":
        return get_direct_connect_log()
    if source == "upload":
        return get_upload_log()
    if source == "admin":
        return get_admin_log()

    # Fallback: show whichever log currently has entries.
    for candidate, getter in (
        ("direct_connect", get_direct_connect_log),
        ("upload", get_upload_log),
        ("admin", get_admin_log),
    ):
        entries = getter()
        if entries:
            set_processing_log_source(candidate)
            return entries
    return []


def register_upload_event(event_id: str) -> bool:
    """Return True if the upload event id has not been seen before."""
    ensure_upload_log_state()
    if not event_id:
        return False
    seen: List[str] = st.session_state.get(UPLOAD_EVENT_IDS_KEY, [])
    if event_id in seen:
        return False
    seen.append(event_id)
    st.session_state[UPLOAD_EVENT_IDS_KEY] = seen
    return True


def set_upload_log_expanded(expanded: bool) -> None:
    ensure_upload_log_state()
    st.session_state["upload_log_expanded"] = expanded


def is_upload_log_expanded() -> bool:
    ensure_upload_log_state()
    return bool(st.session_state.get("upload_log_expanded", True))


def set_default_openwebui_values(host: str, api_key: str) -> None:
    """Ensure session defaults for direct connect inputs."""
    prev_host_default = st.session_state.get("_openwebui_hostname_default")
    current_hostname = st.session_state.get("openwebui_hostname")
    if prev_host_default != host:
        if current_hostname in (None, "", prev_host_default):
            st.session_state["openwebui_hostname"] = host
        st.session_state["_openwebui_hostname_default"] = host
    else:
        st.session_state.setdefault("openwebui_hostname", host)

    prev_api_key_default = st.session_state.get("_openwebui_api_key_default")
    current_api_key = st.session_state.get("openwebui_api_key")
    if prev_api_key_default != api_key:
        if current_api_key in (None, "", prev_api_key_default):
            st.session_state["openwebui_api_key"] = api_key
        st.session_state["_openwebui_api_key_default"] = api_key
    else:
        st.session_state.setdefault("openwebui_api_key", api_key)


def update_dataset_signature(dataset_id: str, chats_df: Any, messages_df: Any) -> bool:
    """Return True if dataset signature changed, used to reset filters."""
    signature = (
        dataset_id,
        len(chats_df) if chats_df is not None else 0,
        len(messages_df) if messages_df is not None else 0,
    )
    changed = st.session_state.get("dataset_signature") != signature
    st.session_state["dataset_signature"] = signature
    return changed


def reset_model_filter() -> None:
    st.session_state["model_filter"] = ALL_MODELS_OPTION
    st.session_state["page"] = 1


def reset_browse_page() -> None:
    st.session_state["page"] = 1


def ensure_filter_defaults() -> None:
    st.session_state.setdefault("user_filter", ALL_USERS_OPTION)
    st.session_state.setdefault("model_filter", ALL_MODELS_OPTION)
    st.session_state.setdefault("page", 1)
    st.session_state.setdefault("last_filter_key", "")


def update_filter_key(filter_key: str) -> None:
    previous = st.session_state.get("last_filter_key")
    if previous != filter_key:
        st.session_state["page"] = 1
        st.session_state["last_filter_key"] = filter_key


def get_threads_per_page() -> int:
    return int(st.session_state.get("threads_per_page", 10))


def set_threads_per_page(value: int) -> None:
    st.session_state["threads_per_page"] = int(value)


def set_dataset_meta(meta: DatasetMeta) -> None:
    st.session_state["dataset_meta"] = meta


def get_dataset_meta(default: Optional[DatasetMeta] = None) -> Optional[DatasetMeta]:
    return st.session_state.get("dataset_meta", default)


def set_processed_data(data: ProcessedData) -> None:
    st.session_state["processed_data"] = data


def get_processed_data() -> Optional[ProcessedData]:
    return st.session_state.get("processed_data")
