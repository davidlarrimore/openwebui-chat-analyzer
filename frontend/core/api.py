from __future__ import annotations

import io
import json
import time
from typing import Any, Callable, Dict, Iterable, Optional

import pandas as pd
import requests

from .config import get_config, get_api_base_url_candidates
from .models import DatasetMeta, ProcessedData, SummaryStatus, UploadResult


class BackendError(Exception):
    """Raised when the backend API is unreachable or returns an error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.cause = cause


def is_backend_unavailable_error(error: Optional[BackendError]) -> bool:
    """Return True if the error likely represents a transient backend outage."""
    if error is None:
        return False

    if error.status_code in {502, 503, 504}:
        return True

    cause = getattr(error, "cause", None)
    transient_exceptions: Iterable[type[BaseException]] = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ConnectTimeout,
        requests.exceptions.ReadTimeout,
    )
    return isinstance(cause, tuple(transient_exceptions))


def _request(
    path: str,
    *,
    method: str = "GET",
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    """Execute an HTTP request against the backend service."""
    config = get_config()
    timeout = timeout or config.request_timeout
    last_exc: Optional[BaseException] = None

    for base_url in get_api_base_url_candidates():
        url = f"{base_url.rstrip('/')}{path}"
        try:
            response = requests.request(method=method.upper(), url=url, timeout=timeout, **kwargs)
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            detail = ""
            if exc.response is not None:
                status_code = exc.response.status_code
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
            else:
                status_code = None

            message = f"Backend request failed: {exc}"
            if detail:
                message = f"{message} - {detail}"
            raise BackendError(message, status_code=status_code, cause=exc) from exc
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            continue
        else:
            break
    else:
        raise BackendError(f"Backend request failed: {last_exc}", cause=last_exc) from last_exc

    if response.status_code == 204 or method.upper() == "HEAD":
        return {}

    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        return response.json()

    try:
        return response.json()
    except ValueError as exc:
        raise BackendError("Backend returned an invalid JSON response.", cause=exc) from exc


def get_dataset_meta() -> DatasetMeta:
    """Fetch dataset metadata."""
    payload = _request("/api/v1/datasets/meta")
    return DatasetMeta.from_dict(payload)


def get_summary_status() -> SummaryStatus:
    """Fetch the summarizer job status."""
    payload = _request("/api/v1/summaries/status")
    return SummaryStatus.from_dict(payload)


def trigger_summary_rebuild() -> SummaryStatus:
    """Request the backend to rebuild chat summaries."""
    payload = _request("/api/v1/summaries/rebuild", method="POST")
    if isinstance(payload, dict):
        status_payload = payload.get("status", payload)
    else:
        status_payload = {}
    return SummaryStatus.from_dict(status_payload)


def poll_summary_status(
    *,
    timeout: Optional[int] = None,
    poll_interval: Optional[float] = None,
    on_update: Optional[Callable[[SummaryStatus], None]] = None,
) -> SummaryStatus:
    """Poll the backend summarizer status until it completes or times out."""
    config = get_config()
    timeout = timeout or config.summary_timeout_seconds
    poll_interval = poll_interval or config.summary_poll_interval

    start_time = time.monotonic()
    status = SummaryStatus(state=None)

    while True:
        status = get_summary_status()
        if on_update:
            try:
                on_update(status)
            except Exception:
                pass

        if status.state in {None, "idle", "completed", "failed", "cancelled"}:
            return status

        if time.monotonic() - start_time >= timeout:
            return status

        time.sleep(poll_interval)


def get_chats(params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Fetch chats as a dataframe."""
    payload = _request("/api/v1/chats", params=params or {})
    chats_df = pd.DataFrame(payload)
    if chats_df.empty:
        return chats_df

    if "summary_128" not in chats_df.columns:
        chats_df["summary_128"] = ""
    else:
        chats_df["summary_128"] = chats_df["summary_128"].fillna("")

    for column in ("created_at", "updated_at"):
        if column in chats_df.columns:
            chats_df[column] = pd.to_datetime(chats_df[column], errors="coerce")

    if "chat_id" in chats_df.columns:
        chats_df["chat_id"] = chats_df["chat_id"].astype(str)

    return chats_df


def get_messages(params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Fetch messages as a dataframe."""
    payload = _request("/api/v1/messages", params=params or {})
    messages_df = pd.DataFrame(payload)
    if messages_df.empty:
        return messages_df

    if "timestamp" in messages_df.columns:
        messages_df["timestamp"] = pd.to_datetime(messages_df["timestamp"], errors="coerce")
    if "chat_id" in messages_df.columns:
        messages_df["chat_id"] = messages_df["chat_id"].astype(str)
    if "model" in messages_df.columns:
        messages_df["model"] = messages_df["model"].fillna("")

    return messages_df


def get_users() -> pd.DataFrame:
    """Fetch users as a dataframe."""
    payload = _request("/api/v1/users")
    users_df = pd.DataFrame(payload)
    if users_df.empty:
        return pd.DataFrame(columns=["user_id", "name"])

    users_df["user_id"] = users_df["user_id"].astype(str)
    users_df["name"] = users_df["name"].astype(str)
    return users_df


def reset_dataset() -> UploadResult:
    """Delete all stored dataset artifacts and reset metadata."""
    payload = _request("/api/v1/datasets/reset", method="POST")
    return UploadResult.from_dict(payload)


def upload_chat_export(file_obj: io.BytesIO) -> UploadResult:
    """Upload a chat export JSON file."""
    filename = getattr(file_obj, "name", "chat_export.json")
    filetype = getattr(file_obj, "type", None) or "application/json"
    file_bytes = file_obj.getvalue() if hasattr(file_obj, "getvalue") else file_obj.read()
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    files = {"file": (filename, file_bytes, filetype)}
    payload = _request("/api/v1/uploads/chat-export", method="POST", files=files)
    return UploadResult.from_dict(payload)


def upload_users_csv(file_obj: io.BytesIO) -> UploadResult:
    """Upload a CSV of user metadata."""
    filename = getattr(file_obj, "name", "users.csv")
    filetype = getattr(file_obj, "type", None) or "text/csv"
    file_bytes = file_obj.getvalue() if hasattr(file_obj, "getvalue") else file_obj.read()
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    files = {"file": (filename, file_bytes, filetype)}
    payload = _request("/api/v1/uploads/users", method="POST", files=files)
    return UploadResult.from_dict(payload)


def sync_openwebui_dataset(hostname: str, api_key: str) -> UploadResult:
    """Trigger a dataset sync directly from an Open WebUI instance."""
    payload = {"hostname": hostname}
    token = (api_key or "").strip()
    if token:
        payload["api_key"] = token
    response = _request("/api/v1/openwebui/sync", method="POST", json=payload)
    return UploadResult.from_dict(response)


def build_processed_data() -> ProcessedData:
    """Fetch chats, messages, and users and prepare helper mappings."""
    chats_df = get_chats()
    messages_df = get_messages()
    users_df = get_users()

    if not chats_df.empty:
        chats_df["user_id"] = chats_df["user_id"].fillna("").astype(str)
    else:
        chats_df["user_id"] = pd.Series(dtype=str)

    fallback_ids = chats_df.get("user_id", pd.Series(dtype=str)).replace({"nan": "", "None": ""})
    user_display_map: Dict[str, str] = {}

    if not users_df.empty:
        users_df["user_id"] = users_df["user_id"].astype(str)
        user_display_map = dict(zip(users_df["user_id"], users_df["name"]))

    if "user_id" in chats_df.columns:
        chats_df["user_display"] = chats_df["user_id"].map(user_display_map)
        chats_df["user_display"] = chats_df["user_display"].fillna(fallback_ids)
        chats_df.loc[chats_df["user_display"].isin(["", "nan", "None"]), "user_display"] = "User"

    chat_user_map = {}
    if "chat_id" in chats_df.columns and "user_display" in chats_df.columns:
        chat_user_map = chats_df.set_index("chat_id")["user_display"].to_dict()

    if "chat_id" in messages_df.columns:
        messages_df["chat_id"] = messages_df["chat_id"].astype(str)
        messages_df["chat_user_display"] = messages_df["chat_id"].map(chat_user_map)
        messages_df["chat_user_display"] = messages_df["chat_user_display"].fillna("User")

    processed = ProcessedData(
        chats=chats_df,
        messages=messages_df,
        users=users_df,
        chat_user_map=chat_user_map,
    )
    return processed
