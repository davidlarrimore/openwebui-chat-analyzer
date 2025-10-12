"""Data service for the Open WebUI Chat Analyzer backend."""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone, date
from io import BytesIO
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
import requests
from urllib.parse import urlparse, urlunparse

from .config import DATA_DIR
from .models import AppMetadata, DatasetMeta
from .summarizer import summarize_chats


class SummaryJobCancelled(Exception):
    """Raised when a summary job is cancelled due to a newer dataset."""


SUMMARY_EXPORT_FIELD = "owca_summary_128"


def _coerce_timestamp(value: Any) -> Optional[datetime]:
    """Convert export timestamps into timezone-aware datetimes."""
    if value in (None, "", 0):
        return None

    if isinstance(value, (int, float)):
        if not math.isfinite(value) or value <= 0:
            return None
        return datetime.fromtimestamp(value, tz=timezone.utc)

    if isinstance(value, str):
        candidate = value.strip()
        if candidate == "":
            return None
        # Try numeric string first
        try:
            numeric_value = float(candidate)
        except ValueError:
            numeric_value = None
        if numeric_value is not None and math.isfinite(numeric_value) and numeric_value > 0:
            return datetime.fromtimestamp(numeric_value, tz=timezone.utc)
        # Fallback to ISO 8601 parsing
        try:
            # Replace trailing Z with UTC offset for fromisoformat compatibility
            normalized = candidate.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None

    return None


def _ensure_list(value: Any) -> List[Any]:
    """Normalize list-like values into plain lists."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _parse_chat_export(raw_bytes: bytes) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse the Open WebUI chat export JSON into chat/message payloads."""
    if not raw_bytes:
        raise ValueError("Chat export file is empty.")

    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid JSON chat export.") from exc

    if not isinstance(data, Iterable):
        raise ValueError("Chat export must be a JSON array.")

    chats: List[Dict[str, Any]] = []
    messages: List[Dict[str, Any]] = []

    for item in data:
        if not isinstance(item, dict):
            continue
        chat_meta = item.get("chat", {}) if isinstance(item.get("chat"), dict) else {}
        files = chat_meta.get("files", [])

        chat_info: Dict[str, Any] = {
            "chat_id": str(item.get("id", "")),
            "user_id": str(item.get("user_id", "")) if item.get("user_id") is not None else "",
            "title": item.get("title") or "",
            "created_at": _coerce_timestamp(item.get("created_at")),
            "updated_at": _coerce_timestamp(item.get("updated_at")),
            "archived": bool(item.get("archived", False)),
            "pinned": bool(item.get("pinned", False)),
            "tags": _ensure_list(item.get("meta", {}).get("tags") if isinstance(item.get("meta"), dict) else []),
            "files_uploaded": len(files) if isinstance(files, list) else 0,
            "files": files if isinstance(files, list) else [],
        }
        summary_value = item.get(SUMMARY_EXPORT_FIELD)
        chat_summary_value = None
        if isinstance(summary_value, str):
            chat_summary_value = summary_value
        elif isinstance(chat_meta, dict):
            nested_summary = chat_meta.get(SUMMARY_EXPORT_FIELD)
            if isinstance(nested_summary, str):
                chat_summary_value = nested_summary
        if chat_summary_value is not None:
            chat_info["summary_128"] = str(chat_summary_value)

        chats.append(chat_info)

        chat_messages = chat_meta.get("messages", [])
        if isinstance(chat_messages, list):
            for msg in chat_messages:
                if not isinstance(msg, dict):
                    continue
                message_info: Dict[str, Any] = {
                    "chat_id": chat_info["chat_id"],
                    "message_id": str(msg.get("id", "")),
                    "parent_id": str(msg.get("parentId")) if msg.get("parentId") not in (None, "") else None,
                    "role": str(msg.get("role", "")),
                    "content": msg.get("content") or "",
                    "timestamp": _coerce_timestamp(msg.get("timestamp")),
                    "model": msg.get("model") or "",
                    "models": _ensure_list(msg.get("models")),
                }
                messages.append(message_info)

    return chats, messages


def _parse_users_csv(raw_bytes: bytes) -> List[Dict[str, str]]:
    """Parse a CSV file into normalized user payloads."""
    if not raw_bytes:
        raise ValueError("Users CSV file is empty.")

    try:
        csv_buffer = BytesIO(raw_bytes)
        users_df = pd.read_csv(csv_buffer)
    except Exception as exc:  # pandas raises various exceptions
        raise ValueError("Unable to parse users CSV.") from exc

    if users_df.empty:
        return []

    # Normalize column names
    users_df.columns = [str(col).strip().lower() for col in users_df.columns]

    id_col = next((col for col in ("user_id", "id") if col in users_df.columns), None)
    if id_col is None:
        raise ValueError("Users CSV must include a 'user_id' or 'id' column.")

    name_col = next(
        (col for col in ("name", "full_name", "display_name", "username") if col in users_df.columns), None
    )
    if name_col is None:
        raise ValueError("Users CSV must include a name-like column (e.g. 'name').")

    normalized = (
        users_df[[id_col, name_col]]
        .rename(columns={id_col: "user_id", name_col: "name"})
        .dropna(subset=["user_id"])
        .drop_duplicates(subset=["user_id"], keep="first")
    )

    normalized["user_id"] = normalized["user_id"].astype(str)
    normalized["name"] = normalized["name"].astype(str).str.strip()

    return normalized.to_dict(orient="records")


def _normalize_hostname(hostname: str) -> str:
    """Cleanse and normalize the provided hostname into a usable base URL."""
    if not isinstance(hostname, str):
        raise ValueError("Hostname must be a string.")
    candidate = hostname.strip()
    if not candidate:
        raise ValueError("Hostname must be a non-empty string.")
    if not candidate.startswith(("http://", "https://")):
        candidate = f"http://{candidate}"
    return candidate.rstrip("/")


def _replace_hostname(url: str, hostname: str) -> str:
    """Return a URL with the hostname replaced while preserving other components."""
    parsed = urlparse(url)
    netloc = hostname
    if parsed.port:
        netloc = f"{hostname}:{parsed.port}"
    if parsed.username:
        credentials = parsed.username
        if parsed.password:
            credentials = f"{credentials}:{parsed.password}"
        netloc = f"{credentials}@{netloc}"
    return urlunparse(parsed._replace(netloc=netloc))


def _build_openwebui_base_candidates(base_url: str) -> List[str]:
    """Generate candidate base URLs to try when reaching Open WebUI."""
    cleaned = base_url.rstrip("/")
    candidates: List[str] = []
    seen: set[str] = set()

    def _add(url: str) -> None:
        normalized = url.rstrip("/")
        if normalized and normalized not in seen:
            seen.add(normalized)
            candidates.append(normalized)

    _add(cleaned)

    parsed = urlparse(cleaned)
    if parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        docker_host = os.getenv("OWUI_DOCKER_HOSTNAME", "host.docker.internal").strip()
        if docker_host:
            try:
                replaced = _replace_hostname(cleaned, docker_host)
            except ValueError:
                replaced = ""
            if replaced:
                _add(replaced)

    return candidates


def _fetch_openwebui_json(session: requests.Session, url: str) -> Any:
    """Fetch JSON payloads from Open WebUI endpoints."""
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:  # pragma: no cover - network errors
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}") from exc


def _normalize_openwebui_users(payload: Any) -> List[Dict[str, str]]:
    """Transform Open WebUI user payloads into analyzer-compatible structures."""
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            entries = payload.get("data", [])
        elif isinstance(payload.get("users"), list):
            entries = payload.get("users", [])
        else:
            entries = []
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []

    normalized: List[Dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        user_id = (
            entry.get("user_id")
            or entry.get("id")
            or entry.get("_id")
            or entry.get("uuid")
            or entry.get("uid")
        )
        if user_id in (None, "", []):
            continue

        name = (
            entry.get("name")
            or entry.get("full_name")
            or entry.get("display_name")
            or entry.get("username")
            or entry.get("email")
            or entry.get("handle")
            or str(user_id)
        )

        normalized.append({"user_id": str(user_id), "name": str(name)})
    return normalized


def _stringify_content(value: Any) -> str:
    """Convert message content into a JSON-friendly string representation."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _normalize_messages(messages_raw: Any) -> List[Dict[str, Any]]:
    """Normalize messages returned by Open WebUI into export-compatible payloads."""
    if isinstance(messages_raw, dict):
        candidates = list(messages_raw.values())
    elif isinstance(messages_raw, list):
        candidates = messages_raw
    else:
        candidates = []

    normalized_messages: List[Dict[str, Any]] = []
    for msg in candidates:
        if not isinstance(msg, dict):
            continue
        message_id = (
            msg.get("id")
            or msg.get("message_id")
            or msg.get("_id")
            or msg.get("uuid")
            or msg.get("key")
            or ""
        )
        parent_raw = msg.get("parentId") or msg.get("parent_id") or msg.get("parent")
        parent_id = None
        if parent_raw not in (None, "", []):
            parent_id = str(parent_raw)

        raw_models = _ensure_list(msg.get("models"))
        models = [str(model) for model in raw_models if model not in (None, "")]

        normalized_messages.append(
            {
                "id": str(message_id),
                "parentId": parent_id,
                "role": str(msg.get("role") or msg.get("type") or msg.get("sender") or ""),
                "content": _stringify_content(
                    msg.get("content") or msg.get("message") or msg.get("text") or msg.get("body")
                ),
                "timestamp": msg.get("timestamp") or msg.get("created_at") or msg.get("updated_at"),
                "model": str(msg.get("model") or msg.get("model_id") or msg.get("assistant") or ""),
                "models": models,
            }
        )
    return normalized_messages


def _build_export_payload_from_openwebui(payload: Any) -> List[Dict[str, Any]]:
    """Convert Open WebUI chat payloads into the export format expected by the analyzer."""
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            entries = payload.get("data", [])
        elif isinstance(payload.get("chats"), list):
            entries = payload.get("chats", [])
        else:
            entries = []
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []

    export_ready: List[Dict[str, Any]] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        base_chat = entry.get("chat") if isinstance(entry.get("chat"), dict) else entry
        chat_id = (
            entry.get("id")
            or base_chat.get("id")
            or base_chat.get("_id")
            or base_chat.get("uuid")
            or entry.get("_id")
        )
        if chat_id in (None, "", []):
            continue
        chat_id = str(chat_id)

        user_id = entry.get("user_id") or base_chat.get("user_id") or ""

        meta_source: Any = {}
        if isinstance(entry.get("meta"), dict):
            meta_source = entry.get("meta")
        elif isinstance(base_chat.get("meta"), dict):
            meta_source = base_chat.get("meta")
        meta = dict(meta_source) if isinstance(meta_source, dict) else {}

        files_source: Any = []
        if isinstance(base_chat.get("files"), list):
            files_source = base_chat.get("files")
        elif isinstance(entry.get("files"), list):
            files_source = entry.get("files")
        files = list(files_source) if isinstance(files_source, list) else []

        normalized_messages = _normalize_messages(
            base_chat.get("messages") if isinstance(base_chat, dict) else []
        )
        if not normalized_messages and isinstance(entry.get("messages"), (list, dict)):
            normalized_messages = _normalize_messages(entry.get("messages"))

        export_ready.append(
            {
                "id": chat_id,
                "user_id": str(user_id) if user_id not in (None, "", []) else "",
                "title": base_chat.get("title") or entry.get("title") or "",
                "updated_at": base_chat.get("updated_at") or entry.get("updated_at"),
                "created_at": base_chat.get("created_at") or entry.get("created_at"),
                "share_id": entry.get("share_id") or base_chat.get("share_id"),
                "archived": bool(base_chat.get("archived") or entry.get("archived", False)),
                "pinned": bool(base_chat.get("pinned") or entry.get("pinned", False)),
                "meta": meta,
                "folder_id": entry.get("folder_id") or base_chat.get("folder_id"),
                "chat": {
                    "id": chat_id,
                    "title": base_chat.get("title") or entry.get("title") or "",
                    "updated_at": base_chat.get("updated_at") or entry.get("updated_at"),
                    "created_at": base_chat.get("created_at") or entry.get("created_at"),
                    "archived": bool(base_chat.get("archived") or entry.get("archived", False)),
                    "pinned": bool(base_chat.get("pinned") or entry.get("pinned", False)),
                    "meta": meta,
                    "messages": normalized_messages,
                    "files": files,
                },
            }
        )

    return export_ready


def _apply_summaries_to_export_payload(
    payload: Any,
    summary_map: Dict[str, str],
) -> Any:
    """Inject analyzer summaries into the raw export payload."""
    if not isinstance(payload, list) or not summary_map:
        return payload

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        chat_id = entry.get("id") or entry.get("_id") or entry.get("chat_id")
        chat_id_str = str(chat_id) if chat_id not in (None, "", []) else None
        if not chat_id_str:
            continue
        summary = summary_map.get(chat_id_str)
        if summary is None:
            continue
        entry.pop(SUMMARY_EXPORT_FIELD, None)
        entry[SUMMARY_EXPORT_FIELD] = summary
        chat_section = entry.get("chat")
        if isinstance(chat_section, dict):
            chat_section.pop(SUMMARY_EXPORT_FIELD, None)
    return payload


def _annotate_export_bytes(raw_bytes: bytes, summary_map: Dict[str, str]) -> bytes:
    """Return raw bytes with analyzer summaries injected; fallback to original bytes on failure."""
    if not summary_map:
        return raw_bytes
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except Exception:
        return raw_bytes
    annotated = _apply_summaries_to_export_payload(payload, summary_map)
    try:
        return json.dumps(annotated, ensure_ascii=False, indent=2).encode("utf-8")
    except Exception:
        return raw_bytes


class DataService:
    """In-memory data repository backed by chat export files."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self._lock = Lock()
        if data_dir is not None:
            self._data_dir = Path(data_dir).resolve()
        else:
            self._data_dir = Path(__file__).resolve().parent.parent / "data"
        self._chats: List[Dict[str, Any]] = []
        self._messages: List[Dict[str, Any]] = []
        self._users: List[Dict[str, str]] = []
        self._dataset_id = uuid4().hex
        self._source = "no dataset loaded"
        self._last_updated = datetime.utcnow().replace(tzinfo=timezone.utc)
        self._app_metadata_path = self._data_dir / "app.json"
        self._dataset_source_override: Optional[str] = None
        self._dataset_pulled_at: Optional[datetime] = None
        self._chat_uploaded_at: Optional[datetime] = None
        self._users_uploaded_at: Optional[datetime] = None
        self._first_chat_day: Optional[date] = None
        self._last_chat_day: Optional[date] = None
        self._summary_state_lock = Lock()
        self._summary_state: Dict[str, Any] = {
            "state": "idle",
            "total": 0,
            "completed": 0,
            "message": "",
            "started_at": None,
            "finished_at": None,
            "updated_at": None,
            "last_event": None,
        }
        self._summary_thread: Optional[Thread] = None
        self._summary_target_dataset_id: Optional[str] = None
        self._summary_job_counter: int = 0
        self._summary_active_job_id: Optional[int] = None
        self._load_app_metadata()
        self.load_initial_data()
        with self._lock:
            should_persist = not self._app_metadata_path.exists()
            self._refresh_app_metadata(persist=should_persist)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _ensure_data_dir(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.utcnow().replace(tzinfo=timezone.utc)

    @staticmethod
    def _normalize_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if not isinstance(value, str):
            return None
        candidate = value.strip()
        if not candidate:
            return None
        normalized = candidate.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        return DataService._normalize_utc(parsed)

    @staticmethod
    def _parse_date(value: Any) -> Optional[date]:
        if not isinstance(value, str):
            return None
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return date.fromisoformat(candidate)
        except ValueError:
            return None

    @staticmethod
    def _serialize_datetime(value: Optional[datetime]) -> Optional[str]:
        if value is None:
            return None
        normalized = DataService._normalize_utc(value)
        return normalized.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _serialize_date(value: Optional[date]) -> Optional[str]:
        if value is None:
            return None
        return value.isoformat()

    def _save_app_metadata(self, payload: Dict[str, Any]) -> None:
        self._ensure_data_dir()
        self._app_metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_app_metadata(self) -> None:
        if not self._app_metadata_path.exists():
            return
        try:
            payload = json.loads(self._app_metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return

        dataset_source = payload.get("dataset_source")
        if isinstance(dataset_source, str) and dataset_source.strip():
            self._dataset_source_override = dataset_source.strip()

        self._dataset_pulled_at = self._parse_datetime(payload.get("dataset_pulled_at"))
        chats_section = payload.get("chats") or {}
        users_section = payload.get("users") or {}

        self._chat_uploaded_at = self._parse_datetime(chats_section.get("uploaded_at"))
        self._users_uploaded_at = self._parse_datetime(users_section.get("uploaded_at"))
        self._first_chat_day = self._parse_date(chats_section.get("first_day"))
        self._last_chat_day = self._parse_date(chats_section.get("last_day"))

    def _determine_dataset_source(self) -> str:
        source = (self._source or "").strip()
        if source.startswith("openwebui:"):
            return source.split(":", 1)[1].strip() or "local upload"
        if not self._chats:
            return "not loaded"
        return "local upload"

    def _compute_dataset_source_display(self) -> str:
        if self._dataset_source_override:
            return self._dataset_source_override
        return self._determine_dataset_source()

    def _calculate_chat_day_range(self) -> Tuple[Optional[date], Optional[date]]:
        timestamps: List[datetime] = []
        for message in self._messages:
            ts = message.get("timestamp")
            if isinstance(ts, datetime):
                timestamps.append(self._normalize_utc(ts))
        if not timestamps:
            for chat in self._chats:
                for field in ("created_at", "updated_at"):
                    candidate = chat.get(field)
                    if isinstance(candidate, datetime):
                        timestamps.append(self._normalize_utc(candidate))
        if not timestamps:
            return None, None
        first_ts = min(timestamps)
        last_ts = max(timestamps)
        return first_ts.date(), last_ts.date()

    def _refresh_app_metadata(
        self,
        chat_updated: bool = False,
        users_updated: bool = False,
        persist: bool = True,
    ) -> None:
        now = self._now_utc()
        if chat_updated:
            self._chat_uploaded_at = now
        if users_updated:
            self._users_uploaded_at = now

        chat_count = len(self._chats)
        user_count = len(self._users)

        first_day: Optional[date]
        last_day: Optional[date]
        if chat_count == 0:
            first_day = None
            last_day = None
            self._chat_uploaded_at = None
        else:
            first_day, last_day = self._calculate_chat_day_range()
            self._chat_uploaded_at = self._chat_uploaded_at or now

        self._first_chat_day = first_day
        self._last_chat_day = last_day

        if user_count == 0:
            self._users_uploaded_at = None
        else:
            self._users_uploaded_at = self._users_uploaded_at or now

        if chat_count == 0 and user_count == 0:
            self._dataset_pulled_at = None
        elif chat_updated or users_updated:
            self._dataset_pulled_at = now
        elif self._dataset_pulled_at is None:
            self._dataset_pulled_at = self._chat_uploaded_at or self._users_uploaded_at

        should_recalculate_source = chat_updated or users_updated or self._dataset_source_override is None
        if should_recalculate_source:
            display_source = self._determine_dataset_source()
            self._dataset_source_override = display_source
        else:
            display_source = self._compute_dataset_source_display()

        if persist:
            payload = {
                "dataset_source": display_source,
                "dataset_pulled_at": self._serialize_datetime(self._dataset_pulled_at),
                "chats": {
                    "count": chat_count,
                    "uploaded_at": self._serialize_datetime(self._chat_uploaded_at),
                    "first_day": self._serialize_date(self._first_chat_day),
                    "last_day": self._serialize_date(self._last_chat_day),
                },
                "users": {
                    "count": user_count,
                    "uploaded_at": self._serialize_datetime(self._users_uploaded_at),
                },
            }
            self._save_app_metadata(payload)

    def _build_app_metadata(self) -> AppMetadata:
        return AppMetadata(
            dataset_source=self._compute_dataset_source_display(),
            dataset_pulled_at=self._dataset_pulled_at,
            chat_uploaded_at=self._chat_uploaded_at,
            users_uploaded_at=self._users_uploaded_at,
            first_chat_day=self._first_chat_day,
            last_chat_day=self._last_chat_day,
            chat_count=len(self._chats),
            user_count=len(self._users),
        )

    def _summary_now_iso(self) -> str:
        return self._now_utc().isoformat().replace("+00:00", "Z")

    def _update_summary_state(
        self,
        *,
        state: Optional[str] = None,
        message: Optional[str] = None,
        total: Optional[int] = None,
        completed: Optional[int] = None,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
        event: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._summary_state_lock:
            if state is not None:
                self._summary_state["state"] = state
            if message is not None:
                self._summary_state["message"] = message
            if total is not None:
                self._summary_state["total"] = total
            if completed is not None:
                self._summary_state["completed"] = completed
            if started_at is not None:
                self._summary_state["started_at"] = started_at
            if finished_at is not None:
                self._summary_state["finished_at"] = finished_at
            if event is not None:
                self._summary_state["last_event"] = event
            self._summary_state["updated_at"] = self._summary_now_iso()

    def get_summary_status(self) -> Dict[str, Any]:
        with self._summary_state_lock:
            return dict(self._summary_state)

    def _enqueue_summary_job(self, reason: str) -> Dict[str, Any]:
        with self._lock:
            chats_snapshot = [chat.copy() for chat in self._chats]
            messages_snapshot = [message.copy() for message in self._messages]
            export_path = self._find_latest_chat_export()
            dataset_id = self._dataset_id

        total = len(chats_snapshot)
        if total == 0:
            self._update_summary_state(
                state="idle",
                message="No chats available for summarization.",
                total=0,
                completed=0,
                started_at=None,
                finished_at=self._summary_now_iso(),
            )
            return self.get_summary_status()

        logger = logging.getLogger(__name__)
        with self._summary_state_lock:
            if self._summary_thread and self._summary_thread.is_alive():
                logger.info("Cancelling in-flight summary job in favor of new request (%s)", reason)
            self._summary_job_counter += 1
            job_id = self._summary_job_counter
            self._summary_target_dataset_id = dataset_id
            self._summary_active_job_id = job_id

        self._update_summary_state(
            state="running",
            message=f"Summarizing {total} chats ({reason})",
            total=total,
            completed=0,
            started_at=self._summary_now_iso(),
            finished_at=None,
            event={
                "type": "start",
                "event_id": f"start-{job_id}",
                "reason": reason,
                "total": total,
                "timestamp": self._summary_now_iso(),
                "message": f"Starting summarizer job ({reason})",
            },
        )

        def progress(
            current: int,
            total_count: int,
            chat_id: str,
            outcome: str,
            details: Optional[Dict[str, Any]] = None,
        ) -> None:
            def _format_message(payload: Dict[str, Any], default_outcome: str) -> str:
                payload_type = payload.get("type")
                if payload_type == "chat":
                    outcome = payload.get("outcome", "generated")
                    chat_id_value = payload.get("chat_id") or "unknown"
                    if outcome == "generated":
                        return f"Generated summary for chat {chat_id_value}"
                    if outcome == "failed":
                        return f"Failed to summarize chat {chat_id_value}"
                    return f"{outcome.capitalize()} chat {chat_id_value}"
                if payload_type == "skip":
                    return f"Skipped existing summary for chat {payload.get('chat_id') or 'unknown'}"
                if payload_type == "invalid_chat":
                    return "Skipped chat without a valid chat_id"
                if payload_type == "empty_context":
                    return f"No summarizable content for chat {payload.get('chat_id') or 'unknown'}"
                return default_outcome.capitalize()

            with self._summary_state_lock:
                if self._summary_active_job_id != job_id:
                    raise SummaryJobCancelled()

            message = None
            event_payload = None
            if details:
                event_payload = dict(details)
                event_payload.setdefault("event_id", f"auto-{uuid4().hex}")
                event_payload["timestamp"] = self._summary_now_iso()
                message = _format_message(event_payload, outcome)
                if message is not None:
                    event_payload["message"] = message
            self._update_summary_state(
                state="running",
                total=total_count,
                completed=current,
                message=message,
                event=event_payload,
            )

        worker = Thread(
            target=self._summary_worker,
            args=(chats_snapshot, messages_snapshot, export_path, dataset_id, reason, job_id, progress),
            daemon=True,
        )
        with self._summary_state_lock:
            self._summary_thread = worker
        worker.start()
        return self.get_summary_status()

    def _summary_worker(
        self,
        chats_snapshot: List[Dict[str, Any]],
        messages_snapshot: List[Dict[str, Any]],
        export_path: Optional[Path],
        target_dataset_id: str,
        reason: str,
        job_id: int,
        progress_cb: Callable[[int, int, str, str, Optional[Dict[str, Any]]], None],
    ) -> None:
        logger = logging.getLogger(__name__)
        logger.info(
            "Summary job %s started for %d chats (%s)",
            job_id,
            len(chats_snapshot),
            reason,
        )
        try:
            summary_map, stats = summarize_chats(
                chats_snapshot,
                messages_snapshot,
                on_progress=progress_cb,
            )

            with self._summary_state_lock:
                current_target = self._summary_target_dataset_id
                active_job_id = self._summary_active_job_id

            if active_job_id != job_id:
                logger.info("Summary job %s superseded by a newer request; exiting early.", job_id)
                return

            if current_target != target_dataset_id:
                logger.info(
                    "Summary job %s skipped writing results because dataset changed (expected %s, current %s)",
                    job_id,
                    target_dataset_id,
                    current_target,
                )
                return

            summary_map = {chat_id: summary for chat_id, summary in summary_map.items() if summary}

            with self._lock:
                for chat in self._chats:
                    chat_id = str(chat.get("chat_id") or "")
                    if chat_id in summary_map:
                        chat["summary_128"] = summary_map[chat_id]

                if export_path is not None and export_path.exists():
                    try:
                        raw_bytes = export_path.read_bytes()
                        updated_bytes = _annotate_export_bytes(raw_bytes, summary_map)
                        export_path.write_bytes(updated_bytes)
                    except Exception as exc:
                        logger.warning("Failed to persist summaries to %s: %s", export_path, exc)

                self._bump_version()
                self._refresh_app_metadata(persist=False)

            logger.info("Summary job %s finished successfully.", job_id)
            self._update_summary_state(
                state="completed",
                message="Summary job complete.",
                total=stats["total"],
                completed=stats["total"],
                finished_at=self._summary_now_iso(),
            )
        except SummaryJobCancelled:
            logger.info("Summary job %s cancelled before completion.", job_id)
        except Exception as exc:
            logger.exception("Summary job failed: %s", exc)
            self._update_summary_state(
                state="failed",
                message=str(exc),
                finished_at=self._summary_now_iso(),
            )
        finally:
            with self._summary_state_lock:
                if self._summary_active_job_id == job_id:
                    self._summary_thread = None
                    self._summary_target_dataset_id = None
                    self._summary_active_job_id = None

    def _build_target_path(
        self,
        original_filename: Optional[str],
        prefix: str,
        suffix: str,
    ) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        default_name = f"{prefix}-{timestamp}{suffix}"

        if original_filename:
            candidate = Path(original_filename).name.strip()
            if candidate:
                stem = Path(candidate).stem or prefix
                extension = suffix if suffix.startswith(".") else f".{suffix}"
                if candidate.lower().endswith(extension.lower()):
                    name = candidate
                else:
                    name = f"{stem}{extension}"
                target = self._data_dir / name
                if target.exists():
                    name = f"{stem}-{timestamp}{extension}"
                return self._data_dir / name

        return self._data_dir / default_name

    def _persist_file(
        self,
        raw_bytes: bytes,
        original_filename: Optional[str],
        prefix: str,
        suffix: str,
        exclusive: bool = False,
    ) -> Path:
        self._ensure_data_dir()
        if exclusive:
            # Remove any existing files that share the prefix/suffix.
            extension = suffix if suffix.startswith(".") else f".{suffix}"
            pattern = f"{prefix}*{extension}"
            for existing in self._data_dir.glob(pattern):
                existing.unlink(missing_ok=True)
            target_path = self._data_dir / f"{prefix}{extension}"
        else:
            target_path = self._build_target_path(original_filename, prefix, suffix)
        target_path.write_bytes(raw_bytes)
        return target_path

    def _write_users_json(self, users: List[Dict[str, Any]]) -> Path:
        self._ensure_data_dir()
        target_path = self._data_dir / "users.json"
        target_path.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")
        return target_path

    def _load_users_from_json(
        self,
        json_path: Path,
        source_label: str,
        bump_version: bool = False,
        persist_metadata: bool = True,
    ) -> None:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid users JSON at {json_path}") from exc

        if not isinstance(payload, list):
            raise ValueError("Users JSON must be a list of objects.")

        normalized: List[Dict[str, str]] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            user_id = entry.get("user_id")
            name = entry.get("name")
            if user_id is None or name is None:
                continue
            normalized.append({"user_id": str(user_id), "name": str(name)})

        with self._lock:
            self._users = normalized
            self._source = source_label
            if bump_version:
                self._bump_version()
            else:
                self._last_updated = self._now_utc()
            self._refresh_app_metadata(persist=persist_metadata)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def load_initial_data(self) -> None:
        """Load default dataset from the data directory if present."""
        default_chat_export = self._find_latest_chat_export()
        if default_chat_export is not None:
            try:
                raw_bytes = default_chat_export.read_bytes()
                self.update_chat_export(
                    raw_bytes,
                    f"default:{default_chat_export.name}",
                    bump_version=False,
                    record_upload=False,
                    persist_metadata=False,
                    run_summarizer=False,
                )
            except Exception:
                # Leave dataset empty but continue boot if default fails
                self._source = f"failed to load {default_chat_export.name}"

        default_users_json = self._find_users_json()
        if default_users_json is not None:
            try:
                self._load_users_from_json(
                    default_users_json,
                    f"default:{default_users_json.name}",
                    bump_version=False,
                    persist_metadata=False,
                )
            except Exception:
                pass
        else:
            default_users_csv = self._find_users_file()
            if default_users_csv is not None:
                try:
                    raw_bytes = default_users_csv.read_bytes()
                    self.update_users(
                        raw_bytes,
                        f"default:{default_users_csv.name}",
                        bump_version=False,
                        persist_filename=default_users_csv.name,
                        record_upload=False,
                        persist_metadata=False,
                    )
                except Exception:
                    pass

        # Always ensure metadata is current even if no files loaded
        self._last_updated = datetime.utcnow().replace(tzinfo=timezone.utc)

    def _find_latest_chat_export(self) -> Optional[Path]:
        if not self._data_dir.exists():
            return None
        candidates = sorted(
            self._data_dir.glob("all-chats-export*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _find_users_file(self) -> Optional[Path]:
        if not self._data_dir.exists():
            return None
        users_file = self._data_dir / "users.csv"
        return users_file if users_file.exists() else None

    def _find_users_json(self) -> Optional[Path]:
        if not self._data_dir.exists():
            return None
        users_json = self._data_dir / "users.json"
        return users_json if users_json.exists() else None

    # ------------------------------------------------------------------
    # Dataset mutations
    # ------------------------------------------------------------------
    def update_chat_export(
        self,
        raw_bytes: bytes,
        source_label: Optional[str],
        bump_version: bool = True,
        persist_filename: Optional[str] = None,
        record_upload: bool = True,
        persist_metadata: bool = True,
        run_summarizer: bool = True,
    ) -> None:
        """Replace the current chat/messages dataset."""
        logging.getLogger(__name__).info(
            "update_chat_export invoked (run_summarizer=%s, source=%s)", run_summarizer, source_label
        )
        chats, messages = _parse_chat_export(raw_bytes)
        summary_map: Dict[str, str] = {}
        for chat in chats:
            chat_id = str(chat.get("chat_id") or "")
            if not chat_id:
                continue
            summary_text = str(chat.get("summary_128") or "").strip()
            if summary_text:
                summary_map[chat_id] = summary_text
        logging.getLogger(__name__).info(
            "Summaries prepared for %d chats (total=%d)", len(summary_map), len(chats)
        )
        export_bytes = _annotate_export_bytes(raw_bytes, summary_map)
        with self._lock:
            if record_upload:
                self._dataset_source_override = None
            self._chats = chats
            self._messages = messages
            if persist_filename is not None:
                saved_path = self._persist_file(
                    export_bytes,
                    persist_filename,
                    "all-chats-export",
                    ".json",
                    exclusive=True,
                )
                self._source = f"upload:{saved_path.name}"
            else:
                self._source = source_label or "chat export upload"
            if bump_version:
                self._bump_version()
            else:
                self._last_updated = self._now_utc()
            self._refresh_app_metadata(chat_updated=record_upload, persist=persist_metadata)

        if run_summarizer:
            self._enqueue_summary_job("chat export upload")

    def update_users(
        self,
        raw_bytes: bytes,
        source_label: Optional[str],
        bump_version: bool = True,
        persist_filename: Optional[str] = None,
        record_upload: bool = True,
        persist_metadata: bool = True,
    ) -> None:
        """Replace the current user metadata dataset."""
        users = _parse_users_csv(raw_bytes)
        with self._lock:
            self._users = users
            if persist_filename is not None:
                saved_path = self._persist_file(
                    raw_bytes,
                    persist_filename,
                    "users",
                    ".csv",
                    exclusive=True,
                )
            else:
                saved_path = None

            json_path = self._write_users_json(users)

            if saved_path is not None:
                self._source = f"upload:{json_path.name}"
            else:
                self._source = source_label or f"json:{json_path.name}"
            if bump_version:
                self._bump_version()
            else:
                self._last_updated = self._now_utc()
            self._refresh_app_metadata(users_updated=record_upload, persist=persist_metadata)

    def clear_users(self) -> None:
        """Clear user metadata."""
        with self._lock:
            self._users = []
            self._users_uploaded_at = None
            self._bump_version()
            self._refresh_app_metadata()

    def reset_dataset(self) -> DatasetMeta:
        """Remove all stored dataset artifacts and reset in-memory state."""
        now_iso = self._summary_now_iso()
        with self._summary_state_lock:
            self._summary_target_dataset_id = None
            self._summary_active_job_id = None
            self._summary_thread = None
            self._summary_state.update(
                {
                    "state": "idle",
                    "total": 0,
                    "completed": 0,
                    "message": "Dataset cleared. No data loaded.",
                    "last_event": None,
                    "started_at": None,
                    "finished_at": now_iso,
                    "updated_at": now_iso,
                }
            )

        with self._lock:
            if self._data_dir.exists():
                for pattern in ("all-chats-export*.json", "users.csv", "users.json"):
                    for candidate in self._data_dir.glob(pattern):
                        try:
                            candidate.unlink(missing_ok=True)
                        except OSError:
                            pass
                if self._app_metadata_path.exists():
                    try:
                        self._app_metadata_path.unlink(missing_ok=True)
                    except OSError:
                        pass

            self._chats = []
            self._messages = []
            self._users = []
            self._dataset_source_override = None
            self._dataset_pulled_at = None
            self._chat_uploaded_at = None
            self._users_uploaded_at = None
            self._first_chat_day = None
            self._last_chat_day = None
            self._source = "not loaded"
            self._bump_version()
            self._refresh_app_metadata(persist=True)

        return self.get_meta()

    def sync_from_openwebui(self, hostname: str, api_key: Optional[str]) -> DatasetMeta:
        """Pull chats and users directly from an Open WebUI instance."""
        base_url = _normalize_hostname(hostname)
        headers = {"Accept": "application/json"}
        token = (api_key or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        base_candidates = _build_openwebui_base_candidates(base_url)
        if not base_candidates:
            raise RuntimeError("No valid hostname candidates available for Open WebUI.")

        attempt_messages: List[str] = []
        chats_payload: Any = None
        users_payload: Any = None

        with requests.Session() as session:
            session.headers.update(headers)
            for candidate in base_candidates:
                chats_endpoint = f"{candidate}/api/v1/chats/all/db"
                users_endpoint = f"{candidate}/api/v1/users/all"
                try:
                    chats_payload = _fetch_openwebui_json(session, chats_endpoint)
                    users_payload = _fetch_openwebui_json(session, users_endpoint)
                except RuntimeError as exc:
                    attempt_messages.append(f"{candidate}: {exc}")
                    underlying = exc.__cause__
                    if isinstance(underlying, requests.exceptions.RequestException):
                        # Connection-level issue; try next candidate if available.
                        chats_payload = None
                        users_payload = None
                        continue
                    raise RuntimeError(
                        "Open WebUI responded with an unexpected payload."
                    ) from exc
                else:
                    break

        if chats_payload is None or users_payload is None:
            attempts_detail = "; ".join(attempt_messages) or "no attempts recorded"
            raise RuntimeError(
                f"Unable to connect to Open WebUI with the provided hostname. Attempts: {attempts_detail}"
            )

        export_payload = _build_export_payload_from_openwebui(chats_payload)
        export_bytes = json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8")
        chats, messages = _parse_chat_export(export_bytes)
        summary_map: Dict[str, str] = {}
        for chat in chats:
            chat_id = str(chat.get("chat_id") or "")
            if not chat_id:
                continue
            summary_text = str(chat.get("summary_128") or "").strip()
            if summary_text:
                summary_map[chat_id] = summary_text
        _apply_summaries_to_export_payload(export_payload, summary_map)
        export_bytes = json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8")

        normalized_users = _normalize_openwebui_users(users_payload)

        with self._lock:
            self._dataset_source_override = None
            self._chats = chats
            self._messages = messages
            self._users = normalized_users
            self._ensure_data_dir()
            self._persist_file(
                export_bytes,
                "all-chats-export-openwebui.json",
                "all-chats-export",
                ".json",
                exclusive=True,
            )
            self._write_users_json(normalized_users)
            self._source = f"openwebui:{base_url}"
            self._bump_version()
            self._refresh_app_metadata(chat_updated=True, users_updated=True)

        self._enqueue_summary_job("Open WebUI sync")
        return self.get_meta()

    def rebuild_summaries(self) -> Dict[str, Any]:
        """Trigger asynchronous summarization on the current dataset."""
        return self._enqueue_summary_job("manual rebuild")

    def _bump_version(self) -> None:
        self._dataset_id = uuid4().hex
        self._last_updated = self._now_utc()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_chats(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [chat.copy() for chat in self._chats]

    def get_messages(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [message.copy() for message in self._messages]

    def get_users(self) -> List[Dict[str, str]]:
        with self._lock:
            return [user.copy() for user in self._users]

    def get_meta(self) -> DatasetMeta:
        with self._lock:
            return DatasetMeta(
                dataset_id=self._dataset_id,
                source=self._source,
                last_updated=self._last_updated,
                chat_count=len(self._chats),
                message_count=len(self._messages),
                user_count=len(self._users),
                app_metadata=self._build_app_metadata(),
            )


# Singleton instance shared across the application
data_service = DataService(data_dir=DATA_DIR)


def get_data_service() -> DataService:
    """FastAPI dependency to retrieve the shared data service."""
    return data_service
