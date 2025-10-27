"""Data service for the Open WebUI Chat Analyzer backend."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import hmac
import math
import os
import secrets
from collections import deque
from datetime import datetime, timezone, date, timedelta
from io import BytesIO
from pathlib import Path
from threading import Lock, Thread, RLock
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple
from uuid import uuid4

import pandas as pd
import requests
from urllib.parse import urlparse, urlunparse

from .config import AUTH_TOKEN_HASH_SECRET, AUTH_TOKEN_TTL_SECONDS, DATA_DIR
from .models import AppMetadata, DatasetMeta, DatasetSyncStats
from .summarizer import summarize_chats, set_summary_model, get_summary_model, set_summary_temperature, get_summary_temperature
from .storage import DatabaseState, DatabaseStorage

LOGGER = logging.getLogger(__name__)


class SummaryJobCancelled(Exception):
    """Raised when a summary job is cancelled due to a newer dataset."""


SUMMARY_EXPORT_FIELD = "owca_gen_chat_summary"
SUMMARY_FIELD = "gen_chat_summary"
DEFAULT_DIRECT_CONNECT_HOST = "http://localhost:4000"
SUMMARY_EVENT_HISTORY_LIMIT = 500
SUMMARIZER_MODEL_SETTING_KEY = "SUMMARIZER_PRIMARY_MODEL"
SUMMARIZER_TEMPERATURE_SETTING_KEY = "SUMMARIZER_TEMPERATURE"


def _strip_encapsulating_quotes(value: str) -> str:
    """Remove leading and trailing double quotes from a string value.

    This function defensively strips quotes that may have been accidentally
    serialized when storing configuration values. It only removes quotes
    if they appear at both the beginning and end of the string.

    Args:
        value: The string value to process.

    Returns:
        The string with encapsulating quotes removed, or the original value.

    Examples:
        >>> _strip_encapsulating_quotes('"http://localhost:4000"')
        'http://localhost:4000'
        >>> _strip_encapsulating_quotes('http://localhost:4000')
        'http://localhost:4000'
        >>> _strip_encapsulating_quotes('""')
        ''
    """
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
        return stripped[1:-1]
    return stripped


def _get_chat_summary(payload: Dict[str, Any]) -> str:
    """Retrieve the normalized chat summary from a payload."""
    value = payload.get(SUMMARY_FIELD)
    if value is None:
        return ""
    return str(value).strip()


def _set_chat_summary(payload: Dict[str, Any], summary: str) -> None:
    """Persist a chat summary to a payload."""
    payload[SUMMARY_FIELD] = summary


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


def _normalize_bool(value: Any) -> Optional[bool]:
    """Interpret loose boolean-like values and return a strict boolean when possible."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in {"true", "1", "yes", "y"}:
            return True
        if candidate in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    return None


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

    def _build_message(chat_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None

        message_id_raw = (
            payload.get("id")
            or payload.get("message_id")
            or payload.get("_id")
            or payload.get("uuid")
        )
        if message_id_raw in (None, "", []):
            return None
        message_id = str(message_id_raw)

        parent_raw = payload.get("parentId") or payload.get("parent_id") or payload.get("parent")
        parent_id = None
        if parent_raw not in (None, "", []):
            parent_id = str(parent_raw)

        children_ids: List[str] = []
        for child in _ensure_list(payload.get("childrenIds") or payload.get("children_ids")):
            child_str = str(child).strip()
            if child_str and child_str.lower() != "none":
                children_ids.append(child_str)

        follow_ups: List[str] = []
        for follow_up in _ensure_list(payload.get("followUps")):
            if isinstance(follow_up, str):
                follow_candidate = follow_up.strip()
                if follow_candidate:
                    follow_ups.append(follow_candidate)

        status_history = [
            entry for entry in _ensure_list(payload.get("statusHistory")) if isinstance(entry, dict)
        ]
        sources = [
            entry for entry in _ensure_list(payload.get("sources")) if isinstance(entry, dict)
        ]
        attached_files = [
            entry for entry in _ensure_list(payload.get("files")) if isinstance(entry, dict)
        ]

        annotation = payload.get("annotation")
        if not isinstance(annotation, dict):
            annotation = None

        error_value = payload.get("error")
        if isinstance(error_value, str):
            error_value = {"message": error_value}
        elif not isinstance(error_value, dict):
            error_value = None

        done = _normalize_bool(payload.get("done"))
        favorite = _normalize_bool(payload.get("favorite"))

        feedback_id = payload.get("feedbackId")
        if feedback_id in (None, "", []):
            feedback_id = None
        else:
            feedback_id = str(feedback_id)

        model_name = payload.get("modelName")
        if model_name in (None, "", []):
            model_name = None
        else:
            model_name = str(model_name)

        model_index: Optional[int]
        model_idx_raw = payload.get("modelIdx")
        try:
            model_index = int(model_idx_raw)
        except (TypeError, ValueError):
            model_index = None

        last_sentence = payload.get("lastSentence")
        if last_sentence in (None, "", []):
            last_sentence = None
        else:
            last_sentence = str(last_sentence)

        models: List[str] = []
        for model_entry in _ensure_list(payload.get("models")):
            model_str = str(model_entry).strip()
            if model_str:
                models.append(model_str)

        timestamp = _coerce_timestamp(
            payload.get("timestamp") or payload.get("created_at") or payload.get("updated_at")
        )
        content_value = (
            payload.get("content")
            or payload.get("message")
            or payload.get("text")
            or payload.get("body")
        )

        message_info: Dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "parent_id": parent_id,
            "role": str(payload.get("role") or payload.get("type") or payload.get("sender") or ""),
            "content": _stringify_content(content_value),
            "timestamp": timestamp,
            "model": str(payload.get("model") or payload.get("model_id") or payload.get("assistant") or ""),
            "models": models,
            "children_ids": children_ids,
            "follow_ups": follow_ups,
            "status_history": status_history,
            "sources": sources,
            "files": attached_files,
            "annotation": annotation,
            "model_name": model_name,
            "model_index": model_index,
            "last_sentence": last_sentence,
            "done": done,
            "favorite": favorite,
            "feedback_id": feedback_id,
            "error": error_value,
        }
        return message_info

    for item in data:
        if not isinstance(item, dict):
            continue
        chat_meta = item.get("chat", {}) if isinstance(item.get("chat"), dict) else {}
        files_raw = chat_meta.get("files")
        files: List[Any] = list(files_raw) if isinstance(files_raw, list) else []

        meta_sections: List[Dict[str, Any]] = []
        if isinstance(item.get("meta"), dict):
            meta_sections.append(dict(item["meta"]))
        chat_meta_meta = chat_meta.get("meta")
        if isinstance(chat_meta_meta, dict):
            meta_sections.append(dict(chat_meta_meta))

        merged_meta: Dict[str, Any] = {}
        tags: List[str] = []
        for meta_section in meta_sections:
            merged_meta.update(meta_section)
            for tag in _ensure_list(meta_section.get("tags")):
                tag_str = str(tag).strip()
                if tag_str and tag_str not in tags:
                    tags.append(tag_str)
        for tag in _ensure_list(chat_meta.get("tags")):
            tag_str = str(tag).strip()
            if tag_str and tag_str not in tags:
                tags.append(tag_str)

        chat_id_raw = item.get("id") or chat_meta.get("id")
        chat_id = str(chat_id_raw or "")

        user_id_raw = item.get("user_id") or chat_meta.get("user_id")
        user_id = str(user_id_raw) if user_id_raw not in (None, "", []) else ""

        share_id_raw = item.get("share_id") or chat_meta.get("share_id")
        share_id = str(share_id_raw) if share_id_raw not in (None, "", []) else None

        folder_id_raw = item.get("folder_id") or chat_meta.get("folder_id")
        folder_id = str(folder_id_raw) if folder_id_raw not in (None, "", []) else None

        models: List[str] = []
        for model_entry in _ensure_list(chat_meta.get("models")):
            model_str = str(model_entry).strip()
            if model_str:
                models.append(model_str)

        params = dict(chat_meta.get("params")) if isinstance(chat_meta.get("params"), dict) else {}

        created_at = _coerce_timestamp(chat_meta.get("created_at") or item.get("created_at"))
        updated_at = _coerce_timestamp(chat_meta.get("updated_at") or item.get("updated_at"))
        timestamp_value = _coerce_timestamp(chat_meta.get("timestamp") or item.get("timestamp"))

        archived = bool(chat_meta.get("archived") or item.get("archived", False))
        pinned = bool(chat_meta.get("pinned") or item.get("pinned", False))

        history = chat_meta.get("history") if isinstance(chat_meta.get("history"), dict) else {}
        history_current_id_raw = history.get("currentId")
        history_current_id = (
            str(history_current_id_raw) if history_current_id_raw not in (None, "", []) else None
        )
        history_messages_raw = history.get("messages")
        history_messages: Dict[str, Dict[str, Any]] = {}
        if isinstance(history_messages_raw, dict):
            for msg_id, payload in history_messages_raw.items():
                if not isinstance(payload, dict):
                    continue
                msg_key = str(msg_id) if msg_id not in (None, "", []) else str(payload.get("id", ""))
                if not msg_key:
                    continue
                merged_payload = dict(payload)
                merged_payload.setdefault("id", msg_key)
                history_messages[msg_key] = merged_payload

        chat_info: Dict[str, Any] = {
            "chat_id": chat_id,
            "user_id": user_id,
            "title": item.get("title") or chat_meta.get("title") or "",
            "created_at": created_at,
            "updated_at": updated_at,
            "timestamp": timestamp_value,
            "archived": archived,
            "pinned": pinned,
            "tags": tags,
            "files_uploaded": len(files),
            "files": files,
            "meta": merged_meta,
            "models": models,
            "params": params,
            "share_id": share_id,
            "folder_id": folder_id,
            "history_current_id": history_current_id,
        }
        chat_summary_value = item.get(SUMMARY_EXPORT_FIELD)
        if not isinstance(chat_summary_value, str):
            chat_summary_value = chat_meta.get(SUMMARY_EXPORT_FIELD)
        if isinstance(chat_summary_value, str):
            _set_chat_summary(chat_info, str(chat_summary_value))

        chats.append(chat_info)

        processed_ids: Set[str] = set()
        chat_messages = chat_meta.get("messages", [])
        if isinstance(chat_messages, list):
            for msg in chat_messages:
                if not isinstance(msg, dict):
                    continue
                msg_id_raw = msg.get("id")
                msg_id = str(msg_id_raw) if msg_id_raw not in (None, "", []) else None
                merged_payload = dict(history_messages.get(msg_id, {}))
                merged_payload.update(msg)
                if msg_id:
                    merged_payload["id"] = msg_id
                message_info = _build_message(chat_id, merged_payload)
                if message_info:
                    messages.append(message_info)
                    if msg_id:
                        processed_ids.add(msg_id)

        for msg_id, payload in history_messages.items():
            if msg_id in processed_ids:
                continue
            message_info = _build_message(chat_id, payload)
            if message_info:
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

    # Extract email and role columns if they exist
    email_col = next((col for col in ("email",) if col in users_df.columns), None)
    role_col = next((col for col in ("role",) if col in users_df.columns), None)

    # Build the list of columns to extract
    cols_to_extract = [id_col, name_col]
    rename_map = {id_col: "user_id", name_col: "name"}

    if email_col:
        cols_to_extract.append(email_col)
        rename_map[email_col] = "email"

    if role_col:
        cols_to_extract.append(role_col)
        rename_map[role_col] = "role"

    normalized = (
        users_df[cols_to_extract]
        .rename(columns=rename_map)
        .dropna(subset=["user_id"])
        .drop_duplicates(subset=["user_id"], keep="first")
    )

    normalized["user_id"] = normalized["user_id"].astype(str)
    normalized["name"] = normalized["name"].astype(str).str.strip()

    if "email" in normalized.columns:
        normalized["email"] = normalized["email"].astype(str).replace({"nan": None, "": None})

    if "role" in normalized.columns:
        normalized["role"] = normalized["role"].astype(str).replace({"nan": None, "": None})

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

        # Extract email and role if available
        email = entry.get("email")
        role = entry.get("role")

        user_dict = {"user_id": str(user_id), "name": str(name)}
        if email:
            user_dict["email"] = str(email)
        if role:
            user_dict["role"] = str(role)

        normalized.append(user_dict)
    return normalized


def _normalize_models_payload(payload: Any) -> List[Dict[str, Any]]:
    """Transform Open WebUI model payloads into analyzer-compatible structures."""
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            entries = payload.get("data", [])
        elif isinstance(payload.get("models"), list):
            entries = payload.get("models", [])
        else:
            entries = []
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []

    normalized: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        info_section = entry.get("info") if isinstance(entry.get("info"), dict) else {}
        model_id = (
            entry.get("model_id")
            or entry.get("id")
            or info_section.get("model_id")
            or info_section.get("id")
        )
        if model_id in (None, "", []):
            continue
        model_id_str = str(model_id)
        if model_id_str in seen:
            continue
        seen.add(model_id_str)
        name = entry.get("name") or info_section.get("name") or model_id_str
        owned_by = entry.get("owned_by") or info_section.get("owned_by")
        connection_type = entry.get("connection_type") or info_section.get("connection_type")
        object_type = entry.get("object") or info_section.get("object")
        raw_payload = entry.get("raw") if isinstance(entry.get("raw"), dict) else entry
        normalized.append(
            {
                "model_id": model_id_str,
                "name": str(name) if name not in (None, "") else model_id_str,
                "owned_by": str(owned_by) if owned_by not in (None, "") else None,
                "connection_type": str(connection_type) if connection_type not in (None, "") else None,
                "object": str(object_type) if object_type not in (None, "") else None,
                "raw": raw_payload,
            }
        )
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


def _extract_chat_id_from_export_entry(entry: Any) -> Optional[str]:
    """Return the chat identifier from an export entry if available."""
    if not isinstance(entry, dict):
        return None
    chat_id = entry.get("id") or entry.get("_id") or entry.get("chat_id")
    if chat_id in (None, "", []):
        chat_section = entry.get("chat")
        if isinstance(chat_section, dict):
            chat_id = chat_section.get("id")
    if chat_id in (None, "", []):
        return None
    return str(chat_id)


def _normalize_source_for_compare(value: Optional[str]) -> Optional[Tuple[str, str]]:
    """Normalize URL-like values to support source comparisons."""
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    try:
        normalized = _normalize_hostname(candidate)
    except ValueError:
        return None
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower()
    if not host:
        return None
    if host in {"host.docker.internal", "127.0.0.1", "::1"}:
        host = "localhost"
    port = parsed.port
    netloc = f"{host}:{port}" if port else host
    path = parsed.path.rstrip("/") or ""
    return netloc, path


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
        entry[SUMMARY_EXPORT_FIELD] = summary
        chat_section = entry.get("chat")
        if isinstance(chat_section, dict):
            chat_section[SUMMARY_EXPORT_FIELD] = summary
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
        # Use a re-entrant lock so helper methods that update shared state can
        # safely call each other without deadlocking during nested acquisitions.
        self._lock = RLock()
        if data_dir is not None:
            self._data_dir = Path(data_dir).resolve()
        else:
            self._data_dir = Path(__file__).resolve().parent.parent / "data"
        LOGGER.info("Initializing DataService (data_dir=%s)", self._data_dir)
        self._chats: List[Dict[str, Any]] = []
        self._messages: List[Dict[str, Any]] = []
        self._users: List[Dict[str, str]] = []
        self._models: List[Dict[str, Any]] = []
        self._auth_users: Dict[str, Dict[str, Any]] = {}
        self._auth_users_updated_at: Optional[datetime] = None
        self._auth_token_ttl = timedelta(seconds=AUTH_TOKEN_TTL_SECONDS)
        self._auth_token_hash_secret = (
            AUTH_TOKEN_HASH_SECRET.encode("utf-8") if AUTH_TOKEN_HASH_SECRET else None
        )
        self._dataset_id = uuid4().hex
        self._source = "no dataset loaded"
        self._source_origin: Optional[Tuple[str, str]] = None
        self._last_updated = datetime.utcnow().replace(tzinfo=timezone.utc)
        self._app_metadata_path = self._data_dir / "app.json"
        self._dataset_source_override: Optional[str] = None
        self._dataset_pulled_at: Optional[datetime] = None
        self._chat_uploaded_at: Optional[datetime] = None
        self._users_uploaded_at: Optional[datetime] = None
        self._models_uploaded_at: Optional[datetime] = None
        self._first_chat_day: Optional[date] = None
        self._last_chat_day: Optional[date] = None
        self._summary_state_lock = Lock()
        self._settings: Dict[str, Any] = {}
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
        self._summary_events: Deque[Dict[str, Any]] = deque(maxlen=SUMMARY_EVENT_HISTORY_LIMIT)
        self._process_logs: Deque[Dict[str, Any]] = deque(maxlen=200)  # Keep last 200 log events
        self._summary_thread: Optional[Thread] = None
        self._summary_target_dataset_id: Optional[str] = None
        self._summary_job_counter: int = 0
        self._summary_active_job_id: Optional[int] = None
        # Scheduler state (loaded from settings)
        self._scheduler_enabled: bool = False
        self._scheduler_interval_minutes: int = 60  # Default 1 hour
        self._scheduler_last_run_at: Optional[datetime] = None
        try:
            self._storage = DatabaseStorage()
        except Exception:  # pylint: disable=broad-except
            LOGGER.exception("Failed to initialise database storage layer")
            raise
        self._hydrate_from_storage()
        self._ensure_direct_connect_defaults_seeded()
        self._load_scheduler_config()
        with self._lock:
            self._refresh_app_metadata(persist=True)
        self._log_dataset_summary("DataService ready")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _snapshot_counts(self) -> Dict[str, int]:
        with self._lock:
            return {
                "chats": len(self._chats),
                "messages": len(self._messages),
                "users": len(self._users),
                "models": len(self._models),
            }

    def _log_dataset_summary(self, prefix: str) -> None:
        counts = self._snapshot_counts()
        LOGGER.info(
            "%s: %d chats, %d messages, %d users, %d models",
            prefix,
            counts["chats"],
            counts["messages"],
            counts["users"],
            counts["models"],
        )

    def _emit_log(
        self,
        level: str,
        phase: str,
        message: str,
        job_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a structured log event to the process log buffer.

        Args:
            level: Log severity level (debug, info, warning, error)
            phase: Current operation phase (connect, fetch, persist, summarize, done, error)
            message: Human-readable log message (secrets must be pre-redacted)
            job_id: Optional job identifier for filtering
            details: Optional additional structured details
        """
        event = {
            "timestamp": self._now_utc(),
            "level": level,
            "phase": phase,
            "message": message,
            "job_id": job_id,
            "details": details,
        }
        with self._lock:
            self._process_logs.append(event)

        # Also log to standard logger for server-side visibility
        log_level = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }.get(level, logging.INFO)
        LOGGER.log(log_level, "[%s] %s", phase.upper(), message)

    def get_process_logs(
        self,
        job_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent process log events from the buffer.

        Args:
            job_id: Optional job ID to filter logs
            limit: Maximum number of logs to return (most recent first)

        Returns:
            List of log event dictionaries, ordered newest to oldest
        """
        with self._lock:
            # Convert deque to list (newest last in deque)
            all_logs = list(self._process_logs)

        # Filter by job_id if specified
        if job_id is not None:
            filtered = [log for log in all_logs if log.get("job_id") == job_id]
        else:
            filtered = all_logs

        # Reverse to get newest first
        filtered.reverse()

        # Apply limit
        if limit is not None and limit > 0:
            filtered = filtered[:limit]

        return filtered

    def _apply_database_state(self, state: DatabaseState) -> None:
        """Populate in-memory caches from the hydrated database state."""
        legacy_updates: Dict[str, Any] = {}
        with self._lock:
            self._chats = [chat.copy() for chat in state.chats]
            for chat in self._chats:
                summary_value = _get_chat_summary(chat)
                _set_chat_summary(chat, summary_value)
            self._messages = [message.copy() for message in state.messages]
            self._users = [user.copy() for user in state.users]
            self._models = [model.copy() for model in state.models]

            self._auth_users = {}
            latest_auth_ts: Optional[datetime] = None
            for account in state.accounts:
                username = account.get("username")
                password_hash = account.get("password_hash")
                if not isinstance(username, str) or not isinstance(password_hash, str):
                    continue
                display_name = account.get("display_name") or username
                created_at = account.get("created_at")
                updated_at = account.get("updated_at") or created_at
                if isinstance(updated_at, datetime):
                    candidate_ts = updated_at
                elif isinstance(created_at, datetime):
                    candidate_ts = created_at
                else:
                    candidate_ts = None
                if candidate_ts is not None and (
                    latest_auth_ts is None or candidate_ts > latest_auth_ts
                ):
                    latest_auth_ts = candidate_ts
                self._auth_users[username] = {
                    "username": username,
                    "display_name": display_name,
                    "password_hash": password_hash,
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            self._auth_users_updated_at = latest_auth_ts

            snapshot = state.snapshot or {}
            self._dataset_source_override = snapshot.get("dataset_source")
            self._dataset_pulled_at = snapshot.get("dataset_pulled_at")
            self._chat_uploaded_at = snapshot.get("chats_uploaded_at")
            self._users_uploaded_at = snapshot.get("users_uploaded_at")
            self._models_uploaded_at = snapshot.get("models_uploaded_at")
            self._first_chat_day = snapshot.get("first_chat_day")
            self._last_chat_day = snapshot.get("last_chat_day")

            # Preserve the last known dataset source for logging / API payloads.
            if isinstance(self._dataset_source_override, str):
                self._source = self._dataset_source_override
            elif snapshot.get("dataset_source"):
                self._source = snapshot["dataset_source"]

            # Note: dataset_source_display setting has been deprecated
            # The dataset source is now derived from OWUI_DIRECT_HOST

            self._settings = {key: value for key, value in state.settings.items()}
            legacy_host = self._settings.get("owui_direct_host")
            if legacy_host is not None and "OWUI_DIRECT_HOST" not in self._settings:
                self._settings["OWUI_DIRECT_HOST"] = legacy_host
                legacy_updates["OWUI_DIRECT_HOST"] = legacy_host
            legacy_api_key = self._settings.get("owui_direct_api_key")
            if legacy_api_key is not None and "OWUI_DIRECT_API_KEY" not in self._settings:
                self._settings["OWUI_DIRECT_API_KEY"] = legacy_api_key
                legacy_updates["OWUI_DIRECT_API_KEY"] = legacy_api_key

            self._last_updated = self._now_utc()

        if legacy_updates:
            self._storage.write_settings(legacy_updates)
        self._log_dataset_summary("Dataset hydration complete")
        self._apply_summarizer_setting_from_state()

    def _ensure_direct_connect_defaults_seeded(self) -> None:
        """Ensure the settings table stores Direct Connect defaults."""
        env_host = os.getenv("OWUI_DIRECT_HOST", "").strip()
        env_api_key = os.getenv("OWUI_DIRECT_API_KEY", "").strip()
        default_host = env_host or DEFAULT_DIRECT_CONNECT_HOST

        with self._lock:
            has_host_setting = any(
                key in self._settings for key in ("OWUI_DIRECT_HOST", "owui_direct_host")
            )
            has_key_setting = any(
                key in self._settings for key in ("OWUI_DIRECT_API_KEY", "owui_direct_api_key")
            )

        updates: Dict[str, Any] = {}
        if not has_host_setting:
            updates["OWUI_DIRECT_HOST"] = default_host
        if not has_key_setting:
            updates["OWUI_DIRECT_API_KEY"] = env_api_key

        if updates:
            self._storage.write_settings(updates)
            with self._lock:
                self._settings.update(updates)
            LOGGER.info(
                "Seeded Direct Connect defaults (host=%s, api_key_present=%s)",
                updates.get("OWUI_DIRECT_HOST"),
                bool(updates.get("OWUI_DIRECT_API_KEY")),
            )

    def _apply_summarizer_setting_from_state(self) -> None:
        """Ensure the summarizer module reflects stored or environment defaults."""
        with self._lock:
            raw_setting = self._settings.get(SUMMARIZER_MODEL_SETTING_KEY)
        stored_value = (
            _strip_encapsulating_quotes(raw_setting.strip()) if isinstance(raw_setting, str) else ""
        )
        env_value = _strip_encapsulating_quotes(os.getenv("OLLAMA_SUMMARY_MODEL", "").strip())
        if stored_value:
            active = stored_value
            source = "database"
        elif env_value:
            active = env_value
            source = "environment"
        else:
            active = ""
            source = "default"
        set_summary_model(active or None)
        if active:
            LOGGER.info("Configured summarizer primary model (model=%s, source=%s)", active, source)
        else:
            LOGGER.info("Using default summarizer primary model (model=%s)", get_summary_model())

    def _seed_initial_state_from_files(self) -> Optional[DatabaseState]:
        """Populate the database from legacy app.json metadata if necessary."""
        if not self._app_metadata_path.exists():
            return None
        LOGGER.info(
            "Attempting to seed database from legacy metadata file: %s",
            self._app_metadata_path,
        )
        try:
            payload = json.loads(self._app_metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Failed to read legacy metadata file: %s", exc)
            return None

        settings_payload: Dict[str, Any] = {}
        if isinstance(payload, dict):
            for key in ("owui_direct_host", "owui_direct_api_key"):
                value = payload.get(key)
                if value is not None:
                    settings_payload[key.upper()] = value

        snapshot_payload: Dict[str, Any] = {}
        if isinstance(payload, dict):
            snapshot_payload = {
                "dataset_source": payload.get("dataset_source"),
                "dataset_pulled_at": self._parse_datetime(payload.get("dataset_pulled_at")),
                "chats_uploaded_at": self._parse_datetime(
                    (payload.get("chats") or {}).get("uploaded_at")
                ),
                "users_uploaded_at": self._parse_datetime(
                    (payload.get("users") or {}).get("uploaded_at")
                ),
                "models_uploaded_at": self._parse_datetime(
                    (payload.get("models") or {}).get("uploaded_at")
                ),
                "first_chat_day": self._parse_date((payload.get("chats") or {}).get("first_day")),
                "last_chat_day": self._parse_date((payload.get("chats") or {}).get("last_day")),
                "chat_count": (payload.get("chats") or {}).get("count") or 0,
                "user_count": (payload.get("users") or {}).get("count") or 0,
                "model_count": (payload.get("models") or {}).get("count") or 0,
                "message_count": (payload.get("messages") or {}).get("count") or 0,
            }

        accounts_payload: List[Dict[str, Any]] = []
        auth_section = payload.get("auth_users") if isinstance(payload, dict) else None
        if isinstance(auth_section, dict):
            for record in auth_section.get("records") or []:
                if not isinstance(record, dict):
                    continue
                username = record.get("username")
                password_hash = record.get("password_hash")
                if not username or not password_hash:
                    continue
                accounts_payload.append(
                    {
                        "username": str(username),
                        "display_name": str(record.get("name") or username),
                        "password_hash": str(password_hash),
                        "is_active": True,
                    }
                )

        if settings_payload:
            self._storage.write_settings(settings_payload)
        if snapshot_payload:
            self._storage.record_snapshot(snapshot_payload)
        if accounts_payload:
            self._storage.replace_accounts(accounts_payload)

        LOGGER.info("Legacy metadata import succeeded; reloading state from database")
        return self._storage.load_state()

    def _hydrate_from_storage(self) -> None:
        """Ensure the in-memory caches reflect the persistent database state."""
        LOGGER.info("Hydrating DataService state from database")
        try:
            state = self._storage.load_state()
        except Exception:  # pylint: disable=broad-except
            LOGGER.exception("Failed to load dataset state from database")
            raise
        if (
            not state.chats
            and not state.users
            and not state.models
            and not state.accounts
            and state.snapshot is None
        ):
            LOGGER.info("Database state empty; attempting to seed from legacy metadata")
            seeded_state = self._seed_initial_state_from_files()
            if seeded_state is not None:
                state = seeded_state
            else:
                LOGGER.info("No legacy metadata available; continuing with empty dataset")
        self._apply_database_state(state)

    def _persist_auth_users(self) -> None:
        accounts: List[Dict[str, Any]] = []
        for record in sorted(self._auth_users.values(), key=lambda item: item["username"]):
            accounts.append(
                {
                    "username": record["username"],
                    "display_name": record.get("display_name") or record["username"],
                    "password_hash": record["password_hash"],
                    "created_at": record.get("created_at"),
                    "updated_at": record.get("updated_at"),
                    "is_active": True,
                }
            )
        self._storage.replace_accounts(accounts)

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

    @staticmethod
    def _normalize_username(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Username must be a string.")
        username = value.strip().lower()
        if not username:
            raise ValueError("Username or email is required.")
        return username

    @staticmethod
    def _hash_password(password: str) -> str:
        if not isinstance(password, str) or not password:
            raise ValueError("Password must not be empty.")
        salt = secrets.token_bytes(16)
        iterations = 150_000
        key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        salt_b64 = base64.urlsafe_b64encode(salt).decode("utf-8").rstrip("=")
        key_b64 = base64.urlsafe_b64encode(key).decode("utf-8").rstrip("=")
        return f"pbkdf2_sha256${iterations}${salt_b64}${key_b64}"

    @staticmethod
    def _verify_password(password: str, encoded: str) -> bool:
        if not isinstance(password, str) or not isinstance(encoded, str):
            return False
        try:
            algorithm, iterations_str, salt_b64, key_b64 = encoded.split("$", 3)
        except ValueError:
            return False
        if algorithm != "pbkdf2_sha256":
            return False
        try:
            iterations = int(iterations_str)
        except ValueError:
            return False
        padding = "=" * (-len(salt_b64) % 4)
        key_padding = "=" * (-len(key_b64) % 4)
        try:
            salt = base64.urlsafe_b64decode(salt_b64 + padding)
            expected_key = base64.urlsafe_b64decode(key_b64 + key_padding)
        except (ValueError, TypeError):
            return False
        candidate_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(candidate_key, expected_key)

    def _hash_access_token(self, token: str) -> str:
        if not isinstance(token, str) or not token:
            raise ValueError("Token must be a non-empty string.")
        token_bytes = token.encode("utf-8")
        secret = self._auth_token_hash_secret
        if secret:
            return hmac.new(secret, token_bytes, hashlib.sha256).hexdigest()
        return hashlib.sha256(token_bytes).hexdigest()

    def _save_app_metadata(self, payload: Dict[str, Any]) -> None:
        """Persist aggregate dataset metadata into the database."""
        chats_section = payload.get("chats") or {}
        users_section = payload.get("users") or {}
        models_section = payload.get("models") or {}
        snapshot_payload = {
            "dataset_source": payload.get("dataset_source"),
            "dataset_pulled_at": self._parse_datetime(payload.get("dataset_pulled_at")),
            "chats_uploaded_at": self._parse_datetime(chats_section.get("uploaded_at")),
            "users_uploaded_at": self._parse_datetime(users_section.get("uploaded_at")),
            "models_uploaded_at": self._parse_datetime(models_section.get("uploaded_at")),
            "first_chat_day": self._parse_date(chats_section.get("first_day")),
            "last_chat_day": self._parse_date(chats_section.get("last_day")),
            "chat_count": chats_section.get("count") or 0,
            "user_count": users_section.get("count") or 0,
            "model_count": models_section.get("count") or 0,
            "message_count": payload.get("message_count") or len(self._messages),
        }
        self._storage.record_snapshot(snapshot_payload)
        # Note: dataset_source_display setting has been deprecated
        # The dataset source is now derived from OWUI_DIRECT_HOST

    def _determine_dataset_source(self) -> str:
        source = (self._source or "").strip()
        if source.startswith("openwebui:"):
            return source.split(":", 1)[1].strip() or "local upload"
        if not self._chats:
            return "not loaded"
        return "local upload"

    def _get_dataset_source_host(self) -> str:
        """Get the dataset source from OWUI_DIRECT_HOST setting.

        This replaces the deprecated dataset_source_display setting.
        Returns the configured OpenWebUI host URL.
        """
        settings = self.get_direct_connect_settings()
        return settings.get("host", "not configured")

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
        models_updated: bool = False,
        persist: bool = True,
        force_timestamp: bool = False,
    ) -> None:
        now = self._now_utc()
        chat_count = len(self._chats)
        user_count = len(self._users)
        model_count = len(self._models)

        update_chat_timestamp = chat_updated or (force_timestamp and chat_count > 0)
        update_user_timestamp = users_updated or (force_timestamp and user_count > 0)
        update_model_timestamp = models_updated or (force_timestamp and model_count > 0)

        if update_chat_timestamp:
            self._chat_uploaded_at = now
        if update_user_timestamp:
            self._users_uploaded_at = now
        if update_model_timestamp:
            self._models_uploaded_at = now

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
        if model_count == 0:
            self._models_uploaded_at = None
        else:
            self._models_uploaded_at = self._models_uploaded_at or now

        if chat_count == 0 and user_count == 0 and model_count == 0:
            self._dataset_pulled_at = None
        elif chat_updated or users_updated or models_updated or force_timestamp:
            self._dataset_pulled_at = now
        elif self._dataset_pulled_at is None:
            self._dataset_pulled_at = (
                self._chat_uploaded_at
                or self._users_uploaded_at
                or self._models_uploaded_at
            )

        should_recalculate_source = (
            chat_updated
            or users_updated
            or models_updated
            or force_timestamp
            or self._dataset_source_override is None
        )
        if should_recalculate_source:
            display_source = self._determine_dataset_source()
            self._dataset_source_override = display_source
        else:
            display_source = self._get_dataset_source_host()

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
                "models": {
                    "count": model_count,
                    "uploaded_at": self._serialize_datetime(self._models_uploaded_at),
                },
            }
            self._save_app_metadata(payload)

    def _build_app_metadata(self) -> AppMetadata:
        return AppMetadata(
            dataset_source=self._get_dataset_source_host(),
            dataset_pulled_at=self._dataset_pulled_at,
            chat_uploaded_at=self._chat_uploaded_at,
            users_uploaded_at=self._users_uploaded_at,
            models_uploaded_at=self._models_uploaded_at,
            first_chat_day=self._first_chat_day,
            last_chat_day=self._last_chat_day,
            chat_count=len(self._chats),
            user_count=len(self._users),
            model_count=len(self._models),
            auth_user_count=len(self._auth_users),
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
                event_payload = dict(event)
                event_id_raw = event_payload.get("event_id")
                if not isinstance(event_id_raw, str) or not event_id_raw.strip():
                    event_payload["event_id"] = f"evt-{uuid4().hex}"
                timestamp_raw = event_payload.get("timestamp")
                if not isinstance(timestamp_raw, str) or not timestamp_raw.strip():
                    event_payload["timestamp"] = self._summary_now_iso()
                event_payload.setdefault("job_id", self._summary_active_job_id)
                self._summary_state["last_event"] = event_payload
                self._summary_events.append(dict(event_payload))
            self._summary_state["updated_at"] = self._summary_now_iso()

    def get_summary_status(self) -> Dict[str, Any]:
        with self._summary_state_lock:
            return dict(self._summary_state)

    def get_summary_events(self, after: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Return a slice of recent summary events queued by the worker.

        Args:
            after: Optional cursor referencing the last seen event identifier.
            limit: Optional maximum number of events to return (most recent first).
        Returns:
            Dict[str, Any]: Payload containing events and a reset flag when cursors are stale.
        """
        with self._summary_state_lock:
            events_snapshot = [dict(event) for event in self._summary_events]

        reset = False
        cursor = (after or "").strip()
        if cursor:
            idx = None
            for position, event in enumerate(events_snapshot):
                event_id = str(event.get("event_id") or "")
                if event_id == cursor:
                    idx = position
                    break
            if idx is None:
                reset = True
            else:
                events_snapshot = events_snapshot[idx + 1 :]

        if limit is not None and isinstance(limit, int) and limit > 0:
            events_snapshot = events_snapshot[-limit:]

        return {
            "events": events_snapshot,
            "reset": reset,
        }

    def _enqueue_summary_job(
        self,
        reason: str,
        chat_ids: Optional[Iterable[str]] = None,
        *,
        force_resummarize: bool = False,
    ) -> Dict[str, Any]:
        chat_id_filter: Optional[Set[str]] = None
        if chat_ids is not None:
            chat_id_filter = {str(chat_id).strip() for chat_id in chat_ids if str(chat_id).strip()}
            if not chat_id_filter:
                self._update_summary_state(
                    state="idle",
                    message="No eligible chats queued for summarization.",
                    total=0,
                    completed=0,
                    started_at=None,
                    finished_at=self._summary_now_iso(),
                )
                return self.get_summary_status()

        with self._lock:
            if chat_id_filter is not None:
                chats_snapshot = [
                    chat.copy()
                    for chat in self._chats
                    if str(chat.get("chat_id") or "").strip() in chat_id_filter
                ]
                messages_snapshot = [
                    message.copy()
                    for message in self._messages
                    if str(message.get("chat_id") or "").strip() in chat_id_filter
                ]
            else:
                chats_snapshot = [chat.copy() for chat in self._chats]
                messages_snapshot = [message.copy() for message in self._messages]
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

        logger = LOGGER
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
                "job_id": job_id,
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
                if payload_type == "chunk":
                    chat_id_value = payload.get("chat_id") or "unknown"
                    chunk_index = payload.get("chunk_index")
                    chunk_count = payload.get("chunk_count")
                    if isinstance(chunk_index, int) and isinstance(chunk_count, int) and chunk_count > 0:
                        return f"Summarizing chunk {chunk_index}/{chunk_count} for chat {chat_id_value}"
                    return f"Summarizing chat {chat_id_value}"
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
                event_payload["job_id"] = job_id
                event_payload.setdefault("completed", current)
                event_payload.setdefault("total", total_count)
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
            args=(
                chats_snapshot,
                messages_snapshot,
                dataset_id,
                reason,
                job_id,
                force_resummarize,
                progress,
            ),
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
        target_dataset_id: str,
        reason: str,
        job_id: int,
        force_resummarize: bool,
        progress_cb: Callable[[int, int, str, str, Optional[Dict[str, Any]]], None],
    ) -> None:
        logger = LOGGER
        logger.info(
            "Summary job %s started for %d chats (%s)",
            job_id,
            len(chats_snapshot),
            reason,
        )

        def chat_persist_callback(chat_summary: Dict[str, str], chat_outcomes: Dict[str, int]) -> None:
            """Immediately persist individual chat summary and outcome to database and update in-memory records.

            Args:
                chat_summary: Dict mapping chat_id to summary text
                chat_outcomes: Dict mapping chat_id to outcome score (1-5)
            """
            if not chat_summary:
                return

            # Update in-memory chat records with both summary and outcome
            with self._lock:
                for chat in self._chats:
                    chat_id = str(chat.get("chat_id") or "")
                    if chat_id in chat_summary:
                        _set_chat_summary(chat, chat_summary[chat_id])
                        # Update outcome if present
                        if chat_id in chat_outcomes:
                            chat["gen_chat_outcome"] = chat_outcomes[chat_id]

            # Persist to database immediately with both summary and outcome
            self._storage.update_chat_summaries(chat_summary, chat_outcomes)

        try:
            summary_map, stats = summarize_chats(
                chats_snapshot,
                messages_snapshot,
                on_progress=progress_cb,
                replace_existing=force_resummarize,
                on_chat_complete=chat_persist_callback,
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

            # Collect outcomes from the chats that were just summarized
            outcome_map: Dict[str, int] = {}
            with self._lock:
                for chat in self._chats:
                    chat_id = str(chat.get("chat_id") or "")
                    if chat_id in summary_map:
                        _set_chat_summary(chat, summary_map[chat_id])
                        # Collect outcome if present
                        outcome_value = chat.get("gen_chat_outcome")
                        if isinstance(outcome_value, int) and 1 <= outcome_value <= 5:
                            outcome_map[chat_id] = outcome_value
                self._bump_version()
                self._refresh_app_metadata(persist=False)

            self._storage.update_chat_summaries(summary_map, outcome_map)

            logger.info("Summary job %s finished successfully.", job_id)
            completed_at = self._summary_now_iso()
            self._update_summary_state(
                state="completed",
                message="Summary job complete.",
                total=stats["total"],
                completed=stats["total"],
                finished_at=completed_at,
                event={
                    "type": "complete",
                    "event_id": f"complete-{job_id}-{uuid4().hex[:8]}",
                    "timestamp": completed_at,
                    "message": "Summary job complete.",
                    "job_id": job_id,
                    "total": stats["total"],
                    "completed": stats["total"],
                },
            )
        except SummaryJobCancelled:
            logger.info("Summary job %s cancelled before completion.", job_id)
        except Exception as exc:
            logger.exception("Summary job failed: %s", exc)
            failed_at = self._summary_now_iso()
            self._update_summary_state(
                state="failed",
                message=str(exc),
                finished_at=failed_at,
                event={
                    "type": "error",
                    "event_id": f"error-{job_id}-{uuid4().hex[:8]}",
                    "timestamp": failed_at,
                    "message": str(exc),
                    "job_id": job_id,
                },
            )
        finally:
            with self._summary_state_lock:
                if self._summary_active_job_id == job_id:
                    self._summary_thread = None
                    self._summary_target_dataset_id = None
                    self._summary_active_job_id = None

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def load_initial_data(self) -> None:
        """Hydrate the in-memory caches from the database."""
        LOGGER.info("Refreshing DataService state during startup")
        self._hydrate_from_storage()
        self._last_updated = datetime.utcnow().replace(tzinfo=timezone.utc)
        self._log_dataset_summary("Initial data load")

    # ------------------------------------------------------------------
    # Dataset mutations
    # ------------------------------------------------------------------
    def _merge_openwebui_dataset(
        self,
        base_url: str,
        chats: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        normalized_users: List[Dict[str, str]],
        normalized_models: List[Dict[str, Any]],
    ) -> Tuple[List[str], int, List[str], List[str], List[str], bool]:
        new_chat_ids: List[str] = []
        new_user_ids: List[str] = []
        new_model_ids: List[str] = []
        new_message_count = 0
        chats_missing_summary: Set[str] = set()
        models_changed = False

        with self._lock:
            existing_chat_ids: Set[str] = set()
            existing_chats_index: Dict[str, int] = {}
            for idx, chat in enumerate(self._chats):
                chat_id_raw = chat.get("chat_id")
                chat_id = str(chat_id_raw) if chat_id_raw not in (None, "", []) else ""
                if chat_id:
                    existing_chat_ids.add(chat_id)
                    existing_chats_index[chat_id] = idx

            existing_message_ids: Set[str] = set()
            for message in self._messages:
                message_id_raw = message.get("message_id")
                message_id = str(message_id_raw) if message_id_raw not in (None, "", []) else ""
                if message_id:
                    existing_message_ids.add(message_id)

            for chat in chats:
                chat_id_raw = chat.get("chat_id")
                chat_id = str(chat_id_raw) if chat_id_raw not in (None, "", []) else ""
                if not chat_id:
                    continue
                summary_text = _get_chat_summary(chat)
                _set_chat_summary(chat, summary_text)
                if chat_id in existing_chat_ids:
                    existing_chat = self._chats[existing_chats_index[chat_id]]
                    previous_summary = _get_chat_summary(existing_chat)
                    existing_chat.update(chat)
                    current_summary = _get_chat_summary(existing_chat)
                    if previous_summary and not current_summary:
                        _set_chat_summary(existing_chat, previous_summary)
                        current_summary = previous_summary
                    if not summary_text and not current_summary:
                        chats_missing_summary.add(chat_id)
                else:
                    self._chats.append(chat)
                    existing_chat_ids.add(chat_id)
                    existing_chats_index[chat_id] = len(self._chats) - 1
                    new_chat_ids.append(chat_id)
                    if not summary_text:
                        chats_missing_summary.add(chat_id)

            for message in messages:
                message_id_raw = message.get("message_id")
                message_id = str(message_id_raw) if message_id_raw not in (None, "", []) else ""
                if not message_id or message_id in existing_message_ids:
                    continue
                self._messages.append(message)
                existing_message_ids.add(message_id)
                new_message_count += 1
                chat_id_value = str(message.get("chat_id") or "").strip()
                if chat_id_value:
                    target_idx = existing_chats_index.get(chat_id_value)
                    if target_idx is not None:
                        target_chat = self._chats[target_idx]
                        target_summary = _get_chat_summary(target_chat)
                        if not target_summary:
                            chats_missing_summary.add(chat_id_value)

            existing_user_ids: Set[str] = set()
            for user in self._users:
                user_id_raw = user.get("user_id")
                user_id = str(user_id_raw) if user_id_raw not in (None, "", []) else ""
                if user_id:
                    existing_user_ids.add(user_id)

            for user in normalized_users:
                user_id_raw = user.get("user_id")
                user_id = str(user_id_raw) if user_id_raw not in (None, "", []) else ""
                if not user_id or user_id in existing_user_ids:
                    continue
                name = str(user.get("name") or "")
                self._users.append({"user_id": user_id, "name": name})
                existing_user_ids.add(user_id)
                new_user_ids.append(user_id)

            existing_models_index: Dict[str, int] = {}
            updated_models: List[Dict[str, Any]] = list(self._models)
            for idx, model in enumerate(updated_models):
                model_id_raw = model.get("model_id")
                model_id = str(model_id_raw) if model_id_raw not in (None, "", []) else ""
                if model_id:
                    existing_models_index[model_id] = idx

            new_model_id_set: Set[str] = set()
            for model in normalized_models:
                model_id_raw = model.get("model_id")
                model_id = str(model_id_raw) if model_id_raw not in (None, "", []) else ""
                if not model_id:
                    continue
                existing_idx = existing_models_index.get(model_id)
                if existing_idx is None:
                    updated_models.append(model)
                    existing_models_index[model_id] = len(updated_models) - 1
                    if model_id not in new_model_id_set:
                        new_model_ids.append(model_id)
                        new_model_id_set.add(model_id)
                    models_changed = True
                else:
                    if updated_models[existing_idx] != model:
                        updated_models[existing_idx] = model
                        models_changed = True

            if models_changed:
                self._models = updated_models

            self._dataset_source_override = None
            self._source = f"openwebui:{base_url}"
            if new_chat_ids or new_user_ids or new_model_ids:
                self._bump_version()
            else:
                self._last_updated = self._now_utc()

            force_timestamp = not new_chat_ids and not new_user_ids and not new_model_ids
            self._refresh_app_metadata(
                chat_updated=bool(new_chat_ids),
                users_updated=bool(new_user_ids),
                models_updated=bool(new_model_ids) or models_changed,
                force_timestamp=force_timestamp,
            )

        return (
            new_chat_ids,
            new_message_count,
            new_user_ids,
            new_model_ids,
            sorted(chats_missing_summary),
            models_changed,
        )

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
        LOGGER.info(
            "update_chat_export invoked (run_summarizer=%s, source=%s)", run_summarizer, source_label
        )
        chats, messages = _parse_chat_export(raw_bytes)
        summary_map: Dict[str, str] = {}
        for chat in chats:
            chat_id = str(chat.get("chat_id") or "")
            if not chat_id:
                continue
            summary_text = _get_chat_summary(chat)
            _set_chat_summary(chat, summary_text)
            if summary_text:
                summary_map[chat_id] = summary_text
        LOGGER.info(
            "Summaries prepared for %d chats (total=%d)", len(summary_map), len(chats)
        )
        self._storage.replace_dataset(chats, messages)
        with self._lock:
            if record_upload:
                self._dataset_source_override = None
            self._chats = chats
            self._messages = messages
            self._source = source_label or "chat export upload"
            self._source_origin = None
            if bump_version:
                self._bump_version()
            else:
                self._last_updated = self._now_utc()
            self._refresh_app_metadata(chat_updated=record_upload, persist=persist_metadata)
            missing_summary_chat_ids = [
                str(chat.get("chat_id") or "").strip()
                for chat in self._chats
                if str(chat.get("chat_id") or "").strip()
                and not _get_chat_summary(chat)
            ]
        self._storage.record_ingest(
            operation="chat_export_replace",
            source=self._source,
            record_count=len(chats),
            details={
                "messages": len(messages),
                "source_label": source_label,
            },
        )

        if run_summarizer and missing_summary_chat_ids:
            self._enqueue_summary_job("chat export upload", chat_ids=missing_summary_chat_ids)

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
        self._storage.replace_users(users)
        with self._lock:
            self._users = users
            self._source = source_label or "users upload"
            if bump_version:
                self._bump_version()
            else:
                self._last_updated = self._now_utc()
            self._refresh_app_metadata(users_updated=record_upload, persist=persist_metadata)
        self._storage.record_ingest(
            operation="users_replace",
            source=self._source,
            record_count=len(users),
            details={"source_label": source_label},
        )

    def clear_users(self) -> None:
        """Clear user metadata."""
        self._storage.replace_users([])
        with self._lock:
            self._users = []
            self._users_uploaded_at = None
            self._bump_version()
            self._refresh_app_metadata()

    def update_models(
        self,
        raw_bytes: bytes,
        source_label: Optional[str],
        bump_version: bool = True,
        record_upload: bool = True,
        persist_metadata: bool = True,
    ) -> None:
        """Replace the current model metadata dataset."""
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("Invalid models JSON.") from exc

        normalized = _normalize_models_payload(payload)
        self._storage.replace_models(normalized)

        with self._lock:
            self._models = normalized
            if source_label:
                self._source = source_label
            if bump_version:
                self._bump_version()
            else:
                self._last_updated = self._now_utc()
            models_present = bool(normalized)
            self._refresh_app_metadata(
                models_updated=record_upload and models_present,
                persist=persist_metadata,
                force_timestamp=record_upload and models_present,
            )
        self._storage.record_ingest(
            operation="models_replace",
            source=self._source,
            record_count=len(normalized),
            details={"source_label": source_label},
        )

    def clear_models(self) -> None:
        """Clear model metadata."""
        self._storage.replace_models([])
        with self._lock:
            self._models = []
            self._models_uploaded_at = None
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

        self._storage.replace_dataset([])
        self._storage.replace_users([])
        self._storage.replace_models([])

        with self._lock:
            self._chats = []
            self._messages = []
            self._users = []
            self._models = []
            self._dataset_source_override = None
            self._dataset_pulled_at = None
            self._chat_uploaded_at = None
            self._users_uploaded_at = None
            self._models_uploaded_at = None
            self._first_chat_day = None
            self._last_chat_day = None
            self._source = "not loaded"
            self._source_origin = None
            self._bump_version()
            self._refresh_app_metadata(persist=True)

        self._storage.record_ingest(
            operation="dataset_reset",
            source="manual-reset",
            record_count=0,
            details=None,
        )

        return self.get_meta()

    def sync_from_openwebui(self, hostname: str, api_key: Optional[str]) -> Tuple[DatasetMeta, DatasetSyncStats]:
        """Pull chats, users, and models directly from an Open WebUI instance."""
        # Generate job ID for this sync operation
        job_id = uuid4().hex[:12]

        try:
            self._emit_log("info", "connect", f"Starting sync from {hostname}", job_id=job_id)

            base_url = _normalize_hostname(hostname)
            headers = {"Accept": "application/json"}
            token = (api_key or "").strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"
                self._emit_log("debug", "connect", "Using authenticated connection", job_id=job_id)

            base_candidates = _build_openwebui_base_candidates(base_url)
            if not base_candidates:
                raise RuntimeError("No valid hostname candidates available for Open WebUI.")

            attempt_messages: List[str] = []
            chats_payload: Any = None
            users_payload: Any = None
            models_payload: Any = None

            with requests.Session() as session:
                session.headers.update(headers)
                for candidate in base_candidates:
                    self._emit_log("debug", "connect", f"Trying endpoint: {candidate}", job_id=job_id)
                    chats_endpoint = f"{candidate}/api/v1/chats/all/db"
                    users_endpoint = f"{candidate}/api/v1/users/all"
                    models_endpoint = f"{candidate}/api/v1/models"
                    try:
                        chats_payload = _fetch_openwebui_json(session, chats_endpoint)
                        users_payload = _fetch_openwebui_json(session, users_endpoint)
                        try:
                            models_payload = _fetch_openwebui_json(session, models_endpoint)
                        except RuntimeError as models_exc:
                            LOGGER.warning(
                                "Model inventory request failed for %s: %s", candidate, models_exc
                            )
                            self._emit_log("warning", "fetch", f"Models endpoint failed: {models_exc}", job_id=job_id)
                            models_payload = []
                    except RuntimeError as exc:
                        attempt_messages.append(f"{candidate}: {exc}")
                        self._emit_log("warning", "connect", f"Connection attempt failed: {exc}", job_id=job_id)
                        underlying = exc.__cause__
                        if isinstance(underlying, requests.exceptions.RequestException):
                            # Connection-level issue; try next candidate if available.
                            chats_payload = None
                            users_payload = None
                            models_payload = None
                            continue
                        raise RuntimeError(
                            "Open WebUI responded with an unexpected payload."
                        ) from exc
                    else:
                        self._emit_log("info", "fetch", f"Successfully connected to {candidate}", job_id=job_id)
                        break

            if chats_payload is None or users_payload is None:
                attempts_detail = "; ".join(attempt_messages) or "no attempts recorded"
                raise RuntimeError(
                    f"Unable to connect to Open WebUI with the provided hostname. Attempts: {attempts_detail}"
                )

            self._emit_log("info", "fetch", f"Fetched {len(chats_payload) if isinstance(chats_payload, list) else '?'} chats", job_id=job_id, details={"users": len(users_payload) if isinstance(users_payload, list) else 0})

            export_payload = _build_export_payload_from_openwebui(chats_payload)
            export_bytes = json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8")
            chats, messages = _parse_chat_export(export_bytes)

            normalized_users = _normalize_openwebui_users(users_payload)
            normalized_models = _normalize_models_payload(models_payload)

            requested_origin = _normalize_source_for_compare(base_url)
            with self._lock:
                if self._source_origin is None:
                    origin_seed = None
                    if isinstance(self._source, str) and self._source.startswith("openwebui:"):
                        parts = self._source.split(":", 1)
                        if len(parts) == 2:
                            origin_seed = parts[1]
                    if origin_seed is None:
                        display_candidate = self._get_dataset_source_host()
                        origin_seed = display_candidate
                    self._source_origin = _normalize_source_for_compare(origin_seed)
                current_origin = self._source_origin
                current_source_display = self._get_dataset_source_host()

            same_source = False
            if requested_origin is not None and current_origin is not None:
                requested_netloc, _ = requested_origin
                current_netloc, _ = current_origin
                same_source = requested_netloc == current_netloc

            if same_source:
                self._emit_log("info", "persist", "Performing incremental sync (same source)", job_id=job_id)
                (
                    new_chat_ids,
                    new_message_count,
                    new_user_ids,
                    new_model_ids,
                    chats_missing_summary,
                    models_changed,
                ) = self._merge_openwebui_dataset(
                    base_url,
                    chats,
                    messages,
                    normalized_users,
                    normalized_models,
                )
                self._storage.replace_dataset(self._chats, self._messages)
                self._storage.replace_users(self._users)
                self._storage.replace_models(self._models)
                self._storage.record_ingest(
                    operation="openwebui_incremental",
                    source=f"openwebui:{base_url}",
                    record_count=len(self._chats),
                    details={
                        "new_chats": len(new_chat_ids),
                        "new_messages": new_message_count,
                        "new_users": len(new_user_ids),
                        "new_models": len(new_model_ids),
                        "models_changed": models_changed,
                    },
                )
                summarizer_enqueued = False
                if chats_missing_summary:
                    self._enqueue_summary_job(
                        f"Open WebUI sync (+{len(new_chat_ids)} new chats)",
                        chat_ids=chats_missing_summary,
                    )
                    summarizer_enqueued = True
                elif new_user_ids or new_model_ids or models_changed:
                    self._update_summary_state(
                        state="idle",
                        message="No new chats detected; user/model data updated.",
                        total=0,
                        completed=0,
                        started_at=None,
                        finished_at=self._summary_now_iso(),
                    )
                else:
                    self._update_summary_state(
                        state="idle",
                        message="No new chats detected; dataset already up to date.",
                        total=0,
                        completed=0,
                        started_at=None,
                        finished_at=self._summary_now_iso(),
                    )
                mode = (
                    "incremental"
                    if (new_chat_ids or new_user_ids or new_model_ids or models_changed)
                    else "noop"
                )
                dataset_meta = self.get_meta()
                stats = DatasetSyncStats(
                    mode=mode,
                    source_matched=True,
                    submitted_hostname=str(hostname),
                    normalized_hostname=base_url,
                    source_display=dataset_meta.source,
                    new_chats=len(new_chat_ids),
                    new_messages=new_message_count,
                    new_users=len(new_user_ids),
                    new_models=len(new_model_ids),
                    models_changed=models_changed,
                    summarizer_enqueued=summarizer_enqueued,
                    total_chats=dataset_meta.chat_count,
                    total_messages=dataset_meta.message_count,
                    total_users=dataset_meta.user_count,
                    total_models=dataset_meta.model_count,
                    queued_chat_ids=chats_missing_summary if summarizer_enqueued else None,
                )
                with self._lock:
                    self._source_origin = requested_origin
                self._emit_log("info", "done", f"Incremental sync complete: {len(new_chat_ids)} new chats, {new_message_count} new messages", job_id=job_id)
                return dataset_meta, stats

            self._emit_log("info", "persist", "Performing full sync (different source or first sync)", job_id=job_id)
            self._storage.replace_dataset(chats, messages)
            self._storage.replace_users(normalized_users)
            self._storage.replace_models(normalized_models)

            with self._lock:
                self._dataset_source_override = None
                self._chats = chats
                self._messages = messages
                self._users = normalized_users
                self._models = normalized_models
                missing_summary_chat_ids = [
                    str(chat.get("chat_id") or "").strip()
                    for chat in self._chats
                    if str(chat.get("chat_id") or "").strip()
                    and not _get_chat_summary(chat)
                ]
                self._source = f"openwebui:{base_url}"
                self._source_origin = requested_origin
                self._bump_version()
                self._refresh_app_metadata(chat_updated=True, users_updated=True, models_updated=bool(self._models))

            dataset_meta = self.get_meta()
            summarizer_enqueued = False
            if missing_summary_chat_ids:
                self._enqueue_summary_job("Open WebUI sync", chat_ids=missing_summary_chat_ids)
                summarizer_enqueued = True
            self._storage.record_ingest(
                operation="openwebui_full",
                source=f"openwebui:{base_url}",
                record_count=len(chats),
                details={
                    "messages": len(messages),
                    "users": len(normalized_users),
                    "models": len(normalized_models),
                },
            )
            stats = DatasetSyncStats(
                mode="full",
                source_matched=False,
                submitted_hostname=str(hostname),
                normalized_hostname=base_url,
                source_display=dataset_meta.source,
                new_chats=len(chats),
                new_messages=len(messages),
                new_users=len(normalized_users),
                new_models=len(normalized_models),
                models_changed=bool(normalized_models),
                summarizer_enqueued=summarizer_enqueued,
                total_chats=dataset_meta.chat_count,
                total_messages=dataset_meta.message_count,
                total_users=dataset_meta.user_count,
                total_models=dataset_meta.model_count,
                queued_chat_ids=missing_summary_chat_ids if summarizer_enqueued else None,
            )
            self._emit_log("info", "done", f"Full sync complete: {len(chats)} chats, {len(messages)} messages", job_id=job_id)
            return dataset_meta, stats
        except Exception as exc:
            self._emit_log("error", "error", f"Sync failed: {str(exc)}", job_id=job_id)
            raise

    def rebuild_summaries(self) -> Dict[str, Any]:
        """Trigger asynchronous summarization on the current dataset."""
        return self._enqueue_summary_job("manual rebuild", force_resummarize=True)

    # ------------------------------------------------------------------
    # Admin settings
    # ------------------------------------------------------------------
    def get_direct_connect_settings(self) -> Dict[str, Any]:
        """Return Direct Connect defaults with their origin metadata."""
        env_host = _strip_encapsulating_quotes(os.getenv("OWUI_DIRECT_HOST", "").strip())
        env_api_key = _strip_encapsulating_quotes(os.getenv("OWUI_DIRECT_API_KEY", "").strip())

        with self._lock:
            stored_host = self._settings.get("OWUI_DIRECT_HOST") or self._settings.get("owui_direct_host")
            stored_api_key = self._settings.get("OWUI_DIRECT_API_KEY") or self._settings.get("owui_direct_api_key")

        # Strip quotes defensively from stored values
        host_value = _strip_encapsulating_quotes((stored_host or "").strip()) if isinstance(stored_host, str) else ""
        api_key_value = _strip_encapsulating_quotes((stored_api_key or "").strip()) if isinstance(stored_api_key, str) else ""

        if host_value:
            host = host_value
            host_source = "database"
        elif env_host:
            host = env_host
            host_source = "environment"
        else:
            host = DEFAULT_DIRECT_CONNECT_HOST
            host_source = "default"

        if api_key_value:
            api_key = api_key_value
            api_key_source = "database"
        elif env_api_key:
            api_key = env_api_key
            api_key_source = "environment"
        else:
            api_key = ""
            api_key_source = "empty"

        database_host = host_value if host_value else ""
        database_api_key = api_key_value if api_key_value else ""

        return {
            "host": host,
            "api_key": api_key,
            "database_host": database_host,
            "database_api_key": database_api_key,
            "host_source": host_source,
            "api_key_source": api_key_source,
        }

    def get_summarizer_settings(self) -> Dict[str, Any]:
        """Return the active summarizer configuration with its origin metadata."""
        env_model = _strip_encapsulating_quotes(os.getenv("OLLAMA_SUMMARY_MODEL", "").strip())
        with self._lock:
            stored_raw = self._settings.get(SUMMARIZER_MODEL_SETTING_KEY)
        stored_model = (
            _strip_encapsulating_quotes(stored_raw.strip()) if isinstance(stored_raw, str) else ""
        )

        if stored_model:
            model = stored_model
            model_source = "database"
        elif env_model:
            model = env_model
            model_source = "environment"
        else:
            model = get_summary_model()
            model_source = "default"

        # Handle temperature setting
        env_temp_str = os.getenv("OLLAMA_DEFAULT_TEMPERATURE", "").strip()
        env_temp = None
        if env_temp_str:
            try:
                env_temp = float(env_temp_str)
            except ValueError:
                env_temp = None

        with self._lock:
            stored_temp_raw = self._settings.get(SUMMARIZER_TEMPERATURE_SETTING_KEY)
        stored_temp = None
        if isinstance(stored_temp_raw, (int, float)):
            stored_temp = float(stored_temp_raw)

        if stored_temp is not None:
            temperature = stored_temp
            temperature_source = "database"
        elif env_temp is not None:
            temperature = env_temp
            temperature_source = "environment"
        else:
            temperature = get_summary_temperature()
            temperature_source = "default"

        return {
            "model": model,
            "temperature": temperature,
            "model_source": model_source,
            "temperature_source": temperature_source,
        }

    def update_direct_connect_settings(self, *, host: Optional[str], api_key: Optional[str]) -> Dict[str, Any]:
        """Persist updated Direct Connect defaults and return the effective values.

        If the hostname changes, this triggers a transactional wipe and full reload:
        1. Detect hostname change
        2. Wipe all OpenWebUI-sourced data (chats, messages, users, models)
        3. Trigger a full sync from the new hostname
        4. Emit structured log events throughout the process
        """
        # Get current settings to detect changes
        current_settings = self.get_direct_connect_settings()
        current_host = current_settings.get("host", "")

        updates: Dict[str, Any] = {}
        hostname_changed = False
        new_hostname = None

        if host is not None:
            # Strip quotes and whitespace from host
            normalized_host = _strip_encapsulating_quotes(host.strip())
            updates["OWUI_DIRECT_HOST"] = normalized_host
            new_hostname = normalized_host

            # Detect hostname change (normalize both for comparison)
            current_normalized = _normalize_hostname(current_host) if current_host else ""
            new_normalized = _normalize_hostname(normalized_host) if normalized_host else ""
            hostname_changed = (current_normalized != new_normalized) and current_normalized != ""

            LOGGER.info("Updating Direct Connect host setting (host=%s, changed=%s)", normalized_host, hostname_changed)

        if api_key is not None:
            # Strip quotes and whitespace from API key
            normalized_api_key = _strip_encapsulating_quotes(api_key.strip())
            updates["OWUI_DIRECT_API_KEY"] = normalized_api_key
            # Log API key update without exposing the actual value
            redacted = f"{normalized_api_key[:4]}...{normalized_api_key[-4:]}" if len(normalized_api_key) > 8 else "***"
            LOGGER.info("Updating Direct Connect API key setting (redacted=%s)", redacted)

        if updates:
            self._storage.write_settings(updates)
            with self._lock:
                self._settings.update(updates)

        # If hostname changed, trigger wipe and reload
        if hostname_changed and new_hostname:
            job_id = uuid4().hex[:12]
            self._emit_log("info", "persist", f"Hostname changed to {new_hostname}, triggering wipe and reload", job_id=job_id)

            try:
                # Step 1: Wipe existing data
                self._emit_log("info", "persist", "Starting transactional wipe of OpenWebUI data", job_id=job_id)
                self._storage.wipe_openwebui_data()

                # Clear in-memory state
                with self._lock:
                    self._chats = []
                    self._messages = []
                    self._users = []
                    self._models = []
                    self._source_origin = None
                    self._dataset_source_override = None
                    self._dataset_pulled_at = None
                    self._bump_version()
                    self._refresh_app_metadata(chat_updated=True, users_updated=True, models_updated=True)

                self._emit_log("info", "persist", "Wipe complete", job_id=job_id)

                # Step 2: Trigger full reload
                self._emit_log("info", "fetch", f"Starting full reload from {new_hostname}", job_id=job_id)

                # Get the API key (either the new one or existing one)
                reload_api_key = api_key if api_key is not None else current_settings.get("api_key", "")

                # Trigger sync (this will emit its own log events)
                self.sync_from_openwebui(new_hostname, reload_api_key)

                self._emit_log("info", "done", "Hostname change complete: wipe and reload finished", job_id=job_id)

            except Exception as exc:
                self._emit_log("error", "error", f"Wipe and reload failed: {str(exc)}", job_id=job_id)
                LOGGER.exception("Failed to wipe and reload after hostname change")
                # Re-raise to let the caller handle the error
                raise

        return self.get_direct_connect_settings()

    def update_summarizer_settings(self, *, model: Optional[str] = None, temperature: Optional[float] = None) -> Dict[str, Any]:
        """Persist the summarizer model and/or temperature preference and return the effective values."""
        updates: Dict[str, Any] = {}

        if model is not None:
            normalized_model = _strip_encapsulating_quotes((model or "").strip())
            if not normalized_model:
                raise ValueError("Model must not be empty.")
            updates[SUMMARIZER_MODEL_SETTING_KEY] = normalized_model

        if temperature is not None:
            if not (0.0 <= temperature <= 2.0):
                raise ValueError("Temperature must be between 0.0 and 2.0.")
            updates[SUMMARIZER_TEMPERATURE_SETTING_KEY] = temperature

        if not updates:
            raise ValueError("At least one of model or temperature must be provided.")

        self._storage.write_settings(updates)
        with self._lock:
            self._settings.update(updates)

        if model is not None:
            set_summary_model(updates[SUMMARIZER_MODEL_SETTING_KEY])
            LOGGER.info("Updated summarizer primary model (model=%s)", updates[SUMMARIZER_MODEL_SETTING_KEY])

        if temperature is not None:
            set_summary_temperature(updates[SUMMARIZER_TEMPERATURE_SETTING_KEY])
            LOGGER.info("Updated summarizer temperature (temperature=%s)", updates[SUMMARIZER_TEMPERATURE_SETTING_KEY])

        return self.get_summarizer_settings()

    def _build_public_user(self, record: Dict[str, Any]) -> Dict[str, str]:
        return {
            "id": record["username"],
            "username": record["username"],
            "email": record["username"],
            "name": record["display_name"],
        }

    def issue_access_token(self, username: str) -> str:
        normalized = self._normalize_username(username)
        token = secrets.token_urlsafe(48)
        now = self._now_utc()
        expires_at = now + self._auth_token_ttl
        token_hash = self._hash_access_token(token)
        try:
            self._storage.prune_expired_tokens(reference=now)
            self._storage.create_access_token(
                token_hash=token_hash,
                username=normalized,
                issued_at=now,
                expires_at=expires_at,
            )
        except Exception:  # pylint: disable=broad-except
            LOGGER.exception("Failed to issue access token for user %s", normalized)
            raise
        return token

    def resolve_user_from_token(self, token: str) -> Optional[Dict[str, str]]:
        if not token:
            return None
        now = self._now_utc()
        try:
            token_hash = self._hash_access_token(token)
        except ValueError:
            return None
        token_state = self._storage.fetch_access_token(token_hash)
        if token_state is None:
            return None
        if token_state.revoked_at is not None:
            return None
        expires_at = self._normalize_utc(token_state.expires_at)
        if expires_at <= now:
            self._storage.delete_access_token(token_hash)
            return None
        try:
            normalized_username = self._normalize_username(token_state.username)
        except ValueError:
            return None
        with self._lock:
            record = self._auth_users.get(normalized_username)
        if record is None:
            # The cache might be stale; attempt to refresh from storage once.
            self._hydrate_from_storage()
            with self._lock:
                record = self._auth_users.get(normalized_username)
        if record is None:
            # If the user no longer exists, revoke the token to prevent reuse.
            self._storage.revoke_access_token(token_hash, revoked_at=now)
            return None
        return self._build_public_user(record)

    def revoke_token(self, token: str) -> bool:
        """Revoke an access token, effectively logging out the user.

        Returns:
            bool: True if token was revoked, False if it didn't exist.
        """
        if not token:
            return False
        try:
            token_hash = self._hash_access_token(token)
        except ValueError:
            return False
        now = self._now_utc()
        if self._storage.revoke_access_token(token_hash, revoked_at=now):
            return True
        return self._storage.delete_access_token(token_hash)

    def has_auth_users(self) -> bool:
        count = self._storage.auth_user_count()
        if count < 0:
            LOGGER.warning("Unable to determine auth user count; preserving cached state.")
            with self._lock:
                return bool(self._auth_users)
        if count == 0:
            with self._lock:
                if self._auth_users:
                    self._auth_users.clear()
                    self._auth_users_updated_at = None
            return False

        with self._lock:
            if self._auth_users:
                return True

        # The database reports stored accounts but our cache is empty; refresh to stay consistent.
        self._hydrate_from_storage()
        with self._lock:
            return bool(self._auth_users)

    def list_auth_users(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "username": record["username"],
                    "name": record["display_name"],
                    "created_at": record.get("created_at"),
                    "updated_at": record.get("updated_at"),
                }
                for record in self._auth_users.values()
            ]

    def auth_user_count(self) -> int:
        with self._lock:
            return len(self._auth_users)

    def create_auth_user(self, username: str, password: str, *, name: Optional[str] = None) -> Dict[str, Any]:
        normalized_username = self._normalize_username(username)
        stored_name = (name or normalized_username).strip()
        if not stored_name:
            stored_name = normalized_username
        password_hash = self._hash_password(password)
        now = self._now_utc()
        with self._lock:
            if normalized_username in self._auth_users:
                raise ValueError("A user with that email already exists.")
            record = {
                "username": normalized_username,
                "display_name": stored_name,
                "password_hash": password_hash,
                "created_at": now,
                "updated_at": now,
            }
            self._auth_users[normalized_username] = record
            self._auth_users_updated_at = now
            self._persist_auth_users()
            self._refresh_app_metadata(persist=True, force_timestamp=False)
            return {
                "username": normalized_username,
                "name": stored_name,
                "created_at": now,
            }

    def authenticate_credentials(self, username: str, password: str) -> Optional[Dict[str, str]]:
        try:
            normalized_username = self._normalize_username(username)
        except ValueError:
            return None
        with self._lock:
            record = self._auth_users.get(normalized_username)
            if record is None:
                return None
            if not self._verify_password(password, record["password_hash"]):
                return None
            return self._build_public_user(record)

    def _bump_version(self) -> None:
        self._dataset_id = uuid4().hex
        self._last_updated = self._now_utc()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_ingest_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._storage.fetch_ingest_logs(limit)

    def get_chats(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [chat.copy() for chat in self._chats]

    def get_messages(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [message.copy() for message in self._messages]

    def get_users(self) -> List[Dict[str, str]]:
        with self._lock:
            return [user.copy() for user in self._users]

    def get_models(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [model.copy() for model in self._models]

    def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status and watermark information.

        Returns:
            Dictionary with last_sync_at, last_watermark, has_data, recommended_mode,
            is_stale, staleness_threshold_hours, and local_counts.
        """
        # Get staleness threshold from settings (default 6 hours)
        staleness_threshold_hours = self._settings.get("SYNC_STALENESS_THRESHOLD_HOURS", 6)
        if not isinstance(staleness_threshold_hours, int):
            staleness_threshold_hours = 6

        with self._lock:
            has_data = len(self._chats) > 0
            last_sync_at = self._dataset_pulled_at

            # Determine recommended mode
            recommended_mode: str = "incremental" if has_data else "full"

            # Calculate staleness
            is_stale = False
            if last_sync_at is not None and isinstance(last_sync_at, datetime):
                time_since_sync = self._now_utc() - last_sync_at
                staleness_threshold = timedelta(hours=staleness_threshold_hours)
                is_stale = time_since_sync > staleness_threshold
            elif has_data:
                # If we have data but no sync timestamp, consider it stale
                is_stale = True

            # Calculate watermark (max updated_at from chats)
            last_watermark: Optional[str] = None
            if has_data and self._chats:
                max_timestamp = None
                for chat in self._chats:
                    updated_ts = chat.get("updated_at")
                    if isinstance(updated_ts, datetime):
                        # Normalize to UTC to avoid timezone comparison errors
                        normalized_ts = self._normalize_utc(updated_ts)
                        if max_timestamp is None or normalized_ts > max_timestamp:
                            max_timestamp = normalized_ts

                if max_timestamp:
                    last_watermark = self._serialize_datetime(max_timestamp)

            # Local counts
            local_counts = {
                "chats": len(self._chats),
                "messages": len(self._messages),
                "users": len(self._users),
                "models": len(self._models),
            }

            return {
                "last_sync_at": self._serialize_datetime(last_sync_at),
                "last_watermark": last_watermark,
                "has_data": has_data,
                "recommended_mode": recommended_mode,
                "is_stale": is_stale,
                "staleness_threshold_hours": staleness_threshold_hours,
                "local_counts": local_counts,
            }

    def _load_scheduler_config(self) -> None:
        """Load scheduler configuration from settings on startup."""
        with self._lock:
            self._scheduler_enabled = bool(self._settings.get("SYNC_SCHEDULER_ENABLED", False))
            interval = self._settings.get("SYNC_SCHEDULER_INTERVAL_MINUTES", 60)
            if isinstance(interval, int) and 5 <= interval <= 1440:
                self._scheduler_interval_minutes = interval
            else:
                self._scheduler_interval_minutes = 60

            # Load last run timestamp if available
            last_run_str = self._settings.get("SYNC_SCHEDULER_LAST_RUN_AT")
            if isinstance(last_run_str, str):
                try:
                    self._scheduler_last_run_at = datetime.fromisoformat(last_run_str)
                except (ValueError, TypeError):
                    self._scheduler_last_run_at = None

        LOGGER.info(
            "Scheduler config loaded: enabled=%s, interval=%d minutes",
            self._scheduler_enabled,
            self._scheduler_interval_minutes,
        )

    def get_scheduler_config(self) -> Dict[str, Any]:
        """Get current scheduler configuration and state."""
        with self._lock:
            next_run_at = None
            if self._scheduler_enabled and self._scheduler_last_run_at:
                next_run_at = self._scheduler_last_run_at + timedelta(minutes=self._scheduler_interval_minutes)

            return {
                "enabled": self._scheduler_enabled,
                "interval_minutes": self._scheduler_interval_minutes,
                "last_run_at": self._scheduler_last_run_at,
                "next_run_at": next_run_at,
            }

    def update_scheduler_config(
        self,
        *,
        enabled: Optional[bool] = None,
        interval_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update scheduler configuration and persist to settings.

        Args:
            enabled: Enable or disable the scheduler
            interval_minutes: Update the sync interval (5-1440 minutes)

        Returns:
            Updated scheduler configuration
        """
        updates: Dict[str, Any] = {}

        if enabled is not None:
            with self._lock:
                self._scheduler_enabled = enabled
            updates["SYNC_SCHEDULER_ENABLED"] = enabled
            LOGGER.info("Scheduler %s", "enabled" if enabled else "disabled")

        if interval_minutes is not None:
            if not (5 <= interval_minutes <= 1440):
                raise ValueError("Interval must be between 5 and 1440 minutes")
            with self._lock:
                self._scheduler_interval_minutes = interval_minutes
            updates["SYNC_SCHEDULER_INTERVAL_MINUTES"] = interval_minutes
            LOGGER.info("Scheduler interval updated to %d minutes", interval_minutes)

        if updates:
            self._storage.write_settings(updates)

        return self.get_scheduler_config()

    def get_meta(self) -> DatasetMeta:
        with self._lock:
            return DatasetMeta(
                dataset_id=self._dataset_id,
                source=self._source,
                last_updated=self._last_updated,
                chat_count=len(self._chats),
                message_count=len(self._messages),
                user_count=len(self._users),
                model_count=len(self._models),
                app_metadata=self._build_app_metadata(),
            )


# Singleton instance shared across the application
data_service = DataService(data_dir=DATA_DIR)


def get_data_service() -> DataService:
    """FastAPI dependency to retrieve the shared data service."""
    return data_service
