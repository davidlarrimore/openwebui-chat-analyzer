from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class DatasetMeta:
    """Dataset metadata returned by the backend."""

    dataset_id: str
    chat_count: int
    message_count: int
    user_count: int
    source: Optional[str] = None
    app_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DatasetMeta":
        payload = payload or {}
        app_metadata = payload.get("app_metadata") or {}
        return cls(
            dataset_id=str(payload.get("dataset_id", "")),
            chat_count=int(payload.get("chat_count", 0) or 0),
            message_count=int(payload.get("message_count", 0) or 0),
            user_count=int(payload.get("user_count", 0) or 0),
            source=payload.get("source"),
            app_metadata=app_metadata,
        )


@dataclass(frozen=True)
class SummaryStatus:
    """Background summarizer status response."""

    state: Optional[str]
    total: int = 0
    completed: int = 0
    message: Optional[str] = None
    last_event: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SummaryStatus":
        payload = payload or {}
        return cls(
            state=payload.get("state"),
            total=int(payload.get("total", 0) or 0),
            completed=int(payload.get("completed", 0) or 0),
            message=payload.get("message"),
            last_event=payload.get("last_event"),
        )


@dataclass(frozen=True)
class UploadResult:
    """Response payload from upload endpoints."""

    detail: Optional[str] = None
    dataset: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "UploadResult":
        payload = payload or {}
        return cls(
            detail=payload.get("detail"),
            dataset=payload.get("dataset") or {},
        )


@dataclass(frozen=True)
class DatasetSourceInfo:
    """Derived information about the active dataset source."""

    label: str
    detail: str = ""


@dataclass(frozen=True)
class DatasetPanel:
    """Structured HTML fragments for the dataset summary card."""

    source_html: str
    chat_html: str
    user_html: str


@dataclass
class ProcessedData:
    """Container for the hydrated frontend dataframes."""

    chats: pd.DataFrame
    messages: pd.DataFrame
    users: pd.DataFrame
    chat_user_map: Dict[str, str] = field(default_factory=dict)
