from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import patch

import io

import pytest
import requests

from frontend.core.api import (
    BackendError,
    get_chats,
    get_dataset_meta,
    get_messages,
    poll_summary_status,
    upload_chat_export,
)
from frontend.core.models import DatasetMeta


class DummyResponse:
    def __init__(self, payload: Any, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self) -> Any:
        return self._payload

    @property
    def text(self) -> str:
        return str(self._payload)


def test_get_dataset_meta_success(monkeypatch) -> None:
    payload = {
        "dataset_id": "abc123",
        "chat_count": 10,
        "message_count": 100,
        "user_count": 4,
        "source": "upload:sample.json",
        "app_metadata": {"dataset_source": "Local Upload"},
    }

    def fake_request(method: str, url: str, timeout: float = 30.0, **kwargs: Any) -> DummyResponse:
        assert url.endswith("/api/v1/datasets/meta")
        return DummyResponse(payload)

    with patch("frontend.core.api.requests.request", side_effect=fake_request):
        meta = get_dataset_meta()
        assert isinstance(meta, DatasetMeta)
        assert meta.dataset_id == "abc123"
        assert meta.chat_count == 10
        assert meta.app_metadata["dataset_source"] == "Local Upload"


def test_get_chats_returns_dataframe(monkeypatch) -> None:
    payload = [
        {"chat_id": "chat-1", "summary_128": "hi", "created_at": "2024-01-01T00:00:00Z", "updated_at": "2024-01-01T01:00:00Z"},
        {"chat_id": "chat-2", "summary_128": None, "created_at": "2024-01-02T00:00:00Z", "updated_at": "2024-01-02T01:00:00Z"},
    ]

    def fake_request(method: str, url: str, timeout: float = 30.0, **kwargs: Any) -> DummyResponse:
        assert url.endswith("/api/v1/chats")
        return DummyResponse(payload)

    with patch("frontend.core.api.requests.request", side_effect=fake_request):
        chats_df = get_chats()
        assert not chats_df.empty
        assert chats_df.loc[0, "summary_128"] == "hi"
        assert chats_df.loc[1, "summary_128"] == ""
        assert str(chats_df.loc[0, "created_at"].tzinfo)


def test_get_messages_parses_timestamps(monkeypatch) -> None:
    payload = [
        {"chat_id": "chat-1", "timestamp": "2024-01-01T00:00:00Z", "role": "user", "model": ""},
    ]

    def fake_request(method: str, url: str, timeout: float = 30.0, **kwargs: Any) -> DummyResponse:
        assert url.endswith("/api/v1/messages")
        return DummyResponse(payload)

    with patch("frontend.core.api.requests.request", side_effect=fake_request):
        messages_df = get_messages()
        assert not messages_df.empty
        assert str(messages_df.loc[0, "timestamp"].tzinfo)


def test_upload_chat_export_raises_backend_error(monkeypatch) -> None:
    def fake_request(method: str, url: str, timeout: float = 30.0, **kwargs: Any) -> DummyResponse:
        raise requests.exceptions.ConnectionError("offline")

    with patch("frontend.core.api.requests.request", side_effect=fake_request):
        with pytest.raises(BackendError):
            upload_chat_export(io.BytesIO(b"{}"))


def test_poll_summary_status_calls_callback(monkeypatch) -> None:
    responses: List[Dict[str, Any]] = [
        {"state": "running", "completed": 1, "total": 2},
        {"state": "completed", "completed": 2, "total": 2},
    ]

    def fake_request(method: str, url: str, timeout: float = 30.0, **kwargs: Any) -> DummyResponse:
        assert url.endswith("/api/v1/summaries/status")
        payload = responses.pop(0)
        return DummyResponse(payload)

    observed_states: List[str] = []

    with patch("frontend.core.api.requests.request", side_effect=fake_request), patch(
        "frontend.core.api.time.sleep", return_value=None
    ):
        status = poll_summary_status(on_update=lambda s: observed_states.append(s.state or "idle"))

    assert observed_states == ["running", "completed"]
    assert status.state == "completed"
