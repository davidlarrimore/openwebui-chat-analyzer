"""Tests for chat route payloads."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from backend.app import app
from backend.models import AuthUserPublic


def _fake_auth_user() -> AuthUserPublic:
    return AuthUserPublic(id="tester", username="tester", email="tester@example.com", name="Test User")


def test_list_chats_includes_generated_summary(monkeypatch):
    """Ensure /chats returns gen_chat_summary field."""

    class FakeService:
        def get_chats(self):
            return [
                {
                    "chat_id": "chat-123",
                    "user_id": "user-1",
                    "title": "Planning discussion",
                    "gen_chat_summary": "Discussed API field mappings for summaries.",
                    "created_at": datetime(2024, 1, 1, 15, 30, tzinfo=timezone.utc),
                }
            ]

    fake_service = FakeService()

    monkeypatch.setattr("backend.app.data_service.load_initial_data", lambda: None)
    monkeypatch.setattr("backend.routes.require_authenticated_user", lambda: _fake_auth_user())
    monkeypatch.setattr("backend.routes.get_data_service", lambda: fake_service)

    with TestClient(app) as client:
        response = client.get("/api/v1/chats")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload, "Expected at least one chat record"

    chat = payload[0]
    assert chat["gen_chat_summary"] == "Discussed API field mappings for summaries."
