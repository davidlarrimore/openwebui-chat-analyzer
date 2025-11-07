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


def test_list_users_returns_pseudonyms_by_default(monkeypatch):
    """Users endpoint should expose pseudonyms in the name field by default."""

    class FakeService:
        def __init__(self) -> None:
            self._show_real = False

        def get_users(self):
            return [
                {"user_id": "user-123", "name": "Alice Example", "pseudonym": "Nebula"},
                {"user_id": "user-456", "name": "Bob Example", "pseudonym": "Star-Lord"},
            ]

        def should_expose_real_names(self) -> bool:
            return self._show_real

    fake_service = FakeService()

    monkeypatch.setattr("backend.app.data_service.load_initial_data", lambda: None)
    monkeypatch.setattr("backend.routes.require_authenticated_user", lambda: _fake_auth_user())
    monkeypatch.setattr("backend.routes.get_data_service", lambda: fake_service)

    with TestClient(app) as client:
        response = client.get("/api/v1/users")

    assert response.status_code == 200
    users = response.json()
    assert users[0]["user_id"] == "user-123"
    assert users[0]["name"] == "Nebula"
    assert users[0]["pseudonym"] == "Nebula"
    assert users[0]["real_name"] == "Alice Example"


def test_list_users_can_expose_real_names(monkeypatch):
    """Users endpoint should switch to real names when the setting flag is enabled."""

    class FakeService:
        def __init__(self) -> None:
            self._show_real = True

        def get_users(self):
            return [{"user_id": "user-999", "name": "Charlie Example", "pseudonym": "Rocket"}]

        def should_expose_real_names(self) -> bool:
            return self._show_real

    fake_service = FakeService()

    monkeypatch.setattr("backend.app.data_service.load_initial_data", lambda: None)
    monkeypatch.setattr("backend.routes.require_authenticated_user", lambda: _fake_auth_user())
    monkeypatch.setattr("backend.routes.get_data_service", lambda: fake_service)

    with TestClient(app) as client:
        response = client.get("/api/v1/users")

    assert response.status_code == 200
    users = response.json()
    assert users[0]["name"] == "Charlie Example"
    assert users[0]["pseudonym"] == "Rocket"
    assert users[0]["real_name"] == "Charlie Example"


def test_rebuild_summaries_errors_when_disabled(monkeypatch):
    """POST /summaries/rebuild should surface disabled-state errors as HTTP 400."""

    class FakeService:
        def rebuild_summaries(self):
            raise ValueError("Cannot rebuild summaries: summarizer is disabled")

    fake_service = FakeService()

    monkeypatch.setattr("backend.app.data_service.load_initial_data", lambda: None)
    monkeypatch.setattr("backend.routes.require_authenticated_user", lambda: _fake_auth_user())
    monkeypatch.setattr("backend.routes.get_data_service", lambda: fake_service)

    with TestClient(app) as client:
        response = client.post("/api/v1/summaries/rebuild")

    assert response.status_code == 400
    assert response.json()["detail"] == "Cannot rebuild summaries: summarizer is disabled"
