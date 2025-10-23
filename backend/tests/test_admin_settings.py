"""Tests for admin settings API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import app
from backend.models import AuthUserPublic


def _fake_auth_user() -> AuthUserPublic:
    return AuthUserPublic(id="tester", username="tester", email="tester", name="Test User")


def test_get_direct_connect_settings(monkeypatch):
    """Admin settings endpoint should return service-provided defaults."""

    class FakeService:
        def get_direct_connect_settings(self):
            return {
                "host": "http://example.com",
                "api_key": "secret",
                "database_host": "http://example.com",
                "database_api_key": "secret",
                "host_source": "database",
                "api_key_source": "environment",
            }

    fake_service = FakeService()

    monkeypatch.setattr("backend.app.data_service.load_initial_data", lambda: None)
    monkeypatch.setattr("backend.routes.require_authenticated_user", lambda: _fake_auth_user())
    monkeypatch.setattr("backend.routes.get_data_service", lambda: fake_service)

    with TestClient(app) as client:
        response = client.get("/api/v1/admin/settings/direct-connect")

    assert response.status_code == 200
    assert response.json() == {
        "host": "http://example.com",
        "api_key": "secret",
        "database_host": "http://example.com",
        "database_api_key": "secret",
        "host_source": "database",
        "api_key_source": "environment",
    }


def test_update_direct_connect_settings(monkeypatch):
    """Updating admin settings should delegate to the service layer."""

    class FakeService:
        def __init__(self) -> None:
            self.updated_with: dict[str, str | None] | None = None

        def get_direct_connect_settings(self):
            return {
                "host": "http://fallback",
                "api_key": "",
                "database_host": "",
                "database_api_key": "",
                "host_source": "default",
                "api_key_source": "empty",
            }

        def update_direct_connect_settings(self, *, host, api_key):
            self.updated_with = {"host": host, "api_key": api_key}
            return {
                "host": host or "",
                "api_key": api_key or "",
                "database_host": host or "",
                "database_api_key": api_key or "",
                "host_source": "database",
                "api_key_source": "database" if api_key else "empty",
            }

    fake_service = FakeService()

    monkeypatch.setattr("backend.app.data_service.load_initial_data", lambda: None)
    monkeypatch.setattr("backend.routes.require_authenticated_user", lambda: _fake_auth_user())
    monkeypatch.setattr("backend.routes.get_data_service", lambda: fake_service)

    with TestClient(app) as client:
        response = client.put(
            "/api/v1/admin/settings/direct-connect",
            json={"host": "http://new-host", "api_key": "new-key"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "host": "http://new-host",
        "api_key": "new-key",
        "database_host": "http://new-host",
        "database_api_key": "new-key",
        "host_source": "database",
        "api_key_source": "database",
    }
    assert fake_service.updated_with == {"host": "http://new-host", "api_key": "new-key"}
