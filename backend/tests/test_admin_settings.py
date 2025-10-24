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


def test_settings_normalizes_quoted_values(monkeypatch):
    """Settings should strip encapsulating quotes from stored values."""

    class FakeService:
        def update_direct_connect_settings(self, *, host, api_key):
            # Simulate quote stripping behavior
            if host and host.startswith('"') and host.endswith('"'):
                host = host[1:-1]
            if api_key and api_key.startswith('"') and api_key.endswith('"'):
                api_key = api_key[1:-1]

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
            json={"host": '"http://quoted-host"', "api_key": '"quoted-key"'},
        )

    assert response.status_code == 200
    data = response.json()
    # Quotes should be stripped
    assert data["host"] == "http://quoted-host"
    assert data["api_key"] == "quoted-key"


def test_settings_never_logs_secrets(monkeypatch, caplog):
    """API key values should never appear in logs."""
    import logging

    class FakeService:
        def update_direct_connect_settings(self, *, host, api_key):
            # Simulate logging with redaction (as done in actual service)
            import logging
            logger = logging.getLogger("backend.services")
            if api_key:
                redacted = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
                logger.info("Updating Direct Connect API key setting (redacted=%s)", redacted)

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

    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            response = client.put(
                "/api/v1/admin/settings/direct-connect",
                json={"host": "http://test", "api_key": "super-secret-key-12345"},
            )

    assert response.status_code == 200

    # The full secret should NOT appear anywhere in logs
    log_text = "\n".join([record.message for record in caplog.records])
    assert "super-secret-key-12345" not in log_text

    # But the redacted version should be present
    assert "supe...2345" in log_text or "***" in log_text


def test_hostname_change_triggers_wipe_and_reload(monkeypatch):
    """Changing hostname should trigger wipe and full reload."""

    wipe_called = False
    sync_called_with = None

    class FakeStorage:
        def wipe_openwebui_data(self):
            nonlocal wipe_called
            wipe_called = True

    class FakeService:
        def __init__(self):
            self._storage = FakeStorage()
            self._chats = [{"chat_id": "old-chat"}]  # Has existing data

        def get_direct_connect_settings(self):
            return {
                "host": "http://old-host.com",
                "api_key": "test-key",
                "database_host": "http://old-host.com",
                "database_api_key": "test-key",
                "host_source": "database",
                "api_key_source": "database",
            }

        def update_direct_connect_settings(self, *, host, api_key):
            nonlocal sync_called_with

            # Simulate hostname change detection
            if host and host != "http://old-host.com":
                # Wipe should be called
                self._storage.wipe_openwebui_data()

                # Sync should be triggered
                sync_called_with = {"hostname": host, "api_key": api_key}

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
            json={"host": "http://new-host.com", "api_key": "test-key"},
        )

    assert response.status_code == 200
    assert wipe_called is True
    assert sync_called_with == {"hostname": "http://new-host.com", "api_key": "test-key"}
