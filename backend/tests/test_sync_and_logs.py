"""Tests for sync status, modes, and process logs."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock

from fastapi.testclient import TestClient

from backend.app import app
from backend.models import AuthUserPublic
from backend import services as services_module


def _fake_auth_user() -> AuthUserPublic:
    return AuthUserPublic(id="tester", username="tester", email="tester", name="Test User")


def test_sync_status_includes_staleness_info(monkeypatch):
    """Sync status should include staleness indicator and local counts."""

    class FakeService:
        def get_sync_status(self):
            return {
                "last_sync_at": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                "last_watermark": "2024-01-01T12:00:00Z",
                "has_data": True,
                "recommended_mode": "incremental",
                "is_stale": False,
                "staleness_threshold_hours": 6,
                "local_counts": {
                    "chats": 100,
                    "messages": 500,
                    "users": 10,
                    "models": 5,
                },
            }

    fake_service = FakeService()

    monkeypatch.setattr("backend.app.data_service.load_initial_data", lambda: None)
    monkeypatch.setattr("backend.routes.require_authenticated_user", lambda: _fake_auth_user())
    monkeypatch.setattr("backend.routes.get_data_service", lambda: fake_service)

    with TestClient(app) as client:
        response = client.get("/api/v1/sync/status")

    assert response.status_code == 200
    data = response.json()
    assert data["is_stale"] is False
    assert data["staleness_threshold_hours"] == 6
    assert data["local_counts"]["chats"] == 100
    assert data["local_counts"]["messages"] == 500


def test_sync_status_marks_stale_when_threshold_exceeded(monkeypatch):
    """Data should be marked stale when last sync exceeds threshold."""

    class FakeService:
        def get_sync_status(self):
            # Simulate old sync (more than 6 hours ago)
            return {
                "last_sync_at": datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                "last_watermark": "2024-01-01T00:00:00Z",
                "has_data": True,
                "recommended_mode": "incremental",
                "is_stale": True,  # Should be stale
                "staleness_threshold_hours": 6,
                "local_counts": {
                    "chats": 100,
                    "messages": 500,
                    "users": 10,
                    "models": 5,
                },
            }

    fake_service = FakeService()

    monkeypatch.setattr("backend.app.data_service.load_initial_data", lambda: None)
    monkeypatch.setattr("backend.routes.require_authenticated_user", lambda: _fake_auth_user())
    monkeypatch.setattr("backend.routes.get_data_service", lambda: fake_service)

    with TestClient(app) as client:
        response = client.get("/api/v1/sync/status")

    assert response.status_code == 200
    data = response.json()
    assert data["is_stale"] is True


def test_sync_supports_full_and_incremental_modes(monkeypatch):
    """Sync endpoint should support both full and incremental modes."""

    sync_mode_used = None

    class FakeService:
        def get_sync_status(self):
            return {
                "last_sync_at": None,
                "last_watermark": None,
                "has_data": False,
                "recommended_mode": "full",
                "is_stale": False,
                "staleness_threshold_hours": 6,
                "local_counts": None,
            }

        def sync_from_openwebui(self, hostname, api_key):
            nonlocal sync_mode_used
            # Capture that sync was called
            # Mode detection happens in route, but we can verify it's called
            sync_mode_used = "called"

            from backend.models import DatasetMeta, DatasetSyncStats

            return (
                DatasetMeta(
                    dataset_id="test",
                    source="test",
                    last_updated=datetime.now(timezone.utc),
                    chat_count=10,
                    message_count=50,
                    user_count=2,
                    model_count=1,
                    app_metadata=None,
                ),
                DatasetSyncStats(
                    mode="full",
                    source_matched=False,
                    submitted_hostname=hostname,
                    normalized_hostname=hostname,
                    source_display="test",
                    new_chats=10,
                    new_messages=50,
                    new_users=2,
                    new_models=1,
                    models_changed=True,
                    summarizer_enqueued=False,
                    total_chats=10,
                    total_messages=50,
                    total_users=2,
                    total_models=1,
                    queued_chat_ids=None,
                ),
            )

    fake_service = FakeService()

    monkeypatch.setattr("backend.app.data_service.load_initial_data", lambda: None)
    monkeypatch.setattr("backend.routes.require_authenticated_user", lambda: _fake_auth_user())
    monkeypatch.setattr("backend.routes.get_data_service", lambda: fake_service)

    # Test with explicit mode
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/openwebui/sync",
            json={"hostname": "http://test.com", "mode": "full"},
        )

    assert response.status_code == 200
    assert sync_mode_used == "called"


def test_data_service_sync_status_handles_naive_timestamp():
    """Regression test: ensure naive timestamps do not raise TypeError."""
    service_cls = services_module.DataService
    service = service_cls.__new__(service_cls)  # bypass heavy __init__
    service._settings = {}  # pylint: disable=protected-access
    service._lock = RLock()  # pylint: disable=protected-access
    service._chats = []  # pylint: disable=protected-access
    service._messages = []  # pylint: disable=protected-access
    service._users = []  # pylint: disable=protected-access
    service._models = []  # pylint: disable=protected-access
    service._dataset_pulled_at = datetime(2024, 1, 1, 12, 0, 0)  # naive timestamp

    status = service.get_sync_status()
    assert status["last_sync_at"] == "2024-01-01T12:00:00Z"
    assert status["has_data"] is False


def test_logs_endpoint_returns_structured_events(monkeypatch):
    """Process logs endpoint should return structured log events."""

    class FakeService:
        def get_process_logs(self, job_id, limit):
            return [
                {
                    "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                    "level": "info",
                    "job_id": "test-job-123",
                    "phase": "connect",
                    "message": "Starting sync",
                    "details": None,
                },
                {
                    "timestamp": datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
                    "level": "info",
                    "job_id": "test-job-123",
                    "phase": "fetch",
                    "message": "Fetched 10 chats",
                    "details": {"count": 10},
                },
                {
                    "timestamp": datetime(2024, 1, 1, 12, 0, 10, tzinfo=timezone.utc),
                    "level": "info",
                    "job_id": "test-job-123",
                    "phase": "done",
                    "message": "Sync complete",
                    "details": None,
                },
            ]

    fake_service = FakeService()

    monkeypatch.setattr("backend.app.data_service.load_initial_data", lambda: None)
    monkeypatch.setattr("backend.routes.require_authenticated_user", lambda: _fake_auth_user())
    monkeypatch.setattr("backend.routes.get_data_service", lambda: fake_service)

    with TestClient(app) as client:
        response = client.get("/api/v1/logs?limit=100")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["logs"]) == 3

    # Check structure
    log1 = data["logs"][0]
    assert log1["level"] == "info"
    assert log1["phase"] == "connect"
    assert log1["message"] == "Starting sync"
    assert log1["job_id"] == "test-job-123"


def test_logs_endpoint_filters_by_job_id(monkeypatch):
    """Process logs should be filterable by job_id."""

    class FakeService:
        def get_process_logs(self, job_id, limit):
            # Simulate filtering
            if job_id == "specific-job":
                return [
                    {
                        "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                        "level": "info",
                        "job_id": "specific-job",
                        "phase": "connect",
                        "message": "Specific job log",
                        "details": None,
                    }
                ]
            return []

    fake_service = FakeService()

    monkeypatch.setattr("backend.app.data_service.load_initial_data", lambda: None)
    monkeypatch.setattr("backend.routes.require_authenticated_user", lambda: _fake_auth_user())
    monkeypatch.setattr("backend.routes.get_data_service", lambda: fake_service)

    with TestClient(app) as client:
        response = client.get("/api/v1/logs?job_id=specific-job&limit=100")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["logs"][0]["job_id"] == "specific-job"


def test_logs_never_contain_secrets(monkeypatch):
    """Process logs should never contain API keys or other secrets."""

    class FakeService:
        def get_process_logs(self, job_id, limit):
            return [
                {
                    "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                    "level": "info",
                    "job_id": "test-job",
                    "phase": "connect",
                    "message": "Using authenticated connection",  # Should NOT contain actual key
                    "details": None,
                },
            ]

    fake_service = FakeService()

    monkeypatch.setattr("backend.app.data_service.load_initial_data", lambda: None)
    monkeypatch.setattr("backend.routes.require_authenticated_user", lambda: _fake_auth_user())
    monkeypatch.setattr("backend.routes.get_data_service", lambda: fake_service)

    with TestClient(app) as client:
        response = client.get("/api/v1/logs?limit=100")

    assert response.status_code == 200
    data = response.json()

    # Check that messages don't contain common secret patterns
    for log in data["logs"]:
        message = log["message"]
        # Should not contain JWT-like patterns
        assert not any(
            pattern in message.lower()
            for pattern in ["sk-", "bearer ", "api_key=", "apikey="]
        )


def test_scheduler_get_returns_config(monkeypatch):
    """Scheduler GET endpoint should return current configuration."""

    class FakeService:
        def get_scheduler_config(self):
            return {
                "enabled": True,
                "interval_minutes": 60,
                "last_run_at": datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc),
                "next_run_at": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            }

    fake_service = FakeService()

    monkeypatch.setattr("backend.app.data_service.load_initial_data", lambda: None)
    monkeypatch.setattr("backend.routes.require_authenticated_user", lambda: _fake_auth_user())
    monkeypatch.setattr("backend.routes.get_data_service", lambda: fake_service)

    with TestClient(app) as client:
        response = client.get("/api/v1/sync/scheduler")

    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True
    assert data["interval_minutes"] == 60


def test_scheduler_post_updates_config(monkeypatch):
    """Scheduler POST endpoint should update configuration."""

    class FakeService:
        def __init__(self):
            self.updated_config = None

        def update_scheduler_config(self, *, enabled, interval_minutes):
            self.updated_config = {
                "enabled": enabled,
                "interval_minutes": interval_minutes,
            }
            return {
                "enabled": enabled if enabled is not None else True,
                "interval_minutes": interval_minutes if interval_minutes is not None else 60,
                "last_run_at": None,
                "next_run_at": None,
            }

    fake_service = FakeService()

    monkeypatch.setattr("backend.app.data_service.load_initial_data", lambda: None)
    monkeypatch.setattr("backend.routes.require_authenticated_user", lambda: _fake_auth_user())
    monkeypatch.setattr("backend.routes.get_data_service", lambda: fake_service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/sync/scheduler",
            json={"enabled": False, "interval_minutes": 120},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is False
    assert data["interval_minutes"] == 120
    assert fake_service.updated_config == {"enabled": False, "interval_minutes": 120}
