"""Tests for backend health probing utilities and endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import app
from backend.health import HealthResult, poll_health


def test_poll_health_success(monkeypatch):
    attempts = []

    def succeed():
        attempts.append(1)
        return {"foo": "bar"}

    result = poll_health(
        succeed,
        service="demo",
        interval_seconds=0.01,
        timeout_seconds=0.05,
    )

    assert result.status == "ok"
    assert result.attempts == 1
    assert result.meta == {"foo": "bar"}


def test_poll_health_failure(monkeypatch):
    call_count = 0

    def fail():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("boom")

    result = poll_health(
        fail,
        service="demo",
        interval_seconds=0.01,
        timeout_seconds=0.03,
    )

    assert result.status == "error"
    assert result.detail == "boom"
    assert result.attempts == 3
    assert call_count == 3


def test_health_endpoint_uses_result(monkeypatch):
    monkeypatch.setattr("backend.app.data_service.load_initial_data", lambda: None)

    fake_result = HealthResult(
        service="ollama",
        status="ok",
        attempts=2,
        elapsed_seconds=0.2,
        meta={"model_count": 3},
    )

    monkeypatch.setattr("backend.routes.check_ollama_health", lambda: fake_result)

    with TestClient(app) as client:
        response = client.get("/api/v1/health/ollama")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["meta"] == {"model_count": 3}


def test_backend_health_endpoint(monkeypatch):
    monkeypatch.setattr("backend.app.data_service.load_initial_data", lambda: None)

    fake_result = HealthResult(
        service="backend",
        status="ok",
        attempts=1,
        elapsed_seconds=0.01,
        meta={"response": "ok"},
    )

    monkeypatch.setattr("backend.routes.check_backend_health", lambda: fake_result)

    with TestClient(app) as client:
        response = client.get("/api/v1/health/backend")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "backend"
    assert payload["meta"] == {"response": "ok"}
