"""Health check utilities for backend dependencies."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from sqlalchemy import text

from .clients import get_ollama_client
from .db import engine

HEALTH_CHECK_INTERVAL_SECONDS = 5.0
HEALTH_CHECK_TIMEOUT_SECONDS = 50.0


@dataclass
class HealthResult:
    """Structured payload describing the outcome of a health probe."""

    service: str
    status: str
    attempts: int
    elapsed_seconds: float
    detail: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result into a serialisable dictionary."""
        payload: Dict[str, Any] = {
            "service": self.service,
            "status": self.status,
            "attempts": self.attempts,
            "elapsed_seconds": self.elapsed_seconds,
        }
        if self.meta:
            payload["meta"] = self.meta
        if self.detail:
            payload["detail"] = self.detail
        return payload


def poll_health(
    probe: Callable[[], Optional[Dict[str, Any]]],
    *,
    service: str,
    interval_seconds: float = HEALTH_CHECK_INTERVAL_SECONDS,
    timeout_seconds: float = HEALTH_CHECK_TIMEOUT_SECONDS,
) -> HealthResult:
    """Repeatedly invoke a health probe until it succeeds or the timeout expires."""
    start_time = time.perf_counter()
    max_attempts = max(1, int(math.ceil(timeout_seconds / interval_seconds)))
    last_error: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        try:
            meta = probe() or {}
        except Exception as exc:  # pylint: disable=broad-except
            last_error = str(exc) or exc.__class__.__name__
            if attempt < max_attempts:
                time.sleep(interval_seconds)
            continue

        elapsed = time.perf_counter() - start_time
        return HealthResult(
            service=service,
            status="ok",
            attempts=attempt,
            elapsed_seconds=elapsed,
            meta=meta,
        )

    elapsed = time.perf_counter() - start_time
    return HealthResult(
        service=service,
        status="error",
        attempts=max_attempts,
        elapsed_seconds=elapsed,
        detail=last_error or "Unhealthy",
    )


def _ping_ollama() -> Dict[str, Any]:
    """Send a lightweight request to the Ollama service."""
    client = get_ollama_client()
    models = client.list_models()
    return {"model_count": len(models)}


def _ping_database() -> Dict[str, Any]:
    """Run a trivial query to confirm the database connection is healthy."""
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        scalar = result.scalar_one_or_none()
    return {"result": scalar}


def check_ollama_health(
    *,
    interval_seconds: float = HEALTH_CHECK_INTERVAL_SECONDS,
    timeout_seconds: float = HEALTH_CHECK_TIMEOUT_SECONDS,
) -> HealthResult:
    """Probe the Ollama service until it responds or the timeout elapses."""
    return poll_health(
        _ping_ollama,
        service="ollama",
        interval_seconds=interval_seconds,
        timeout_seconds=timeout_seconds,
    )


def check_database_health(
    *,
    interval_seconds: float = HEALTH_CHECK_INTERVAL_SECONDS,
    timeout_seconds: float = HEALTH_CHECK_TIMEOUT_SECONDS,
) -> HealthResult:
    """Probe the database until it responds or the timeout elapses."""
    return poll_health(
        _ping_database,
        service="database",
        interval_seconds=interval_seconds,
        timeout_seconds=timeout_seconds,
    )
