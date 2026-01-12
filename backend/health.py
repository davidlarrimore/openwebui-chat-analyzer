"""Health check utilities for backend dependencies."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import requests
from sqlalchemy import text

from .clients import get_ollama_client
from .db import engine

LOGGER = logging.getLogger(__name__)

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


def _ping_backend() -> Dict[str, Any]:
    """Return a lightweight payload confirming the API is responsive."""
    return {"response": "ok"}


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


def check_backend_health(
    *,
    interval_seconds: float = HEALTH_CHECK_INTERVAL_SECONDS,
    timeout_seconds: float = HEALTH_CHECK_TIMEOUT_SECONDS,
) -> HealthResult:
    """Confirm the backend API is reachable."""
    return poll_health(
        _ping_backend,
        service="backend",
        interval_seconds=interval_seconds,
        timeout_seconds=timeout_seconds,
    )


def _ping_openwebui(host: str, api_key: Optional[str]) -> Dict[str, Any]:
    """Test connectivity to an OpenWebUI instance.

    Args:
        host: The OpenWebUI host URL
        api_key: Optional API key for authentication

    Returns:
        Dictionary with version and sample counts if successful

    Raises:
        RuntimeError: If connection fails or authentication is invalid
    """
    # Normalize host URL
    base_url = host.strip()
    if not base_url:
        raise RuntimeError("Host URL is required")

    # Ensure URL has scheme
    if not base_url.startswith(("http://", "https://")):
        base_url = f"https://{base_url}"

    # Remove trailing slashes
    base_url = base_url.rstrip("/")

    # Build headers
    headers = {"Accept": "application/json"}
    token = (api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
        # Log that we're using authentication, but never log the key itself
        LOGGER.debug("Testing OpenWebUI connection with authentication")
    else:
        LOGGER.debug("Testing OpenWebUI connection without authentication")

    # Try to fetch basic endpoints
    result: Dict[str, Any] = {"host": base_url, "authenticated": bool(token)}

    try:
        with requests.Session() as session:
            session.headers.update(headers)

            # Lightweight availability check
            health_url = f"{base_url}/api/health"
            try:
                response = session.get(health_url, timeout=5)
                response.raise_for_status()
                result["reachable"] = True
            except requests.exceptions.RequestException as exc:
                LOGGER.debug("Could not fetch health from %s: %s", health_url, exc)
                result["reachable"] = False

            # Try to get version/config info
            config_url = f"{base_url}/api/config"
            try:
                response = session.get(config_url, timeout=5)
                response.raise_for_status()
                config_data = response.json()
                if isinstance(config_data, dict):
                    result["version"] = config_data.get("version", "unknown")
            except (requests.exceptions.RequestException, ValueError) as exc:
                LOGGER.debug("Could not fetch config from %s: %s", config_url, exc)

            # Try a lightweight authenticated endpoint instead of full chat export
            models_url = f"{base_url}/api/v1/models"
            try:
                response = session.get(models_url, timeout=10)
                response.raise_for_status()
                models_data = response.json()
                if isinstance(models_data, list):
                    result["model_count"] = len(models_data)
            except requests.exceptions.HTTPError as exc:
                if exc.response.status_code == 401:
                    raise RuntimeError(f"Authentication failed for {base_url}. Please check your API key.")
                raise RuntimeError(f"HTTP {exc.response.status_code} from {base_url}: {exc.response.reason}")
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(f"Connection failed for {base_url}: {str(exc)}")
            except ValueError as exc:
                raise RuntimeError(f"Invalid JSON response from {base_url}: {str(exc)}")

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Cannot connect to {base_url}. Please check the hostname. ({exc})")
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Connection to {base_url} timed out.")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Request failed for {base_url}: {str(exc)}")

    return result


def check_openwebui_health(
    host: str,
    api_key: Optional[str],
    *,
    interval_seconds: float = HEALTH_CHECK_INTERVAL_SECONDS,
    timeout_seconds: float = HEALTH_CHECK_TIMEOUT_SECONDS,
) -> HealthResult:
    """Test connectivity and authentication to an OpenWebUI instance.

    Args:
        host: The OpenWebUI host URL
        api_key: Optional API key for authentication
        interval_seconds: Interval between retry attempts
        timeout_seconds: Maximum time to wait for success

    Returns:
        HealthResult with status and metadata
    """
    def probe() -> Dict[str, Any]:
        return _ping_openwebui(host, api_key)

    return poll_health(
        probe,
        service="openwebui",
        interval_seconds=interval_seconds,
        timeout_seconds=timeout_seconds,
    )
