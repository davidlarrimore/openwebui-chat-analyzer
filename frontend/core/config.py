from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

# Load environment variables once so both Streamlit and tests share the same defaults.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Frontend configuration derived from the environment."""

    api_base_url: str
    fallback_api_urls: Tuple[str, ...]
    request_timeout: float = 30.0
    summary_poll_interval: float = 2.0
    summary_timeout_seconds: int = 180
    direct_host_default: str = "http://localhost:4000"
    direct_api_key_default: str = ""

    @property
    def api_base_url_candidates(self) -> Tuple[str, ...]:
        """Primary API base URL followed by fallbacks."""
        return (self.api_base_url, *self.fallback_api_urls)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return the cached application configuration."""
    primary = os.getenv("OWUI_API_BASE_URL", "").strip()
    fallback_env = os.getenv("OWUI_API_FALLBACKS", "").strip()

    fallbacks = []
    if fallback_env:
        fallbacks.extend(url.strip() for url in fallback_env.split(",") if url.strip())

    # These fallbacks mirror docker-compose defaults and local development.
    fallbacks.extend(
        [
            "http://backend:8502",
            "http://localhost:8502",
        ]
    )

    direct_host = os.getenv("OWUI_DIRECT_HOST", "http://localhost:4000").strip() or "http://localhost:4000"
    direct_api_key = os.getenv("OWUI_DIRECT_API_KEY", "").strip()

    base_url = primary or fallbacks[0]

    return AppConfig(
        api_base_url=base_url,
        fallback_api_urls=tuple(dict.fromkeys(fallbacks)),  # preserve order, remove duplicates
        direct_host_default=direct_host,
        direct_api_key_default=direct_api_key,
    )


def get_api_base_url_candidates() -> Tuple[str, ...]:
    """Convenience helper for the API client."""
    return get_config().api_base_url_candidates
