"""Configuration helpers for the Open WebUI Chat Analyzer backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

# Load environment variables from a .env file if present.
load_dotenv()


def _split_origins(value: str) -> List[str]:
    """Convert a comma-separated origin string into a clean list.

    Args:
        value (str): One or many origins separated by commas.
    Returns:
        List[str]: Normalized origin values with whitespace removed.
    """
    return [origin.strip() for origin in value.split(",") if origin.strip()]


DEFAULT_ALLOWED_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8503"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_OPENWEBUI_HOST = "http://localhost:4000"

API_HOST = os.getenv("OWUI_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("OWUI_API_PORT", "8502"))
API_ALLOWED_ORIGINS = _split_origins(os.getenv("OWUI_API_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS))
# Resolve the data directory eagerly so downstream code can rely on absolute paths.
DATA_DIR = Path(os.getenv("OWUI_DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()

def _detect_containerized_default_ollama_url() -> str:
    """Return a reasonable default Ollama URL depending on the runtime host."""
    env_url = os.getenv("OLLAMA_BASE_URL")
    if env_url:
        return env_url.rstrip("/")

    docker_marker = Path("/.dockerenv")
    if docker_marker.exists():
        # When running inside Docker we need to reach the host-bound Ollama instance.
        return "http://host.docker.internal:11434"
    return "http://localhost:11434"


OLLAMA_BASE_URL = _detect_containerized_default_ollama_url()
OLLAMA_SUMMARY_MODEL = ""
OLLAMA_LONGFORM_MODEL = ""
OLLAMA_SUMMARY_FALLBACK_MODEL = ""
OLLAMA_EMBED_MODEL = ""
OLLAMA_DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_DEFAULT_TEMPERATURE", "0.2"))
OLLAMA_TIMEOUT_SECONDS = 30.0  # Fixed at 30 seconds
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")

# LiteLLM Configuration
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", "http://localhost:4000").rstrip("/")

OPENWEBUI_TIMEOUT_SECONDS = float(os.getenv("OWUI_OPENWEBUI_TIMEOUT_SECONDS", "30"))

# =========================================================================
# Summarizer Configuration
# =========================================================================

# Legacy retry configuration (for backward compatibility)
SUMMARIZER_OLLAMA_RETRY_ATTEMPTS = max(1, int(os.getenv("SUMMARIZER_OLLAMA_RETRY_ATTEMPTS", "3")))
SUMMARIZER_OLLAMA_RETRY_DELAY_SECONDS = max(0.0, float(os.getenv("SUMMARIZER_OLLAMA_RETRY_DELAY_SECONDS", "3.0")))

# Exponential backoff configuration
# When enabled, retry delays increase exponentially: base_delay * (2 ** attempt) + jitter
SUMMARIZER_USE_EXPONENTIAL_BACKOFF = os.getenv("SUMMARIZER_USE_EXPONENTIAL_BACKOFF", "true").strip().lower() in {"1", "true", "yes", "on"}
SUMMARIZER_RETRY_MAX_ATTEMPTS = max(1, int(os.getenv("SUMMARIZER_RETRY_MAX_ATTEMPTS", "5")))
SUMMARIZER_RETRY_BASE_DELAY = max(0.1, float(os.getenv("SUMMARIZER_RETRY_BASE_DELAY", "1.0")))
SUMMARIZER_RETRY_MAX_DELAY = max(1.0, float(os.getenv("SUMMARIZER_RETRY_MAX_DELAY", "60.0")))

# Parse retry configuration
# Number of times to retry LLM call if JSON parsing fails (enables JSON mode on retry)
SUMMARIZER_PARSE_RETRY_ATTEMPTS = max(1, int(os.getenv("SUMMARIZER_PARSE_RETRY_ATTEMPTS", "2")))

# Error logging configuration
# When enabled, preserves full prompts and responses for debugging (can be large)
SUMMARIZER_PRESERVE_FULL_ERRORS = os.getenv("SUMMARIZER_PRESERVE_FULL_ERRORS", "true").strip().lower() in {"1", "true", "yes", "on"}
# Maximum size for preserved errors (in characters) to prevent memory issues
SUMMARIZER_MAX_ERROR_SIZE = max(1000, int(os.getenv("SUMMARIZER_MAX_ERROR_SIZE", "10000")))

# Sprint 3: Quality validation and drop-off detection configuration
# Minimum keyword overlap ratio for summary quality validation (0.0-1.0)
SUMMARIZER_MIN_KEYWORD_OVERLAP = max(0.0, min(1.0, float(os.getenv("SUMMARIZER_MIN_KEYWORD_OVERLAP", "0.15"))))
# Enable quality validation for summaries (adds quality_score and quality_notes to results)
SUMMARIZER_ENABLE_QUALITY_VALIDATION = os.getenv("SUMMARIZER_ENABLE_QUALITY_VALIDATION", "true").strip().lower() in {"1", "true", "yes", "on"}
# Enable drop-off detection (analyzes whether conversations ended naturally)
SUMMARIZER_ENABLE_DROPOFF_DETECTION = os.getenv("SUMMARIZER_ENABLE_DROPOFF_DETECTION", "true").strip().lower() in {"1", "true", "yes", "on"}
# Number of recent messages to analyze for drop-off detection
SUMMARIZER_DROPOFF_LOOKBACK_MESSAGES = max(2, int(os.getenv("SUMMARIZER_DROPOFF_LOOKBACK_MESSAGES", "4")))

# Sprint 5: Advanced resilience and observability configuration
# Enable detailed logging of all summarizer operations (prompt/response pairs)
SUMMARIZER_ENABLE_DETAILED_LOGGING = os.getenv("SUMMARIZER_ENABLE_DETAILED_LOGGING", "false").strip().lower() in {"1", "true", "yes", "on"}
# Hours to retain detailed logs before rotation (0 = keep forever)
SUMMARIZER_LOG_RETENTION_HOURS = max(0, int(os.getenv("SUMMARIZER_LOG_RETENTION_HOURS", "72")))
# Maximum retry attempts for failed metric extractions
SUMMARIZER_MAX_RETRIES = max(0, min(5, int(os.getenv("SUMMARIZER_MAX_RETRIES", "2"))))
# Enable automatic retry with simplified prompts on failure
SUMMARIZER_ENABLE_FALLBACK_PROMPTS = os.getenv("SUMMARIZER_ENABLE_FALLBACK_PROMPTS", "true").strip().lower() in {"1", "true", "yes", "on"}
# Enable graceful degradation (continue with other metrics if one fails)
SUMMARIZER_ENABLE_GRACEFUL_DEGRADATION = os.getenv("SUMMARIZER_ENABLE_GRACEFUL_DEGRADATION", "true").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_openwebui_base(value: Optional[str]) -> str:
    """Normalize a configured Open WebUI host into a usable base URL."""
    candidate = (value or "").strip()
    if not candidate:
        return ""
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    parsed = urlparse(candidate)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return candidate.rstrip("/")


_OPENWEBUI_API_BASE_OVERRIDE: Optional[str] = None
_OPENWEBUI_API_KEY_OVERRIDE: Optional[str] = None


def get_openwebui_api_base() -> str:
    """Return the configured Open WebUI base URL."""
    if _OPENWEBUI_API_BASE_OVERRIDE is not None:
        return _OPENWEBUI_API_BASE_OVERRIDE

    direct_env = _normalize_openwebui_base(os.getenv("OWUI_DIRECT_HOST"))
    return direct_env or _normalize_openwebui_base(DEFAULT_OPENWEBUI_HOST)


def get_openwebui_api_key() -> str:
    """Return the configured Open WebUI API key."""
    if _OPENWEBUI_API_KEY_OVERRIDE is not None:
        return _OPENWEBUI_API_KEY_OVERRIDE

    direct_key = os.getenv("OWUI_DIRECT_API_KEY", "").strip()
    return direct_key


def set_openwebui_api_config(base_url: Optional[str], api_key: Optional[str]) -> bool:
    """Override the Open WebUI base URL and API key at runtime.

    Returns:
        True if the effective configuration changed, False otherwise.
    """
    global _OPENWEBUI_API_BASE_OVERRIDE, _OPENWEBUI_API_KEY_OVERRIDE

    normalized_base = _normalize_openwebui_base(base_url)
    normalized_key = (api_key or "").strip()

    changed = False
    if normalized_base != _OPENWEBUI_API_BASE_OVERRIDE:
        _OPENWEBUI_API_BASE_OVERRIDE = normalized_base
        changed = True
    if normalized_key != _OPENWEBUI_API_KEY_OVERRIDE:
        _OPENWEBUI_API_KEY_OVERRIDE = normalized_key
        changed = True

    return changed


AUTH_TOKEN_TTL_SECONDS = int(os.getenv("AUTH_TOKEN_TTL_SECONDS", str(60 * 60 * 24)))
AUTH_TOKEN_HASH_SECRET = os.getenv("AUTH_TOKEN_HASH_SECRET", "")
