"""Configuration helpers for the authentication subsystem."""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()


class AuthMode(str, Enum):
    """Supported authentication operating modes."""

    DEFAULT = "DEFAULT"
    HYBRID = "HYBRID"
    OAUTH = "OAUTH"


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_base_url(value: Optional[str], fallback: str) -> str:
    candidate = (value or fallback or "").strip()
    if not candidate:
        return fallback
    parsed = urlparse(candidate if "://" in candidate else f"https://{candidate}")
    if not parsed.scheme or not parsed.netloc:
        return fallback
    return f"{parsed.scheme}://{parsed.netloc}"


APP_BASE_URL = _normalize_base_url(os.getenv("APP_BASE_URL"), "http://localhost:3000")
APP_HOSTNAME = urlparse(APP_BASE_URL).hostname or "localhost"

AUTH_MODE = AuthMode(os.getenv("AUTH_MODE", AuthMode.DEFAULT.value).upper())

SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-session-secret")
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "analyzer_session")
REFRESH_COOKIE_NAME = os.getenv("REFRESH_COOKIE_NAME", "analyzer_refresh")
PKCE_COOKIE_NAME = os.getenv("OIDC_PKCE_COOKIE_NAME", "analyzer_oidc_pkce")

SESSION_COOKIE_DOMAIN = os.getenv("SESSION_COOKIE_DOMAIN") or APP_HOSTNAME
SESSION_COOKIE_PATH = os.getenv("SESSION_COOKIE_PATH", "/")
SESSION_COOKIE_SECURE = _bool_env("SESSION_COOKIE_SECURE", default=APP_BASE_URL.startswith("https://"))
SESSION_COOKIE_HTTPONLY = _bool_env("SESSION_COOKIE_HTTPONLY", default=True)
SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "Lax").capitalize()

SESSION_ACCESS_TTL_MINUTES = int(os.getenv("SESSION_ACCESS_TTL_MINUTES", "30"))
SESSION_REFRESH_TTL_HOURS = int(os.getenv("SESSION_REFRESH_TTL_HOURS", "12"))
SESSION_IDLE_EXTENSION_MINUTES = int(os.getenv("SESSION_IDLE_EXTENSION_MINUTES", "5"))

SESSION_MAX_DEVICES = int(os.getenv("SESSION_MAX_DEVICES", "5"))
SESSION_IDLE_REAP_SECONDS = int(os.getenv("SESSION_IDLE_REAP_SECONDS", "300"))

OIDC_ISSUER = os.getenv("OIDC_ISSUER")
OIDC_CLIENT_ID = os.getenv("OIDC_CLIENT_ID")
OIDC_CLIENT_SECRET = os.getenv("OIDC_CLIENT_SECRET")
OIDC_SCOPES = os.getenv("OIDC_SCOPES", "openid profile email offline_access")
OIDC_PROMPT = os.getenv("OIDC_PROMPT", "select_account")
OIDC_REDIRECT_PATH = os.getenv("OIDC_REDIRECT_PATH", "/api/backend/auth/oidc/callback")
OIDC_REDIRECT_URI = f"{APP_BASE_URL.rstrip('/')}{OIDC_REDIRECT_PATH}"


def oidc_enabled() -> bool:
    """Return True if OIDC flows are configured and allowed."""

    if AUTH_MODE == AuthMode.DEFAULT:
        return False
    return bool(OIDC_ISSUER and OIDC_CLIENT_ID)
