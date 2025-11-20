"""Cookie helpers for session management."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Response

from . import config
from .service import SessionTokens


def _max_age(target: datetime) -> int:
    now = datetime.now(timezone.utc)
    delta = int((target - now).total_seconds())
    return max(delta, 60)


def attach_session_cookies(response: Response, tokens: SessionTokens) -> None:
    response.set_cookie(
        key=config.SESSION_COOKIE_NAME,
        value=tokens.session_id,
        domain=config.SESSION_COOKIE_DOMAIN,
        path=config.SESSION_COOKIE_PATH,
        httponly=config.SESSION_COOKIE_HTTPONLY,
        secure=config.SESSION_COOKIE_SECURE,
        samesite=config.SESSION_COOKIE_SAMESITE,
        max_age=_max_age(tokens.expires_at),
    )
    response.set_cookie(
        key=config.REFRESH_COOKIE_NAME,
        value=tokens.refresh_token,
        domain=config.SESSION_COOKIE_DOMAIN,
        path=config.SESSION_COOKIE_PATH,
        httponly=config.SESSION_COOKIE_HTTPONLY,
        secure=config.SESSION_COOKIE_SECURE,
        samesite=config.SESSION_COOKIE_SAMESITE,
        max_age=_max_age(tokens.refresh_expires_at),
    )


def clear_session_cookies(response: Response) -> None:
    for cookie_name in (config.SESSION_COOKIE_NAME, config.REFRESH_COOKIE_NAME, config.PKCE_COOKIE_NAME):
        response.delete_cookie(
            key=cookie_name,
            domain=config.SESSION_COOKIE_DOMAIN,
            path=config.SESSION_COOKIE_PATH,
        )
