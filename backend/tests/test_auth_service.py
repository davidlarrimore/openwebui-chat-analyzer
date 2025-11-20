"""Tests for the backend authentication service."""

from __future__ import annotations

from contextlib import contextmanager

import importlib

from starlette.requests import Request
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import Base
from backend.auth import config
from backend.auth import models as auth_models  # noqa: F401


def _build_request(*, cookies: dict[str, str] | None = None) -> Request:
    cookie_header = "".join(
        [f"{key}={value}; " for key, value in (cookies or {}).items()]
    ).strip()
    headers = [
        (b"user-agent", b"pytest"),
        (b"x-forwarded-for", b"127.0.0.1"),
    ]
    if cookie_header:
        headers.append((b"cookie", cookie_header.encode("ascii")))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/backend/auth/test",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


def _build_service(monkeypatch):
    from backend.auth import service as auth_service_module

    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    @contextmanager
    def _session_scope():
        session = SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    importlib.reload(auth_service_module)
    monkeypatch.setattr(auth_service_module, "session_scope", _session_scope)
    monkeypatch.setattr(auth_service_module, "init_database", lambda: None)
    return auth_service_module.AuthService()


def test_local_login_refresh_and_logout(monkeypatch):
    service = _build_service(monkeypatch)
    user = service.create_initial_user(email="test@example.com", password="super-secret", name="Tester")

    login_request = _build_request()
    session, tokens = service.establish_session(request=login_request, user=user)

    assert session.user_id == user.id
    assert tokens.session_id
    assert tokens.refresh_token

    refresh_request = _build_request(
        cookies={
            config.SESSION_COOKIE_NAME: tokens.session_id,
            config.REFRESH_COOKIE_NAME: tokens.refresh_token,
        }
    )
    new_session, new_tokens = service.refresh_session(refresh_request)

    assert new_session.id != session.id
    assert new_tokens.session_id != tokens.session_id

    logout_request = _build_request(
        cookies={config.SESSION_COOKIE_NAME: new_tokens.session_id}
    )
    assert service.logout(logout_request) is True
