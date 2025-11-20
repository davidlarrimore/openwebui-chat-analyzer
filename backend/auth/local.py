"""Local email/password authentication helpers."""

from __future__ import annotations

from fastapi import HTTPException, status

from .config import AUTH_MODE, AuthMode
from .service import AuthService


def ensure_local_enabled() -> None:
    if AUTH_MODE == AuthMode.OAUTH:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Local authentication is disabled")


def login_with_credentials(auth_service: AuthService, *, email: str, password: str):
    ensure_local_enabled()
    user = auth_service.authenticate_local(email=email, password=password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return user
