"""FastAPI dependencies for session-aware routes."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from ..models import AuthUserPublic
from .service import AuthenticatedSession, AuthService, get_auth_service


def get_authenticated_session(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthenticatedSession:
    principal = auth_service.validate_request(request)
    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return principal


def get_optional_session(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthenticatedSession | None:
    return auth_service.validate_request(request)


def require_user(principal: AuthenticatedSession = Depends(get_authenticated_session)) -> AuthUserPublic:
    auth_service = get_auth_service()
    return auth_service.serialize_user(principal.user)


def resolve_optional_user(
    principal: AuthenticatedSession | None = Depends(get_optional_session),
) -> AuthUserPublic | None:
    if principal is None:
        return None
    auth_service = get_auth_service()
    return auth_service.serialize_user(principal.user)
