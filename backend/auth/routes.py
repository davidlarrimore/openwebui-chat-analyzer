"""FastAPI routes that expose session-based authentication flows."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse

from ..models import AuthUserPublic
from . import config
from .cookies import attach_session_cookies, clear_session_cookies
from .deps import get_authenticated_session
from .local import ensure_local_enabled, login_with_credentials
from .oidc import build_authorization_url, finalize_oidc_login
from .schemas import (
    AuthStatusResponse,
    BootstrapRequest,
    LoginRequest,
    LogoutResponse,
    SessionEnvelope,
    SessionMeta,
    SessionUser,
)
from .service import AuthService, AuthenticatedSession, get_auth_service

router = APIRouter(prefix="/api/backend/auth", tags=["auth"])


def _session_meta(session) -> SessionMeta:
    return SessionMeta(
        session_id=session.id,
        expires_at=session.expires_at,
        refresh_expires_at=session.refresh_expires_at,
    )


def _session_user(user: AuthUserPublic, *, is_admin: bool, provider: str, tenant: Optional[str]) -> SessionUser:
    return SessionUser(
        id=user.id,
        email=user.email,
        name=user.name,
        is_admin=is_admin,
        provider=provider,
        tenant=tenant,
    )


def _build_envelope(user_obj, session_obj, *, callback_url: Optional[str] = None) -> SessionEnvelope:
    auth_service = get_auth_service()
    public = auth_service.serialize_user(user_obj)
    payload_user = _session_user(
        public,
        is_admin=getattr(user_obj, "is_admin", False),
        provider=getattr(user_obj, "provider", "local"),
        tenant=getattr(user_obj, "tenant", None),
    )
    return SessionEnvelope(
        user=payload_user,
        session=_session_meta(session_obj),
        callback_url=callback_url,
    )


@router.get("/status", response_model=AuthStatusResponse)
def auth_status(auth_service: AuthService = Depends(get_auth_service)) -> AuthStatusResponse:
    return AuthStatusResponse(has_users=auth_service.has_any_users())


@router.post("/bootstrap", response_model=SessionEnvelope, status_code=status.HTTP_201_CREATED)
def bootstrap(
    payload: BootstrapRequest,
    request: Request,
    response: Response,
    auth_service: AuthService = Depends(get_auth_service),
) -> SessionEnvelope:
    ensure_local_enabled()
    try:
        user = auth_service.create_initial_user(email=payload.email, password=payload.password, name=payload.name)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    session, tokens = auth_service.establish_session(request=request, user=user)
    attach_session_cookies(response, tokens)
    response.headers["Cache-Control"] = "no-store"
    return _build_envelope(user, session, callback_url="/dashboard")


@router.post("/login", response_model=SessionEnvelope)
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    auth_service: AuthService = Depends(get_auth_service),
) -> SessionEnvelope:
    user = login_with_credentials(auth_service, email=payload.email, password=payload.password)
    session, tokens = auth_service.establish_session(request=request, user=user)
    attach_session_cookies(response, tokens)
    response.headers["Cache-Control"] = "no-store"
    callback = payload.callback_url or request.query_params.get("callbackUrl")
    return _build_envelope(user, session, callback_url=callback)


@router.post("/logout", response_model=LogoutResponse)
def logout(request: Request, response: Response, auth_service: AuthService = Depends(get_auth_service)) -> LogoutResponse:
    auth_service.logout(request)
    clear_session_cookies(response)
    return LogoutResponse()


@router.post("/refresh", response_model=SessionEnvelope)
def refresh(
    request: Request,
    response: Response,
    auth_service: AuthService = Depends(get_auth_service),
) -> SessionEnvelope:
    try:
        session, tokens = auth_service.refresh_session(request)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(error)) from error
    user = auth_service.get_user_by_id(session.user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    attach_session_cookies(response, tokens)
    response.headers["Cache-Control"] = "no-store"
    return _build_envelope(user, session)


@router.get("/session", response_model=SessionEnvelope)
def session_info(principal: AuthenticatedSession = Depends(get_authenticated_session)) -> SessionEnvelope:
    return _build_envelope(principal.user, principal.session)


@router.get("/oidc/login")
async def oidc_login(
    request: Request,
    response: Response,
    callbackUrl: Optional[str] = None,
):
    url = await build_authorization_url(request, response, callback_url=callbackUrl)
    return RedirectResponse(url, status_code=status.HTTP_302_FOUND)


@router.get("/oidc/callback")
async def oidc_callback(
    request: Request,
    response: Response,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    auth_service: AuthService = Depends(get_auth_service),
):
    if error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)
    if not code or not state:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing code/state")
    user_info, callback_url = await finalize_oidc_login(code=code, state=state, request=request)
    user = auth_service.ensure_oidc_user(
        subject=user_info["subject"],
        email=user_info["email"],
        name=user_info["name"],
        tenant=user_info["tenant"],
    )
    session, tokens = auth_service.establish_session(request=request, user=user)
    attach_session_cookies(response, tokens)
    response.delete_cookie(key=config.PKCE_COOKIE_NAME, domain=config.SESSION_COOKIE_DOMAIN, path=config.SESSION_COOKIE_PATH)
    return RedirectResponse(callback_url or "/dashboard", status_code=status.HTTP_302_FOUND)


def _require_admin(principal: AuthenticatedSession = Depends(get_authenticated_session)) -> AuthenticatedSession:
    if not getattr(principal.user, "is_admin", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    return principal


@router.get("/sessions")
def list_sessions(
    _: AuthenticatedSession = Depends(_require_admin),
    auth_service: AuthService = Depends(get_auth_service),
):
    return {"sessions": auth_service.list_sessions()}


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_session(
    session_id: str,
    _: AuthenticatedSession = Depends(_require_admin),
    auth_service: AuthService = Depends(get_auth_service),
):
    if not auth_service.revoke_session(session_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
