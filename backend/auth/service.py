"""Authentication service coordinating users, sessions, and providers."""

from __future__ import annotations

import logging
import hmac
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Request
from sqlalchemy import func, or_, select

from ..db import init_database, session_scope
from ..models import AuthUserPublic
from . import config
from .crypto import (
    derive_ip_prefix,
    fingerprint_user_agent,
    generate_token,
    hash_password,
    hash_token,
    verify_password,
)
from .models import AuthSession, AuthUser

LOGGER = logging.getLogger(__name__)


@dataclass
class SessionContext:
    user_agent: Optional[str]
    ip_address: Optional[str]
    ip_prefix: Optional[str]
    trusted_internal: bool = False


@dataclass
class SessionTokens:
    session_id: str
    refresh_token: str
    expires_at: datetime
    refresh_expires_at: datetime


@dataclass
class AuthenticatedSession:
    user: AuthUser
    session: AuthSession


class AuthService:
    """Central authority for authentication flows."""

    def __init__(self) -> None:
        init_database()
        self._access_ttl = timedelta(minutes=config.SESSION_ACCESS_TTL_MINUTES)
        self._refresh_ttl = timedelta(hours=config.SESSION_REFRESH_TTL_HOURS)
        self._idle_extension = timedelta(minutes=config.SESSION_IDLE_EXTENSION_MINUTES)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _normalize_dt(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _session_context_from_request(self, request: Request) -> SessionContext:
        user_agent = request.headers.get("user-agent")
        ip_address = request.headers.get("x-forwarded-for")
        if ip_address and "," in ip_address:
            ip_address = ip_address.split(",", 1)[0].strip()
        if not ip_address and request.client:
            ip_address = request.client.host
        internal_header = (request.headers.get("x-analyzer-internal") or "").strip().lower()
        return SessionContext(
            user_agent=user_agent,
            ip_address=ip_address,
            ip_prefix=derive_ip_prefix(ip_address),
            trusted_internal=internal_header in {"true", "1", "yes"},
        )

    def has_any_users(self) -> bool:
        with session_scope() as db:
            result = db.execute(select(func.count()).select_from(AuthUser))
            count = result.scalar() or 0
        return count > 0

    def create_initial_user(self, *, email: str, password: str, name: Optional[str] = None) -> AuthUser:
        if self.has_any_users():
            raise ValueError("Users already provisioned.")
        return self._create_local_user(email=email, password=password, name=name, is_admin=True)

    def _create_local_user(
        self,
        *,
        email: str,
        password: str,
        name: Optional[str],
        is_admin: bool,
    ) -> AuthUser:
        normalized = email.strip().lower()
        display_name = (name or normalized).strip() or normalized
        with session_scope() as db:
            existing = db.execute(select(AuthUser).where(AuthUser.email == normalized)).scalar_one_or_none()
            if existing:
                raise ValueError("User already exists")
            user = AuthUser(
                email=normalized,
                display_name=display_name,
                password_hash=hash_password(password),
                is_admin=is_admin,
                provider="local",
            )
            db.add(user)
            db.flush()
            return user

    def authenticate_local(self, *, email: str, password: str) -> Optional[AuthUser]:
        normalized = email.strip().lower()
        with session_scope() as db:
            user = db.execute(select(AuthUser).where(AuthUser.email == normalized)).scalar_one_or_none()
            if user is None or not user.is_active or not user.password_hash:
                return None
            if not verify_password(password, user.password_hash):
                return None
            user.last_login_at = self._now()
            db.add(user)
            return user

    def ensure_oidc_user(
        self,
        *,
        subject: str,
        email: str,
        name: Optional[str],
        tenant: Optional[str],
    ) -> AuthUser:
        normalized = email.strip().lower()
        with session_scope() as db:
            existing = db.execute(
                select(AuthUser).where(
                    or_(
                        AuthUser.provider_subject == subject,
                        AuthUser.email == normalized,
                    )
                )
            ).scalar_one_or_none()
            if existing:
                existing.provider = "oidc"
                existing.provider_subject = subject
                existing.tenant = tenant
                if name:
                    existing.display_name = name
                if existing.email != normalized:
                    existing.email = normalized
                db.add(existing)
                return existing
            user = AuthUser(
                email=normalized,
                display_name=name or normalized,
                password_hash=None,
                is_admin=True,
                provider="oidc",
                provider_subject=subject,
                tenant=tenant,
            )
            db.add(user)
            db.flush()
            return user

    def serialize_user(self, user: AuthUser) -> AuthUserPublic:
        return AuthUserPublic(
            id=user.id,
            username=user.email,
            email=user.email,
            name=user.display_name or user.email,
        )

    def session_metadata(self, tokens: SessionTokens) -> Dict[str, datetime | str]:
        return {
            "session_id": tokens.session_id,
            "expires_at": tokens.expires_at,
            "refresh_expires_at": tokens.refresh_expires_at,
        }

    def _create_session(self, *, user: AuthUser, context: SessionContext) -> Tuple[AuthSession, SessionTokens]:
        now = self._now()
        session_id = generate_token(32)
        refresh_token = generate_token(48)
        session = AuthSession(
            id=session_id,
            user_id=user.id,
            expires_at=now + self._access_ttl,
            refresh_expires_at=now + self._refresh_ttl,
            refresh_token_hash=hash_token(refresh_token),
            user_agent_hash=fingerprint_user_agent(context.user_agent),
            ip_prefix=context.ip_prefix,
            last_seen_at=now,
        )
        with session_scope() as db:
            db.add(session)
        tokens = SessionTokens(
            session_id=session_id,
            refresh_token=refresh_token,
            expires_at=session.expires_at,
            refresh_expires_at=session.refresh_expires_at,
        )
        return session, tokens

    def establish_session(self, *, request: Request, user: AuthUser) -> Tuple[AuthSession, SessionTokens]:
        context = self._session_context_from_request(request)
        self._enforce_device_limit(user.id)
        return self._create_session(user=user, context=context)

    def _enforce_device_limit(self, user_id: str) -> None:
        limit = config.SESSION_MAX_DEVICES
        if limit <= 0:
            return
        with session_scope() as db:
            sessions = (
                db.execute(
                    select(AuthSession)
                    .where(AuthSession.user_id == user_id, AuthSession.revoked_at.is_(None))
                    .order_by(AuthSession.created_at.desc())
                )
                .scalars()
                .all()
            )
            if len(sessions) <= limit:
                return
            for stale in sessions[limit:]:
                stale.revoked_at = self._now()
                db.add(stale)

    def validate_request(self, request: Request) -> Optional[AuthenticatedSession]:
        session_id = request.cookies.get(config.SESSION_COOKIE_NAME)
        if not session_id:
            return None
        context = self._session_context_from_request(request)
        return self.validate_session(session_id=session_id, context=context)

    def validate_session(self, *, session_id: str, context: SessionContext) -> Optional[AuthenticatedSession]:
        now = self._now()
        with session_scope() as db:
            session = db.get(AuthSession, session_id)
            if session is None or session.revoked_at is not None:
                return None
            if self._normalize_dt(session.refresh_expires_at) <= now:
                session.revoked_at = now
                db.add(session)
                return None
            if self._normalize_dt(session.expires_at) <= now:
                return None
            request_ua_hash = fingerprint_user_agent(context.user_agent)
            if (
                not context.trusted_internal
                and session.user_agent_hash
                and request_ua_hash
                and request_ua_hash != session.user_agent_hash
            ):
                return None
            if (
                not context.trusted_internal
                and session.ip_prefix
                and context.ip_prefix
                and session.ip_prefix != context.ip_prefix
            ):
                return None
            user = db.get(AuthUser, session.user_id)
            if user is None or not user.is_active:
                return None
            last_seen = self._normalize_dt(session.last_seen_at)
            if now - last_seen >= self._idle_extension:
                session.last_seen_at = now
                session.expires_at = now + self._access_ttl
                db.add(session)
            return AuthenticatedSession(user=user, session=session)

    def refresh_session(self, request: Request) -> Tuple[AuthSession, SessionTokens]:
        session_id = request.cookies.get(config.SESSION_COOKIE_NAME)
        refresh_token = request.cookies.get(config.REFRESH_COOKIE_NAME)
        if not session_id or not refresh_token:
            raise ValueError("Missing refresh context")
        context = self._session_context_from_request(request)
        now = self._now()
        with session_scope() as db:
            current = db.get(AuthSession, session_id)
            if current is None or current.revoked_at is not None:
                raise ValueError("Session revoked")
            if self._normalize_dt(current.refresh_expires_at) <= now:
                raise ValueError("Refresh token expired")
            if not hmac_compare(current.refresh_token_hash, hash_token(refresh_token)):
                raise ValueError("Refresh token mismatch")
            current.revoked_at = now
            current.rotated_at = now
            db.add(current)
            user_id = current.user_id
        user = self.get_user_by_id(user_id)
        if user is None:
            raise ValueError("User does not exist")
        return self._create_session(user=user, context=context)

    def get_user_by_id(self, user_id: str) -> Optional[AuthUser]:
        with session_scope() as db:
            return db.get(AuthUser, user_id)

    def logout(self, request: Request) -> bool:
        session_id = request.cookies.get(config.SESSION_COOKIE_NAME)
        if not session_id:
            return False
        now = self._now()
        with session_scope() as db:
            session = db.get(AuthSession, session_id)
            if session is None:
                return False
            session.revoked_at = now
            db.add(session)
        return True

    def revoke_session(self, session_id: str) -> bool:
        now = self._now()
        with session_scope() as db:
            session = db.get(AuthSession, session_id)
            if session is None:
                return False
            session.revoked_at = now
            db.add(session)
            return True

    def list_sessions(self, *, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with session_scope() as db:
            stmt = select(AuthSession)
            if user_id:
                stmt = stmt.where(AuthSession.user_id == user_id)
            sessions = db.execute(stmt.order_by(AuthSession.created_at.desc())).scalars().all()
            payload: List[Dict[str, Any]] = []
            for record in sessions:
                payload.append(
                    {
                        "session_id": record.id,
                        "user_id": record.user_id,
                        "created_at": record.created_at,
                        "last_seen_at": record.last_seen_at,
                        "expires_at": record.expires_at,
                        "refresh_expires_at": record.refresh_expires_at,
                        "revoked_at": record.revoked_at,
                        "rotated_at": record.rotated_at,
                        "ip_prefix": record.ip_prefix,
                    }
                )
            return payload


def hmac_compare(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)


_AUTH_SERVICE: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    global _AUTH_SERVICE
    if _AUTH_SERVICE is None:
        _AUTH_SERVICE = AuthService()
    return _AUTH_SERVICE
