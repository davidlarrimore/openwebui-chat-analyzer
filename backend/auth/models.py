"""SQLAlchemy models for the authentication subsystem."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import relationship

from ..db import Base
from ..db_models import TimestampMixin


class AuthUser(TimestampMixin, Base):
    """Application user that can authenticate with the analyzer."""

    __tablename__ = "auth_users"

    id = Column(String(40), primary_key=True, default=lambda: uuid4().hex)
    email = Column(String(320), unique=True, nullable=False, index=True)
    password_hash = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, server_default="true")
    is_admin = Column(Boolean, nullable=False, server_default="false")
    provider = Column(String(32), nullable=False, server_default="local")
    provider_subject = Column(String(255), nullable=True, index=True)
    tenant = Column(String(255), nullable=True)
    display_name = Column(String(320), nullable=True)
    last_login_at = Column(DateTime(timezone=True), nullable=True)

    sessions = relationship("AuthSession", back_populates="user", cascade="all, delete-orphan")


class AuthSession(TimestampMixin, Base):
    """Persistent session with refresh token binding."""

    __tablename__ = "auth_sessions"

    id = Column(String(72), primary_key=True, default=lambda: uuid4().hex)
    user_id = Column(String(40), ForeignKey("auth_users.id", ondelete="CASCADE"), nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    refresh_expires_at = Column(DateTime(timezone=True), nullable=False)
    refresh_token_hash = Column(String(128), nullable=False)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    rotated_at = Column(DateTime(timezone=True), nullable=True)
    last_seen_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    user_agent_hash = Column(String(128), nullable=True)
    ip_prefix = Column(String(64), nullable=True)

    user = relationship("AuthUser", back_populates="sessions")


Index("ix_auth_sessions_expiry", AuthSession.expires_at)
Index("ix_auth_sessions_refresh_expiry", AuthSession.refresh_expires_at)
