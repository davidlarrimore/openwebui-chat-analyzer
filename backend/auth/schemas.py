"""Pydantic schemas for authentication routes."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field


class SessionUser(BaseModel):
    id: str
    email: EmailStr
    name: str
    is_admin: bool = Field(default=False)
    provider: Literal["local", "oidc"]
    tenant: Optional[str] = None


class SessionMeta(BaseModel):
    session_id: str = Field(..., description="Opaque session identifier")
    expires_at: datetime
    refresh_expires_at: datetime


class SessionEnvelope(BaseModel):
    user: SessionUser
    session: SessionMeta
    callback_url: Optional[str] = None


class AuthStatusResponse(BaseModel):
    has_users: bool


class BootstrapRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    callback_url: Optional[str] = None


class RefreshResponse(SessionEnvelope):
    pass


class LogoutResponse(BaseModel):
    success: bool = True
