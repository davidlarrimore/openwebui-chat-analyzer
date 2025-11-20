"""Authentication package for the Open WebUI Chat Analyzer backend."""

from .routes import router as auth_router
from .service import AuthService, get_auth_service

__all__ = [
    "AuthService",
    "auth_router",
    "get_auth_service",
]
