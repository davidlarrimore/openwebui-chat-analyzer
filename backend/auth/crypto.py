"""Cryptographic helpers for the authentication subsystem."""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import os
import secrets
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from .config import SESSION_SECRET

_PASSWORD_HASHER = PasswordHasher()


def hash_password(password: str) -> str:
    """Hash a plain text password using Argon2."""

    return _PASSWORD_HASHER.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password hash."""

    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def generate_token(length: int = 48) -> str:
    """Generate a URL-safe random token."""

    return secrets.token_urlsafe(length)


def hash_token(value: str, *, secret: str = SESSION_SECRET) -> str:
    """Create a deterministic HMAC hash of a token."""

    return hmac.new(secret.encode("utf-8"), msg=value.encode("utf-8"), digestmod=hashlib.sha256).hexdigest()


def fingerprint_user_agent(user_agent: Optional[str]) -> Optional[str]:
    if not user_agent:
        return None
    digest = hashlib.sha256(user_agent.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def derive_ip_prefix(ip_address: Optional[str]) -> Optional[str]:
    if not ip_address:
        return None
    try:
        parsed = ipaddress.ip_address(ip_address)
    except ValueError:
        return None
    if isinstance(parsed, ipaddress.IPv4Address):
        octets = str(parsed).split(".")
        return ".".join(octets[:3]) + ".0/24"
    if isinstance(parsed, ipaddress.IPv6Address):
        hextets = parsed.exploded.split(":")
        return ":".join(hextets[:4]) + "::/64"
    return None


def generate_pkce_verifier() -> str:
    raw = os.urandom(32)
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def generate_pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
