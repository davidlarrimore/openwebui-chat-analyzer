"""OIDC helpers for Microsoft Entra ID sign-in flows."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException, Request, Response, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from jose import jwk, jwt

from . import config
from .crypto import generate_pkce_challenge, generate_pkce_verifier, generate_token

LOGGER = logging.getLogger(__name__)

_serializer = URLSafeTimedSerializer(config.SESSION_SECRET, salt="oidc-state")
_metadata_cache: Dict[str, Any] | None = None
_metadata_cached_at: Optional[datetime] = None
_jwks_cache: Dict[str, Any] | None = None
_jwks_cached_at: Optional[datetime] = None
_cache_lock = asyncio.Lock()


async def _fetch_json(url: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


async def _metadata() -> Dict[str, Any]:
    global _metadata_cache, _metadata_cached_at
    async with _cache_lock:
        if _metadata_cache and _metadata_cached_at and datetime.now(timezone.utc) - _metadata_cached_at < timedelta(hours=1):
            return _metadata_cache
        if not config.OIDC_ISSUER:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OIDC issuer missing")
        well_known = config.OIDC_ISSUER.rstrip("/") + "/.well-known/openid-configuration"
        _metadata_cache = await _fetch_json(well_known)
        _metadata_cached_at = datetime.now(timezone.utc)
        return _metadata_cache


async def _jwks() -> Dict[str, Any]:
    global _jwks_cache, _jwks_cached_at
    async with _cache_lock:
        if _jwks_cache and _jwks_cached_at and datetime.now(timezone.utc) - _jwks_cached_at < timedelta(hours=4):
            return _jwks_cache
        metadata = await _metadata()
        jwks_uri = metadata.get("jwks_uri")
        if not jwks_uri:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OIDC JWKS missing")
        _jwks_cache = await _fetch_json(jwks_uri)
        _jwks_cached_at = datetime.now(timezone.utc)
        return _jwks_cache


def _serialize_state(callback_url: Optional[str], nonce: str) -> str:
    payload = {"callback": callback_url or "/dashboard", "nonce": nonce}
    return _serializer.dumps(payload)


def _deserialize_state(state: str) -> Dict[str, Any]:
    try:
        return _serializer.loads(state, max_age=600)
    except SignatureExpired as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OIDC state expired") from exc
    except BadSignature as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OIDC state invalid") from exc


async def build_authorization_url(request: Request, response: Response, *, callback_url: Optional[str]) -> str:
    if not config.oidc_enabled():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="OIDC disabled")
    metadata = await _metadata()
    authorization_endpoint = metadata.get("authorization_endpoint")
    if not authorization_endpoint:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OIDC authorization endpoint missing")
    nonce = generate_token(8)
    state = _serialize_state(callback_url, nonce)
    verifier = generate_pkce_verifier()
    challenge = generate_pkce_challenge(verifier)
    response.set_cookie(
        key=config.PKCE_COOKIE_NAME,
        value=verifier,
        httponly=True,
        secure=config.SESSION_COOKIE_SECURE,
        samesite=config.SESSION_COOKIE_SAMESITE,
        domain=config.SESSION_COOKIE_DOMAIN,
        path=config.SESSION_COOKIE_PATH,
        max_age=600,
    )
    query = {
        "client_id": config.OIDC_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": config.OIDC_REDIRECT_URI,
        "scope": config.OIDC_SCOPES,
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    if config.OIDC_PROMPT:
        query["prompt"] = config.OIDC_PROMPT
    return f"{authorization_endpoint}?{urlencode(query)}"


async def exchange_code(*, code: str, verifier: str) -> Dict[str, Any]:
    metadata = await _metadata()
    token_endpoint = metadata.get("token_endpoint")
    if not token_endpoint:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OIDC token endpoint missing")
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": config.OIDC_REDIRECT_URI,
        "client_id": config.OIDC_CLIENT_ID,
        "code_verifier": verifier,
    }
    if config.OIDC_CLIENT_SECRET:
        data["client_secret"] = config.OIDC_CLIENT_SECRET
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(token_endpoint, data=data)
        if resp.status_code != 200:
            LOGGER.error("OIDC token exchange failed: %s", resp.text)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OIDC token exchange failed")
        return resp.json()


async def decode_id_token(id_token: str, *, nonce: str) -> Dict[str, Any]:
    keys = await _jwks()
    header = jwt.get_unverified_header(id_token)
    kid = header.get("kid")
    if not kid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OIDC token missing kid")
    key_data = next((key for key in keys.get("keys", []) if key.get("kid") == kid), None)
    if not key_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OIDC signing key not found")
    public_key = jwk.construct(key_data).to_pem().decode("utf-8")
    claims = jwt.decode(
        id_token,
        public_key,
        algorithms=[header.get("alg", "RS256")],
        audience=config.OIDC_CLIENT_ID,
        issuer=config.OIDC_ISSUER,
    )
    if claims.get("nonce") != nonce:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OIDC nonce mismatch")
    return claims


def parse_user_info(claims: Dict[str, Any]) -> Dict[str, Optional[str]]:
    email = claims.get("preferred_username") or claims.get("email")
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OIDC email missing")
    return {
        "subject": claims.get("sub"),
        "email": email,
        "name": claims.get("name") or email,
        "tenant": claims.get("tid") or claims.get("tenantid"),
    }


async def finalize_oidc_login(
    *,
    code: str,
    state: str,
    request: Request,
) -> tuple[Dict[str, Optional[str]], str]:
    payload = _deserialize_state(state)
    verifier = request.cookies.get(config.PKCE_COOKIE_NAME)
    if not verifier:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OIDC PKCE verifier missing")
    token_response = await exchange_code(code=code, verifier=verifier)
    id_token = token_response.get("id_token")
    if not id_token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OIDC id_token missing")
    claims = await decode_id_token(id_token, nonce=payload["nonce"])
    user_info = parse_user_info(claims)
    return user_info, payload.get("callback") or "/dashboard"
