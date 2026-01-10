"""HTTP client for interacting with an Ollama instance."""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import OLLAMA_BASE_URL, OLLAMA_TIMEOUT_SECONDS

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO)

_JSON_DEBUG_MAX_LENGTH = 2048


def _format_debug_json(payload: Any) -> str:
    """Return a truncated JSON string suitable for debug logging."""
    if payload is None:
        return "<none>"
    try:
        encoded = json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        encoded = str(payload)
    if len(encoded) > _JSON_DEBUG_MAX_LENGTH:
        return f"{encoded[:_JSON_DEBUG_MAX_LENGTH]}…(truncated)"
    return encoded


class OllamaClientError(RuntimeError):
    """Raised when the Ollama client cannot complete a request."""


class OllamaOutOfMemoryError(OllamaClientError):
    """Raised when the Ollama server reports insufficient memory for a request."""


def _build_retry() -> Retry:
    return Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=0.5,
        status_forcelist=(408, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )


def _normalize_path(path: str) -> str:
    if not path.startswith("/"):
        return "/" + path
    return path


@dataclass
class OllamaGenerateResult:
    """Response returned from a generate request."""

    model: str
    response: str
    raw: Dict[str, Any]


@dataclass
class OllamaChatResult:
    """Response returned from a chat request."""

    model: str
    message: Dict[str, Any]
    raw: Dict[str, Any]


@dataclass
class OllamaEmbedResult:
    """Response returned from an embed request."""

    model: str
    embeddings: List[List[float]]
    raw: Dict[str, Any]


class OllamaClient:
    """Lightweight wrapper around the Ollama REST API."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        *,
        timeout: float = OLLAMA_TIMEOUT_SECONDS,
        session: Optional[Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()
        adapter = HTTPAdapter(max_retries=_build_retry())
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
    ) -> Response:
        url = f"{self.base_url}{_normalize_path(path)}"
        LOGGER.debug(
            "Ollama request -> %s %s (payload_keys=%s)",
            method,
            url,
            tuple(sorted(json.keys())) if isinstance(json, dict) else (),
        )
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("Ollama request payload: %s", _format_debug_json(json))
        try:
            response = self._session.request(
                method=method,
                url=url,
                json=json,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise OllamaClientError(f"Ollama request failed: {exc}") from exc

        LOGGER.debug(
            "Ollama response <- %s %s status=%s",
            method,
            url,
            response.status_code,
        )
        if LOGGER.isEnabledFor(logging.DEBUG):
            body_preview = response.text
            if len(body_preview) > _JSON_DEBUG_MAX_LENGTH:
                body_preview = f"{body_preview[:_JSON_DEBUG_MAX_LENGTH]}…(truncated)"
            LOGGER.debug("Ollama raw response body: %s", body_preview)
        if response.status_code >= 400:
            text = response.text
            error_detail = ""
            try:
                payload = response.json()
            except ValueError:
                payload = None
            else:
                if isinstance(payload, dict):
                    error_detail = str(payload.get("error") or "")

            normalized_error = (error_detail or text).lower()
            msg = (
                f"Ollama responded with {response.status_code} for {method} {path}: {text}"
            )
            if "requires more system memory" in normalized_error:
                raise OllamaOutOfMemoryError(msg)
            raise OllamaClientError(msg)

        return response

    def generate(
        self,
        *,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: str | int | None = None,
        format: Optional[str] = None,
    ) -> OllamaGenerateResult:
        """Generate a completion from Ollama.

        Args:
            prompt: The prompt to send to the model
            model: Model name to use
            system: Optional system prompt
            options: Optional generation parameters (temperature, num_predict, etc.)
            keep_alive: Keep-alive duration for model in memory
            format: Response format - set to "json" for JSON mode (Ollama 0.1.34+)

        Returns:
            OllamaGenerateResult with model response

        Note:
            The format parameter enables structured JSON output (requires compatible model).
        """
        payload: Dict[str, Any] = {
            "model": model or "",
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
        if format:
            payload["format"] = format
        if keep_alive is not None:
            normalized_keep_alive: Any
            if isinstance(keep_alive, str):
                normalized_candidate = keep_alive.strip()
                try:
                    normalized_keep_alive = int(normalized_candidate)
                except ValueError:
                    normalized_keep_alive = normalized_candidate
            else:
                normalized_keep_alive = keep_alive
            payload["keep_alive"] = normalized_keep_alive
        LOGGER.debug(
            "Calling Ollama generate model=%s prompt_chars=%d options_keys=%s keep_alive=%s format=%s",
            payload.get("model") or "",
            len(prompt),
            tuple(sorted((options or {}).keys())),
            payload.get("keep_alive"),
            format,
        )

        response = self._request("POST", "/api/generate", json=payload)
        data = response.json()
        LOGGER.debug(
            "Ollama generate returned model=%s response_chars=%d",
            data.get("model") or payload["model"],
            len(str(data.get("response", "") or "")),
        )
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("Ollama generate JSON response: %s", _format_debug_json(data))
        return OllamaGenerateResult(
            model=data.get("model") or payload["model"],
            response=str(data.get("response", "")).strip(),
            raw=data,
        )

    def chat(
        self,
        *,
        messages: Sequence[Dict[str, Any]],
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> OllamaChatResult:
        payload: Dict[str, Any] = {
            "model": model or "",
            "messages": list(messages),
            "stream": False,
        }
        if options:
            payload["options"] = options

        LOGGER.debug(
            "Calling Ollama chat model=%s message_count=%d options_keys=%s",
            payload.get("model") or "",
            len(payload["messages"]),
            tuple(sorted((options or {}).keys())),
        )
        response = self._request("POST", "/api/chat", json=payload)
        data = response.json()
        message = data.get("message") or {}
        if not isinstance(message, dict):
            message = {"role": "assistant", "content": str(message)}
        LOGGER.debug(
            "Ollama chat returned model=%s message_chars=%d",
            data.get("model") or payload["model"],
            len(str(message.get("content") or "")),
        )
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("Ollama chat JSON response: %s", _format_debug_json(data))
        return OllamaChatResult(
            model=data.get("model") or payload["model"],
            message=message,
            raw=data,
        )

    def embed(
        self,
        *,
        inputs: Sequence[str],
        model: Optional[str] = None,
    ) -> OllamaEmbedResult:
        payload = {
            "model": model or "",
            "input": list(inputs),
        }
        LOGGER.debug(
            "Calling Ollama embed model=%s input_count=%d",
            payload.get("model"),
            len(payload["input"]),
        )
        response = self._request("POST", "/api/embed", json=payload)
        data = response.json()
        embeddings = data.get("embeddings") or []
        if not isinstance(embeddings, list):
            raise OllamaClientError("Invalid embeddings payload returned by Ollama.")
        LOGGER.debug(
            "Ollama embed returned model=%s embedding_count=%d",
            data.get("model") or payload["model"],
            len(embeddings) if isinstance(embeddings, list) else 0,
        )
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("Ollama embed JSON response: %s", _format_debug_json(data))
        return OllamaEmbedResult(
            model=data.get("model") or payload["model"],
            embeddings=embeddings,
            raw=data,
        )

    def list_models(self) -> List[Dict[str, Any]]:
        response = self._request("GET", "/api/tags")
        data = response.json()
        results = data.get("models") or data.get("data") or data
        if isinstance(results, dict):
            results = results.get("models") or []
        if not isinstance(results, list):
            raise OllamaClientError("Unexpected payload when listing models.")
        return [model for model in results if isinstance(model, dict)]


@lru_cache(maxsize=1)
def get_ollama_client() -> OllamaClient:
    """Return a cached Ollama client instance."""
    return OllamaClient()
