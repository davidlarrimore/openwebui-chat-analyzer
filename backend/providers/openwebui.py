"""Open WebUI provider implementation.

This module provides integration with Open WebUI's external models API
for discovering and using models from Open WebUI instances.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from .. import config
from .base import GenerateResult, LLMProvider, ModelInfo

LOGGER = logging.getLogger(__name__)

_MAX_LOG_BODY_CHARS = 512


def _truncate(text: str, limit: int = _MAX_LOG_BODY_CHARS) -> str:
    """Return a safely truncated representation of a response body."""
    if text is None:
        return "<no-body>"
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[:limit]}…(truncated)"


class OpenWebUIProvider(LLMProvider):
    """Open WebUI external models provider implementation.

    This provider discovers external models from an Open WebUI instance
    and delegates generation to those external providers (via Open WebUI).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize Open WebUI provider.

        Args:
            base_url: Open WebUI base URL (defaults to config.get_openwebui_api_base())
            api_key: API key for authentication (defaults to config.get_openwebui_api_key())
            timeout: Request timeout in seconds
        """
        resolved_base = base_url if base_url is not None else config.get_openwebui_api_base()
        self.base_url = resolved_base.rstrip("/") if resolved_base else ""
        self.api_key = api_key if api_key is not None else config.get_openwebui_api_key()
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if Open WebUI provider is configured and reachable.

        Returns:
            True if base URL is set and endpoint responds, False otherwise.
        """
        if not self.base_url:
            LOGGER.debug("OpenWebUI provider unavailable: Base URL not configured")
            return False

        try:
            # Test connection with GET /api/models
            response = requests.get(
                f"{self.base_url}/api/models",
                headers=self._get_headers(),
                timeout=10,  # Quick check
            )
            if response.status_code == 401:
                LOGGER.warning("OpenWebUI provider unavailable: Authentication failed")
                return False
            if response.status_code == 200:
                LOGGER.debug("OpenWebUI provider available")
                return True
            LOGGER.warning(
                "OpenWebUI provider check returned status %d", response.status_code
            )
            return False
        except requests.exceptions.RequestException as exc:
            LOGGER.warning("OpenWebUI provider unavailable: %s", exc)
            return False

    def get_unavailable_reason(self) -> Optional[str]:
        """Get human-readable reason why provider is unavailable.

        Returns:
            Error message if unavailable, None if available.
        """
        if not self.base_url:
            return "Base URL not configured (set OWUI_DIRECT_HOST)"

        try:
            response = requests.get(
                f"{self.base_url}/api/models",
                headers=self._get_headers(),
                timeout=10,
            )
            if response.status_code == 401:
                return "Authentication failed (check OWUI_DIRECT_API_KEY)"
            if response.status_code != 200:
                return f"Service returned status {response.status_code}"
            return None
        except requests.exceptions.Timeout:
            return "Service timeout (unreachable)"
        except requests.exceptions.ConnectionError:
            return "Connection failed (check OWUI_DIRECT_HOST)"
        except requests.exceptions.RequestException as exc:
            return f"Service error: {exc}"

    def list_models(self) -> List[ModelInfo]:
        """Discover external models from Open WebUI.

        Returns:
            List of ModelInfo objects for external models only.

        Raises:
            requests.exceptions.RequestException: If API call fails.

        Note:
            Only includes models where connection_type == "external".
            This filters out Open WebUI's built-in models to avoid conflicts
            with direct provider connections.
        """
        LOGGER.info("Discovering models from Open WebUI at %s", self.base_url)

        response = requests.get(
            f"{self.base_url}/api/models",
            headers=self._get_headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        models_data = data.get("data", []) if isinstance(data, dict) else data

        models = []
        for model_obj in models_data:
            # Only include external models (not local Ollama models)
            if model_obj.get("connection_type") != "external":
                continue

            model_id = model_obj.get("id") or model_obj.get("name")
            if not model_id:
                continue

            # Human friendly display name falls back to identifier if not provided
            display_name = model_obj.get("name") or model_id

            models.append(
                ModelInfo(
                    name=model_id,
                    display_name=display_name,
                    provider="openwebui",
                    supports_completions=None,  # Not validated yet
                    metadata={
                        "id": model_id,
                        "base_model_id": model_obj.get("base_model_id"),
                        "connection_type": model_obj.get("connection_type"),
                        "display_name": display_name,
                    },
                )
            )

        LOGGER.info("Discovered %d external models from Open WebUI", len(models))
        return models

    def validate_completion(self, model: str) -> bool:
        """Test if a model supports chat completion via Open WebUI.

        Args:
            model: Model name to test

        Returns:
            True if model can generate completions, False otherwise.

        Note:
            Sends a minimal test prompt through Open WebUI's chat API.
            Uses stream=false for simpler response parsing.
        """
        LOGGER.info("Validating completion support for Open WebUI model: %s", model)

        endpoint = f"{self.base_url}/api/chat/completions"
        LOGGER.info(
            "Validating OpenWebUI completion support (model=%s url=%s)",
            model,
            endpoint,
        )
        try:
            response = requests.post(
                endpoint,
                headers=self._get_headers(),
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False,
                    "temperature": 0,
                    "max_tokens": 5,
                },
                timeout=self.timeout,
            )
            LOGGER.debug(
                "OpenWebUI validation response status=%s body=%s",
                response.status_code,
                _truncate(response.text),
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content")
                if content:
                    LOGGER.info("Model %s supports completions", model)
                    return True

            LOGGER.warning(
                "Model %s validation failed with status %d: %s",
                model,
                response.status_code,
                _truncate(response.text),
            )
            return False

        except requests.exceptions.RequestException as exc:
            LOGGER.error(
                "Model %s validation request to %s failed: %s",
                model,
                endpoint,
                exc,
            )
            return False

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GenerateResult:
        """Generate a chat completion via Open WebUI.

        Args:
            model: Model name to use
            prompt: User message content
            system: Optional system message
            options: Optional parameters (temperature, max_tokens, etc.)

        Returns:
            GenerateResult with generated content and metadata.

        Raises:
            requests.exceptions.RequestException: If API call fails.

        Note:
            Uses Open WebUI's chat completions endpoint which proxies to
            the external model provider.
        """
        LOGGER.debug("Generating completion with Open WebUI model: %s", model)

        options = options or {}

        # Build messages array
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Build request body
        request_body = {
            "model": model,
            "messages": messages,
            "stream": False,  # Non-streaming for simpler handling
        }

        # Add temperature if provided
        if "temperature" in options:
            request_body["temperature"] = options["temperature"]

        # Map num_predict to max_tokens
        if "num_predict" in options:
            request_body["max_tokens"] = options["num_predict"]
        elif "max_tokens" in options:
            request_body["max_tokens"] = options["max_tokens"]

        # Pass through other options
        for key, value in options.items():
            if key not in ("temperature", "num_predict", "max_tokens"):
                request_body[key] = value

        response = requests.post(
            f"{self.base_url}/api/chat/completions",
            headers=self._get_headers(),
            json=request_body,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()

        # Extract generated content
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Extract metadata
        metadata = {
            "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
            "usage": data.get("usage", {}),
            "model_used": data.get("model"),
        }

        return GenerateResult(
            content=content,
            model=model,
            provider="openwebui",
            metadata=metadata,
        )

    def get_provider_name(self) -> str:
        """Return provider identifier.

        Returns:
            "openwebui"
        """
        return "openwebui"

    def _get_headers(self) -> Dict[str, str]:
        """Build HTTP headers for Open WebUI requests.

        Returns:
            Dictionary with Authorization and Content-Type headers.
        """
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
