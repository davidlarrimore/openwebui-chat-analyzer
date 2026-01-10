"""LiteLLM provider implementation.

This module provides integration with LiteLLM's unified API for model access
across multiple providers (OpenAI, Anthropic, Azure, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from .base import GenerateResult, LLMProvider, ModelInfo

LOGGER = logging.getLogger(__name__)


class LiteLLMProvider(LLMProvider):
    """LiteLLM proxy provider implementation.

    Supports LiteLLM's /v1/chat/completions and /v1/models endpoints.
    Compatible with OpenAI API format.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize LiteLLM provider.

        Args:
            api_key: LiteLLM API key (defaults to config.LITELLM_API_KEY)
            base_url: API base URL (defaults to config.LITELLM_API_BASE)
            timeout: Request timeout in seconds
        """
        # Import here to avoid circular dependency
        from .. import config

        self.api_key = api_key or config.LITELLM_API_KEY
        self.base_url = (base_url or config.LITELLM_API_BASE).rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if LiteLLM provider is configured and reachable."""
        if not self.api_key or not self.base_url:
            LOGGER.debug("LiteLLM provider unavailable: API key or base URL not configured")
            return False

        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._get_headers(),
                timeout=10,
            )
            if response.status_code == 401:
                LOGGER.warning("LiteLLM provider unavailable: Authentication failed")
                return False
            if response.status_code == 200:
                LOGGER.debug("LiteLLM provider available")
                return True
            LOGGER.warning("LiteLLM provider check returned status %d", response.status_code)
            return False
        except requests.exceptions.RequestException as exc:
            LOGGER.warning("LiteLLM provider unavailable: %s", exc)
            return False

    def get_unavailable_reason(self) -> Optional[str]:
        """Get human-readable reason why provider is unavailable."""
        if not self.api_key:
            return "API key not configured (set LITELLM_API_KEY)"
        if not self.base_url:
            return "Base URL not configured (set LITELLM_API_BASE)"

        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._get_headers(),
                timeout=10,
            )
            if response.status_code == 401:
                return "Authentication failed (invalid API key)"
            if response.status_code != 200:
                return f"Service returned status {response.status_code}"
            return None
        except requests.exceptions.Timeout:
            return "Service timeout (unreachable)"
        except requests.exceptions.ConnectionError:
            return "Connection failed (check LITELLM_API_BASE)"
        except requests.exceptions.RequestException as exc:
            return f"Service error: {exc}"

    def list_models(self) -> List[ModelInfo]:
        """Discover available models from LiteLLM."""
        LOGGER.info("Discovering models from LiteLLM at %s", self.base_url)

        response = requests.get(
            f"{self.base_url}/models",
            headers=self._get_headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        models_data = self._extract_models_data(data)

        models = []
        for model_obj in models_data:
            model_id = self._extract_model_id(model_obj)
            if not model_id:
                continue

            models.append(
                ModelInfo(
                    name=model_id,
                    display_name=model_id,
                    provider="litellm",
                    supports_completions=None,
                    metadata=(
                        {
                            "created": model_obj.get("created"),
                            "owned_by": model_obj.get("owned_by"),
                        }
                        if isinstance(model_obj, dict)
                        else {}
                    ),
                )
            )

        LOGGER.info("Discovered %d LiteLLM models", len(models))
        return models

    def validate_completion(self, model: str) -> bool:
        """Test if a model supports chat completion."""
        LOGGER.info("Validating completion support for LiteLLM model: %s", model)

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
            "max_tokens": 5,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                timeout=min(self.timeout, 10.0),
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content")
                if content or data.get("choices"):
                    LOGGER.info("Model %s supports completions", model)
                    return True

            LOGGER.warning(
                "Model %s validation failed with status %d: %s",
                model,
                response.status_code,
                response.text[:200],
            )
            return False

        except requests.exceptions.RequestException as exc:
            LOGGER.error("Model %s validation error: %s", model, exc)
            return False

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GenerateResult:
        """Generate a chat completion from LiteLLM.

        Args:
            model: Model name/identifier to use
            prompt: User message content
            system: Optional system message
            options: Optional parameters (temperature, max_tokens, json_mode, etc.)

        Returns:
            GenerateResult with generated content and metadata.

        Note:
            Supports json_mode option - sets response_format={"type": "json_object"}.
        """
        LOGGER.debug("Generating completion with LiteLLM model: %s", model)

        options = options or {}

        # Build messages array
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Map common options to LiteLLM parameters
        request_body = {
            "model": model,
            "messages": messages,
        }

        if "temperature" in options:
            request_body["temperature"] = options["temperature"]

        if "num_predict" in options:
            request_body["max_tokens"] = options["num_predict"]
        elif "max_tokens" in options:
            request_body["max_tokens"] = options["max_tokens"]

        # Enable JSON mode if requested
        if options.get("json_mode"):
            request_body["response_format"] = {"type": "json_object"}
            LOGGER.debug("Enabled JSON mode for model %s", model)

        # Pass through other options
        for key, value in options.items():
            if key not in ("temperature", "num_predict", "max_tokens", "json_mode"):
                request_body[key] = value

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._get_headers(),
            json=request_body,
            timeout=self.timeout,
        )

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            detail = response.text.strip()
            raise requests.exceptions.HTTPError(f"{exc} Response: {detail}") from exc

        data = response.json()

        # Extract generated content
        content = ""
        choices = data.get("choices") or []
        if choices:
            primary = choices[0] or {}
            message = primary.get("message") or {}
            message_content = message.get("content")
            if isinstance(message_content, str):
                content = message_content
            elif isinstance(message_content, list):
                parts = []
                for part in message_content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
                content = "".join(parts).strip()

            if not content:
                LOGGER.warning(
                    "LiteLLM completion returned empty content; finish_reason=%s",
                    primary.get("finish_reason"),
                )

        # Extract metadata
        metadata = {
            "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
            "usage": data.get("usage", {}),
            "model_used": data.get("model"),
        }

        return GenerateResult(
            content=content,
            model=model,
            provider="litellm",
            metadata=metadata,
        )

    def get_provider_name(self) -> str:
        """Return provider identifier."""
        return "litellm"

    def supports_json_mode(self) -> bool:
        """Check if provider supports native JSON-structured output.

        Returns:
            True - LiteLLM supports response_format passthrough to underlying providers.

        Note:
            LiteLLM acts as a proxy and passes response_format to the underlying provider
            (OpenAI, Anthropic, Azure, etc.). Actual support depends on the backend model.
        """
        return True

    def _get_headers(self) -> Dict[str, str]:
        """Build HTTP headers for LiteLLM requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _extract_models_data(payload: Any) -> List[Any]:
        """Extract models list from provider response."""
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("models"), list):
            return payload["models"]
        return []

    @staticmethod
    def _extract_model_id(model_obj: Any) -> Optional[str]:
        """Extract model identifier from model object."""
        if isinstance(model_obj, str):
            return model_obj
        if not isinstance(model_obj, dict):
            return None
        return (
            model_obj.get("id")
            or model_obj.get("model")
            or model_obj.get("name")
        )
