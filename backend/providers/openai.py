"""OpenAI provider implementation.

This module provides integration with OpenAI's API for model discovery,
validation, and text generation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from .. import config
from .base import GenerateResult, LLMProvider, ModelInfo

LOGGER = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation.

    Supports OpenAI-compatible APIs by allowing custom base URLs.
    Authentication is via Bearer token (API key).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to config.OPENAI_API_KEY)
            base_url: API base URL (defaults to config.OPENAI_API_BASE)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.base_url = (base_url or config.OPENAI_API_BASE).rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if OpenAI provider is configured and reachable.

        Returns:
            True if API key is set and endpoint responds, False otherwise.
        """
        if not self.api_key:
            LOGGER.debug("OpenAI provider unavailable: API key not configured")
            return False

        try:
            # Test connection with GET /models (lightweight check)
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._get_headers(),
                timeout=10,  # Quick check
            )
            if response.status_code == 401:
                LOGGER.warning("OpenAI provider unavailable: Authentication failed")
                return False
            if response.status_code == 200:
                LOGGER.debug("OpenAI provider available")
                return True
            LOGGER.warning(
                "OpenAI provider check returned status %d", response.status_code
            )
            return False
        except requests.exceptions.RequestException as exc:
            LOGGER.warning("OpenAI provider unavailable: %s", exc)
            return False

    def get_unavailable_reason(self) -> Optional[str]:
        """Get human-readable reason why provider is unavailable.

        Returns:
            Error message if unavailable, None if available.
        """
        if not self.api_key:
            return "API key not configured (set OPENAI_API_KEY)"

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
            return "Connection failed (check OPENAI_API_BASE)"
        except requests.exceptions.RequestException as exc:
            return f"Service error: {exc}"

    def list_models(self) -> List[ModelInfo]:
        """Discover available models from OpenAI.

        Returns:
            List of ModelInfo objects for all available models.

        Raises:
            requests.exceptions.RequestException: If API call fails.

        Note:
            Filters to only include models with object type "model".
            Some deployments may return other objects (e.g., "deployment").
        """
        LOGGER.info("Discovering models from OpenAI at %s", self.base_url)

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

            if isinstance(model_obj, dict):
                object_type = model_obj.get("object")
                if object_type in ("list", "model_list"):
                    continue

            # Filter to chat/completion models (exclude embedding, audio, etc.)
            # Most OpenAI models support chat, but we can check ownership for hints
            if self._is_likely_chat_model(model_id, model_obj if isinstance(model_obj, dict) else {}):
                models.append(
                    ModelInfo(
                        name=model_id,
                        display_name=model_id,
                        provider="openai",
                        supports_completions=None,  # Not validated yet
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

        LOGGER.info("Discovered %d OpenAI models", len(models))
        return models

    def validate_completion(self, model: str) -> bool:
        """Test if a model supports chat completion.

        Args:
            model: Model ID to test

        Returns:
            True if model can generate completions, False otherwise.

        Note:
            Sends a minimal test prompt to verify the model works.
            Uses temperature=0 and max_tokens=5 for fast, deterministic test.
        """
        LOGGER.info("Validating completion support for OpenAI model: %s", model)

        validation_timeout = min(self.timeout, 10.0)
        payloads = [
            {
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0,
                "max_tokens": 5,
            },
            {
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        ]

        for attempt, payload in enumerate(payloads, start=1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=validation_timeout,
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content")
                    if content:
                        LOGGER.info("Model %s supports completions", model)
                        return True
                    if data.get("choices"):
                        LOGGER.info(
                            "Model %s returned empty content during validation; treating as valid.",
                            model,
                        )
                        return True

                if response.status_code == 400 and attempt < len(payloads):
                    LOGGER.info(
                        "Model %s validation attempt %d failed with 400; retrying with minimal payload.",
                        model,
                        attempt,
                    )
                    continue

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

        return False

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GenerateResult:
        """Generate a chat completion from OpenAI.

        Args:
            model: Model ID to use
            prompt: User message content
            system: Optional system message
            options: Optional parameters (temperature, max_tokens, etc.)

        Returns:
            GenerateResult with generated content and metadata.

        Raises:
            requests.exceptions.RequestException: If API call fails.

        Note:
            Maps common options to OpenAI parameter names:
            - temperature: temperature (0.0-2.0)
            - num_predict: max_tokens
            - json_mode: If True, sets response_format={"type": "json_object"}
            - Additional OpenAI-specific options can be passed directly
        """
        LOGGER.debug("Generating completion with OpenAI model: %s", model)

        options = options or {}

        # Build messages array
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Map common options to OpenAI parameters
        request_body = {
            "model": model,
            "messages": messages,
        }

        # Map temperature (already compatible)
        if "temperature" in options:
            request_body["temperature"] = options["temperature"]

        # Map num_predict to max_tokens
        if "num_predict" in options:
            request_body["max_tokens"] = options["num_predict"]
        elif "max_tokens" in options:
            request_body["max_tokens"] = options["max_tokens"]

        # Enable JSON mode if requested
        if options.get("json_mode"):
            request_body["response_format"] = {"type": "json_object"}
            LOGGER.debug("Enabled JSON mode for model %s", model)

        # Pass through any other OpenAI-specific options
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

        # Extract generated content (handle string, list-of-parts, or legacy text field)
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
                text_candidate = primary.get("text")
                if isinstance(text_candidate, str):
                    content = text_candidate

            if not content:
                LOGGER.warning(
                    "OpenAI completion returned empty content; finish_reason=%s keys=%s",
                    primary.get("finish_reason"),
                    list(primary.keys()),
                )

        # Extract metadata
        metadata = {
            "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
            "usage": data.get("usage", {}),
            "model_used": data.get("model"),  # Actual model (may differ for aliases)
        }

        return GenerateResult(
            content=content,
            model=model,
            provider="openai",
            metadata=metadata,
        )

    def get_provider_name(self) -> str:
        """Return provider identifier.

        Returns:
            "openai"
        """
        return "openai"

    def supports_json_mode(self) -> bool:
        """Check if provider supports native JSON-structured output.

        Returns:
            True - OpenAI supports response_format={"type": "json_object"}.

        Note:
            JSON mode is supported by:
            - gpt-4o and later (all variants)
            - gpt-4-turbo-preview and later
            - gpt-3.5-turbo-1106 and later
            Older models may not support this feature but will gracefully ignore it.
        """
        return True

    def _get_headers(self) -> Dict[str, str]:
        """Build HTTP headers for OpenAI requests.

        Returns:
            Dictionary with Authorization and Content-Type headers.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _is_likely_chat_model(self, model_id: str, model_obj: Dict[str, Any]) -> bool:
        """Heuristic to filter for chat/completion models.

        Args:
            model_id: Model identifier
            model_obj: Full model object from API

        Returns:
            True if model is likely a chat/completion model.

        Note:
            Filters out:
            - Embedding models (text-embedding-*)
            - Audio models (whisper-*, tts-*)
            - Image models (dall-e-*)
            - Moderation models (text-moderation-*)
        """
        model_lower = model_id.lower()

        # Exclude known non-chat model prefixes
        excluded_prefixes = (
            "text-embedding-",
            "whisper-",
            "tts-",
            "dall-e-",
            "text-moderation-",
            "davinci-",  # Legacy completion models (not chat)
            "curie-",
            "babbage-",
            "ada-",
        )

        if any(model_lower.startswith(prefix) for prefix in excluded_prefixes):
            return False

        # Include GPT models (gpt-3.5, gpt-4, etc.)
        if "gpt" in model_lower:
            return True

        # Include o1 models
        if model_lower.startswith("o1"):
            return True

        # Default to including unknown models (can be validated later)
        return True

    @staticmethod
    def _extract_models_data(payload: Any) -> List[Any]:
        """Return the models list from a provider-specific payload."""
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("models"), list):
            return payload["models"]
        nested = payload.get("data")
        if isinstance(nested, dict) and isinstance(nested.get("data"), list):
            return nested["data"]
        return []

    @staticmethod
    def _extract_model_id(model_obj: Any) -> Optional[str]:
        """Return a model identifier from an entry."""
        if isinstance(model_obj, str):
            return model_obj
        if not isinstance(model_obj, dict):
            return None
        return (
            model_obj.get("id")
            or model_obj.get("model")
            or model_obj.get("name")
        )
