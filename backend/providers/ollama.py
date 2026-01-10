"""Ollama provider implementation.

This module provides a unified provider interface for the existing Ollama client,
allowing it to be used interchangeably with other LLM providers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .. import config
from ..clients.ollama import (
    OllamaClient,
    OllamaClientError,
    get_ollama_client,
)
from .base import GenerateResult, LLMProvider, ModelInfo

LOGGER = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama provider implementation.

    Wraps the existing OllamaClient to conform to the unified provider interface.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        client: Optional[OllamaClient] = None,
    ):
        """Initialize Ollama provider.

        Args:
            base_url: Ollama base URL (defaults to config.OLLAMA_BASE_URL)
            timeout: Request timeout in seconds (defaults to config.OLLAMA_TIMEOUT_SECONDS)
            client: Optional OllamaClient instance (for testing/DI)
        """
        if client:
            self._client = client
        elif base_url or timeout:
            # Custom configuration
            self._client = OllamaClient(
                base_url=base_url or config.OLLAMA_BASE_URL,
                timeout=timeout or config.OLLAMA_TIMEOUT_SECONDS,
            )
        else:
            # Use singleton from factory
            self._client = get_ollama_client()

    def is_available(self) -> bool:
        """Check if Ollama provider is reachable.

        Returns:
            True if Ollama responds to requests, False otherwise.
        """
        try:
            # Test with lightweight request
            self._client.list_models()
            LOGGER.debug("Ollama provider available")
            return True
        except OllamaClientError as exc:
            LOGGER.warning("Ollama provider unavailable: %s", exc)
            return False

    def get_unavailable_reason(self) -> Optional[str]:
        """Get human-readable reason why provider is unavailable.

        Returns:
            Error message if unavailable, None if available.
        """
        try:
            self._client.list_models()
            return None
        except OllamaClientError as exc:
            error_str = str(exc).lower()
            if "connection" in error_str or "refused" in error_str:
                return f"Connection failed (check OLLAMA_BASE_URL: {self._client.base_url})"
            if "timeout" in error_str:
                return "Service timeout (Ollama not responding)"
            return f"Service error: {exc}"

    def list_models(self) -> List[ModelInfo]:
        """Discover available models from Ollama.

        Returns:
            List of ModelInfo objects for all Ollama models.

        Raises:
            OllamaClientError: If API call fails.

        Note:
            The supports_completions field is None for all models.
            Use validate_completion() to test support.
        """
        LOGGER.info("Discovering models from Ollama at %s", self._client.base_url)

        models_data = self._client.list_models()

        models = []
        for model_obj in models_data:
            model_name = model_obj.get("name")
            if not model_name:
                continue

            models.append(
                ModelInfo(
                    name=model_name,
                    display_name=model_name,
                    provider="ollama",
                    supports_completions=None,  # Not validated yet
                    metadata={
                        "model": model_obj.get("model"),
                        "size": model_obj.get("size"),
                        "modified_at": model_obj.get("modified_at"),
                        "digest": model_obj.get("digest"),
                        "details": model_obj.get("details", {}),
                    },
                )
            )

        LOGGER.info("Discovered %d Ollama models", len(models))
        return models

    def validate_completion(self, model: str) -> bool:
        """Test if a model supports completion generation.

        Args:
            model: Model name to test

        Returns:
            True if model successfully generates a completion, False otherwise.

        Note:
            Sends a minimal test prompt: "Hello"
            Uses num_predict=5 for fast execution.
        """
        LOGGER.info("Validating completion support for Ollama model: %s", model)

        try:
            result = self._client.generate(
                model=model,
                prompt="Hello",
                options={"num_predict": 5},
            )

            if result.response:
                LOGGER.info("Model %s supports completions", model)
                return True

            LOGGER.warning("Model %s returned empty response", model)
            return False

        except OllamaClientError as exc:
            LOGGER.error("Model %s validation error: %s", model, exc)
            return False

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GenerateResult:
        """Generate a completion from Ollama.

        Args:
            model: Model name to use
            prompt: User prompt
            system: Optional system prompt
            options: Optional generation parameters (temperature, num_predict, json_mode, etc.)

        Returns:
            GenerateResult with generated content and metadata.

        Raises:
            OllamaClientError: If generation fails.

        Note:
            - Passes options directly to Ollama (temperature, num_predict, num_ctx, etc.)
            - Supports json_mode option - sets format="json" for structured output
        """
        LOGGER.debug("Generating completion with Ollama model: %s", model)

        options = options or {}

        # Extract json_mode from options (not passed to Ollama options)
        format_param = "json" if options.get("json_mode") else None
        if format_param:
            LOGGER.debug("Enabled JSON mode for Ollama model %s", model)

        # Filter out json_mode from options (it's not an Ollama option)
        ollama_options = {k: v for k, v in options.items() if k != "json_mode"}

        result = self._client.generate(
            model=model,
            prompt=prompt,
            system=system,
            options=ollama_options if ollama_options else None,
            keep_alive=config.OLLAMA_KEEP_ALIVE,
            format=format_param,
        )

        return GenerateResult(
            content=result.response,
            model=result.model,
            provider="ollama",
            metadata=result.raw,
        )

    def get_provider_name(self) -> str:
        """Return provider identifier.

        Returns:
            "ollama"
        """
        return "ollama"

    def supports_json_mode(self) -> bool:
        """Check if provider supports native JSON-structured output.

        Returns:
            True - Ollama supports format="json" parameter (v0.1.34+).

        Note:
            JSON mode is supported by most modern Ollama models when using
            the format="json" parameter. The model must support structured output.
        """
        return True
