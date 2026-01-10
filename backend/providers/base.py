"""Base provider interface for LLM providers.

This module defines the abstract base class and data structures for implementing
LLM provider integrations (Ollama, OpenAI, Open WebUI, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelInfo:
    """Provider-agnostic model information.

    Attributes:
        name: Unique identifier for the model (e.g., "gpt-4o-mini", "llama3:8b")
        display_name: Human-readable name for UI display
        provider: Provider identifier ("ollama" | "openai" | "openwebui")
        supports_completions: Whether model has been validated for completions (None = unknown)
        metadata: Additional provider-specific data (size, parameters, etc.)
    """
    name: str
    display_name: str
    provider: str
    supports_completions: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerateResult:
    """Result from a completion/generation request.

    Attributes:
        content: Generated text content
        model: Model name that was used
        provider: Provider that generated the response
        metadata: Additional response metadata (tokens, timing, etc.)
    """
    content: str
    model: str
    provider: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations (Ollama, OpenAI, Open WebUI) must implement
    this interface to ensure consistent behavior across the application.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and reachable.

        Returns:
            True if provider can be used, False otherwise.

        Note:
            This should perform a lightweight check (e.g., verify API key exists,
            test network connectivity). It should NOT enumerate all models.
        """
        pass

    @abstractmethod
    def get_unavailable_reason(self) -> Optional[str]:
        """Get human-readable reason why provider is unavailable.

        Returns:
            None if available, otherwise a message like:
            - "API key not configured"
            - "Service unreachable"
            - "Authentication failed"
        """
        pass

    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """Discover available models from provider.

        Returns:
            List of models available from this provider.

        Raises:
            Exception: If provider is unavailable or API call fails.

        Note:
            The supports_completions field may be None for models that
            haven't been validated yet. Call validate_completion() to test.
        """
        pass

    @abstractmethod
    def validate_completion(self, model: str) -> bool:
        """Test if a model supports completion/chat functionality.

        Args:
            model: Model name/identifier to test

        Returns:
            True if model successfully generated a test completion, False otherwise.

        Note:
            This should send a minimal test prompt (e.g., "Hello") and verify
            a valid response is returned. The test should be fast (<5 seconds).
        """
        pass

    @abstractmethod
    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GenerateResult:
        """Generate a completion from the model.

        Args:
            model: Model name/identifier to use
            prompt: User prompt/message
            system: Optional system prompt
            options: Provider-specific options (temperature, max_tokens, etc.)

        Returns:
            GenerateResult with the generated content and metadata.

        Raises:
            Exception: If generation fails (model not found, API error, etc.)
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider identifier.

        Returns:
            One of: "ollama", "openai", "openwebui", "litellm"
        """
        pass

    @abstractmethod
    def supports_json_mode(self) -> bool:
        """Check if provider supports native JSON-structured output.

        Returns:
            True if provider supports response_format or similar JSON mode,
            False if provider relies on prompt engineering only.

        Note:
            - OpenAI: Returns True (supports response_format={"type": "json_object"})
            - LiteLLM: Returns True (supports response_format passthrough)
            - Ollama: Returns True if version supports format="json" parameter
            - OpenWebUI: Returns False (no native structured output support)
        """
        pass
