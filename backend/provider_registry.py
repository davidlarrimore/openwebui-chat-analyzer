"""Provider registry for managing multiple LLM providers.

This module provides a centralized registry for discovering, validating,
and using models from multiple LLM providers (Ollama, OpenAI, Open WebUI).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config
from .models_registry import get_models_registry
from .providers import (
    LLMProvider,
    ModelInfo,
    OllamaProvider,
    OpenAIProvider,
    OpenWebUIProvider,
)

LOGGER = logging.getLogger(__name__)
LEGACY_MODELS_REGISTRY_PATH = Path(__file__).with_name("data").joinpath("models_registry.json")


class ProviderRegistry:
    """Centralized registry for managing multiple LLM providers.

    This class provides a unified interface for:
    - Discovering which providers are available and configured
    - Listing models from each provider
    - Validating model capabilities
    - Retrieving the appropriate provider for generation

    Providers are initialized lazily on first access.
    """

    def __init__(self):
        """Initialize the provider registry.

        Providers are registered but not initialized until first use.
        """
        self._providers: Dict[str, LLMProvider] = {}
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of providers.

        Only creates provider instances when first accessed.
        Providers that are not configured (missing API keys, etc.)
        are still registered but will report as unavailable.
        """
        if self._initialized:
            return

        # Always register Ollama (local, no auth required)
        self._providers["ollama"] = OllamaProvider()
        LOGGER.info("Registered Ollama provider")

        # Register OpenAI if API key is configured
        if config.OPENAI_API_KEY:
            self._providers["openai"] = OpenAIProvider()
            LOGGER.info("Registered OpenAI provider")
        else:
            LOGGER.debug("OpenAI provider not registered (API key not configured)")

        # Register OpenWebUI if base URL is configured
        openwebui_base = config.get_openwebui_api_base()
        openwebui_key = config.get_openwebui_api_key()
        if openwebui_base:
            self._providers["openwebui"] = OpenWebUIProvider(
                base_url=openwebui_base,
                api_key=openwebui_key,
            )
            LOGGER.info("Registered OpenWebUI provider")
        else:
            LOGGER.debug(
                "OpenWebUI provider not registered (base URL not configured)"
            )

        self._initialized = True

    def refresh_configuration(self) -> None:
        """Rebuild provider instances based on current configuration."""
        self._providers.clear()
        self._initialized = False
        self._ensure_initialized()

    def get_available_connections(self) -> List[Dict[str, Any]]:
        """Get status of all available provider connections.

        Returns:
            List of connection status objects with keys:
            - type: Provider identifier ("ollama" | "openai" | "openwebui")
            - available: Boolean indicating if provider is usable
            - reason: Optional string explaining why unavailable

        Example:
            [
                {"type": "ollama", "available": True, "reason": None},
                {"type": "openai", "available": False, "reason": "API key not configured"},
                {"type": "openwebui", "available": False, "reason": "Base URL not configured"}
            ]
        """
        self._ensure_initialized()

        connections = []

        # Always include Ollama (even if not responding)
        ollama = self._providers.get("ollama")
        if ollama:
            connections.append({
                "type": "ollama",
                "available": ollama.is_available(),
                "reason": ollama.get_unavailable_reason(),
            })
        else:
            connections.append({
                "type": "ollama",
                "available": False,
                "reason": "Provider not initialized",
            })

        # OpenAI (only if configured)
        openai = self._providers.get("openai")
        if openai:
            connections.append({
                "type": "openai",
                "available": openai.is_available(),
                "reason": openai.get_unavailable_reason(),
            })
        else:
            connections.append({
                "type": "openai",
                "available": False,
                "reason": "API key not configured (set OPENAI_API_KEY)",
            })

        # OpenWebUI (only if configured)
        openwebui = self._providers.get("openwebui")
        if openwebui:
            connections.append({
                "type": "openwebui",
                "available": openwebui.is_available(),
                "reason": openwebui.get_unavailable_reason(),
            })
        else:
            connections.append({
                "type": "openwebui",
                "available": False,
                "reason": "Base URL not configured (set OWUI_DIRECT_HOST)",
            })

        return connections

    def get_models_for_connection(
        self,
        connection_type: str,
        include_unvalidated: bool = True,
        *,
        auto_validate_missing: bool = False,
        force_auto_validate: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get models available from a specific provider.

        Args:
            connection_type: Provider identifier ("ollama" | "openai" | "openwebui")
            include_unvalidated: If True, include models not yet validated for completions
            auto_validate_missing: When True, automatically validates models if no validated
                entries exist for the provider.
            force_auto_validate: When True, re-run validation even if validated models exist.

        Returns:
            List of model objects with keys:
            - name: Model identifier
            - display_name: Human-readable name
            - provider: Provider identifier
            - validated: Boolean indicating if model is confirmed for completions
            - metadata: Provider-specific metadata (optional)

        Raises:
            ValueError: If connection_type is unknown or unavailable
        """
        self._ensure_initialized()

        provider = self._providers.get(connection_type)
        if not provider:
            raise ValueError(f"Unknown connection type: {connection_type}")

        if not provider.is_available():
            reason = provider.get_unavailable_reason()
            raise ValueError(
                f"Provider '{connection_type}' is not available: {reason}"
            )

        # Discover models from provider
        models_info = provider.list_models()
        total_discovered = len(models_info)

        # Load validated models registry
        registry = get_models_registry()

        # Enrich with validation status
        result = []
        for model_info in models_info:
            is_validated = registry.is_validated(model_info.name)

            # Skip unvalidated if requested
            if not include_unvalidated and not is_validated:
                continue

            result.append({
                "name": model_info.name,
                "display_name": model_info.display_name,
                "provider": model_info.provider,
                "validated": is_validated,
                "metadata": model_info.metadata,
            })

        validated_count = sum(1 for m in result if m["validated"])
        if (
            auto_validate_missing
            and not include_unvalidated
            and total_discovered > 0
            and (force_auto_validate or validated_count == 0)
        ):
            LOGGER.warning(
                "No validated models found for %s; auto validating %d candidates",
                connection_type,
                total_discovered,
            )
            for model_info in models_info:
                if registry.is_validated(model_info.name):
                    continue
                try:
                    self.validate_model(connection_type, model_info.name)
                except Exception:  # pylint: disable=broad-except
                    LOGGER.exception(
                        "Auto validation failed for %s on provider %s",
                        model_info.name,
                        connection_type,
                    )
            # Rebuild result list with updated registry state
            result = []
            for model_info in models_info:
                is_validated = registry.is_validated(model_info.name)
                if not include_unvalidated and not is_validated:
                    continue
                result.append({
                    "name": model_info.name,
                    "display_name": model_info.display_name,
                    "provider": model_info.provider,
                    "validated": is_validated,
                    "metadata": model_info.metadata,
                })
            validated_count = sum(1 for m in result if m["validated"])

        LOGGER.info(
            "Retrieved %d models from %s (%d validated, include_unvalidated=%s, discovered=%d, auto_validate=%s)",
            len(result),
            connection_type,
            validated_count,
            include_unvalidated,
            total_discovered,
            auto_validate_missing,
        )
        if not result and total_discovered > 0 and not include_unvalidated:
            LOGGER.warning(
                "Provider %s returned %d models but none are validated. Run /admin/summarizer/validate-model first.",
                connection_type,
                total_discovered,
            )

        return result

    def validate_model(self, connection_type: str, model_name: str) -> bool:
        """Validate if a model supports text completion.

        Args:
            connection_type: Provider identifier
            model_name: Model name to validate

        Returns:
            True if model successfully generated a test completion, False otherwise.

        Side Effects:
            If validation succeeds, adds model to the global registry.

        Raises:
            ValueError: If connection_type is unknown or unavailable
        """
        self._ensure_initialized()

        provider = self._providers.get(connection_type)
        if not provider:
            raise ValueError(f"Unknown connection type: {connection_type}")

        if not provider.is_available():
            reason = provider.get_unavailable_reason()
            raise ValueError(
                f"Provider '{connection_type}' is not available: {reason}"
            )

        LOGGER.info(
            "Validating model '%s' on provider '%s'", model_name, connection_type
        )

        # Run validation test
        is_valid = provider.validate_completion(model_name)

        # If valid, add to registry and persist
        if is_valid:
            registry = get_models_registry()
            registry.add_model(model_name)
            registry.save()
            _persist_legacy_validation(model_name, True)
            LOGGER.info("Model '%s' validated and added to registry", model_name)
        else:
            _persist_legacy_validation(model_name, False)
            LOGGER.warning(
                "Model '%s' failed validation on provider '%s'. Inspect provider logs for HTTP response details.",
                model_name,
                connection_type,
            )

        return is_valid

    def get_provider(self, connection_type: str) -> LLMProvider:
        """Get a provider instance by type.

        Args:
            connection_type: Provider identifier

        Returns:
            LLMProvider instance

        Raises:
            ValueError: If connection_type is unknown
        """
        self._ensure_initialized()

        provider = self._providers.get(connection_type)
        if not provider:
            raise ValueError(f"Unknown connection type: {connection_type}")

        return provider

    def get_validated_models(self) -> List[str]:
        """Get all models validated for completions across all providers.

        Returns:
            Sorted list of model names from the global registry.
        """
        registry = get_models_registry()
        return registry.get_all_models()


# Singleton instance
_registry: Optional[ProviderRegistry] = None


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry instance.

    Returns:
        Singleton ProviderRegistry instance.
    """
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry
def _persist_legacy_validation(model_name: str, supports_completions: bool) -> None:
    """Store validation metadata in the legacy models_registry.json file."""
    try:
        if LEGACY_MODELS_REGISTRY_PATH.exists():
            with open(LEGACY_MODELS_REGISTRY_PATH, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        else:
            payload = {}
        models_section = payload.get("models", {})
        if not isinstance(models_section, dict):
            models_section = {}
        models_section[model_name] = {
            "supports_completions": supports_completions,
            "tested_at": datetime.now(timezone.utc).isoformat(),
        }
        payload = {"models": models_section}
        LEGACY_MODELS_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LEGACY_MODELS_REGISTRY_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception:  # pylint: disable=broad-except
        LOGGER.warning("Failed to update legacy models registry for %s", model_name, exc_info=True)
