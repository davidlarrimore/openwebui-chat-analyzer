"""LLM Provider abstraction layer.

This package provides a unified interface for interacting with multiple
LLM providers (Ollama, OpenAI, LiteLLM, Open WebUI) for summarization and other
generative AI tasks.
"""

from .base import (
    GenerateResult,
    LLMProvider,
    ModelInfo,
)
from .litellm import LiteLLMProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .openwebui import OpenWebUIProvider

__all__ = [
    "LLMProvider",
    "ModelInfo",
    "GenerateResult",
    "LiteLLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "OpenWebUIProvider",
]
