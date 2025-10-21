"""Client wrappers used by the backend."""

from .ollama import (
    OllamaClient,
    OllamaClientError,
    OllamaOutOfMemoryError,
    OllamaGenerateResult,
    OllamaChatResult,
    OllamaEmbedResult,
    get_ollama_client,
)

__all__ = [
    "OllamaClient",
    "OllamaClientError",
    "OllamaOutOfMemoryError",
    "OllamaGenerateResult",
    "OllamaChatResult",
    "OllamaEmbedResult",
    "get_ollama_client",
]
