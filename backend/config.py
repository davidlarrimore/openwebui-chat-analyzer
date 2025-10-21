"""Configuration helpers for the Open WebUI Chat Analyzer backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load environment variables from a .env file if present.
load_dotenv()


def _split_origins(value: str) -> List[str]:
    """Convert a comma-separated origin string into a clean list.

    Args:
        value (str): One or many origins separated by commas.
    Returns:
        List[str]: Normalized origin values with whitespace removed.
    """
    return [origin.strip() for origin in value.split(",") if origin.strip()]


DEFAULT_ALLOWED_ORIGINS = "http://localhost:8501,http://127.0.0.1:8501"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

API_HOST = os.getenv("OWUI_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("OWUI_API_PORT", "8502"))
API_ALLOWED_ORIGINS = _split_origins(os.getenv("OWUI_API_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS))
# Resolve the data directory eagerly so downstream code can rely on absolute paths.
DATA_DIR = Path(os.getenv("OWUI_DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_SUMMARY_MODEL = os.getenv("OLLAMA_SUMMARY_MODEL", "llama3.1")
OLLAMA_LONGFORM_MODEL = os.getenv("OLLAMA_LONGFORM_MODEL", "phi3:mini")
OLLAMA_SUMMARY_FALLBACK_MODEL = os.getenv("OLLAMA_SUMMARY_FALLBACK_MODEL") or OLLAMA_LONGFORM_MODEL
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_DEFAULT_TEMPERATURE", "0.2"))
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30"))
