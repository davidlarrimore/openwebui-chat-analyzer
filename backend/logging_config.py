"""Logging configuration helpers for the backend service."""

from __future__ import annotations

import logging
import os
from logging.config import dictConfig


def configure_logging() -> None:
    """Apply a consistent logging configuration for the backend service."""
    log_level = os.getenv("OWUI_LOG_LEVEL", "INFO").upper()

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": log_level,
                },
            },
            "root": {"handlers": ["console"], "level": log_level},
            # Ensure uvicorn loggers inherit our formatting.
            "loggers": {
                "uvicorn": {"handlers": ["console"], "level": log_level, "propagate": False},
                "uvicorn.error": {
                    "handlers": ["console"],
                    "level": log_level,
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["console"],
                    "level": os.getenv("UVICORN_ACCESS_LOG_LEVEL", "INFO").upper(),
                    "propagate": False,
                },
            },
        }
    )

    logging.getLogger(__name__).debug("Logging configured at %s level", log_level)
