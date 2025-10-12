from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent formatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def set_level(level: int) -> None:
    """Adjust the root logger level."""
    logging.getLogger().setLevel(level)

