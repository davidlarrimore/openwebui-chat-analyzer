from __future__ import annotations

import os
from typing import Iterator

import pandas as pd
import pytest

from frontend.core import config as core_config


@pytest.fixture(autouse=True)
def reset_config_cache(monkeypatch) -> Iterator[None]:
    """Reset cached configuration and fix the API base URL for tests."""
    monkeypatch.setenv("OWUI_API_BASE_URL", "http://testserver")
    core_config.get_config.cache_clear()
    yield
    core_config.get_config.cache_clear()


@pytest.fixture
def sample_chats_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chat_id": "chat-1",
                "user_id": "user-1",
                "user_display": "Alice",
                "created_at": pd.Timestamp("2024-01-01T10:00:00Z"),
                "updated_at": pd.Timestamp("2024-01-01T10:10:00Z"),
                "files_uploaded": 1,
                "summary_128": "Sample summary",
                "title": "First chat",
                "archived": False,
                "pinned": False,
                "tags": [],
            },
            {
                "chat_id": "chat-2",
                "user_id": "user-2",
                "user_display": "Bob",
                "created_at": pd.Timestamp("2024-01-02T11:00:00Z"),
                "updated_at": pd.Timestamp("2024-01-02T11:05:00Z"),
                "files_uploaded": 0,
                "summary_128": "Another summary",
                "title": "Second chat",
                "archived": False,
                "pinned": False,
                "tags": [],
            },
        ]
    )


@pytest.fixture
def sample_messages_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chat_id": "chat-1",
                "timestamp": pd.Timestamp("2024-01-01T10:00:00Z"),
                "role": "user",
                "content": "Hello assistant!",
                "model": "model-a",
            },
            {
                "chat_id": "chat-1",
                "timestamp": pd.Timestamp("2024-01-01T10:01:00Z"),
                "role": "assistant",
                "content": "Hello Alice, how can I help?",
                "model": "model-a",
            },
            {
                "chat_id": "chat-2",
                "timestamp": pd.Timestamp("2024-01-02T11:00:00Z"),
                "role": "user",
                "content": "Need assistance with my account.",
                "model": "",
            },
        ]
    )

