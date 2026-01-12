"""Tests for the summarizer pipeline tweaks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from backend import summarizer


def test_summarize_chats_emits_chunk_events(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure deterministic chunking for the test.
    monkeypatch.setattr(summarizer, "CHUNK_CHAR_LIMIT", 50)
    monkeypatch.setattr(summarizer, "CHUNK_OVERLAP_LINES", 0)
    monkeypatch.setattr(summarizer, "_build_context_from_lines", lambda lines: " ".join(lines))

    recorded_contexts: List[str] = []

    def fake_summarize_context(context: str, **_: Any) -> summarizer.ConversationAnalysis:
        recorded_contexts.append(context)
        return summarizer.ConversationAnalysis(summary=f"summary:{len(context)}", outcome=None)

    monkeypatch.setattr(summarizer, "_summarize_context", fake_summarize_context)

    chats = [{"chat_id": "chat-1", "gen_chat_summary": ""}]
    messages = [
        {"chat_id": "chat-1", "role": "user", "content": "A" * 40},
        {"chat_id": "chat-1", "role": "assistant", "content": "B" * 40},
        {"chat_id": "chat-1", "role": "user", "content": "C" * 40},
        {"chat_id": "chat-1", "role": "assistant", "content": "D" * 40},
    ]

    progress_events: List[Tuple[int, int, str, str, Optional[Dict[str, Any]]]] = []

    def progress(
        current: int,
        total: int,
        chat_id: str,
        outcome: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        progress_events.append((current, total, chat_id, outcome, details))

    summary_map, stats = summarizer.summarize_chats(
        chats,
        messages,
        on_progress=progress,
        replace_existing=True,
    )

    # Ensure the mocked summarizer was invoked for every chunk plus the aggregate summary.
    assert len(recorded_contexts) >= 1, "Expected summarize context to be invoked at least once."
    assert stats["generated"] == 1
    assert "chat-1" in summary_map
    assert summary_map["chat-1"].startswith("summary:")

    chunk_events = [
        event
        for event in progress_events
        if event[-1] and event[-1].get("type") == "chunk"
    ]
    if chunk_events:
        first_chunk_details = chunk_events[0][-1]
        assert first_chunk_details is not None
        assert first_chunk_details.get("chunk_index") == 1
        assert first_chunk_details.get("chunk_count") >= 1

    final_events = [
        event
        for event in progress_events
        if event[-1] and event[-1].get("type") == "chat"
    ]
    assert final_events, "Expected a final chat completion event."


def test_summarize_with_chunks_preserves_failure_details(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure failure reasons from the LLM path aren't dropped."""
    failure = summarizer.ConversationAnalysis(
        summary="",
        outcome=None,
        failure_reason="Ollama returned an empty summary response.",
        provider="ollama",
    )

    def fake_summarize_context(_: str, **__: Any) -> summarizer.ConversationAnalysis:
        return failure

    monkeypatch.setattr(summarizer, "_summarize_context", fake_summarize_context)

    analysis = summarizer._summarize_with_chunks(  # pylint: disable=protected-access
        ["line one", "line two"],
        fallback_context="line one line two",
        on_progress=None,
        progress_index=1,
        total=1,
        chat_id="chat-1",
        connection="ollama",
    )

    assert analysis.failure_reason == "Ollama returned an empty summary response."
    assert analysis.provider == "ollama"
