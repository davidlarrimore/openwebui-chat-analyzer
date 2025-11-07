"""Unit tests for summarizer failure categorization heuristics."""

from __future__ import annotations

import pytest

from backend import summarizer


@pytest.mark.parametrize(
    ("reason", "response", "expected"),
    [
        (
            "Ollama response parsing failed: LLM response did not contain valid summary JSON.",
            '{"summary": "The agent succeeded at the task", ',
            summarizer._FAILURE_CATEGORY_BAD_FORMATTING,
        ),
        (
            "Ollama returned an empty summary response.",
            "",
            summarizer._FAILURE_CATEGORY_NO_RESPONSE,
        ),
        (
            "Ollama response parsing failed: LLM response did not contain valid summary JSON.",
            "I'm sorry, I can't help with that request.",
            summarizer._FAILURE_CATEGORY_BLOCKED,
        ),
        (
            "Ollama response parsing failed: LLM response did not contain valid summary JSON.",
            "Sure, here is the deployment script you asked for:\n#!/bin/bash",
            summarizer._FAILURE_CATEGORY_PROMPT_GUARDRAIL,
        ),
        (
            "Unexpected error during Ollama summarization: socket timeout",
            None,
            summarizer._FAILURE_CATEGORY_OTHER,
        ),
    ],
)
def test_categorize_failure(reason: str, response: str | None, expected: str) -> None:
    """Ensure failures are categorized into the expected buckets."""
    category = summarizer._categorize_failure(reason, response)
    assert category == expected
