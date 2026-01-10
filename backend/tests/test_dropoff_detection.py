"""Tests for Sprint 3 conversation drop-off detection.

Tests the drop-off detector that identifies conversations that ended
prematurely or were abandoned mid-way.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from backend.metrics.dropoff import DropOffDetector
from backend.metrics.base import MetricResult


class TestDropOffDetector:
    """Test DropOffDetector class."""

    def test_conversation_complete_assistant_last(self) -> None:
        """Conversation ending with assistant response is considered complete."""
        detector = DropOffDetector()

        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "How do I use async in Python?"},
            {"role": "assistant", "content": "Here's a complete guide to async in Python with examples..."},
        ]

        mock_provider = MagicMock()

        result = detector.extract(
            context="test context",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
            messages=messages,
        )

        assert result.success
        assert result.data["conversation_complete"] is True
        assert result.data["last_message_has_questions"] is False
        assert result.data["abandonment_pattern"] == "resolved"

    def test_conversation_incomplete_user_question_last(self) -> None:
        """Conversation ending with user question is incomplete."""
        detector = DropOffDetector()

        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "How do I use async?"},
            {"role": "assistant", "content": "Here's how..."},
            {"role": "user", "content": "What about error handling?"},  # No response
        ]

        mock_provider = MagicMock()

        result = detector.extract(
            context="test context",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
            messages=messages,
        )

        assert result.success
        assert result.data["conversation_complete"] is False
        assert result.data["last_message_has_questions"] is True
        assert result.data["abandonment_pattern"] == "user_disappeared"

    def test_question_markers_detected(self) -> None:
        """Question markers in last message are detected."""
        detector = DropOffDetector()

        # Test question mark
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "Is this correct?"},
        ]

        result = detector.extract(
            context="test", provider=MagicMock(), model="test", provider_name="test", messages=messages
        )

        assert result.data["last_message_has_questions"] is True

        # Test question words
        question_starters = ["How", "What", "When", "Where", "Why", "Can", "Could", "Should"]
        for starter in question_starters:
            messages = [{"role": "user", "content": f"{starter} do I do this"}]
            result = detector.extract(
                context="test",
                provider=MagicMock(),
                model="test",
                provider_name="test",
                messages=messages,
            )
            assert result.data["last_message_has_questions"] is True, f"Failed for: {starter}"

    def test_short_assistant_response_unclear(self) -> None:
        """Short assistant response (<50 chars) is unclear."""
        detector = DropOffDetector()

        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "How do I use async?"},
            {"role": "assistant", "content": "Try this"},  # Too short
        ]

        mock_provider = MagicMock()

        result = detector.extract(
            context="test",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
            messages=messages,
        )

        assert result.success
        # Short response defaults to complete but unclear pattern
        assert result.data["abandonment_pattern"] == "unclear"

    def test_empty_conversation_error(self) -> None:
        """Empty conversation is handled."""
        detector = DropOffDetector()

        messages: List[Dict[str, Any]] = []

        mock_provider = MagicMock()

        result = detector.extract(
            context="test",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
            messages=messages,
        )

        assert result.success
        assert result.data["conversation_complete"] is False
        assert result.data["abandonment_pattern"] == "unclear"

    def test_no_messages_parameter_error(self) -> None:
        """Missing messages parameter returns error."""
        detector = DropOffDetector()

        mock_provider = MagicMock()

        result = detector.extract(
            context="test",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
            messages=None,  # Missing required parameter
        )

        assert not result.success
        assert "requires full message history" in result.error

    def test_multiple_exchanges_last_complete(self) -> None:
        """Multi-turn conversation ending with complete response."""
        detector = DropOffDetector()

        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "How do I use async?"},
            {"role": "assistant", "content": "Here's the basics..."},
            {"role": "user", "content": "What about error handling?"},
            {"role": "assistant", "content": "For error handling in async, you should use try/except blocks around await calls and handle specific exceptions appropriately."},
        ]

        mock_provider = MagicMock()

        result = detector.extract(
            context="test",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
            messages=messages,
        )

        assert result.success
        assert result.data["conversation_complete"] is True
        assert result.data["last_message_has_questions"] is False
        assert result.data["abandonment_pattern"] == "resolved"

    def test_user_statement_last_no_question(self) -> None:
        """User statement (not question) at end defaults to complete."""
        detector = DropOffDetector()

        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "I need help with async"},
            {"role": "assistant", "content": "Here's a guide..."},
            {"role": "user", "content": "Thanks, that helped"},  # Statement, not question
        ]

        mock_provider = MagicMock()

        result = detector.extract(
            context="test",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
            messages=messages,
        )

        assert result.success
        # Statement without question markers defaults to complete
        assert result.data["conversation_complete"] is True
        assert result.data["last_message_has_questions"] is False

    def test_metric_name(self) -> None:
        """Detector has correct metric name."""
        detector = DropOffDetector()
        assert detector.metric_name == "dropoff"


class TestQuestionDetection:
    """Test question marker detection logic."""

    def test_has_question_markers_question_mark(self) -> None:
        """Question mark at end is detected."""
        detector = DropOffDetector()

        assert detector._has_question_markers("Is this correct?")
        assert detector._has_question_markers("How do I do this?")
        assert not detector._has_question_markers("This is a statement.")

    def test_has_question_markers_question_words(self) -> None:
        """Question words at start are detected."""
        detector = DropOffDetector()

        assert detector._has_question_markers("How do I use this")
        assert detector._has_question_markers("What is async")
        assert detector._has_question_markers("Can you help")
        assert detector._has_question_markers("Could this work")
        assert detector._has_question_markers("Should I proceed")

    def test_has_question_markers_case_insensitive(self) -> None:
        """Question detection is case insensitive."""
        detector = DropOffDetector()

        assert detector._has_question_markers("HOW DO I DO THIS")
        assert detector._has_question_markers("how do i do this")
        assert detector._has_question_markers("How Do I Do This")

    def test_has_question_markers_word_boundaries(self) -> None:
        """Question words must be at word boundaries."""
        detector = DropOffDetector()

        # "how" at start of word = match
        assert detector._has_question_markers("how do I proceed")

        # "how" in middle of word = no match (not a question starter)
        assert not detector._has_question_markers("somehow this works")
