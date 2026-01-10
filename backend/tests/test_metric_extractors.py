"""Tests for Sprint 2 metric extractors.

This module tests the individual metric extractor implementations:
- SummaryExtractor
- OutcomeExtractor
- TagsExtractor
- ClassificationExtractor

Each test verifies:
- Successful extraction with valid input
- JSON parsing and validation
- Error handling with malformed responses
- Graceful failure handling
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from backend.metrics.summary import SummaryExtractor
from backend.metrics.outcome import OutcomeExtractor
from backend.metrics.tags import TagsExtractor
from backend.metrics.classification import ClassificationExtractor
from backend.providers.base import GenerateResult


class TestSummaryExtractor:
    """Test SummaryExtractor."""

    def test_extract_success(self) -> None:
        """Successful summary extraction returns summary text."""
        extractor = SummaryExtractor()

        # Mock provider
        mock_provider = MagicMock()
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = GenerateResult(
            content='{"summary": "Python async debugging help"}',
            model="test-model",
            provider="test",
        )

        result = extractor.extract(
            context="user: Help with async\nassistant: Here's how...",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
        )

        assert result.success is True
        assert result.data == {"summary": "Python async debugging help"}
        assert result.provider == "test"
        assert result.model == "test-model"
        assert result.error is None

    def test_extract_json_parse_error(self) -> None:
        """Malformed JSON response returns error."""
        extractor = SummaryExtractor()

        mock_provider = MagicMock()
        mock_provider.supports_json_mode.return_value = False
        mock_provider.generate.return_value = GenerateResult(
            content="This is not valid JSON",
            model="test-model",
            provider="test",
        )

        result = extractor.extract(
            context="user: test\nassistant: response",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
        )

        assert result.success is False
        assert result.data is None
        assert "Failed to parse JSON" in result.error

    def test_extract_missing_summary_field(self) -> None:
        """Response missing 'summary' field returns error."""
        extractor = SummaryExtractor()

        mock_provider = MagicMock()
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = GenerateResult(
            content='{"wrong_field": "value"}',
            model="test-model",
            provider="test",
        )

        result = extractor.extract(
            context="user: test",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
        )

        assert result.success is False
        assert "missing 'summary' field" in result.error

    def test_extract_with_json_mode(self) -> None:
        """Extractor enables JSON mode when provider supports it."""
        extractor = SummaryExtractor()

        mock_provider = MagicMock()
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = GenerateResult(
            content='{"summary": "Test"}',
            model="test-model",
            provider="test",
        )

        extractor.extract(
            context="test context",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
        )

        # Verify JSON mode was enabled in options
        call_kwargs = mock_provider.generate.call_args[1]
        assert call_kwargs["options"]["json_mode"] is True


class TestOutcomeExtractor:
    """Test OutcomeExtractor."""

    def test_extract_success(self) -> None:
        """Successful outcome extraction returns score and reasoning."""
        extractor = OutcomeExtractor()

        mock_provider = MagicMock()
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = GenerateResult(
            content='{"outcome": 5, "reasoning": "Complete answer provided"}',
            model="test-model",
            provider="test",
        )

        result = extractor.extract(
            context="user: question\nassistant: answer",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
        )

        assert result.success is True
        assert result.data == {"outcome": 5, "reasoning": "Complete answer provided"}

    def test_extract_invalid_score_range(self) -> None:
        """Outcome score outside 1-5 range returns error."""
        extractor = OutcomeExtractor()

        mock_provider = MagicMock()
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = GenerateResult(
            content='{"outcome": 10, "reasoning": "Too high"}',
            model="test-model",
            provider="test",
        )

        result = extractor.extract(
            context="test",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
        )

        assert result.success is False
        assert "Invalid outcome score" in result.error

    def test_extract_missing_outcome_field(self) -> None:
        """Response missing 'outcome' field returns error."""
        extractor = OutcomeExtractor()

        mock_provider = MagicMock()
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = GenerateResult(
            content='{"reasoning": "No score"}',
            model="test-model",
            provider="test",
        )

        result = extractor.extract(
            context="test",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
        )

        assert result.success is False
        assert "missing 'outcome' field" in result.error

    def test_extract_reasoning_optional(self) -> None:
        """Reasoning field is optional, defaults to empty string."""
        extractor = OutcomeExtractor()

        mock_provider = MagicMock()
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = GenerateResult(
            content='{"outcome": 3}',
            model="test-model",
            provider="test",
        )

        result = extractor.extract(
            context="test",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
        )

        assert result.success is True
        assert result.data["outcome"] == 3
        assert result.data["reasoning"] == ""


class TestTagsExtractor:
    """Test TagsExtractor."""

    def test_extract_success(self) -> None:
        """Successful tags extraction returns list of tags."""
        extractor = TagsExtractor()

        mock_provider = MagicMock()
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = GenerateResult(
            content='{"tags": ["python", "asyncio", "debugging"]}',
            model="test-model",
            provider="test",
        )

        result = extractor.extract(
            context="user: async bug\nassistant: here's the fix",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
        )

        assert result.success is True
        assert result.data == {"tags": ["python", "asyncio", "debugging"]}

    def test_extract_normalizes_tags(self) -> None:
        """Tags are normalized to lowercase and stripped."""
        extractor = TagsExtractor()

        mock_provider = MagicMock()
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = GenerateResult(
            content='{"tags": ["Python", "  AsyncIO  ", "DEBUGGING"]}',
            model="test-model",
            provider="test",
        )

        result = extractor.extract(
            context="test",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
        )

        assert result.success is True
        assert result.data["tags"] == ["python", "asyncio", "debugging"]

    def test_extract_filters_empty_tags(self) -> None:
        """Empty tags are filtered out."""
        extractor = TagsExtractor()

        mock_provider = MagicMock()
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = GenerateResult(
            content='{"tags": ["python", "", "   ", "asyncio"]}',
            model="test-model",
            provider="test",
        )

        result = extractor.extract(
            context="test",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
        )

        assert result.success is True
        assert result.data["tags"] == ["python", "asyncio"]

    def test_extract_invalid_tags_not_list(self) -> None:
        """Tags field that is not a list returns error."""
        extractor = TagsExtractor()

        mock_provider = MagicMock()
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = GenerateResult(
            content='{"tags": "not a list"}',
            model="test-model",
            provider="test",
        )

        result = extractor.extract(
            context="test",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
        )

        assert result.success is False
        assert "'tags' field must be a list" in result.error


class TestClassificationExtractor:
    """Test ClassificationExtractor."""

    def test_extract_success(self) -> None:
        """Successful classification extraction returns domain and resolution_status."""
        extractor = ClassificationExtractor()

        mock_provider = MagicMock()
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = GenerateResult(
            content='{"domain": "technical", "resolution_status": "resolved"}',
            model="test-model",
            provider="test",
        )

        result = extractor.extract(
            context="user: bug\nassistant: fixed",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
        )

        assert result.success is True
        assert result.data == {"domain": "technical", "resolution_status": "resolved"}

    def test_extract_normalizes_values(self) -> None:
        """Domain and resolution_status are normalized to lowercase."""
        extractor = ClassificationExtractor()

        mock_provider = MagicMock()
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = GenerateResult(
            content='{"domain": "TECHNICAL", "resolution_status": "Resolved"}',
            model="test-model",
            provider="test",
        )

        result = extractor.extract(
            context="test",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
        )

        assert result.success is True
        assert result.data["domain"] == "technical"
        assert result.data["resolution_status"] == "resolved"

    def test_extract_invalid_domain_defaults_to_other(self) -> None:
        """Invalid domain value defaults to 'other'."""
        extractor = ClassificationExtractor()

        mock_provider = MagicMock()
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = GenerateResult(
            content='{"domain": "invalid", "resolution_status": "resolved"}',
            model="test-model",
            provider="test",
        )

        result = extractor.extract(
            context="test",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
        )

        assert result.success is True
        assert result.data["domain"] == "other"

    def test_extract_invalid_resolution_defaults_to_unclear(self) -> None:
        """Invalid resolution_status value defaults to 'unclear'."""
        extractor = ClassificationExtractor()

        mock_provider = MagicMock()
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = GenerateResult(
            content='{"domain": "technical", "resolution_status": "invalid"}',
            model="test-model",
            provider="test",
        )

        result = extractor.extract(
            context="test",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
        )

        assert result.success is True
        assert result.data["resolution_status"] == "unclear"

    def test_extract_missing_required_fields(self) -> None:
        """Response missing required fields returns error."""
        extractor = ClassificationExtractor()

        mock_provider = MagicMock()
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = GenerateResult(
            content='{"domain": "technical"}',
            model="test-model",
            provider="test",
        )

        result = extractor.extract(
            context="test",
            provider=mock_provider,
            model="test-model",
            provider_name="test",
        )

        assert result.success is False
        assert "missing required fields" in result.error
