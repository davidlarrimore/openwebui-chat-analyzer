"""Tests for Sprint 3 quality validation utilities.

Tests the validation utilities used for detecting hallucinations and
validating metric extraction quality.
"""

from __future__ import annotations

import pytest

from backend.metrics.validation import (
    ValidationResult,
    extract_keywords,
    calculate_keyword_overlap,
    validate_summary_against_conversation,
    validate_outcome_reasoning,
)


class TestExtractKeywords:
    """Test keyword extraction from text."""

    def test_extract_keywords_basic(self) -> None:
        """Extract keywords from simple text."""
        text = "Python programming language debugging"
        keywords = extract_keywords(text)

        assert "python" in keywords
        assert "programming" in keywords
        assert "language" in keywords
        assert "debugging" in keywords

    def test_extract_keywords_filters_stop_words(self) -> None:
        """Stop words are filtered out."""
        text = "the quick brown fox jumps over the lazy dog"
        keywords = extract_keywords(text)

        # Stop words should be filtered
        assert "the" not in keywords
        assert "over" not in keywords

        # Content words should remain
        assert "quick" in keywords
        assert "brown" in keywords

    def test_extract_keywords_filters_short_words(self) -> None:
        """Short words (< 4 chars) are filtered by default."""
        text = "SQL and API are short but code is longer"
        keywords = extract_keywords(text, min_length=4)

        assert "sql" not in keywords
        assert "api" not in keywords
        assert "code" in keywords
        assert "longer" in keywords

    def test_extract_keywords_normalizes_case(self) -> None:
        """Keywords are normalized to lowercase."""
        text = "Python PROGRAMMING Language"
        keywords = extract_keywords(text)

        assert "python" in keywords
        assert "PYTHON" not in keywords
        assert "programming" in keywords

    def test_extract_keywords_empty_text(self) -> None:
        """Empty text returns empty set."""
        keywords = extract_keywords("")
        assert len(keywords) == 0


class TestCalculateKeywordOverlap:
    """Test keyword overlap calculation."""

    def test_overlap_identical_texts(self) -> None:
        """Identical texts have 100% overlap."""
        text = "python programming debugging async"
        overlap = calculate_keyword_overlap(text, text)

        assert overlap == 1.0

    def test_overlap_no_common_keywords(self) -> None:
        """No common keywords = 0% overlap."""
        text1 = "python programming"
        text2 = "java development"
        overlap = calculate_keyword_overlap(text1, text2)

        assert overlap == 0.0

    def test_overlap_partial_match(self) -> None:
        """Partial keyword match gives fractional overlap."""
        text1 = "python async programming debugging"
        text2 = "python debugging testing validation"

        # Common: python, debugging (2)
        # Union: python, async, programming, debugging, testing, validation (6)
        # Overlap: 2/6 = 0.333...
        overlap = calculate_keyword_overlap(text1, text2)

        assert 0.3 <= overlap <= 0.4

    def test_overlap_empty_texts(self) -> None:
        """Both empty texts = perfect match."""
        overlap = calculate_keyword_overlap("", "")
        assert overlap == 1.0

    def test_overlap_one_empty(self) -> None:
        """One empty text = no match."""
        overlap1 = calculate_keyword_overlap("python programming", "")
        overlap2 = calculate_keyword_overlap("", "java development")

        assert overlap1 == 0.0
        assert overlap2 == 0.0


class TestValidateSummaryAgainstConversation:
    """Test summary validation against conversation."""

    def test_valid_summary_high_overlap(self) -> None:
        """Summary with high keyword overlap is valid."""
        summary = "Python async debugging help"
        conversation = "user: I need help debugging my Python async code\nassistant: Here's how to debug async..."

        result = validate_summary_against_conversation(summary, conversation, min_overlap=0.15)

        assert result.is_valid
        assert result.confidence_score >= 0.5
        assert len(result.issues) == 0

    def test_invalid_summary_low_overlap(self) -> None:
        """Summary with low keyword overlap is invalid."""
        summary = "Database schema design"
        conversation = "user: How do I debug Python async code?\nassistant: Use asyncio debug mode..."

        result = validate_summary_against_conversation(summary, conversation, min_overlap=0.15)

        assert not result.is_valid
        assert result.confidence_score < 0.5
        assert len(result.issues) > 0
        assert any("overlap" in issue.lower() for issue in result.issues)

    def test_empty_summary_invalid(self) -> None:
        """Empty summary is invalid."""
        summary = ""
        conversation = "user: test\nassistant: response"

        result = validate_summary_against_conversation(summary, conversation)

        assert not result.is_valid
        assert result.confidence_score == 0.0
        assert any("empty" in issue.lower() or "short" in issue.lower() for issue in result.issues)

    def test_generic_summary_warning(self) -> None:
        """Generic summaries get warnings."""
        summary = "User asked a question about programming"
        conversation = "user: How do I use Python async?\nassistant: Here's how..."

        result = validate_summary_against_conversation(summary, conversation)

        # Should have warnings about generic phrases
        assert len(result.warnings) > 0
        assert any("generic" in warning.lower() for warning in result.warnings)

    def test_long_summary_warning(self) -> None:
        """Summaries over 25 words get warnings."""
        summary = "This is a very long summary that contains way too many words and really should be much more concise because the whole point of a summary is to be brief and to the point"
        conversation = "user: test\nassistant: response about something"

        result = validate_summary_against_conversation(summary, conversation)

        # Should have warning about length
        assert any("longer" in warning.lower() for warning in result.warnings)

    def test_confidence_score_proportional_to_overlap(self) -> None:
        """Confidence score scales with keyword overlap."""
        conversation = "user: Python async debugging\nassistant: Here's how to debug async code..."

        summary_high = "Python async debugging help"
        summary_med = "Python programming task"
        summary_low = "Java database schema"

        result_high = validate_summary_against_conversation(summary_high, conversation, min_overlap=0.15)
        result_med = validate_summary_against_conversation(summary_med, conversation, min_overlap=0.15)
        result_low = validate_summary_against_conversation(summary_low, conversation, min_overlap=0.15)

        # Higher overlap should have higher confidence
        assert result_high.confidence_score > result_med.confidence_score
        assert result_med.confidence_score > result_low.confidence_score


class TestValidateOutcomeReasoning:
    """Test outcome reasoning validation."""

    def test_valid_reasoning_with_content(self) -> None:
        """Reasoning with content is valid."""
        outcome_score = 4
        reasoning = "Assistant provided complete answer with code examples"
        conversation = "user: how to use async\nassistant: here's async code example..."

        result = validate_outcome_reasoning(outcome_score, reasoning, conversation)

        assert result.is_valid
        assert result.confidence_score > 0.5
        assert len(result.issues) == 0

    def test_missing_reasoning_warning(self) -> None:
        """Missing or brief reasoning gets warning."""
        outcome_score = 3
        reasoning = ""
        conversation = "user: test\nassistant: response"

        result = validate_outcome_reasoning(outcome_score, reasoning, conversation)

        assert len(result.warnings) > 0
        assert any("missing" in warning.lower() or "brief" in warning.lower() for warning in result.warnings)

    def test_invalid_score_out_of_range(self) -> None:
        """Score out of range (1-5) is invalid."""
        outcome_score = 10
        reasoning = "Complete answer"
        conversation = "user: test\nassistant: response"

        result = validate_outcome_reasoning(outcome_score, reasoning, conversation)

        assert not result.is_valid
        assert result.confidence_score == 0.0
        assert len(result.issues) > 0

    def test_reasoning_with_low_overlap_warning(self) -> None:
        """Reasoning with very low overlap gets warning."""
        outcome_score = 4
        reasoning = "Database schema normalization was excellent"
        conversation = "user: Python async help\nassistant: Here's async code..."

        result = validate_outcome_reasoning(outcome_score, reasoning, conversation)

        # Should warn about low overlap
        assert len(result.warnings) > 0


class TestValidationResult:
    """Test ValidationResult class."""

    def test_to_dict(self) -> None:
        """ValidationResult converts to dictionary."""
        result = ValidationResult(
            is_valid=True,
            confidence_score=0.85,
            issues=["issue1"],
            warnings=["warning1", "warning2"],
        )

        data = result.to_dict()

        assert data["is_valid"] is True
        assert data["confidence_score"] == 0.85
        assert data["issues"] == ["issue1"]
        assert data["warnings"] == ["warning1", "warning2"]

    def test_default_empty_lists(self) -> None:
        """ValidationResult defaults to empty lists."""
        result = ValidationResult(is_valid=True, confidence_score=1.0)

        assert result.issues == []
        assert result.warnings == []
