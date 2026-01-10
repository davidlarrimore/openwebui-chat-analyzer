"""Quality validation utilities for metric extraction.

Provides utilities for validating extracted metrics against source conversations:
- Hallucination detection (claims not supported by conversation)
- Keyword overlap analysis
- Confidence scoring
- Cross-validation with conversation content

Sprint 3 Implementation: LLM as a Judge - Enhanced Evaluation
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

_logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of quality validation check.

    Attributes:
        is_valid: Whether the content passed validation
        confidence_score: 0.0-1.0 confidence in the extracted content
        issues: List of validation issues found
        warnings: List of non-critical warnings
    """

    def __init__(
        self,
        is_valid: bool,
        confidence_score: float,
        issues: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ):
        self.is_valid = is_valid
        self.confidence_score = confidence_score
        self.issues = issues or []
        self.warnings = warnings or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "confidence_score": self.confidence_score,
            "issues": self.issues,
            "warnings": self.warnings,
        }


def extract_keywords(text: str, min_length: int = 4) -> Set[str]:
    """Extract significant keywords from text.

    Args:
        text: Text to extract keywords from
        min_length: Minimum keyword length to consider

    Returns:
        Set of normalized keywords
    """
    # Remove punctuation and split
    words = re.findall(r"\b\w+\b", text.lower())

    # Filter stop words and short words
    stop_words = {
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "just",
        "him",
        "know",
        "take",
        "into",
        "your",
        "some",
        "could",
        "them",
        "than",
        "then",
        "now",
        "only",
        "come",
        "its",
        "over",
        "also",
        "back",
        "after",
        "use",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "most",
        "us",
    }

    keywords = {word for word in words if len(word) >= min_length and word not in stop_words}

    return keywords


def calculate_keyword_overlap(text1: str, text2: str) -> float:
    """Calculate keyword overlap between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Overlap score 0.0-1.0 (Jaccard similarity)
    """
    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)

    if not keywords1 and not keywords2:
        return 1.0  # Both empty = perfect match

    if not keywords1 or not keywords2:
        return 0.0  # One empty = no match

    intersection = len(keywords1 & keywords2)
    union = len(keywords1 | keywords2)

    return intersection / union if union > 0 else 0.0


def validate_summary_against_conversation(
    summary: str,
    conversation_context: str,
    min_overlap: float = 0.15,
) -> ValidationResult:
    """Validate summary against conversation content.

    Checks if the summary is grounded in the actual conversation and
    doesn't contain hallucinated information.

    Args:
        summary: Extracted summary text
        conversation_context: Source conversation text
        min_overlap: Minimum required keyword overlap (default 0.15 = 15%)

    Returns:
        ValidationResult with validation status and confidence score
    """
    issues = []
    warnings = []

    # Check if summary is empty or trivial
    if not summary or len(summary.strip()) < 5:
        issues.append("Summary is empty or too short")
        return ValidationResult(
            is_valid=False,
            confidence_score=0.0,
            issues=issues,
        )

    # Calculate keyword overlap
    overlap = calculate_keyword_overlap(summary, conversation_context)

    # Check for sufficient overlap
    if overlap < min_overlap:
        issues.append(
            f"Low keyword overlap ({overlap:.1%}) suggests summary may not reflect conversation content"
        )
        confidence_score = overlap / min_overlap  # Proportional confidence
    else:
        confidence_score = min(1.0, overlap / min_overlap)

    # Check for overly generic summaries
    generic_phrases = [
        "user asked a question",
        "assistant provided an answer",
        "conversation about",
        "discussion regarding",
        "help with",
        "asked for help",
        "general inquiry",
    ]
    summary_lower = summary.lower()
    for phrase in generic_phrases:
        if phrase in summary_lower:
            warnings.append(f"Summary contains generic phrase: '{phrase}'")
            confidence_score *= 0.9  # Reduce confidence slightly

    # Check summary length (shouldn't be too long)
    word_count = len(summary.split())
    if word_count > 25:
        warnings.append(f"Summary is longer than recommended ({word_count} words)")
        confidence_score *= 0.95

    # Determine validity
    is_valid = len(issues) == 0 and confidence_score >= 0.5

    return ValidationResult(
        is_valid=is_valid,
        confidence_score=max(0.0, min(1.0, confidence_score)),
        issues=issues,
        warnings=warnings,
    )


def validate_outcome_reasoning(
    outcome_score: int,
    reasoning: str,
    conversation_context: str,
) -> ValidationResult:
    """Validate outcome score and reasoning.

    Checks if the reasoning is substantive and relates to the conversation.

    Args:
        outcome_score: The 1-5 outcome score
        reasoning: The reasoning text explaining the score
        conversation_context: Source conversation text

    Returns:
        ValidationResult with validation status
    """
    issues = []
    warnings = []
    confidence_score = 1.0

    # Check if reasoning is provided
    if not reasoning or len(reasoning.strip()) < 10:
        warnings.append("Reasoning is missing or too brief")
        confidence_score *= 0.7

    # Check if score is in valid range (should be caught earlier, but double-check)
    if not (1 <= outcome_score <= 5):
        issues.append(f"Outcome score {outcome_score} is out of valid range (1-5)")
        confidence_score = 0.0

    # Check for keyword overlap with conversation
    if reasoning:
        overlap = calculate_keyword_overlap(reasoning, conversation_context)
        if overlap < 0.05:
            warnings.append("Reasoning has very low overlap with conversation content")
            confidence_score *= 0.8

    is_valid = len(issues) == 0

    return ValidationResult(
        is_valid=is_valid,
        confidence_score=max(0.0, min(1.0, confidence_score)),
        issues=issues,
        warnings=warnings,
    )
