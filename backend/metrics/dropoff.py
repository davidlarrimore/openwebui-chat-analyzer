"""Conversation drop-off detection metric.

Analyzes whether a conversation ended naturally or was abandoned mid-way.
Detects unresolved questions and incomplete interactions to identify
conversations that may need follow-up.

Sprint 3 Implementation: LLM as a Judge - Enhanced Evaluation
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Mapping

from backend.metrics.base import MetricExtractor, MetricResult
from backend.providers.base import LLMProvider

_logger = logging.getLogger(__name__)


class DropOffDetector(MetricExtractor):
    """Detects whether conversation ended naturally or was abandoned.

    Analyzes the last few messages to determine if:
    - User had unresolved questions
    - Assistant was in the middle of helping
    - Conversation reached a natural stopping point

    Output Schema:
        {
            "conversation_complete": true/false,
            "last_message_has_questions": true/false,
            "abandonment_pattern": "user_disappeared" | "resolved" | "unclear",
            "reasoning": "Brief explanation"
        }
    """

    @property
    def metric_name(self) -> str:
        """Metric identifier."""
        return "dropoff"

    def _has_question_markers(self, text: str) -> bool:
        """Check if text contains question markers.

        Args:
            text: Message text to analyze

        Returns:
            True if text appears to contain questions
        """
        # Question mark at end
        if text.rstrip().endswith("?"):
            return True

        # Question words at start
        question_starters = [
            r"\bhow\s",
            r"\bwhat\s",
            r"\bwhen\s",
            r"\bwhere\s",
            r"\bwhy\s",
            r"\bwhich\s",
            r"\bwho\s",
            r"\bcan\s",
            r"\bcould\s",
            r"\bwould\s",
            r"\bshould\s",
            r"\bis\s",
            r"\bare\s",
            r"\bdoes\s",
            r"\bdo\s",
            r"\bdid\s",
        ]
        text_lower = text.lower()
        for pattern in question_starters:
            if re.search(pattern, text_lower):
                return True

        return False

    def _analyze_conversation_ending(
        self,
        messages: Sequence[Mapping[str, Any]],
        analyze_with_llm: bool = True,
    ) -> Dict[str, Any]:
        """Analyze how the conversation ended.

        Args:
            messages: Full conversation history
            analyze_with_llm: If True, use LLM for deep analysis; if False, use heuristics only

        Returns:
            Dictionary with drop-off analysis
        """
        if not messages:
            return {
                "conversation_complete": False,
                "last_message_has_questions": False,
                "abandonment_pattern": "unclear",
                "reasoning": "Empty conversation",
            }

        # Get last few messages
        last_n = 4
        recent_messages = list(messages[-last_n:]) if len(messages) >= last_n else list(messages)

        if not recent_messages:
            return {
                "conversation_complete": False,
                "last_message_has_questions": False,
                "abandonment_pattern": "unclear",
                "reasoning": "No messages to analyze",
            }

        # Heuristic analysis
        last_message = recent_messages[-1]
        last_role = last_message.get("role", "")
        last_content = str(last_message.get("content", ""))

        # Check if last message from user contains questions
        last_message_has_questions = False
        if last_role == "user":
            last_message_has_questions = self._has_question_markers(last_content)

        # Simple heuristics
        if last_role == "assistant" and len(last_content) > 50:
            # Assistant provided substantial response
            return {
                "conversation_complete": True,
                "last_message_has_questions": False,
                "abandonment_pattern": "resolved",
                "reasoning": "Conversation ended with assistant providing complete response",
            }

        if last_role == "user" and last_message_has_questions:
            # User asked something but got no response
            return {
                "conversation_complete": False,
                "last_message_has_questions": True,
                "abandonment_pattern": "user_disappeared",
                "reasoning": "User asked question but conversation ended before assistant could respond",
            }

        # Unclear cases - use heuristics for now (LLM analysis can be added later if needed)
        return {
            "conversation_complete": True,  # Default to complete if no clear abandonment
            "last_message_has_questions": last_message_has_questions,
            "abandonment_pattern": "unclear",
            "reasoning": "Conversation ending pattern unclear; defaulting to complete",
        }

    def extract(
        self,
        context: str,
        provider: LLMProvider,
        model: str,
        provider_name: str,
        messages: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> MetricResult:
        """Detect conversation drop-off.

        Args:
            context: Conversation text (pre-processed, not used for drop-off)
            provider: LLM provider instance (not used for heuristic analysis)
            model: Model name (not used for heuristic analysis)
            provider_name: Provider identifier
            messages: Full conversation history (required for drop-off analysis)

        Returns:
            MetricResult with drop-off analysis
        """
        self._log_extraction_start(provider_name, model)

        try:
            if not messages:
                error_msg = "Drop-off detection requires full message history"
                self._log_extraction_error(provider_name, model, error_msg)
                return MetricResult(
                    metric_name=self.metric_name,
                    success=False,
                    error=error_msg,
                    provider=provider_name,
                    model=model,
                )

            # Analyze conversation ending (heuristic-based for now)
            analysis = self._analyze_conversation_ending(messages, analyze_with_llm=False)

            self._log_extraction_success(provider_name, model)

            return MetricResult(
                metric_name=self.metric_name,
                success=True,
                data=analysis,
                provider=provider_name,
                model=model,
            )

        except Exception as e:
            error_msg = f"Unexpected error during drop-off detection: {e}"
            self._log_extraction_error(provider_name, model, error_msg)
            return MetricResult(
                metric_name=self.metric_name,
                success=False,
                error=error_msg,
                provider=provider_name,
                model=model,
            )
