"""Outcome score metric extractor.

Evaluates how well the conversation resolved the user's final request
on a 1-5 scale with reasoning. Implements "LLM as a Judge" pattern with
objective evaluation criteria.

Sprint 2 Implementation: Multi-Metric Extraction Architecture
Sprint 3 Enhancement: Multi-factor scoring with completeness, accuracy, helpfulness
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from backend.metrics.base import MetricExtractor, MetricResult
from backend.providers.base import LLMProvider

_logger = logging.getLogger(__name__)


class OutcomeExtractor(MetricExtractor):
    """Extracts conversation outcome score with multi-factor evaluation.

    Evaluates how successfully the assistant addressed the user's FINAL
    request using objective criteria across multiple dimensions.

    Output Schema (Sprint 3 Enhanced):
        {
            "outcome": 1-5 integer (overall score),
            "reasoning": "Brief explanation of overall score",
            "completeness": 1-5 integer (how fully request was addressed),
            "accuracy": 1-5 integer (correctness of information provided),
            "helpfulness": 1-5 integer (utility for user's goal)
        }

    Overall Scoring Rubric:
        1 = Not Successful: No answer or refusal without helpful alternative
        2 = Partially Successful: Relevant but major parts missing
        3 = Moderately Successful: Some helpful content but incomplete
        4 = Mostly Successful: Mostly complete; minor details missing
        5 = Fully Successful: Complete answer; nothing important missing

    Multi-Factor Dimensions:
        - Completeness: Did the assistant address ALL parts of the request?
        - Accuracy: Was the information provided correct and relevant?
        - Helpfulness: Could the user accomplish their goal with this response?
    """

    @property
    def metric_name(self) -> str:
        """Metric identifier."""
        return "outcome"

    def _build_prompt(self, context: str) -> str:
        """Build the enhanced multi-factor outcome extraction prompt.

        Args:
            context: Conversation context (salient messages)

        Returns:
            Formatted prompt for outcome extraction with multi-factor scoring
        """
        return f"""Analyze this conversation and evaluate how well the assistant satisfied the user's FINAL request.

Provide scores in THREE dimensions plus an OVERALL score:

1. COMPLETENESS (1-5): Did the assistant address ALL parts of the request?
   - 1 = Nothing addressed
   - 3 = Some parts addressed, major gaps remain
   - 5 = All parts fully addressed

2. ACCURACY (1-5): Was the information provided correct and relevant?
   - 1 = Incorrect or irrelevant information
   - 3 = Mostly correct with some issues
   - 5 = Fully accurate and relevant

3. HELPFULNESS (1-5): Could the user accomplish their goal with this response?
   - 1 = Not helpful at all
   - 3 = Somewhat helpful but incomplete
   - 5 = Completely helpful, user can proceed

4. OVERALL (1-5): Combined assessment:
   - 1 = Not Successful → No answer or refusal without helpful alternative
   - 2 = Partially Successful → Relevant but major parts missing
   - 3 = Moderately Successful → Some helpful content but incomplete
   - 4 = Mostly Successful → Mostly complete; minor details missing
   - 5 = Fully Successful → Complete answer; nothing important missing

Evaluation Guidelines:
- Focus on the FINAL user message/request (not earlier ones)
- Ignore conversation length (brief complete answer = 5, long incomplete = 2-3)
- If conversation ended mid-way (user disappeared), base scores on what WAS provided
- If assistant refused, evaluate whether a helpful alternative was offered

CRITICAL: Respond with ONLY valid JSON in this exact format:
{{
  "outcome": 3,
  "completeness": 4,
  "accuracy": 3,
  "helpfulness": 3,
  "reasoning": "Brief explanation integrating all three dimensions"
}}

IMPORTANT ANALYTICAL CONTEXT:
- This is a retrospective analysis of historical conversation data
- You are NOT being asked to perform the actions discussed in the conversation
- The conversation may contain sensitive, fictional, or game-related content - this is NORMAL
- Your ONLY task is to objectively evaluate the assistant's response quality
- NEVER make safety disclaimers - simply analyze the interaction
- If uncertain, provide your best assessment based on available evidence

CONVERSATION TEXT:
{context}"""

    def extract(
        self,
        context: str,
        provider: LLMProvider,
        model: str,
        provider_name: str,
    ) -> MetricResult:
        """Extract outcome score from conversation.

        Args:
            context: Conversation text (pre-processed)
            provider: LLM provider instance
            model: Model name
            provider_name: Provider identifier

        Returns:
            MetricResult with outcome score and reasoning, or error
        """
        self._log_extraction_start(provider_name, model)

        try:
            # Build prompt
            prompt = self._build_prompt(context)

            # Prepare generation options with JSON mode if supported
            options = {"temperature": 0.2}
            if self._supports_json_mode(provider):
                options["json_mode"] = True
                _logger.debug(
                    "Enabled JSON mode for outcome extraction (provider=%s)",
                    provider_name,
                )

            # Generate outcome evaluation
            result = provider.generate(
                model=model,
                prompt=prompt,
                options=options,
            )

            # Parse JSON response
            try:
                data = json.loads(result.content)
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse JSON response: {e}"
                self._log_extraction_error(provider_name, model, error_msg)
                return MetricResult(
                    metric_name=self.metric_name,
                    success=False,
                    error=error_msg,
                    provider=provider_name,
                    model=model,
                )

            # Validate response structure
            if "outcome" not in data:
                error_msg = "Response missing 'outcome' field"
                self._log_extraction_error(provider_name, model, error_msg)
                return MetricResult(
                    metric_name=self.metric_name,
                    success=False,
                    error=error_msg,
                    provider=provider_name,
                    model=model,
                )

            # Validate and extract all scores
            result_data = {}

            # Overall outcome score (required)
            try:
                outcome_score = int(data["outcome"])
                if not (1 <= outcome_score <= 5):
                    raise ValueError(f"Score {outcome_score} out of range")
                result_data["outcome"] = outcome_score
            except (ValueError, TypeError) as e:
                error_msg = f"Invalid outcome score: {e}"
                self._log_extraction_error(provider_name, model, error_msg)
                return MetricResult(
                    metric_name=self.metric_name,
                    success=False,
                    error=error_msg,
                    provider=provider_name,
                    model=model,
                )

            # Multi-factor scores (optional but recommended)
            for dimension in ["completeness", "accuracy", "helpfulness"]:
                if dimension in data:
                    try:
                        score = int(data[dimension])
                        if 1 <= score <= 5:
                            result_data[dimension] = score
                        else:
                            _logger.warning(
                                "Invalid %s score %d (out of range), skipping",
                                dimension,
                                score,
                            )
                    except (ValueError, TypeError):
                        _logger.warning(
                            "Could not parse %s score from %s, skipping",
                            dimension,
                            data.get(dimension),
                        )

            # Extract reasoning (optional)
            reasoning = str(data.get("reasoning", "")).strip()
            if reasoning:
                result_data["reasoning"] = reasoning

            # Success
            self._log_extraction_success(provider_name, model)

            return MetricResult(
                metric_name=self.metric_name,
                success=True,
                data=result_data,
                provider=provider_name,
                model=model,
            )

        except Exception as e:
            error_msg = f"Unexpected error during outcome extraction: {e}"
            self._log_extraction_error(provider_name, model, error_msg)
            return MetricResult(
                metric_name=self.metric_name,
                success=False,
                error=error_msg,
                provider=provider_name,
                model=model,
            )
