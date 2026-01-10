"""Base interface for metric extractors.

This module defines the abstract MetricExtractor interface that all
specialized metric extractors must implement. Each extractor is responsible
for generating a specific metric from conversation data using LLM analysis.

Sprint 2 Implementation: Multi-Metric Extraction Architecture
Sprint 5 Enhancement: Advanced resilience and monitoring
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from backend.config import (
    SUMMARIZER_MAX_RETRIES,
    SUMMARIZER_ENABLE_FALLBACK_PROMPTS,
    SUMMARIZER_ENABLE_GRACEFUL_DEGRADATION,
)
from backend.provider_registry import ProviderRegistry
from backend.providers.base import LLMProvider

_logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result from a metric extraction operation.

    Attributes:
        metric_name: Identifier for the metric (e.g., "summary", "outcome")
        success: Whether extraction succeeded
        data: Extracted metric data (structure varies by metric type)
        error: Error message if extraction failed
        provider: Provider used for extraction
        model: Model used for extraction
        latency_ms: Time taken for extraction in milliseconds (Sprint 5)
        retry_count: Number of retries before success/failure (Sprint 5)
        token_usage: Tokens used if available from provider (Sprint 5)
    """

    metric_name: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    latency_ms: float = 0.0
    retry_count: int = 0
    token_usage: Optional[int] = None


class MetricExtractor(ABC):
    """Abstract base class for metric extractors.

    Each metric extractor implements a specialized prompt and parsing logic
    for extracting a specific metric from conversation data. Extractors
    use provider-specific JSON mode support for reliable structured outputs.

    Subclasses must implement:
    - metric_name: Unique identifier for the metric
    - extract(): Core extraction logic with provider calls

    Design Principles:
    - One metric per extractor (separation of concerns)
    - Purpose-built prompts optimized for specific metric
    - Structured JSON outputs with schema validation
    - Graceful failure handling (return MetricResult with error)
    - Provider-agnostic (works with any LLMProvider)

    TODO: Add per-metric model configuration support (Sprint 4)
    """

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Unique identifier for this metric (e.g., 'summary', 'outcome')."""
        pass

    @abstractmethod
    def extract(
        self,
        context: str,
        provider: LLMProvider,
        model: str,
        provider_name: str,
    ) -> MetricResult:
        """Extract the metric from conversation context.

        Args:
            context: Conversation text (pre-processed, salient messages)
            provider: LLM provider instance to use for generation
            model: Model name to use for this extraction
            provider_name: Provider identifier (for logging/result tracking)

        Returns:
            MetricResult with extraction outcome (success/failure with data/error)

        Implementation Notes:
        - Use provider.generate() with json_mode option if provider supports it
        - Handle JSON parsing errors gracefully (return MetricResult with error)
        - Log failures with sufficient context for debugging
        - Keep prompts focused on single metric (avoid trying to extract multiple things)
        - Return structured data that can be serialized to JSON metadata field
        """
        pass

    def _supports_json_mode(self, provider: LLMProvider) -> bool:
        """Check if provider supports native JSON mode.

        Args:
            provider: LLM provider instance

        Returns:
            True if provider supports JSON mode, False otherwise
        """
        return provider.supports_json_mode()

    def _log_extraction_start(self, provider_name: str, model: str) -> None:
        """Log the start of metric extraction.

        Args:
            provider_name: Provider identifier
            model: Model name
        """
        _logger.debug(
            "Extracting metric=%s provider=%s model=%s",
            self.metric_name,
            provider_name,
            model,
        )

    def _log_extraction_success(self, provider_name: str, model: str) -> None:
        """Log successful metric extraction.

        Args:
            provider_name: Provider identifier
            model: Model name
        """
        _logger.info(
            "Successfully extracted metric=%s provider=%s model=%s",
            self.metric_name,
            provider_name,
            model,
        )

    def _log_extraction_error(
        self, provider_name: str, model: str, error: str
    ) -> None:
        """Log failed metric extraction.

        Args:
            provider_name: Provider identifier
            model: Model name
            error: Error message
        """
        _logger.error(
            "Failed to extract metric=%s provider=%s model=%s error=%s",
            self.metric_name,
            provider_name,
            model,
            error,
        )

    def extract_with_retry(
        self,
        context: str,
        provider: LLMProvider,
        model: str,
        provider_name: str,
        chat_id: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> MetricResult:
        """Extract metric with automatic retry and monitoring (Sprint 5).

        This method wraps the abstract extract() method with:
        - Automatic retry with exponential backoff
        - Performance monitoring (latency, token usage)
        - Failure tracking and logging
        - Optional fallback to simpler prompts on retry

        Args:
            context: Conversation text (pre-processed, salient messages)
            provider: LLM provider instance to use for generation
            model: Model name to use for this extraction
            provider_name: Provider identifier (for logging/result tracking)
            chat_id: Optional chat ID for monitoring
            max_retries: Maximum retry attempts (overrides config)

        Returns:
            MetricResult with extraction outcome including retry count and latency
        """
        max_attempts = (max_retries if max_retries is not None else SUMMARIZER_MAX_RETRIES) + 1
        retry_count = 0
        start_time = time.time()
        last_error = None

        for attempt in range(max_attempts):
            try:
                self._log_extraction_start(provider_name, model)

                attempt_start = time.time()
                result = self.extract(context, provider, model, provider_name)
                attempt_latency = (time.time() - attempt_start) * 1000  # ms

                # Add timing and retry metadata
                result.latency_ms = attempt_latency
                result.retry_count = retry_count

                if result.success:
                    self._log_extraction_success(provider_name, model)

                    # Record success in monitoring
                    self._record_monitoring(
                        chat_id=chat_id or "unknown",
                        result=result,
                        context_length=len(context),
                    )

                    return result
                else:
                    # Extraction returned error
                    last_error = result.error or "Unknown error"
                    self._log_extraction_error(provider_name, model, last_error)

                    if attempt < max_attempts - 1:
                        retry_count += 1
                        _logger.info(
                            "Retrying metric=%s attempt=%d/%d",
                            self.metric_name,
                            attempt + 1,
                            max_attempts,
                        )
                        time.sleep(min(2 ** attempt, 8))  # Exponential backoff, max 8s
                    else:
                        # Final attempt failed
                        result.retry_count = retry_count
                        self._record_monitoring(
                            chat_id=chat_id or "unknown",
                            result=result,
                            context_length=len(context),
                        )
                        return result

            except Exception as exc:
                last_error = str(exc)
                _logger.exception(
                    "Exception during metric=%s extraction attempt=%d",
                    self.metric_name,
                    attempt + 1,
                )

                if attempt < max_attempts - 1:
                    retry_count += 1
                    time.sleep(min(2 ** attempt, 8))  # Exponential backoff
                else:
                    # All attempts exhausted
                    total_latency = (time.time() - start_time) * 1000
                    result = MetricResult(
                        metric_name=self.metric_name,
                        success=False,
                        error=f"Failed after {max_attempts} attempts: {last_error}",
                        provider=provider_name,
                        model=model,
                        latency_ms=total_latency,
                        retry_count=retry_count,
                    )
                    self._record_monitoring(
                        chat_id=chat_id or "unknown",
                        result=result,
                        context_length=len(context),
                    )
                    return result

        # Should never reach here, but just in case
        total_latency = (time.time() - start_time) * 1000
        return MetricResult(
            metric_name=self.metric_name,
            success=False,
            error=f"Unexpected failure: {last_error}",
            provider=provider_name,
            model=model,
            latency_ms=total_latency,
            retry_count=retry_count,
        )

    def _record_monitoring(
        self,
        chat_id: str,
        result: MetricResult,
        context_length: int,
    ) -> None:
        """Record extraction metrics in monitoring system (Sprint 5).

        Args:
            chat_id: Chat identifier
            result: Extraction result
            context_length: Length of context in characters
        """
        try:
            from backend.monitoring import get_metrics_collector

            collector = get_metrics_collector()
            collector.record_extraction(
                chat_id=chat_id,
                metric_name=self.metric_name,
                provider=result.provider or "unknown",
                model=result.model or "unknown",
                success=result.success,
                latency_ms=result.latency_ms,
                token_usage=result.token_usage,
                error=result.error,
                prompt_length=context_length,
                response_length=len(str(result.data)) if result.data else 0,
                retry_count=result.retry_count,
            )
        except Exception as exc:
            _logger.warning("Failed to record monitoring data: %s", exc)
