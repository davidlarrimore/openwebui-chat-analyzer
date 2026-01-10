"""Base interface for metric extractors.

This module defines the abstract MetricExtractor interface that all
specialized metric extractors must implement. Each extractor is responsible
for generating a specific metric from conversation data using LLM analysis.

Sprint 2 Implementation: Multi-Metric Extraction Architecture
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
    """

    metric_name: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None


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
