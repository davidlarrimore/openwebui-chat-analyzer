"""Multi-metric extraction system for conversation analysis.

This package provides specialized metric extractors for different aspects
of conversation analysis, implementing the "LLM as a Judge" pattern with
purpose-built prompts for each metric type.

Architecture:
- Each metric has its own extractor class derived from MetricExtractor
- Separate LLM calls with specialized prompts per metric
- Structured JSON outputs with provider-specific JSON mode support
- Orchestration layer handles sequential extraction with failure recovery
- Selective execution allows users to choose which metrics to run

Available Metrics:
- Summary: One-line conversation description
- Outcome: 1-5 rating with reasoning for conversation success
- Tags: Topic tags and categorical labels
- Classification: Domain type and resolution status

Sprint 2 Implementation: Multi-Metric Extraction Architecture
"""

from backend.metrics.base import MetricExtractor, MetricResult

__all__ = ["MetricExtractor", "MetricResult"]
