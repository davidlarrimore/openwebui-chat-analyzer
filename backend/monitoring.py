"""Sprint 5: Advanced monitoring and observability for summarizer operations.

This module provides:
- Metrics collection and aggregation
- Performance tracking (latency, token usage)
- Failure tracking and analysis
- Structured logging for debugging
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Deque

from backend.config import SUMMARIZER_ENABLE_DETAILED_LOGGING, SUMMARIZER_LOG_RETENTION_HOURS

LOGGER = logging.getLogger(__name__)


@dataclass
class MetricExtractionLog:
    """Log entry for a single metric extraction attempt."""

    timestamp: str
    chat_id: str
    metric_name: str
    provider: str
    model: str
    success: bool
    latency_ms: float
    token_usage: Optional[int] = None
    error: Optional[str] = None
    prompt_length: Optional[int] = None
    response_length: Optional[int] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class MetricStats:
    """Statistics for a single metric type."""

    total_attempts: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_retries: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.total_successes / self.total_attempts if self.total_attempts > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        return self.total_latency_ms / self.total_attempts if self.total_attempts > 0 else 0.0

    @property
    def avg_tokens(self) -> float:
        """Calculate average token usage."""
        return self.total_tokens / self.total_successes if self.total_successes > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_attempts": self.total_attempts,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_tokens": self.avg_tokens if self.total_tokens > 0 else None,
            "total_retries": self.total_retries,
        }


class MetricsCollector:
    """Collects and aggregates metrics for summarizer operations.

    Thread-safe singleton for collecting metrics across all summarizer operations.
    Maintains in-memory statistics and recent failure logs.
    """

    _instance: Optional["MetricsCollector"] = None
    _lock = Lock()

    def __new__(cls) -> "MetricsCollector":
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize metrics collector."""
        if self._initialized:
            return

        self._stats_lock = Lock()
        self._logs_lock = Lock()

        # Per-metric statistics
        self._metric_stats: Dict[str, MetricStats] = defaultdict(MetricStats)

        # Recent logs (fixed-size circular buffer)
        self._recent_logs: Deque[MetricExtractionLog] = deque(maxlen=1000)

        # Recent failures only (for debugging)
        self._recent_failures: Deque[MetricExtractionLog] = deque(maxlen=200)

        # Detailed logging configuration
        self._detailed_logging_enabled = SUMMARIZER_ENABLE_DETAILED_LOGGING
        self._log_file: Optional[Path] = None

        if self._detailed_logging_enabled:
            self._setup_detailed_logging()

        self._initialized = True
        LOGGER.info("MetricsCollector initialized (detailed_logging=%s)", self._detailed_logging_enabled)

    def _setup_detailed_logging(self) -> None:
        """Setup detailed logging to separate file."""
        try:
            log_dir = Path("logs/summarizer")
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d")
            self._log_file = log_dir / f"summarizer_{timestamp}.jsonl"

            LOGGER.info("Detailed summarizer logging enabled: %s", self._log_file)
        except Exception as exc:
            LOGGER.warning("Failed to setup detailed logging: %s", exc)
            self._detailed_logging_enabled = False

    def record_extraction(
        self,
        chat_id: str,
        metric_name: str,
        provider: str,
        model: str,
        success: bool,
        latency_ms: float,
        token_usage: Optional[int] = None,
        error: Optional[str] = None,
        prompt_length: Optional[int] = None,
        response_length: Optional[int] = None,
        retry_count: int = 0,
    ) -> None:
        """Record a metric extraction attempt.

        Args:
            chat_id: Chat identifier
            metric_name: Name of metric being extracted
            provider: LLM provider used
            model: Model name
            success: Whether extraction succeeded
            latency_ms: Time taken in milliseconds
            token_usage: Tokens used (if available)
            error: Error message if failed
            prompt_length: Length of prompt in characters
            response_length: Length of response in characters
            retry_count: Number of retries before success/failure
        """
        log_entry = MetricExtractionLog(
            timestamp=datetime.utcnow().isoformat(),
            chat_id=chat_id,
            metric_name=metric_name,
            provider=provider,
            model=model,
            success=success,
            latency_ms=latency_ms,
            token_usage=token_usage,
            error=error,
            prompt_length=prompt_length,
            response_length=response_length,
            retry_count=retry_count,
        )

        # Update statistics
        with self._stats_lock:
            stats = self._metric_stats[metric_name]
            stats.total_attempts += 1
            if success:
                stats.total_successes += 1
            else:
                stats.total_failures += 1
            stats.total_latency_ms += latency_ms
            if token_usage:
                stats.total_tokens += token_usage
            stats.total_retries += retry_count

        # Store log entry
        with self._logs_lock:
            self._recent_logs.append(log_entry)
            if not success:
                self._recent_failures.append(log_entry)

        # Write to detailed log file
        if self._detailed_logging_enabled and self._log_file:
            try:
                with open(self._log_file, "a") as f:
                    json.dump(log_entry.to_dict(), f)
                    f.write("\n")
            except Exception as exc:
                LOGGER.warning("Failed to write detailed log: %s", exc)

    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall statistics across all metrics."""
        with self._stats_lock:
            total_attempts = sum(s.total_attempts for s in self._metric_stats.values())
            total_successes = sum(s.total_successes for s in self._metric_stats.values())
            total_failures = sum(s.total_failures for s in self._metric_stats.values())
            total_latency = sum(s.total_latency_ms for s in self._metric_stats.values())
            total_tokens = sum(s.total_tokens for s in self._metric_stats.values())
            total_retries = sum(s.total_retries for s in self._metric_stats.values())

            return {
                "total_attempts": total_attempts,
                "total_successes": total_successes,
                "total_failures": total_failures,
                "success_rate": total_successes / total_attempts if total_attempts > 0 else 0.0,
                "avg_latency_ms": total_latency / total_attempts if total_attempts > 0 else 0.0,
                "total_tokens": total_tokens if total_tokens > 0 else None,
                "total_retries": total_retries,
            }

    def get_metric_stats(self, metric_name: str) -> Dict[str, Any]:
        """Get statistics for a specific metric."""
        with self._stats_lock:
            stats = self._metric_stats.get(metric_name)
            if not stats:
                return {
                    "total_attempts": 0,
                    "total_successes": 0,
                    "total_failures": 0,
                    "success_rate": 0.0,
                    "avg_latency_ms": 0.0,
                }
            return stats.to_dict()

    def get_all_metric_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all metrics."""
        with self._stats_lock:
            return {
                metric_name: stats.to_dict()
                for metric_name, stats in self._metric_stats.items()
            }

    def get_recent_failures(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent failure logs.

        Args:
            limit: Maximum number of failures to return

        Returns:
            List of failure log dictionaries
        """
        with self._logs_lock:
            failures = list(self._recent_failures)[-limit:]
            return [log.to_dict() for log in failures]

    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent logs (all attempts).

        Args:
            limit: Maximum number of logs to return

        Returns:
            List of log dictionaries
        """
        with self._logs_lock:
            logs = list(self._recent_logs)[-limit:]
            return [log.to_dict() for log in logs]

    def reset_stats(self) -> None:
        """Reset all statistics (for testing)."""
        with self._stats_lock:
            self._metric_stats.clear()
        with self._logs_lock:
            self._recent_logs.clear()
            self._recent_failures.clear()
        LOGGER.info("MetricsCollector statistics reset")

    def export_logs(self, output_path: str) -> None:
        """Export all recent logs to a file.

        Args:
            output_path: Path to output file
        """
        with self._logs_lock:
            logs = [log.to_dict() for log in self._recent_logs]

        try:
            with open(output_path, "w") as f:
                json.dump(logs, f, indent=2)
            LOGGER.info("Exported %d logs to %s", len(logs), output_path)
        except Exception as exc:
            LOGGER.error("Failed to export logs: %s", exc)
            raise


# Global singleton instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
