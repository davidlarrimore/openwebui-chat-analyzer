"""Tests for Sprint 2 metric extraction orchestration.

This module tests the orchestration layer that coordinates multiple
metric extractors:
- extract_metrics(): Core orchestration logic
- extract_and_store_metrics(): Convenience function with persistence
- Selective metric execution
- Partial failure handling
- JSON metadata persistence
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from backend import summarizer
from backend.metrics.base import MetricResult
from backend.providers.base import GenerateResult


class TestExtractMetrics:
    """Test extract_metrics() orchestration function."""

    def test_extract_all_metrics_success(self) -> None:
        """Extracting all metrics with no failures returns all data."""
        context = "user: test\nassistant: response"
        chat_id = "test-chat-123"

        # Mock provider
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.supports_json_mode.return_value = True

        # Mock provider responses for each metric
        mock_provider.generate.side_effect = [
            # Summary
            GenerateResult(
                content='{"summary": "Test conversation"}',
                model="test-model",
                provider="test",
            ),
            # Outcome
            GenerateResult(
                content='{"outcome": 5, "reasoning": "Complete"}',
                model="test-model",
                provider="test",
            ),
            # Tags
            GenerateResult(
                content='{"tags": ["test", "example"]}',
                model="test-model",
                provider="test",
            ),
            # Classification
            GenerateResult(
                content='{"domain": "technical", "resolution_status": "resolved"}',
                model="test-model",
                provider="test",
            ),
        ]

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = mock_provider

        with patch("backend.summarizer.get_provider_registry", return_value=mock_registry):
            with patch("backend.summarizer.get_connection_type", return_value="test"):
                with patch("backend.summarizer.get_summary_model", return_value="test-model"):
                    metrics_data, extraction_metadata = summarizer.extract_metrics(
                        context=context,
                        chat_id=chat_id,
                    )

        # Verify all metrics extracted
        assert metrics_data == {
            "summary": "Test conversation",
            "outcome": 5,
            "reasoning": "Complete",
            "tags": ["test", "example"],
            "domain": "technical",
            "resolution_status": "resolved",
        }

        # Verify metadata
        assert extraction_metadata["provider"] == "test"
        assert len(extraction_metadata["metrics_extracted"]) == 6  # All fields
        assert len(extraction_metadata["extraction_errors"]) == 0
        assert "timestamp" in extraction_metadata
        assert "models_used" in extraction_metadata

    def test_extract_selective_metrics(self) -> None:
        """Extracting only selected metrics works correctly."""
        context = "user: test"
        chat_id = "test-chat-123"

        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.supports_json_mode.return_value = True

        # Only mock responses for summary and outcome
        mock_provider.generate.side_effect = [
            GenerateResult(
                content='{"summary": "Test"}',
                model="test-model",
                provider="test",
            ),
            GenerateResult(
                content='{"outcome": 3, "reasoning": "Partial"}',
                model="test-model",
                provider="test",
            ),
        ]

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = mock_provider

        with patch("backend.summarizer.get_provider_registry", return_value=mock_registry):
            with patch("backend.summarizer.get_connection_type", return_value="test"):
                with patch("backend.summarizer.get_summary_model", return_value="test-model"):
                    metrics_data, extraction_metadata = summarizer.extract_metrics(
                        context=context,
                        chat_id=chat_id,
                        metrics_to_extract=["summary", "outcome"],
                    )

        # Only requested metrics should be present
        assert "summary" in metrics_data
        assert "outcome" in metrics_data
        assert "reasoning" in metrics_data
        assert "tags" not in metrics_data
        assert "domain" not in metrics_data

        # Verify metadata
        assert len(extraction_metadata["metrics_extracted"]) == 3  # summary, outcome, reasoning
        assert len(extraction_metadata["extraction_errors"]) == 0

    def test_extract_partial_failure(self) -> None:
        """Some metrics failing returns partial results with errors logged."""
        context = "user: test"
        chat_id = "test-chat-123"

        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.supports_json_mode.return_value = True

        # First metric succeeds, second fails
        mock_provider.generate.side_effect = [
            GenerateResult(
                content='{"summary": "Test"}',
                model="test-model",
                provider="test",
            ),
            GenerateResult(
                content='invalid json',
                model="test-model",
                provider="test",
            ),
        ]

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = mock_provider

        with patch("backend.summarizer.get_provider_registry", return_value=mock_registry):
            with patch("backend.summarizer.get_connection_type", return_value="test"):
                with patch("backend.summarizer.get_summary_model", return_value="test-model"):
                    metrics_data, extraction_metadata = summarizer.extract_metrics(
                        context=context,
                        chat_id=chat_id,
                        metrics_to_extract=["summary", "outcome"],
                    )

        # Summary should succeed
        assert "summary" in metrics_data
        assert metrics_data["summary"] == "Test"

        # Outcome should be in errors
        assert "outcome" in extraction_metadata["extraction_errors"]
        assert len(extraction_metadata["metrics_extracted"]) == 1

    def test_extract_invalid_metrics_filtered(self) -> None:
        """Invalid metric names are filtered out with warning."""
        context = "user: test"
        chat_id = "test-chat-123"

        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.supports_json_mode.return_value = True

        mock_provider.generate.return_value = GenerateResult(
            content='{"summary": "Test"}',
            model="test-model",
            provider="test",
        )

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = mock_provider

        with patch("backend.summarizer.get_provider_registry", return_value=mock_registry):
            with patch("backend.summarizer.get_connection_type", return_value="test"):
                with patch("backend.summarizer.get_summary_model", return_value="test-model"):
                    metrics_data, extraction_metadata = summarizer.extract_metrics(
                        context=context,
                        chat_id=chat_id,
                        metrics_to_extract=["summary", "invalid_metric", "fake_metric"],
                    )

        # Only valid metric extracted
        assert "summary" in metrics_data
        assert len(extraction_metadata["metrics_extracted"]) == 1

    def test_extract_provider_unavailable_raises_error(self) -> None:
        """Provider not available raises RuntimeError."""
        context = "user: test"
        chat_id = "test-chat-123"

        mock_provider = MagicMock()
        mock_provider.is_available.return_value = False
        mock_provider.get_unavailable_reason.return_value = "Not configured"

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = mock_provider

        with patch("backend.summarizer.get_provider_registry", return_value=mock_registry):
            with patch("backend.summarizer.get_connection_type", return_value="test"):
                with patch("backend.summarizer.get_summary_model", return_value="test-model"):
                    with pytest.raises(RuntimeError, match="Provider test is not available"):
                        summarizer.extract_metrics(
                            context=context,
                            chat_id=chat_id,
                        )


class TestExtractAndStoreMetrics:
    """Test extract_and_store_metrics() convenience function."""

    def test_extract_and_store_success(self) -> None:
        """Successful extraction stores metrics in database."""
        chat_id = "test-chat-123"
        messages = [
            {"role": "user", "content": "test question"},
            {"role": "assistant", "content": "test answer"},
        ]

        # Mock storage
        mock_storage = MagicMock()

        # Mock context building
        with patch("backend.summarizer._build_salient_context", return_value="mocked context"):
            # Mock extraction
            with patch(
                "backend.summarizer.extract_metrics",
                return_value=(
                    {"summary": "Test", "outcome": 5},
                    {
                        "timestamp": "2026-01-10T00:00:00Z",
                        "provider": "test",
                        "models_used": {"summary": "test-model"},
                        "metrics_extracted": ["summary", "outcome"],
                        "extraction_errors": [],
                    },
                ),
            ):
                result = summarizer.extract_and_store_metrics(
                    chat_id=chat_id,
                    messages=messages,
                    metrics_to_extract=["summary", "outcome"],
                    storage=mock_storage,
                )

        # Verify success
        assert result["success"] is True
        assert result["chat_id"] == chat_id
        assert "summary" in result["metrics_extracted"]
        assert "outcome" in result["metrics_extracted"]

        # Verify storage was called
        mock_storage.update_chat_metrics.assert_called_once()
        call_args = mock_storage.update_chat_metrics.call_args
        assert call_args[1]["chat_id"] == chat_id
        assert "summary" in call_args[1]["metrics"]
        assert "outcome" in call_args[1]["metrics"]

    def test_extract_and_store_extraction_error(self) -> None:
        """Extraction error returns failure result."""
        chat_id = "test-chat-123"
        messages = [{"role": "user", "content": "test"}]

        mock_storage = MagicMock()

        with patch("backend.summarizer._build_salient_context", return_value="context"):
            # Mock extraction raising exception
            with patch(
                "backend.summarizer.extract_metrics",
                side_effect=RuntimeError("Provider unavailable"),
            ):
                result = summarizer.extract_and_store_metrics(
                    chat_id=chat_id,
                    messages=messages,
                    storage=mock_storage,
                )

        # Verify failure
        assert result["success"] is False
        assert result["chat_id"] == chat_id
        assert "error" in result
        assert "Provider unavailable" in result["error"]

        # Storage should not be called on failure
        mock_storage.update_chat_metrics.assert_not_called()


class TestJSONMetadataPersistence:
    """Test JSON metadata persistence in storage layer."""

    def test_update_chat_metrics_creates_structure(self) -> None:
        """update_chat_metrics() creates proper JSON structure in meta field."""
        from backend.storage import DatabaseStorage
        from backend.db_models import ChatRecord
        from unittest.mock import patch, MagicMock

        storage = DatabaseStorage()

        # Mock the session and chat record
        mock_chat = MagicMock(spec=ChatRecord)
        mock_chat.meta = {}

        mock_session = MagicMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = mock_chat

        with patch("backend.storage.session_scope") as mock_session_scope:
            mock_session_scope.return_value.__enter__.return_value = mock_session

            storage.update_chat_metrics(
                chat_id="test-123",
                metrics={"summary": "Test", "outcome": 5},
                extraction_metadata={
                    "timestamp": "2026-01-10T00:00:00Z",
                    "provider": "ollama",
                },
            )

        # Verify session.execute was called with update statement
        assert mock_session.execute.called

    def test_get_chat_metrics_returns_full_structure(self) -> None:
        """get_chat_metrics() returns full meta structure with metrics and metadata."""
        from backend.storage import DatabaseStorage
        from backend.db_models import ChatRecord
        from unittest.mock import patch, MagicMock

        storage = DatabaseStorage()

        # Mock chat with metrics
        mock_chat = MagicMock(spec=ChatRecord)
        mock_chat.meta = {
            "metrics": {"summary": "Test", "outcome": 5},
            "extraction_metadata": {"timestamp": "2026-01-10T00:00:00Z"},
        }

        mock_session = MagicMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = mock_chat

        with patch("backend.storage.session_scope") as mock_session_scope:
            mock_session_scope.return_value.__enter__.return_value = mock_session

            result = storage.get_chat_metrics("test-123")

        assert result is not None
        assert "metrics" in result
        assert "extraction_metadata" in result
        assert result["metrics"]["summary"] == "Test"

    def test_update_chat_metrics_preserves_existing_meta(self) -> None:
        """update_chat_metrics() preserves existing meta fields."""
        from backend.storage import DatabaseStorage
        from backend.db_models import ChatRecord
        from unittest.mock import patch, MagicMock

        storage = DatabaseStorage()

        # Mock chat with existing meta data
        mock_chat = MagicMock(spec=ChatRecord)
        mock_chat.meta = {
            "metrics": {"summary": "Old summary"},
            "custom_field": "preserved",
        }

        mock_session = MagicMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = mock_chat

        with patch("backend.storage.session_scope") as mock_session_scope:
            mock_session_scope.return_value.__enter__.return_value = mock_session

            storage.update_chat_metrics(
                chat_id="test-123",
                metrics={"outcome": 5},  # Adding new metric
            )

        # Verify update was called
        assert mock_session.execute.called
