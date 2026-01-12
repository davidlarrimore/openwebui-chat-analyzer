"""Sprint 6: Integration tests for end-to-end summarization pipeline.

Tests the complete flow from conversation input through metric extraction
to storage, verifying all components work together correctly.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from backend.metrics.base import MetricResult
from backend.summarizer import extract_metrics, extract_and_store_metrics
from backend.storage import DatabaseStorage


class TestEndToEndSummarization:
    """Integration tests for complete summarization pipeline."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.supports_json_mode.return_value = True
        return provider

    @pytest.fixture
    def sample_conversation(self) -> str:
        """Sample conversation for testing."""
        return """User: I need help with Python async programming. How do I handle errors in async functions?
Assistant: In Python async programming, you handle errors using try/except blocks around your await statements. Here's a complete example:

```python
async def fetch_data():
    try:
        result = await some_async_operation()
        return result
    except ConnectionError as e:
        print(f"Connection failed: {e}")
        return None
    except asyncio.TimeoutError:
        print("Operation timed out")
        return None
```

You should wrap await calls in try/except to catch specific exceptions. Common patterns include retrying failed operations and implementing timeouts with asyncio.wait_for().

User: That's helpful! What about handling multiple concurrent operations?
Assistant: For multiple concurrent operations, use asyncio.gather() or asyncio.create_task(). Here's how:

```python
async def main():
    tasks = [
        fetch_data("api1"),
        fetch_data("api2"),
        fetch_data("api3")
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

This runs all operations concurrently and collects results. Use return_exceptions=True to continue even if some tasks fail.

User: Perfect, thanks!"""

    @pytest.fixture
    def sample_messages(self) -> List[Dict[str, Any]]:
        """Sample message list for testing."""
        return [
            {"role": "user", "content": "I need help with Python async programming. How do I handle errors in async functions?"},
            {"role": "assistant", "content": "In Python async programming, you handle errors using try/except blocks around your await statements. Here's a complete example:\n\n```python\nasync def fetch_data():\n    try:\n        result = await some_async_operation()\n        return result\n    except ConnectionError as e:\n        print(f\"Connection failed: {e}\")\n        return None\n    except asyncio.TimeoutError:\n        print(\"Operation timed out\")\n        return None\n```\n\nYou should wrap await calls in try/except to catch specific exceptions. Common patterns include retrying failed operations and implementing timeouts with asyncio.wait_for()."},
            {"role": "user", "content": "That's helpful! What about handling multiple concurrent operations?"},
            {"role": "assistant", "content": "For multiple concurrent operations, use asyncio.gather() or asyncio.create_task(). Here's how:\n\n```python\nasync def main():\n    tasks = [\n        fetch_data(\"api1\"),\n        fetch_data(\"api2\"),\n        fetch_data(\"api3\")\n    ]\n    results = await asyncio.gather(*tasks, return_exceptions=True)\n    return results\n```\n\nThis runs all operations concurrently and collects results. Use return_exceptions=True to continue even if some tasks fail."},
            {"role": "user", "content": "Perfect, thanks!"}
        ]

    @pytest.mark.asyncio
    async def test_extract_all_metrics_success(
        self, mock_provider, sample_conversation, sample_messages
    ):
        """Test extracting all metrics successfully."""
        # Mock provider responses for each metric
        mock_provider.generate.side_effect = [
            '{"summary": "Python async error handling discussion"}',  # summary
            '{"outcome_score": 5, "completeness": 5, "accuracy": 5, "helpfulness": 5, "reasoning": "Complete explanation with code examples"}',  # outcome
            '{"tags": ["python", "async", "error-handling", "asyncio"]}',  # tags
            '{"domain": "technical", "interaction_type": "qa", "resolution_status": "resolved"}',  # classification
        ]

        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            # Extract all metrics
            results = await extract_metrics(
                conversation_text=sample_conversation,
                messages=sample_messages,
                metrics=["summary", "outcome", "tags", "classification"]
            )

        # Verify all metrics extracted
        assert len(results) == 4
        assert "summary" in results
        assert "outcome" in results
        assert "tags" in results
        assert "classification" in results

        # Verify summary
        summary_result = results["summary"]
        assert summary_result.success is True
        assert summary_result.value == "Python async error handling discussion"
        assert summary_result.error is None

        # Verify outcome
        outcome_result = results["outcome"]
        assert outcome_result.success is True
        assert outcome_result.value["outcome_score"] == 5
        assert outcome_result.value["completeness"] == 5

        # Verify tags
        tags_result = results["tags"]
        assert tags_result.success is True
        assert "python" in tags_result.value["tags"]
        assert "async" in tags_result.value["tags"]

        # Verify classification
        class_result = results["classification"]
        assert class_result.success is True
        assert class_result.value["domain"] == "technical"
        assert class_result.value["resolution_status"] == "resolved"

    @pytest.mark.asyncio
    async def test_extract_selective_metrics(
        self, mock_provider, sample_conversation, sample_messages
    ):
        """Test extracting only specific metrics."""
        # Mock responses for only summary and tags
        mock_provider.generate.side_effect = [
            '{"summary": "Python async discussion"}',  # summary
            '{"tags": ["python", "async"]}',  # tags
        ]

        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            results = await extract_metrics(
                conversation_text=sample_conversation,
                messages=sample_messages,
                metrics=["summary", "tags"]
            )

        # Should only have 2 metrics
        assert len(results) == 2
        assert "summary" in results
        assert "tags" in results
        assert "outcome" not in results
        assert "classification" not in results

    @pytest.mark.asyncio
    async def test_extract_metrics_partial_failure(
        self, mock_provider, sample_conversation, sample_messages
    ):
        """Test graceful degradation when some metrics fail."""
        # First metric succeeds, second fails, third succeeds
        mock_provider.generate.side_effect = [
            '{"summary": "Python async discussion"}',  # summary - success
            Exception("Provider timeout"),  # outcome - failure
            '{"tags": ["python", "async"]}',  # tags - success
        ]

        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            results = await extract_metrics(
                conversation_text=sample_conversation,
                messages=sample_messages,
                metrics=["summary", "outcome", "tags"]
            )

        # Should have all 3 results
        assert len(results) == 3

        # Summary should succeed
        assert results["summary"].success is True
        assert results["summary"].value == "Python async discussion"

        # Outcome should fail
        assert results["outcome"].success is False
        assert "Provider timeout" in results["outcome"].error

        # Tags should succeed
        assert results["tags"].success is True
        assert "python" in results["tags"].value["tags"]

    @pytest.mark.asyncio
    async def test_extract_and_store_metrics(
        self, mock_provider, sample_conversation, sample_messages, tmp_path
    ):
        """Test complete pipeline with storage."""
        # Create temporary database
        db_path = tmp_path / "test.db"
        storage = DatabaseStorage(str(db_path))

        # Create a test chat
        chat_id = "test-chat-123"
        storage.db_session.execute(
            """
            INSERT INTO chats (id, title, created_at, updated_at)
            VALUES (?, ?, datetime('now'), datetime('now'))
            """,
            (chat_id, "Test Chat")
        )
        storage.db_session.commit()

        # Mock provider responses
        mock_provider.generate.side_effect = [
            '{"summary": "Python async error handling"}',
            '{"outcome_score": 5, "completeness": 5, "accuracy": 5, "helpfulness": 5, "reasoning": "Complete answer"}',
            '{"tags": ["python", "async"]}',
            '{"domain": "technical", "interaction_type": "qa", "resolution_status": "resolved"}',
        ]

        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            with patch("backend.summarizer._get_storage", return_value=storage):
                # Extract and store
                results = await extract_and_store_metrics(
                    chat_id=chat_id,
                    conversation_text=sample_conversation,
                    messages=sample_messages,
                    metrics=["summary", "outcome", "tags", "classification"]
                )

        # Verify results
        assert len(results) == 4
        assert all(r.success for r in results.values())

        # Verify storage
        chat_metrics = storage.get_chat_metrics(chat_id)
        assert chat_metrics is not None
        assert "metrics" in chat_metrics
        assert chat_metrics["metrics"]["summary"] == "Python async error handling"
        assert chat_metrics["metrics"]["outcome"] == 5
        assert "python" in chat_metrics["metrics"]["tags"]
        assert chat_metrics["metrics"]["domain"] == "technical"

        # Verify backward compatibility fields
        chat = storage.get_chat(chat_id)
        assert chat.gen_chat_summary == "Python async error handling"
        assert chat.gen_chat_outcome == 5

    @pytest.mark.asyncio
    async def test_retry_on_json_parse_failure(
        self, mock_provider, sample_conversation, sample_messages
    ):
        """Test retry logic when JSON parsing fails."""
        # First call returns invalid JSON, second call succeeds
        mock_provider.generate.side_effect = [
            'This is not valid JSON',  # First attempt - parse error
            '{"summary": "Valid summary after retry"}',  # Second attempt - success
        ]

        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            results = await extract_metrics(
                conversation_text=sample_conversation,
                messages=sample_messages,
                metrics=["summary"]
            )

        # Should eventually succeed after retry
        assert results["summary"].success is True
        assert results["summary"].value == "Valid summary after retry"

        # Should have called generate twice (initial + 1 retry)
        assert mock_provider.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_quality_validation_integration(
        self, mock_provider, sample_conversation, sample_messages
    ):
        """Test quality validation is applied to summaries."""
        # Return a summary with good keyword overlap
        mock_provider.generate.return_value = '{"summary": "Python async error handling with try except blocks"}'

        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            results = await extract_metrics(
                conversation_text=sample_conversation,
                messages=sample_messages,
                metrics=["summary"]
            )

        summary_result = results["summary"]
        assert summary_result.success is True

        # If quality validation is enabled, metadata should include quality metrics
        if "quality_score" in summary_result.metadata:
            assert summary_result.metadata["quality_score"] > 0.0
            assert summary_result.metadata["quality_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_dropoff_detection_integration(
        self, mock_provider, sample_messages
    ):
        """Test conversation drop-off detection integration."""
        # Conversation that ends with unanswered question
        incomplete_messages = [
            {"role": "user", "content": "How do I use Python async?"},
            {"role": "assistant", "content": "Use async/await keywords."},
            {"role": "user", "content": "Can you show me an example?"},
            # No response - conversation dropped off
        ]

        incomplete_conv = "User: How do I use Python async?\nAssistant: Use async/await keywords.\nUser: Can you show me an example?"

        mock_provider.generate.return_value = '{"outcome_score": 2, "completeness": 2, "accuracy": 5, "helpfulness": 3, "reasoning": "Question left unanswered"}'

        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            results = await extract_metrics(
                conversation_text=incomplete_conv,
                messages=incomplete_messages,
                metrics=["outcome"]
            )

        outcome_result = results["outcome"]
        assert outcome_result.success is True
        # Low outcome score due to incompleteness
        assert outcome_result.value["outcome_score"] <= 3
        assert outcome_result.value["completeness"] <= 3


class TestProviderCompatibility:
    """Test integration with different provider types."""

    @pytest.mark.asyncio
    async def test_ollama_provider_json_format(self, sample_conversation, sample_messages):
        """Test Ollama provider with format='json' parameter."""
        from backend.providers.ollama import OllamaProvider

        provider = OllamaProvider()

        # Mock the requests to Ollama
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "response": '{"summary": "Test summary"}'
            }

            with patch("backend.summarizer._get_provider", return_value=provider):
                results = await extract_metrics(
                    conversation_text=sample_conversation,
                    messages=sample_messages,
                    metrics=["summary"]
                )

            # Verify format='json' was passed to Ollama
            call_kwargs = mock_post.call_args[1]["json"]
            assert call_kwargs.get("format") == "json"

    @pytest.mark.asyncio
    async def test_openai_provider_response_format(self, sample_conversation, sample_messages):
        """Test OpenAI provider with response_format parameter."""
        from backend.providers.openai import OpenAIProvider

        provider = OpenAIProvider()

        # Mock OpenAI client
        with patch("backend.providers.openai.OpenAI") as mock_openai_class:
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content='{"summary": "Test summary"}'))
            ]
            mock_client.chat.completions.create.return_value = mock_response

            with patch("backend.summarizer._get_provider", return_value=provider):
                results = await extract_metrics(
                    conversation_text=sample_conversation,
                    messages=sample_messages,
                    metrics=["summary"]
                )

            # Verify response_format was passed
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs.get("response_format") == {"type": "json_object"}


class TestMonitoringIntegration:
    """Test monitoring and metrics collection integration."""

    @pytest.mark.asyncio
    async def test_monitoring_records_extraction(
        self, mock_provider, sample_conversation, sample_messages
    ):
        """Test that monitoring records extraction metrics."""
        from backend.monitoring import MetricsCollector

        collector = MetricsCollector()
        collector.clear()  # Reset before test

        mock_provider.generate.return_value = '{"summary": "Test summary"}'

        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            await extract_metrics(
                conversation_text=sample_conversation,
                messages=sample_messages,
                metrics=["summary"]
            )

        # Verify monitoring recorded the extraction
        stats = collector.get_overall_stats()
        assert stats.total_extractions > 0

    @pytest.mark.asyncio
    async def test_monitoring_records_failures(
        self, mock_provider, sample_conversation, sample_messages
    ):
        """Test that monitoring records extraction failures."""
        from backend.monitoring import MetricsCollector

        collector = MetricsCollector()
        collector.clear()

        mock_provider.generate.side_effect = Exception("Test error")

        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            results = await extract_metrics(
                conversation_text=sample_conversation,
                messages=sample_messages,
                metrics=["summary"]
            )

        # Should record the failure
        assert results["summary"].success is False

        stats = collector.get_overall_stats()
        assert stats.total_failures > 0

        # Check recent failures
        recent_failures = collector.get_recent_failures(limit=10)
        assert len(recent_failures) > 0
        assert any("Test error" in f.error_message for f in recent_failures)