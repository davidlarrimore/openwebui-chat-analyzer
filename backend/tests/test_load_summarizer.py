"""Sprint 6: Load tests for summarization pipeline performance.

Tests performance with large conversation datasets, measuring throughput,
latency, memory usage, and identifying bottlenecks.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest
import psutil
import os

from backend.summarizer import extract_metrics, extract_and_store_metrics
from backend.storage import DatabaseStorage


class TestLoadPerformance:
    """Load tests for high-volume summarization scenarios."""

    @pytest.fixture
    def generate_conversations(self) -> List[Dict[str, Any]]:
        """Generate a large dataset of synthetic conversations."""
        conversations = []

        # Mix of conversation types and lengths
        templates = [
            {
                "messages": [
                    {"role": "user", "content": "How do I {action} in {language}?"},
                    {"role": "assistant", "content": "To {action} in {language}, you use the {method} syntax. Here's an example: {code}"},
                    {"role": "user", "content": "Thanks!"}
                ],
                "variables": {
                    "action": ["loop", "sort", "filter", "map", "reduce"],
                    "language": ["Python", "JavaScript", "Go", "Rust"],
                    "method": ["for-in", "built-in function", "standard library"],
                    "code": ["code example here"]
                }
            },
            {
                "messages": [
                    {"role": "user", "content": "I'm getting an error: {error}"},
                    {"role": "assistant", "content": "This error occurs when {cause}. To fix it, {solution}."},
                    {"role": "user", "content": "That worked!"}
                ],
                "variables": {
                    "error": ["NullPointerException", "IndexError", "TypeError"],
                    "cause": ["variable is null", "index out of bounds", "wrong type"],
                    "solution": ["check for null first", "validate indices", "use type checking"]
                }
            },
            {
                "messages": [
                    {"role": "user", "content": "What's the difference between {concept1} and {concept2}?"},
                    {"role": "assistant", "content": "{concept1} is used for {use1}, while {concept2} is for {use2}. Key differences include: {differences}"}
                ],
                "variables": {
                    "concept1": ["async", "concurrency", "parallelism"],
                    "concept2": ["await", "threading", "multiprocessing"],
                    "use1": ["asynchronous operations", "concurrent execution"],
                    "use2": ["waiting for results", "parallel processing"],
                    "differences": ["execution model, resource usage, use cases"]
                }
            }
        ]

        # Generate 100 conversations from templates
        for i in range(100):
            template = templates[i % len(templates)]
            messages = []
            for msg in template["messages"]:
                content = msg["content"]
                for var, values in template["variables"].items():
                    if "{" + var + "}" in content:
                        content = content.replace("{" + var + "}", values[i % len(values)])
                messages.append({"role": msg["role"], "content": content})

            # Convert to conversation text
            conv_text = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in messages])

            conversations.append({
                "id": f"load-test-{i}",
                "messages": messages,
                "text": conv_text
            })

        return conversations

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider with realistic latency."""
        provider = MagicMock()
        provider.supports_json_mode.return_value = True

        # Simulate realistic LLM latency (50-200ms)
        def mock_generate(*args, **kwargs):
            time.sleep(0.05 + (hash(str(args)) % 150) / 1000)  # 50-200ms
            # Return mock responses based on metric type
            if "one-line summary" in args[0] or "summarize" in args[0].lower():
                return '{"summary": "Technical Q&A conversation"}'
            elif "outcome" in args[0].lower() or "score" in args[0].lower():
                return '{"outcome_score": 4, "completeness": 4, "accuracy": 4, "helpfulness": 4, "reasoning": "Good response"}'
            elif "tags" in args[0].lower():
                return '{"tags": ["technical", "programming", "qa"]}'
            else:
                return '{"domain": "technical", "interaction_type": "qa", "resolution_status": "resolved"}'

        provider.generate.side_effect = mock_generate
        return provider

    @pytest.mark.asyncio
    async def test_throughput_100_conversations(self, generate_conversations, mock_provider):
        """Test throughput with 100 conversations."""
        conversations = generate_conversations

        start_time = time.time()
        results = []

        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            # Process all conversations with only summary metric for speed
            for conv in conversations:
                result = await extract_metrics(
                    conversation_text=conv["text"],
                    messages=conv["messages"],
                    metrics=["summary"]
                )
                results.append(result)

        end_time = time.time()
        duration = end_time - start_time
        throughput = len(conversations) / duration

        # Performance assertions
        assert len(results) == 100
        assert all(r["summary"].success for r in results)

        print(f"\n=== Throughput Test Results ===")
        print(f"Conversations processed: {len(conversations)}")
        print(f"Total time: {duration:.2f}s")
        print(f"Throughput: {throughput:.2f} conversations/second")
        print(f"Average latency: {duration/len(conversations)*1000:.2f}ms per conversation")

        # Should process at least 5 conversations per second
        assert throughput > 5.0, f"Throughput too low: {throughput:.2f} conv/s"

    @pytest.mark.asyncio
    async def test_latency_per_metric(self, generate_conversations, mock_provider):
        """Measure latency for each metric type."""
        conversations = generate_conversations[:10]  # Use 10 conversations

        metrics_to_test = ["summary", "outcome", "tags", "classification"]
        latencies = {metric: [] for metric in metrics_to_test}

        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            for conv in conversations:
                for metric in metrics_to_test:
                    start = time.time()
                    result = await extract_metrics(
                        conversation_text=conv["text"],
                        messages=conv["messages"],
                        metrics=[metric]
                    )
                    latency = (time.time() - start) * 1000  # Convert to ms
                    latencies[metric].append(latency)

        print(f"\n=== Per-Metric Latency Results ===")
        for metric, times in latencies.items():
            avg_latency = sum(times) / len(times)
            min_latency = min(times)
            max_latency = max(times)
            print(f"{metric}:")
            print(f"  Avg: {avg_latency:.2f}ms")
            print(f"  Min: {min_latency:.2f}ms")
            print(f"  Max: {max_latency:.2f}ms")

            # Each metric should complete in under 500ms on average
            assert avg_latency < 500, f"{metric} latency too high: {avg_latency:.2f}ms"

    @pytest.mark.asyncio
    async def test_memory_usage_large_batch(self, generate_conversations, mock_provider, tmp_path):
        """Test memory usage during large batch processing."""
        conversations = generate_conversations

        # Create temporary database
        db_path = tmp_path / "load_test.db"
        storage = DatabaseStorage(str(db_path))

        # Create test chats
        for conv in conversations:
            storage.db_session.execute(
                """
                INSERT INTO chats (id, title, created_at, updated_at)
                VALUES (?, ?, datetime('now'), datetime('now'))
                """,
                (conv["id"], f"Test Chat {conv['id']}")
            )
        storage.db_session.commit()

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            with patch("backend.summarizer._get_storage", return_value=storage):
                # Process all conversations with storage
                for conv in conversations:
                    await extract_and_store_metrics(
                        chat_id=conv["id"],
                        conversation_text=conv["text"],
                        messages=conv["messages"],
                        metrics=["summary"]
                    )

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        memory_per_conv = memory_increase / len(conversations)

        print(f"\n=== Memory Usage Results ===")
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")
        print(f"Memory per conversation: {memory_per_conv:.2f} MB")

        # Memory increase should be reasonable (< 100MB for 100 conversations)
        assert memory_increase < 100, f"Memory usage too high: {memory_increase:.2f} MB"

    @pytest.mark.asyncio
    async def test_concurrent_extraction_performance(self, generate_conversations, mock_provider):
        """Test performance with concurrent metric extraction."""
        conversations = generate_conversations[:20]  # Use 20 conversations

        # Sequential processing
        start_sequential = time.time()
        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            for conv in conversations:
                await extract_metrics(
                    conversation_text=conv["text"],
                    messages=conv["messages"],
                    metrics=["summary"]
                )
        sequential_time = time.time() - start_sequential

        # Concurrent processing (process multiple conversations in parallel)
        start_concurrent = time.time()
        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            tasks = []
            for conv in conversations:
                task = extract_metrics(
                    conversation_text=conv["text"],
                    messages=conv["messages"],
                    metrics=["summary"]
                )
                tasks.append(task)
            await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_concurrent

        speedup = sequential_time / concurrent_time

        print(f"\n=== Concurrency Performance Results ===")
        print(f"Sequential time: {sequential_time:.2f}s")
        print(f"Concurrent time: {concurrent_time:.2f}s")
        print(f"Speedup: {speedup:.2f}x")

        # Concurrent should be faster than sequential
        assert concurrent_time < sequential_time, "Concurrent processing should be faster"

        # Should see at least 2x speedup with concurrency
        assert speedup > 2.0, f"Speedup too low: {speedup:.2f}x"

    @pytest.mark.asyncio
    async def test_retry_overhead(self, generate_conversations, mock_provider):
        """Test overhead of retry logic with failures."""
        conversations = generate_conversations[:10]

        # Test without retries (all succeed)
        mock_provider.generate.side_effect = None
        mock_provider.generate.return_value = '{"summary": "Test summary"}'

        start_no_retry = time.time()
        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            for conv in conversations:
                await extract_metrics(
                    conversation_text=conv["text"],
                    messages=conv["messages"],
                    metrics=["summary"]
                )
        time_no_retry = time.time() - start_no_retry

        # Test with retries (first attempt fails, second succeeds)
        mock_provider.generate.side_effect = [
            "Invalid JSON",  # First attempt fails
            '{"summary": "Test summary"}',  # Second attempt succeeds
        ] * 10  # Repeat for all conversations

        start_with_retry = time.time()
        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            for conv in conversations:
                await extract_metrics(
                    conversation_text=conv["text"],
                    messages=conv["messages"],
                    metrics=["summary"]
                )
        time_with_retry = time.time() - start_with_retry

        retry_overhead = time_with_retry - time_no_retry
        overhead_percent = (retry_overhead / time_no_retry) * 100

        print(f"\n=== Retry Overhead Results ===")
        print(f"Time without retries: {time_no_retry:.2f}s")
        print(f"Time with retries: {time_with_retry:.2f}s")
        print(f"Retry overhead: {retry_overhead:.2f}s ({overhead_percent:.1f}%)")

        # Retry overhead should be reasonable (< 200% of base time)
        assert overhead_percent < 200, f"Retry overhead too high: {overhead_percent:.1f}%"

    @pytest.mark.asyncio
    async def test_database_write_performance(self, generate_conversations, mock_provider, tmp_path):
        """Test database write performance during batch processing."""
        conversations = generate_conversations[:50]  # 50 conversations

        db_path = tmp_path / "write_perf_test.db"
        storage = DatabaseStorage(str(db_path))

        # Create test chats
        for conv in conversations:
            storage.db_session.execute(
                """
                INSERT INTO chats (id, title, created_at, updated_at)
                VALUES (?, ?, datetime('now'), datetime('now'))
                """,
                (conv["id"], f"Test Chat {conv['id']}")
            )
        storage.db_session.commit()

        mock_provider.generate.return_value = '{"summary": "Test summary"}'

        write_times = []

        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            with patch("backend.summarizer._get_storage", return_value=storage):
                for conv in conversations:
                    # Measure just the storage time
                    extraction_start = time.time()
                    await extract_and_store_metrics(
                        chat_id=conv["id"],
                        conversation_text=conv["text"],
                        messages=conv["messages"],
                        metrics=["summary"]
                    )
                    extraction_time = time.time() - extraction_start
                    write_times.append(extraction_time * 1000)  # Convert to ms

        avg_write_time = sum(write_times) / len(write_times)
        max_write_time = max(write_times)

        print(f"\n=== Database Write Performance Results ===")
        print(f"Average write time: {avg_write_time:.2f}ms")
        print(f"Max write time: {max_write_time:.2f}ms")
        print(f"Total records written: {len(conversations)}")

        # Average write time should be reasonable (< 100ms)
        assert avg_write_time < 100, f"Database writes too slow: {avg_write_time:.2f}ms"

    @pytest.mark.asyncio
    async def test_scalability_increasing_conversation_length(self, mock_provider):
        """Test scalability with increasing conversation lengths."""
        # Generate conversations of varying lengths
        lengths = [5, 10, 20, 50, 100]  # Number of messages
        results = {}

        mock_provider.generate.return_value = '{"summary": "Test summary"}'

        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            for length in lengths:
                # Create conversation with specified length
                messages = []
                for i in range(length):
                    messages.append({
                        "role": "user" if i % 2 == 0 else "assistant",
                        "content": f"Message {i}: Some content here."
                    })

                conv_text = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in messages])

                # Measure processing time
                start = time.time()
                await extract_metrics(
                    conversation_text=conv_text,
                    messages=messages,
                    metrics=["summary"]
                )
                processing_time = (time.time() - start) * 1000  # ms

                results[length] = processing_time

        print(f"\n=== Scalability Results (by conversation length) ===")
        for length, time_ms in results.items():
            print(f"{length} messages: {time_ms:.2f}ms")

        # Processing time should scale sub-linearly with conversation length
        # (not perfectly linear due to LLM processing)
        ratio_5_to_100 = results[100] / results[5]
        print(f"Time ratio (100/5 messages): {ratio_5_to_100:.2f}x")

        # Should be less than 20x slower for 20x more messages
        assert ratio_5_to_100 < 20, f"Scaling too poorly: {ratio_5_to_100:.2f}x"


class TestStressScenarios:
    """Stress tests for edge cases and failure scenarios."""

    @pytest.mark.asyncio
    async def test_rapid_sequential_calls(self, mock_provider):
        """Test rapid sequential extraction calls."""
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = '{"summary": "Test"}'

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        conv_text = "User: Hello\nAssistant: Hi there!"

        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            # Make 50 rapid calls
            start = time.time()
            for _ in range(50):
                await extract_metrics(
                    conversation_text=conv_text,
                    messages=messages,
                    metrics=["summary"]
                )
            duration = time.time() - start

        print(f"\n=== Rapid Sequential Calls ===")
        print(f"50 calls completed in {duration:.2f}s")
        print(f"Rate: {50/duration:.2f} calls/second")

        # Should handle rapid calls without errors
        assert duration < 10, f"Rapid calls too slow: {duration:.2f}s"

    @pytest.mark.asyncio
    async def test_high_failure_rate_resilience(self, mock_provider):
        """Test system resilience with high failure rate."""
        # 70% of calls fail, 30% succeed
        responses = [
            Exception("Timeout"),
            Exception("Rate limit"),
            '{"summary": "Success"}',
        ] * 20  # Repeat pattern

        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.side_effect = responses

        messages = [{"role": "user", "content": "Test"}]
        conv_text = "User: Test"

        successes = 0
        failures = 0

        with patch("backend.summarizer._get_provider", return_value=mock_provider):
            for _ in range(60):
                result = await extract_metrics(
                    conversation_text=conv_text,
                    messages=messages,
                    metrics=["summary"]
                )
                if result["summary"].success:
                    successes += 1
                else:
                    failures += 1

        print(f"\n=== High Failure Rate Test ===")
        print(f"Successes: {successes}")
        print(f"Failures: {failures}")
        print(f"Success rate: {successes/(successes+failures)*100:.1f}%")

        # Should gracefully handle failures
        assert successes > 0, "Some extractions should succeed"
        assert failures > 0, "Test should have failures"
