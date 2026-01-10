"""Tests for summarizer retry logic and exponential backoff (Sprint 1).

This module tests the enhanced retry mechanisms including:
- Exponential backoff with jitter
- Parse retry logic with JSON mode fallback
- Enhanced error logging with full context preservation
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend import summarizer
from backend.providers.base import GenerateResult


class TestExponentialBackoff:
    """Test exponential backoff delay calculation."""

    def test_fixed_delay_without_exponential(self) -> None:
        """Fixed delay mode should return base_delay + small jitter."""
        base_delay = 2.0
        max_delay = 10.0
        use_exponential = False

        # Run multiple times to test jitter variability
        delays = [
            summarizer._calculate_retry_delay(1, base_delay, max_delay, use_exponential)
            for _ in range(10)
        ]

        # All delays should be close to base_delay (within jitter range)
        for delay in delays:
            assert 2.0 <= delay <= 2.2  # base_delay + 10% jitter
            assert delay != base_delay  # Should have some jitter

    def test_exponential_delay_first_attempt(self) -> None:
        """First exponential retry should be base_delay * 2^0 = base_delay."""
        base_delay = 1.0
        max_delay = 60.0
        use_exponential = True

        delay = summarizer._calculate_retry_delay(1, base_delay, max_delay, use_exponential)

        # First attempt: 1.0 * (2^0) = 1.0, plus up to 10% jitter
        assert 1.0 <= delay <= 1.1

    def test_exponential_delay_second_attempt(self) -> None:
        """Second exponential retry should double the delay."""
        base_delay = 1.0
        max_delay = 60.0
        use_exponential = True

        delay = summarizer._calculate_retry_delay(2, base_delay, max_delay, use_exponential)

        # Second attempt: 1.0 * (2^1) = 2.0, plus up to 10% jitter
        assert 2.0 <= delay <= 2.2

    def test_exponential_delay_third_attempt(self) -> None:
        """Third exponential retry should quadruple the original delay."""
        base_delay = 1.0
        max_delay = 60.0
        use_exponential = True

        delay = summarizer._calculate_retry_delay(3, base_delay, max_delay, use_exponential)

        # Third attempt: 1.0 * (2^2) = 4.0, plus up to 10% jitter
        assert 4.0 <= delay <= 4.4

    def test_exponential_delay_capped_at_max(self) -> None:
        """Exponential delay should be capped at max_delay."""
        base_delay = 10.0
        max_delay = 30.0
        use_exponential = True

        # 10 * (2^5) = 320, should be capped at 30
        delay = summarizer._calculate_retry_delay(6, base_delay, max_delay, use_exponential)

        # Should be capped at max_delay + 10% jitter
        assert 30.0 <= delay <= 33.0

    def test_jitter_variability(self) -> None:
        """Jitter should produce different delays on each call."""
        base_delay = 5.0
        max_delay = 60.0
        use_exponential = True

        delays = [
            summarizer._calculate_retry_delay(3, base_delay, max_delay, use_exponential)
            for _ in range(10)
        ]

        # Not all delays should be identical (jitter should vary)
        unique_delays = set(delays)
        assert len(unique_delays) > 1  # Should have at least some variation


class TestParseRetryLogic:
    """Test parse retry logic with JSON mode fallback."""

    def test_parse_success_on_first_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Successful parse on first attempt should not retry."""
        # Mock provider registry
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.return_value = GenerateResult(
            content='{"summary": "test summary", "outcome": 3}',
            model="test-model",
            provider="test",
        )

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = mock_provider

        with patch("backend.summarizer.get_provider_registry", return_value=mock_registry):
            result = summarizer._headline_with_provider(
                context="test context",
                connection="test",
                model="test-model",
            )

        # Should succeed on first attempt
        assert result.summary == "test summary"
        assert result.outcome == 3
        # Should only call generate once
        assert mock_provider.generate.call_count == 1

    def test_parse_retry_enables_json_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Parse failure on first attempt should retry with JSON mode enabled."""
        monkeypatch.setattr(summarizer, "SUMMARIZER_PARSE_RETRY_ATTEMPTS", 2)

        call_count = 0

        def mock_generate(*args: Any, **kwargs: Any) -> GenerateResult:
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: return malformed response
                return GenerateResult(
                    content="This is not valid JSON",
                    model="test-model",
                    provider="test",
                )
            else:
                # Second call: return valid JSON
                return GenerateResult(
                    content='{"summary": "retry success", "outcome": 4}',
                    model="test-model",
                    provider="test",
                )

        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.side_effect = mock_generate

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = mock_provider

        with patch("backend.summarizer.get_provider_registry", return_value=mock_registry):
            result = summarizer._headline_with_provider(
                context="test context",
                connection="test",
                model="test-model",
            )

        # Should succeed on second attempt
        assert result.summary == "retry success"
        assert result.outcome == 4
        assert mock_provider.generate.call_count == 2

        # Verify second call had json_mode enabled
        second_call_kwargs = mock_provider.generate.call_args_list[1][1]
        assert second_call_kwargs["options"]["json_mode"] is True

    def test_parse_retry_all_attempts_exhausted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All parse retry attempts failing should raise StructuredResponseError."""
        monkeypatch.setattr(summarizer, "SUMMARIZER_PARSE_RETRY_ATTEMPTS", 2)

        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.supports_json_mode.return_value = True
        # Always return malformed response
        mock_provider.generate.return_value = GenerateResult(
            content="This is not valid JSON",
            model="test-model",
            provider="test",
        )

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = mock_provider

        with patch("backend.summarizer.get_provider_registry", return_value=mock_registry):
            with pytest.raises(summarizer.StructuredResponseError) as exc_info:
                summarizer._headline_with_provider(
                    context="test context",
                    connection="test",
                    model="test-model",
                )

        # Should have tried both attempts
        assert mock_provider.generate.call_count == 2
        # Error should contain context
        assert exc_info.value.provider == "test"

    def test_parse_retry_simplified_prompt_on_final_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Final parse retry attempt should use simplified prompt."""
        monkeypatch.setattr(summarizer, "SUMMARIZER_PARSE_RETRY_ATTEMPTS", 3)

        call_prompts = []

        def mock_generate(*args: Any, **kwargs: Any) -> GenerateResult:
            call_prompts.append(kwargs.get("prompt", ""))
            # Always fail to see all retry attempts
            return GenerateResult(
                content="invalid",
                model="test-model",
                provider="test",
            )

        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.supports_json_mode.return_value = True
        mock_provider.generate.side_effect = mock_generate

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = mock_provider

        with patch("backend.summarizer.get_provider_registry", return_value=mock_registry):
            try:
                summarizer._headline_with_provider(
                    context="test context",
                    connection="test",
                    model="test-model",
                )
            except summarizer.StructuredResponseError:
                pass  # Expected

        # Should have made 3 attempts
        assert len(call_prompts) == 3
        # First two prompts should be normal
        assert not call_prompts[0].startswith("CRITICAL:")
        assert not call_prompts[1].startswith("CRITICAL:")
        # Final prompt should be simplified
        assert call_prompts[2].startswith("CRITICAL:")

    def test_parse_retry_no_json_mode_if_unsupported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Provider without JSON mode support should not have json_mode enabled."""
        monkeypatch.setattr(summarizer, "SUMMARIZER_PARSE_RETRY_ATTEMPTS", 2)

        call_count = 0

        def mock_generate(*args: Any, **kwargs: Any) -> GenerateResult:
            nonlocal call_count
            call_count += 1

            # Second call should not have json_mode if provider doesn't support it
            if call_count == 2:
                assert "json_mode" not in kwargs.get("options", {})

            if call_count == 1:
                return GenerateResult(content="invalid", model="test", provider="test")
            else:
                return GenerateResult(
                    content='{"summary": "success", "outcome": 3}',
                    model="test",
                    provider="test",
                )

        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.supports_json_mode.return_value = False  # Does not support
        mock_provider.generate.side_effect = mock_generate

        mock_registry = MagicMock()
        mock_registry.get_provider.return_value = mock_provider

        with patch("backend.summarizer.get_provider_registry", return_value=mock_registry):
            result = summarizer._headline_with_provider(
                context="test context",
                connection="test",
                model="test-model",
            )

        assert result.summary == "success"


class TestErrorLoggingPreservation:
    """Test enhanced error logging with full context preservation."""

    def test_truncate_for_logging_short_text(self) -> None:
        """Short text should not be truncated."""
        text = "short text"
        result = summarizer._truncate_for_logging(text, 100)
        assert result == "short text"

    def test_truncate_for_logging_long_text(self) -> None:
        """Long text should be truncated with indicator."""
        text = "a" * 200
        result = summarizer._truncate_for_logging(text, 100)
        assert len(result) > 100  # Includes truncation message
        assert result.startswith("a" * 100)
        assert "truncated" in result
        assert "200 chars" in result

    def test_truncate_for_logging_empty_text(self) -> None:
        """Empty text should return empty string."""
        result = summarizer._truncate_for_logging("", 100)
        assert result == ""

    def test_preserve_error_details_with_preservation_disabled(self) -> None:
        """With preservation disabled, should return truncated versions."""
        prompt = "p" * 1000
        response = "r" * 1000
        max_size = 500

        preserved_prompt, preserved_response = summarizer._preserve_error_details(
            prompt, response, preserve_full=False, max_size=max_size
        )

        # Should truncate to half of max_size (250 chars)
        assert "truncated" in preserved_prompt
        assert "truncated" in preserved_response
        assert "250" in preserved_prompt  # Half of max_size

    def test_preserve_error_details_with_preservation_enabled(self) -> None:
        """With preservation enabled, should return fuller versions."""
        prompt = "p" * 1000
        response = "r" * 1000
        max_size = 600

        preserved_prompt, preserved_response = summarizer._preserve_error_details(
            prompt, response, preserve_full=True, max_size=max_size
        )

        # Should use full max_size (600 chars)
        assert "truncated" in preserved_prompt
        assert "1000 chars" in preserved_prompt  # Shows original size

    def test_structured_response_error_includes_context(self) -> None:
        """StructuredResponseError should preserve prompt and provider context."""
        error = summarizer.StructuredResponseError(
            "Parse failed",
            response_text='{"invalid"}',
            prompt="test prompt",
            provider="ollama",
        )

        assert str(error) == "Parse failed"
        assert error.response_text == '{"invalid"}'
        assert error.prompt == "test prompt"
        assert error.provider == "ollama"


class TestRetryWithExponentialBackoff:
    """Test the full retry mechanism with exponential backoff."""

    def test_transient_error_triggers_exponential_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Transient connectivity errors should trigger exponential backoff."""
        from backend.clients import OllamaClientError

        monkeypatch.setattr(summarizer, "SUMMARIZER_USE_EXPONENTIAL_BACKOFF", True)
        monkeypatch.setattr(summarizer, "SUMMARIZER_RETRY_MAX_ATTEMPTS", 3)
        monkeypatch.setattr(summarizer, "SUMMARIZER_RETRY_BASE_DELAY", 0.1)  # Fast test

        call_count = 0
        retry_delays = []

        def mock_headline(*args: Any, **kwargs: Any) -> summarizer.ConversationAnalysis:
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                # Fail first two attempts with transient error
                raise OllamaClientError("Connection refused")
            else:
                # Succeed on third attempt
                return summarizer.ConversationAnalysis(summary="success", outcome=5)

        monkeypatch.setattr(summarizer, "_headline_with_provider", mock_headline)

        # Capture time.sleep calls to verify backoff
        original_sleep = time.sleep

        def mock_sleep(seconds: float) -> None:
            retry_delays.append(seconds)
            # Don't actually sleep in tests
            pass

        with patch("time.sleep", side_effect=mock_sleep):
            result = summarizer._call_provider_with_retry(
                context="test",
                connection="ollama",
                model="test",
                prompt="test",
            )

        assert result.summary == "success"
        assert call_count == 3
        # Should have slept twice (between attempts)
        assert len(retry_delays) == 2
        # Second delay should be larger than first (exponential)
        assert retry_delays[1] > retry_delays[0]
