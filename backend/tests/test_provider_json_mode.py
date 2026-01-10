"""Tests for provider JSON mode support (Sprint 1).

This module tests the native structured output capabilities added to all providers,
including the supports_json_mode() interface and json_mode option handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.providers.base import GenerateResult
from backend.providers.ollama import OllamaProvider
from backend.providers.openai import OpenAIProvider
from backend.providers.litellm import LiteLLMProvider
from backend.providers.openwebui import OpenWebUIProvider


class TestProviderJsonModeInterface:
    """Test that all providers implement supports_json_mode() correctly."""

    def test_openai_supports_json_mode(self) -> None:
        """OpenAI provider should support JSON mode."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.supports_json_mode() is True

    def test_litellm_supports_json_mode(self) -> None:
        """LiteLLM provider should support JSON mode."""
        provider = LiteLLMProvider(api_key="test-key")
        assert provider.supports_json_mode() is True

    def test_ollama_supports_json_mode(self) -> None:
        """Ollama provider should support JSON mode (format parameter)."""
        provider = OllamaProvider()
        assert provider.supports_json_mode() is True

    def test_openwebui_does_not_support_json_mode(self) -> None:
        """OpenWebUI provider should not support JSON mode."""
        provider = OpenWebUIProvider()
        assert provider.supports_json_mode() is False


class TestOpenAIJsonMode:
    """Test OpenAI provider JSON mode implementation."""

    def test_generate_without_json_mode(self) -> None:
        """Generate without JSON mode should not include response_format."""
        provider = OpenAIProvider(api_key="test-key")

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "test response"}}],
                "model": "gpt-4",
            }
            mock_post.return_value = mock_response

            result = provider.generate(
                model="gpt-4",
                prompt="test prompt",
                options={"temperature": 0.5},
            )

            # Verify response_format was NOT included
            call_kwargs = mock_post.call_args[1]
            request_body = call_kwargs["json"]
            assert "response_format" not in request_body
            assert request_body["temperature"] == 0.5

    def test_generate_with_json_mode_enabled(self) -> None:
        """Generate with json_mode=True should include response_format."""
        provider = OpenAIProvider(api_key="test-key")

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": '{"summary": "test"}'}}],
                "model": "gpt-4",
            }
            mock_post.return_value = mock_response

            result = provider.generate(
                model="gpt-4",
                prompt="test prompt",
                options={"temperature": 0.5, "json_mode": True},
            )

            # Verify response_format was included
            call_kwargs = mock_post.call_args[1]
            request_body = call_kwargs["json"]
            assert "response_format" in request_body
            assert request_body["response_format"] == {"type": "json_object"}
            assert request_body["temperature"] == 0.5

    def test_json_mode_not_passed_through(self) -> None:
        """json_mode option should be filtered out and not passed to API."""
        provider = OpenAIProvider(api_key="test-key")

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "test"}}],
                "model": "gpt-4",
            }
            mock_post.return_value = mock_response

            provider.generate(
                model="gpt-4",
                prompt="test",
                options={"json_mode": True, "max_tokens": 100},
            )

            # Verify json_mode is not in the request body
            call_kwargs = mock_post.call_args[1]
            request_body = call_kwargs["json"]
            # Should have response_format but not json_mode
            assert "response_format" in request_body
            assert "json_mode" not in request_body
            assert request_body["max_tokens"] == 100


class TestLiteLLMJsonMode:
    """Test LiteLLM provider JSON mode implementation."""

    def test_generate_with_json_mode_enabled(self) -> None:
        """LiteLLM should pass through response_format when json_mode=True."""
        provider = LiteLLMProvider(api_key="test-key", base_url="http://localhost:4000")

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": '{"summary": "test"}'}}],
                "model": "gpt-4",
            }
            mock_post.return_value = mock_response

            result = provider.generate(
                model="gpt-4",
                prompt="test prompt",
                options={"json_mode": True},
            )

            # Verify response_format was included
            call_kwargs = mock_post.call_args[1]
            request_body = call_kwargs["json"]
            assert "response_format" in request_body
            assert request_body["response_format"] == {"type": "json_object"}


class TestOllamaJsonMode:
    """Test Ollama provider JSON mode implementation."""

    def test_generate_without_json_mode(self) -> None:
        """Ollama generate without JSON mode should not include format parameter."""
        from backend.clients.ollama import OllamaClient, OllamaGenerateResult

        mock_client = MagicMock(spec=OllamaClient)
        mock_client.generate.return_value = OllamaGenerateResult(
            model="llama3.2:3b",
            response="test response",
            raw={},
        )

        provider = OllamaProvider(client=mock_client)
        result = provider.generate(
            model="llama3.2:3b",
            prompt="test prompt",
            options={"temperature": 0.5},
        )

        # Verify format parameter was NOT passed
        call_kwargs = mock_client.generate.call_args[1]
        assert call_kwargs.get("format") is None
        assert call_kwargs["options"]["temperature"] == 0.5

    def test_generate_with_json_mode_enabled(self) -> None:
        """Ollama generate with json_mode=True should include format='json'."""
        from backend.clients.ollama import OllamaClient, OllamaGenerateResult

        mock_client = MagicMock(spec=OllamaClient)
        mock_client.generate.return_value = OllamaGenerateResult(
            model="llama3.2:3b",
            response='{"summary": "test"}',
            raw={},
        )

        provider = OllamaProvider(client=mock_client)
        result = provider.generate(
            model="llama3.2:3b",
            prompt="test prompt",
            options={"temperature": 0.5, "json_mode": True},
        )

        # Verify format='json' was passed
        call_kwargs = mock_client.generate.call_args[1]
        assert call_kwargs.get("format") == "json"
        # Verify json_mode was filtered out of options
        assert "json_mode" not in call_kwargs["options"]
        assert call_kwargs["options"]["temperature"] == 0.5

    def test_json_mode_filtered_from_options(self) -> None:
        """json_mode should be filtered out and not passed to Ollama options."""
        from backend.clients.ollama import OllamaClient, OllamaGenerateResult

        mock_client = MagicMock(spec=OllamaClient)
        mock_client.generate.return_value = OllamaGenerateResult(
            model="llama3.2:3b",
            response="test",
            raw={},
        )

        provider = OllamaProvider(client=mock_client)
        provider.generate(
            model="llama3.2:3b",
            prompt="test",
            options={"json_mode": True, "num_predict": 256, "num_ctx": 1024},
        )

        call_kwargs = mock_client.generate.call_args[1]
        # json_mode should not be in options
        assert "json_mode" not in call_kwargs["options"]
        # Other options should be preserved
        assert call_kwargs["options"]["num_predict"] == 256
        assert call_kwargs["options"]["num_ctx"] == 1024


class TestOllamaClientFormatParameter:
    """Test that OllamaClient properly handles the format parameter."""

    def test_ollama_client_generate_with_format(self) -> None:
        """OllamaClient.generate should accept and pass format parameter."""
        from backend.clients.ollama import OllamaClient

        with patch("requests.Session.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "model": "llama3.2:3b",
                "response": '{"test": "response"}',
            }
            mock_request.return_value = mock_response

            client = OllamaClient(base_url="http://localhost:11434")
            result = client.generate(
                model="llama3.2:3b",
                prompt="test",
                format="json",
            )

            # Verify format was included in request payload
            call_args = mock_request.call_args
            json_payload = call_args[1]["json"]
            assert "format" in json_payload
            assert json_payload["format"] == "json"

    def test_ollama_client_generate_without_format(self) -> None:
        """OllamaClient.generate should work without format parameter."""
        from backend.clients.ollama import OllamaClient

        with patch("requests.Session.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "model": "llama3.2:3b",
                "response": "test response",
            }
            mock_request.return_value = mock_response

            client = OllamaClient(base_url="http://localhost:11434")
            result = client.generate(
                model="llama3.2:3b",
                prompt="test",
            )

            # Verify format was NOT included in request payload
            call_args = mock_request.call_args
            json_payload = call_args[1]["json"]
            assert "format" not in json_payload
