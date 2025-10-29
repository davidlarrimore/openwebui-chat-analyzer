"""Tests for the Ollama-backed GenAI API routes."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend import summarizer
from backend.clients.ollama import (
    OllamaChatResult,
    OllamaEmbedResult,
    OllamaGenerateResult,
    OllamaOutOfMemoryError,
)


class DummyOllamaClient:
    """In-memory Ollama client used to avoid network calls in tests."""

    def __init__(self) -> None:
        self.last_generate: Dict[str, Any] | None = None
        self.last_chat: Dict[str, Any] | None = None
        self.last_embed: Dict[str, Any] | None = None
        self.generate_models: List[str] = []
        self.raise_oom_once: bool = False
        self.models: List[Dict[str, Any]] = [
            {"name": "llama3.1", "modified_at": "2024-01-01T00:00:00Z"},
            {"name": "phi3:mini", "modified_at": "2024-01-01T00:00:00Z"},
            {"name": "nomic-embed-text", "modified_at": "2024-01-01T00:00:00Z"},
        ]

    def generate(
        self,
        *,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        options: Dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
    ) -> OllamaGenerateResult:
        attempted_model = model or ""
        self.generate_models.append(attempted_model)
        self.last_generate = {
            "prompt": prompt,
            "model": model,
            "system": system,
            "options": options or {},
            "keep_alive": keep_alive,
        }
        if self.raise_oom_once:
            self.raise_oom_once = False
            raise OllamaOutOfMemoryError(
                "Ollama responded with 500 for POST /api/generate: "
                '{"error":"model requires more system memory (5.6 GiB) than is available (3.8 GiB)"}'
            )
        return OllamaGenerateResult(
            model=model or "llama3.1",
            response="Summarized headline.\nExtra detail.",
            raw={"response": "Summarized headline."},
        )

    def chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str | None = None,
        options: Dict[str, Any] | None = None,
    ) -> OllamaChatResult:
        self.last_chat = {
            "messages": messages,
            "model": model,
            "options": options or {},
        }
        return OllamaChatResult(
            model=model or "phi3:mini",
            message={"role": "assistant", "content": "Assistant reply."},
            raw={"message": {"role": "assistant", "content": "Assistant reply."}},
        )

    def embed(
        self,
        *,
        inputs: List[str],
        model: str | None = None,
    ) -> OllamaEmbedResult:
        self.last_embed = {
            "inputs": inputs,
            "model": model,
        }
        return OllamaEmbedResult(
            model=model or "nomic-embed-text",
            embeddings=[[0.1, 0.2, 0.3] for _ in inputs],
            raw={"embeddings": [[0.1, 0.2, 0.3]]},
        )

    def list_models(self) -> List[Dict[str, Any]]:
        return self.models


@pytest.fixture(scope="function")
def dummy_ollama(monkeypatch: pytest.MonkeyPatch) -> DummyOllamaClient:
    client = DummyOllamaClient()
    monkeypatch.setattr("backend.routes.get_ollama_client", lambda: client)
    monkeypatch.setattr("backend.summarizer.get_ollama_client", lambda: client)
    monkeypatch.setattr("backend.app.data_service.load_initial_data", lambda: None)
    return client


@pytest.fixture(autouse=True)
def reset_summary_defaults() -> None:
    summarizer.set_summary_model("llama3.1")
    summarizer.set_summary_fallback_model("phi3:mini")
    yield
    summarizer.set_summary_model("llama3.1")
    summarizer.set_summary_fallback_model("phi3:mini")


@pytest.fixture(scope="function")
def api_client(dummy_ollama: DummyOllamaClient) -> TestClient:  # noqa: F811
    with TestClient(app) as client:
        yield client


def test_generate_summary_returns_trimmed_text(api_client: TestClient, dummy_ollama: DummyOllamaClient) -> None:
    response = api_client.post(
        "/api/v1/genai/summarize",
        json={"context": "User: Hello\nassistant: Response"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == "Summarized headline."
    assert payload["model"] == "llama3.1"
    assert dummy_ollama.last_generate is not None
    assert "system" in dummy_ollama.last_generate
    assert dummy_ollama.last_generate["keep_alive"] is None
    assert dummy_ollama.last_generate["options"]["num_predict"] == 32
    assert dummy_ollama.last_generate["options"]["num_ctx"] == 1024


def test_generate_summary_falls_back_when_model_runs_out_of_memory(
    api_client: TestClient, dummy_ollama: DummyOllamaClient
) -> None:
    dummy_ollama.raise_oom_once = True

    response = api_client.post(
        "/api/v1/genai/summarize",
        json={"context": "User: Hello\nassistant: Response"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == "Summarized headline."
    assert payload["model"] == "phi3:mini"
    assert dummy_ollama.generate_models == ["llama3.1", "phi3:mini"]
    assert dummy_ollama.last_generate is not None
    assert dummy_ollama.last_generate["keep_alive"] is None


def test_generate_text_uses_default_temperature(api_client: TestClient, dummy_ollama: DummyOllamaClient) -> None:
    response = api_client.post(
        "/api/v1/genai/generate",
        json={"prompt": "Write haiku about testing.", "model": "phi3:mini"},
    )

    assert response.status_code == 200
    assert response.json()["text"].startswith("Summarized")
    assert dummy_ollama.last_generate is not None
    assert dummy_ollama.last_generate["options"]["temperature"] == pytest.approx(0.2)


def test_chat_endpoint_returns_message(api_client: TestClient, dummy_ollama: DummyOllamaClient) -> None:
    response = api_client.post(
        "/api/v1/genai/chat",
        json={
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
            "model": "phi3:mini",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["message"]["content"] == "Assistant reply."
    assert dummy_ollama.last_chat is not None
    assert len(dummy_ollama.last_chat["messages"]) == 3


def test_embed_endpoint_validates_inputs(api_client: TestClient) -> None:
    response = api_client.post(
        "/api/v1/genai/embed",
        json={"inputs": []},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "inputs must not be empty."


def test_list_models_returns_payload(api_client: TestClient) -> None:
    response = api_client.get("/api/v1/genai/models")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert any(model["name"] == "llama3.1" for model in data)
