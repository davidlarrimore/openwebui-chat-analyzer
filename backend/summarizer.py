"""Local two-stage summarizer for chat headlines."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from .clients import OllamaClientError, OllamaOutOfMemoryError, get_ollama_client
from .config import (
    OLLAMA_DEFAULT_TEMPERATURE,
    OLLAMA_SUMMARY_MODEL,
    OLLAMA_SUMMARY_FALLBACK_MODEL,
)

MAX_CHARS = 256
SALIENT_K = int(os.getenv("SALIENT_K", "10"))
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SUMMARY_BATCH_TOKEN_LIMIT = max(1024, int(os.getenv("SUMMARY_BATCH_TOKEN_LIMIT", "16000")))
SUMMARY_BATCH_MAX_OUTPUT_TOKENS = max(256, int(os.getenv("SUMMARY_BATCH_MAX_OUTPUT_TOKENS", "1024")))
SUMMARY_FIELD = "gen_chat_summary"
LEGACY_SUMMARY_FIELD = "summary_128"

def _normalize_base_url(value: str) -> str:
    """Normalize a host string into a usable HTTP base URL."""
    candidate = (value or "").strip()
    if not candidate:
        candidate = "http://localhost:4000"
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    parsed = urlparse(candidate)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid Open WebUI host: {value!r}")
    return candidate.rstrip("/")


try:
    OWUI_BASE_URL = _normalize_base_url(os.getenv("OWUI_DIRECT_HOST", "http://localhost:4000"))
except ValueError:
    OWUI_BASE_URL = "http://localhost:4000"
    logging.getLogger(__name__).warning(
        "Invalid OWUI_DIRECT_HOST provided; falling back to %s", OWUI_BASE_URL
    )

OWUI_API_KEY = os.getenv("OWUI_DIRECT_API_KEY", "").strip()
OWUI_COMPLETIONS_MODEL = (
    os.getenv("OWUI_COMPLETIONS_MODEL")
    or os.getenv("OPENAI_MODEL")
    or "llama3.2:3b-instruct"
).strip()
if not OWUI_COMPLETIONS_MODEL:
    OWUI_COMPLETIONS_MODEL = "llama3.2:3b-instruct"

_OWUI_COMPLETIONS_URL = f"{OWUI_BASE_URL}/api/chat/completions"

_logger = logging.getLogger(__name__)
if not _logger.handlers:
    logging.basicConfig(level=logging.INFO)

_embeddings_model: Optional[SentenceTransformer] = None

def _get_embeddings_model() -> SentenceTransformer:
    """Lazy-load the sentence transformer model to avoid blocking application startup."""
    global _embeddings_model
    if _embeddings_model is None:
        _logger.info("Loading SentenceTransformer model: %s", EMB_MODEL_NAME)
        _embeddings_model = SentenceTransformer(EMB_MODEL_NAME)
    return _embeddings_model


def _get_chat_summary(chat: Mapping[str, object]) -> str:
    """Return the current summary value from a chat payload."""
    primary = chat.get(SUMMARY_FIELD)
    if primary is None or (isinstance(primary, str) and not primary.strip()):
        primary = chat.get(LEGACY_SUMMARY_FIELD)
    return str(primary or "").strip()


def _set_chat_summary(chat: Dict[str, object], value: str) -> None:
    """Persist a summary value to the chat payload, removing legacy keys."""
    chat[SUMMARY_FIELD] = value
    if LEGACY_SUMMARY_FIELD in chat:
        try:
            del chat[LEGACY_SUMMARY_FIELD]  # type: ignore[arg-type]
        except Exception:  # pragma: no cover
            pass

_requests_session = requests.Session()
_requests_session.headers.update({"Accept": "application/json"})
if OWUI_API_KEY:
    _requests_session.headers["Authorization"] = f"Bearer {OWUI_API_KEY}"

_HEADLINE_SYS = (
    "You write concise, single-line conversation summaries that describe both the topic and "
    "the key points discussed."
)
_HEADLINE_USER_TMPL = (
    "Summarize this chat interaction in one line (<=256 characters). No quotes or trailing punctuation.\n"
    "Identify: (1) the user's main intent, (2) whether the request was successfully fulfilled, and (3) the type of information or response provided (e.g., factual, procedural, analytical, creative).\n"
    "Be specific about the topic or subject matter.\n\n"
    "Context:\n{ctx}\n\n"
    "Summary:"
)

_BATCH_HEADLINE_SYS = (
    "You write concise, single-line conversation summaries that describe both the topic and "
    "key points for each chat you receive. You always respond with strictly valid JSON and nothing else."
)
_BATCH_HEADLINE_USER_TMPL = (
    "You will be given multiple chat transcripts as JSON objects with keys `chat_id` and "
    "`lines`. Each `lines` value is an ordered list of \"role: message\" entries representing "
    "a conversation between a user and an assistant.\n\n"
    "For every chat:\n"
    "1. Write a single sentence summary under 256 characters.\n"
    "2. Capture the user's main intent, whether it was fulfilled, and the type of response provided.\n"
    "3. Do not include quotation marks or trailing punctuation at the end of the summary.\n\n"
    "CRITICAL: You must return ONLY a valid JSON array with NO additional text, commentary, or markdown formatting.\n"
    "Each object in the array MUST have exactly this structure:\n"
    "{{\"chat_id\": \"<exact_chat_id_from_input>\", \"summary\": \"<your_summary_here>\"}}\n\n"
    "Ensure every chat_id from the input appears in your response with its corresponding summary.\n\n"
    "Chats:\n{payload}"
)


def _clean(text: str) -> str:
    """Normalize message content so it is safe for downstream processing.

    Args:
        text (str): Raw message text from a conversation export.
    Returns:
        str: Sanitized content with markdown fences removed and whitespace squashed.
    """
    if not text:
        return ""
    text = text.replace("```", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chat_lines(messages: Sequence[Mapping[str, object]]) -> List[str]:
    """Convert message dictionaries into prefixed lines useful for summarization.

    Args:
        messages (Sequence[Mapping[str, object]]): Chat records emitted by the export.
    Returns:
        List[str]: Ordered lines in the format `role: content`.
    """
    lines: List[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        if role not in {"user", "assistant"}:
            continue
        content = _clean(str(message.get("content", "")))
        if content:
            lines.append(f"{role}: {content}")
    return lines


def _build_context_from_lines(lines: Sequence[str]) -> str:
    """Construct a condensed context from pre-formatted chat lines."""
    if not lines:
        return ""

    limit = min(max(SALIENT_K, 8), 12)
    try:
        salient = _select_salient(lines, limit)
    except Exception:
        salient = list(lines[:limit])
    return " ".join(salient)


def _cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute a cosine similarity matrix for ranking diverse snippets."""
    return embeddings @ embeddings.T


def _select_salient(lines: Sequence[str], k: int) -> List[str]:
    """Select the most representative conversation lines while preserving diversity.

    Args:
        lines (Sequence[str]): Candidate message lines derived from a chat.
        k (int): Target number of salient lines to retain.
    Returns:
        List[str]: Lines chosen by maximal marginal relevance heuristics.
    """
    if not lines:
        return []

    limit = min(max(k, 8), 12)
    if len(lines) <= limit:
        return list(lines)

    embeddings = _get_embeddings_model().encode(
        list(lines),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    centroid = np.mean(embeddings, axis=0, keepdims=True)
    centroid /= np.linalg.norm(centroid) + 1e-12

    relevance = (embeddings @ centroid.T).squeeze(-1)
    pre_k = min(len(lines), max(limit * 2, limit + 2))
    pre_idx = np.argsort(-relevance)[:pre_k]

    selected: List[int] = []
    lambda_param = 0.7
    sim_mat = _cosine_sim_matrix(embeddings[pre_idx])

    while len(selected) < limit and len(selected) < len(pre_idx):
        if not selected:
            # Seed the selection with the line most aligned to the centroid embedding.
            best = int(np.argmax(relevance[pre_idx]))
            selected.append(best)
            continue

        scores: List[float] = []
        for candidate in range(len(pre_idx)):
            if candidate in selected:
                scores.append(-1e9)
                continue
            rel = relevance[pre_idx[candidate]]
            div = max(sim_mat[candidate, idx] for idx in selected) if selected else 0.0
            # Score balances relevance to topic centroid with dissimilarity to selected lines.
            scores.append(lambda_param * rel - (1 - lambda_param) * div)

        next_idx = int(np.argmax(scores))
        if scores[next_idx] < -1e8:
            break
        selected.append(next_idx)

    chosen = sorted(pre_idx[idx] for idx in selected)
    return [lines[idx] for idx in chosen]


def _trim_one_line(text: str, max_chars: int = MAX_CHARS) -> str:
    """Return a single-line summary clipped to the configured character limit."""
    cleaned = _clean(text)
    cleaned = cleaned.split("\n", 1)[0]
    cleaned = cleaned.strip('"\'' "“”‘’ ").rstrip(".")
    if len(cleaned) > max_chars:
        return cleaned[: max_chars - 1] + "…"
    return cleaned


def _build_context(messages: Sequence[Mapping[str, object]]) -> str:
    """Build a condensed context window from a chat's messages."""
    lines = _chat_lines(messages)
    return _build_context_from_lines(lines)


def _headline_with_ollama(context: str, *, model: str = OLLAMA_SUMMARY_MODEL) -> str:
    """Generate a headline using the local Ollama service."""
    client = get_ollama_client()
    prompt = _HEADLINE_USER_TMPL.format(ctx=context)
    options = {
        "temperature": OLLAMA_DEFAULT_TEMPERATURE,
        "num_predict": 32,
        "num_ctx": 1024,
    }
    result = client.generate(
        prompt=prompt,
        model=model,
        system=_HEADLINE_SYS,
        options=options,
        keep_alive="-1",
    )
    return _trim_one_line(result.response)


def _headline_with_openwebui(context: str) -> str:
    """Call the legacy Open WebUI completion endpoint to obtain a headline.

    Args:
        context (str): Condensed conversation snippet used as the LLM prompt.
    Returns:
        str: Model-generated summary trimmed to a single line.
    """
    payload = {
        "model": OWUI_COMPLETIONS_MODEL,
        "temperature": 0.1,
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": _HEADLINE_SYS},
            {"role": "user", "content": _HEADLINE_USER_TMPL.format(ctx=context)},
        ],
    }

    try:
        response = _requests_session.post(
            _OWUI_COMPLETIONS_URL,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Open WebUI completion request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Open WebUI completion response was not valid JSON.") from exc

    text = ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        primary = choices[0] or {}
        message = primary.get("message")
        if isinstance(message, dict):
            text = str(message.get("content") or "").strip()
        if not text:
            text_candidate = primary.get("text")
            if isinstance(text_candidate, str):
                text = text_candidate.strip()

    return _trim_one_line(text)


def _headline_with_llm(context: str) -> str:
    """Attempt summarization via Ollama, falling back to Open WebUI if needed."""
    try:
        headline = _headline_with_ollama(context)
        if headline:
            return headline
    except OllamaOutOfMemoryError as exc:  # pragma: no cover - depends on runtime model.
        fallback_model = (
            OLLAMA_SUMMARY_FALLBACK_MODEL
            if OLLAMA_SUMMARY_FALLBACK_MODEL and OLLAMA_SUMMARY_FALLBACK_MODEL != OLLAMA_SUMMARY_MODEL
            else ""
        )
        if fallback_model:
            _logger.warning(
                "Ollama model %s ran out of memory; retrying with fallback model %s.",
                OLLAMA_SUMMARY_MODEL,
                fallback_model,
                exc_info=True,
            )
            try:
                headline = _headline_with_ollama(context, model=fallback_model)
                if headline:
                    return headline
            except OllamaClientError as fallback_exc:
                _logger.warning(
                    "Ollama fallback model %s failed: %s",
                    fallback_model,
                    fallback_exc,
                    exc_info=True,
                )
        else:
            _logger.warning(
                "Ollama summarization failed because the model ran out of memory and no fallback model is configured.",
                exc_info=True,
            )
    except OllamaClientError as exc:  # pragma: no cover - logged for observability.
        _logger.warning(
            "Ollama summarization failed; attempting Open WebUI fallback: %s",
            exc,
            exc_info=True,
        )
    except Exception as exc:  # pragma: no cover - logged for observability.
        _logger.warning(
            "Unexpected error during Ollama summarization; attempting Open WebUI fallback: %s",
            exc,
            exc_info=True,
        )

    try:
        return _headline_with_openwebui(context)
    except Exception as exc:  # pragma: no cover - logged for observability.
        _logger.warning("Open WebUI fallback failed: %s", exc, exc_info=True)
        return ""


def _estimate_token_length(text: str) -> int:
    """Approximate the token length of a prompt using a simple character heuristic."""
    if not text:
        return 0
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


def _build_batch_payload(batch_items: Sequence[Mapping[str, object]]) -> str:
    """Serialize chat batches into JSON payloads for the batch summarization prompt."""
    payload: List[Dict[str, object]] = []
    for item in batch_items:
        chat_id = str(item.get("chat_id") or "").strip()
        raw_lines = item.get("lines")
        if isinstance(raw_lines, list):
            lines = [str(entry) for entry in raw_lines if str(entry).strip()]
        else:
            lines = []
        payload.append({"chat_id": chat_id, "lines": lines})
    return json.dumps(payload, ensure_ascii=False)


def _strip_json_block(text: str) -> str:
    """Remove Markdown code fences or leading commentary from model responses."""
    if not text:
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        segments = cleaned.split("```")
        if len(segments) >= 3:
            cleaned = segments[1]
        else:
            cleaned = segments[-1]
    return cleaned.strip()


def _parse_batch_response(text: str) -> Dict[str, str]:
    """Extract chat summaries from a JSON response payload."""
    cleaned = _strip_json_block(text)
    if not cleaned:
        _logger.debug("Batch response was empty after stripping JSON blocks")
        return {}
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        _logger.debug("No valid JSON array found in batch response")
        return {}
    snippet = cleaned[start : end + 1]
    try:
        data = json.loads(snippet)
    except ValueError as exc:
        _logger.warning("Failed to parse JSON from batch response: %s", exc)
        return {}

    results: Dict[str, str] = {}
    if not isinstance(data, list):
        _logger.warning("Batch response JSON is not an array, got type: %s", type(data).__name__)
        return results

    for entry in data:
        if not isinstance(entry, dict):
            continue
        chat_id = str(entry.get("chat_id") or "").strip()
        summary = str(entry.get("summary") or "").strip()
        if chat_id and summary:
            results[chat_id] = summary
        elif chat_id:
            _logger.debug("Chat %s has empty summary in batch response", chat_id)

    _logger.debug("Parsed %d summaries from batch response", len(results))
    return results


def _batch_headlines_with_ollama(payload: str, *, model: str = OLLAMA_SUMMARY_MODEL) -> str:
    """Generate batch summaries using the local Ollama service."""
    client = get_ollama_client()
    options = {
        "temperature": OLLAMA_DEFAULT_TEMPERATURE,
        "num_predict": SUMMARY_BATCH_MAX_OUTPUT_TOKENS,
        "num_ctx": 1024,
    }
    prompt = _BATCH_HEADLINE_USER_TMPL.format(payload=payload)
    result = client.generate(
        prompt=prompt,
        model=model,
        system=_BATCH_HEADLINE_SYS,
        options=options,
        keep_alive="-1",
    )
    return str(result.response or "").strip()


def _batch_headlines_with_openwebui(payload_json: str) -> str:
    """Call the Open WebUI completion endpoint for batch summarization."""
    payload = {
        "model": OWUI_COMPLETIONS_MODEL,
        "temperature": 0.1,
        "max_tokens": SUMMARY_BATCH_MAX_OUTPUT_TOKENS,
        "messages": [
            {"role": "system", "content": _BATCH_HEADLINE_SYS},
            {"role": "user", "content": _BATCH_HEADLINE_USER_TMPL.format(payload=payload_json)},
        ],
    }

    try:
        response = _requests_session.post(
            _OWUI_COMPLETIONS_URL,
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Open WebUI batch completion request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Open WebUI batch completion response was not valid JSON.") from exc

    text = ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        primary = choices[0] or {}
        message = primary.get("message")
        if isinstance(message, dict):
            text = str(message.get("content") or "").strip()
        if not text:
            text_candidate = primary.get("text")
            if isinstance(text_candidate, str):
                text = text_candidate.strip()
    return text


def _batch_headlines_with_llm(batch_items: Sequence[Dict[str, object]]) -> Dict[str, str]:
    """Request batched chat summaries via Ollama, falling back to Open WebUI when needed."""
    if not batch_items:
        return {}

    prompt_payload = _build_batch_payload(batch_items)
    try:
        response_text = _batch_headlines_with_ollama(prompt_payload)
        parsed = _parse_batch_response(response_text)
        if parsed:
            return parsed
    except OllamaOutOfMemoryError as exc:
        fallback_model = (
            OLLAMA_SUMMARY_FALLBACK_MODEL
            if OLLAMA_SUMMARY_FALLBACK_MODEL and OLLAMA_SUMMARY_FALLBACK_MODEL != OLLAMA_SUMMARY_MODEL
            else ""
        )
        if fallback_model:
            _logger.warning(
                "Ollama batch summarization ran out of memory; retrying with fallback model %s.",
                fallback_model,
                exc_info=True,
            )
            try:
                response_text = _batch_headlines_with_ollama(prompt_payload, model=fallback_model)
                parsed = _parse_batch_response(response_text)
                if parsed:
                    return parsed
            except OllamaClientError as fallback_exc:
                _logger.warning(
                    "Ollama fallback batch model %s failed: %s",
                    fallback_model,
                    fallback_exc,
                    exc_info=True,
                )
        else:
            _logger.warning(
                "Ollama batch summarization failed due to OOM and no fallback model is configured.",
                exc_info=True,
            )
    except OllamaClientError as exc:
        _logger.warning(
            "Ollama batch summarization failed: %s",
            exc,
            exc_info=True,
        )
    except Exception as exc:  # pragma: no cover - logged for observability.
        _logger.warning(
            "Unexpected error during Ollama batch summarization: %s",
            exc,
            exc_info=True,
        )

    try:
        response_text = _batch_headlines_with_openwebui(prompt_payload)
        parsed = _parse_batch_response(response_text)
        if parsed:
            return parsed
    except Exception as exc:  # pragma: no cover - logged for observability.
        _logger.warning("Open WebUI batch summarization failed: %s", exc, exc_info=True)

    return {}


def _summarize_batch_items(batch_items: Sequence[Dict[str, object]]) -> Dict[str, str]:
    """Wrapper used by summarize_chats to facilitate monkeypatching in tests."""
    return _batch_headlines_with_llm(batch_items)


def _summarize_context(context: str) -> str:
    """Generate a summary via the LLM with a deterministic fallback.

    Args:
        context (str): Concatenated salient conversation lines.
    Returns:
        str: Single-line summary, either model generated or clipped context.
    """
    if not context:
        return ""
    try:
        headline = _headline_with_llm(context)
        if headline:
            return headline
        _logger.warning("LLM returned an empty summary; using fallback snippet.")
    except Exception as exc:
        _logger.warning("LLM summary request failed; falling back to snippet: %s", exc, exc_info=True)
    return _trim_one_line(context, MAX_CHARS)


def summarize_chat(messages: Sequence[Mapping[str, object]]) -> str:
    """Summarize a single chat's message history.

    Args:
        messages (Sequence[Mapping[str, object]]): Exported messages for a single chat.
    Returns:
        str: Summary line suitable for display in the UI.
    """
    context = _build_context(messages)
    if not context:
        return ""

    return _summarize_context(context)


ProgressCallback = Optional[Callable[[int, int, str, str, Optional[Dict[str, object]]], None]]
# Optional hook signature used to stream summarization progress back to callers.


def summarize_chats(
    chats: Iterable[Dict[str, object]],
    messages: Sequence[Mapping[str, object]],
    on_progress: ProgressCallback = None,
    *,
    replace_existing: bool = False,
    on_batch_complete: Optional[Callable[[Dict[str, str]], None]] = None,
) -> tuple[Dict[str, str], Dict[str, int]]:
    """Summarize chats and return a map of chat_id -> summary plus stats.

    Args:
        chats (Iterable[Dict[str, object]]): Chat metadata dictionaries.
        messages (Sequence[Mapping[str, object]]): All messages in the dataset.
        on_progress (ProgressCallback, optional): Callback invoked after each chat.
        replace_existing (bool, optional): When True, regenerate summaries even if one already exists.
        on_batch_complete (Callable[[Dict[str, str]], None], optional): Callback invoked after each batch
            is processed with the batch summaries for immediate persistence.
    Returns:
        tuple[Dict[str, str], Dict[str, int]]: Summary map and aggregate statistics.
    """
    chats_list = list(chats)
    messages_list = list(messages)
    total = len(chats_list)

    if total == 0:
        return {}, {"total": 0, "skipped": 0, "failures": 0, "generated": 0}

    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for message in messages_list:
        chat_id_raw = message.get("chat_id")
        chat_id = str(chat_id_raw) if chat_id_raw not in (None, "") else ""
        if not chat_id:
            continue
        # Bucket messages by chat to avoid repeatedly scanning the full list later.
        grouped[chat_id].append(message)

    summary_map: Dict[str, str] = {}
    skipped = 0
    failures = 0
    generated = 0

    pending_batch: List[Dict[str, object]] = []
    pending_tokens = 0

    def _flush_pending_batch() -> None:
        nonlocal pending_batch, pending_tokens, generated, failures
        if not pending_batch:
            return

        batch_size = len(pending_batch)
        batch_chat_ids = [item["chat_id"] for item in pending_batch]
        _logger.info(
            "Processing batch of %d chats (estimated %d tokens)",
            batch_size,
            pending_tokens,
        )

        try:
            batch_results = _summarize_batch_items(pending_batch)
            _logger.info(
                "Batch summarization returned %d summaries for %d chats",
                len(batch_results),
                batch_size,
            )
        except Exception as exc:  # pragma: no cover - logged for observability.
            _logger.warning(
                "Batch summarization failed; falling back to per-chat summaries: %s",
                exc,
                exc_info=True,
            )
            batch_results = {}

        # Collect summaries for this batch to persist immediately
        batch_summaries: Dict[str, str] = {}

        for item in pending_batch:
            chat = item["chat"]
            chat_id = item["chat_id"]
            previous_summary = item["previous_summary"]
            fallback_context = item["context"]

            summary_text = str(batch_results.get(chat_id) or "").strip()
            if summary_text:
                summary_text = _trim_one_line(summary_text, MAX_CHARS)
            elif batch_results:
                # Batch returned results but not for this chat_id - log and fall back
                _logger.debug(
                    "Chat %s missing from batch results; falling back to individual summarization",
                    chat_id,
                )
            if not summary_text:
                summary_text = _summarize_context(fallback_context)

            if summary_text:
                generated += 1
                summary_map[chat_id] = summary_text
                _set_chat_summary(chat, summary_text)
                batch_summaries[chat_id] = summary_text
                outcome = "generated"
            else:
                failures += 1
                _set_chat_summary(chat, previous_summary if previous_summary else "")
                outcome = "failed"

            if on_progress:
                details = {
                    "type": "chat",
                    "chat_id": chat_id,
                    "completed": item["position"],
                    "total": total,
                    "outcome": outcome,
                    "event_id": f"chat-{chat_id or 'unknown'}-{item['position']}",
                }
                try:
                    on_progress(item["position"], total, chat_id, outcome, details)
                except Exception:  # pragma: no cover
                    pass

        # Persist batch summaries immediately to database
        if on_batch_complete and batch_summaries:
            try:
                on_batch_complete(batch_summaries)
                _logger.info("Persisted %d summaries from batch to database", len(batch_summaries))
            except Exception as exc:  # pragma: no cover
                _logger.warning("Failed to persist batch summaries: %s", exc, exc_info=True)

        pending_batch = []
        pending_tokens = 0

    for idx, chat in enumerate(chats_list, start=1):
        chat_id_raw = chat.get("chat_id")
        chat_id = str(chat_id_raw) if chat_id_raw not in (None, "") else ""
        existing_summary = _get_chat_summary(chat)
        has_existing_summary = bool(existing_summary)

        if has_existing_summary and not replace_existing:
            _set_chat_summary(chat, existing_summary)
            skipped += 1
            if on_progress:
                details = {
                    "type": "skip",
                    "chat_id": chat_id,
                    "completed": idx,
                    "total": total,
                    "event_id": f"skip-{chat_id or 'unknown'}-{idx}",
                }
                try:
                    on_progress(idx, total, chat_id, "skipped", details)
                except Exception:  # pragma: no cover
                    pass
            continue

        previous_summary = existing_summary if has_existing_summary else ""

        if not chat_id:
            _set_chat_summary(chat, previous_summary if previous_summary else "")
            failures += 1
            if on_progress:
                details = {
                    "type": "invalid_chat",
                    "chat_id": chat_id,
                    "completed": idx,
                    "total": total,
                    "event_id": f"invalid-{idx}",
                }
                try:
                    on_progress(idx, total, chat_id, "failed", details)
                except Exception:  # pragma: no cover
                    pass
            continue

        chat_messages = grouped.get(chat_id, [])
        lines = _chat_lines(chat_messages)
        if not lines:
            _set_chat_summary(chat, previous_summary if previous_summary else "")
            failures += 1
            if on_progress:
                details = {
                    "type": "empty_context",
                    "chat_id": chat_id,
                    "completed": idx,
                    "total": total,
                    "event_id": f"empty-{chat_id or 'unknown'}-{idx}",
                }
                try:
                    on_progress(idx, total, chat_id, "failed", details)
                except Exception:  # pragma: no cover
                    pass
            continue

        context = _build_context_from_lines(lines)
        if not context:
            context = " ".join(lines)
        if not context:
            context = " ".join(lines[: SALIENT_K])

        prompt_lines = "\n".join(lines)
        token_estimate = _estimate_token_length(prompt_lines) + 64
        if pending_batch and pending_tokens + token_estimate > SUMMARY_BATCH_TOKEN_LIMIT:
            _flush_pending_batch()

        pending_batch.append(
            {
                "chat": chat,
                "chat_id": chat_id,
                "context": context,
                "lines": lines,
                "position": idx,
                "previous_summary": previous_summary,
            }
        )
        pending_tokens += min(token_estimate, SUMMARY_BATCH_TOKEN_LIMIT)

    _flush_pending_batch()

    stats = {
        "total": total,
        "skipped": skipped,
        "failures": failures,
        "generated": generated,
    }
    _logger.info(
        "Summarizer completed. total=%d generated=%d skipped=%d failed=%d",
        stats["total"],
        stats["generated"],
        stats["skipped"],
        stats["failures"],
    )
    return summary_map, stats


def attach_summaries(
    chats: Iterable[Dict[str, object]],
    messages: Sequence[Mapping[str, object]],
) -> None:
    """Convenience wrapper that mutates chat dictionaries with generated summaries."""
    summarize_chats(chats, messages)
