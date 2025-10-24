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
    OLLAMA_KEEP_ALIVE,
    OLLAMA_SUMMARY_MODEL,
    OLLAMA_SUMMARY_FALLBACK_MODEL,
)

SALIENT_K = int(os.getenv("SALIENT_K", "10"))
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SUMMARY_FIELD = "gen_chat_summary"
LEGACY_SUMMARY_FIELD = "summary_128"
CHUNK_CHAR_LIMIT = max(64, int(os.getenv("SUMMARY_CHUNK_CHAR_LIMIT", "2048")))
CHUNK_OVERLAP_LINES = max(0, int(os.getenv("SUMMARY_CHUNK_OVERLAP_LINES", "2")))

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


_DEBUG_JSON_LIMIT = 2048


def _format_debug_json(payload: object) -> str:
    """Return a truncated JSON serialization for debug logging."""
    try:
        encoded = json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        encoded = str(payload)
    if len(encoded) > _DEBUG_JSON_LIMIT:
        return f"{encoded[:_DEBUG_JSON_LIMIT]}…(truncated)"
    return encoded


def _format_debug_text(text: str) -> str:
    """Return a truncated text preview for debug logging."""
    if text is None:
        return "<none>"
    if len(text) > _DEBUG_JSON_LIMIT:
        return f"{text[:_DEBUG_JSON_LIMIT]}…(truncated)"
    return text


def _env_flag(name: str, default: bool) -> bool:
    """Return True/False based on common string representations."""
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


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
OWUI_FALLBACK_ENABLED = _env_flag("OWUI_FALLBACK_ENABLED", default=False)

_logger = logging.getLogger(__name__)
if not _logger.handlers:
    logging.basicConfig(level=logging.INFO)
if not OWUI_FALLBACK_ENABLED:
    _logger.info(
        "Open WebUI fallback disabled via OWUI_FALLBACK_ENABLED; summarizer will skip Open WebUI requests."
    )


def _disable_openwebui_fallback(reason: str) -> None:
    """Disable Open WebUI fallback attempts for the current process."""
    global OWUI_FALLBACK_ENABLED
    if OWUI_FALLBACK_ENABLED:
        OWUI_FALLBACK_ENABLED = False
        _logger.warning(
            "Disabling Open WebUI fallback after runtime failure: %s",
            reason,
        )

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


def _chunk_lines(
    lines: Sequence[str],
    *,
    char_limit: int,
    overlap_lines: int,
) -> List[List[str]]:
    """Split normalized chat lines into overlapping chunks within a char budget."""
    normalized_limit = max(64, char_limit)
    normalized_overlap = max(0, overlap_lines)

    chunks: List[List[str]] = []
    buffer: List[str] = []
    buffer_chars = 0

    for raw_line in lines:
        line = str(raw_line or "").strip()
        if not line:
            continue

        line_len = len(line)

        # Create a new chunk when the current buffer would overflow the limit.
        prospective = buffer_chars + (1 if buffer else 0) + line_len
        if buffer and prospective > normalized_limit:
            chunks.append(buffer)
            if normalized_overlap > 0:
                overlap = buffer[-normalized_overlap:]
                buffer = overlap.copy()
                buffer_chars = sum(len(entry) for entry in buffer)
                if buffer:
                    buffer_chars += len(buffer) - 1
            else:
                buffer = []
                buffer_chars = 0

        if buffer:
            buffer_chars += 1  # account for separating space
        buffer.append(line)
        buffer_chars += line_len

    if buffer:
        chunks.append(buffer)

    return chunks or [[]]

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


def _trim_one_line(text: str) -> str:
    """Return a single-line summary with no enforced character limit."""
    cleaned = _clean(text)
    cleaned = cleaned.split("\n", 1)[0]
    cleaned = cleaned.strip('"\'' "“”‘’ ").rstrip(".")
    return cleaned


def _build_context(messages: Sequence[Mapping[str, object]]) -> str:
    """Build a condensed context window from a chat's messages."""
    lines = _chat_lines(messages)
    return _build_context_from_lines(lines)


def _headline_with_ollama(context: str, *, model: str = OLLAMA_SUMMARY_MODEL) -> str:
    """Generate a headline using the local Ollama service."""
    client = get_ollama_client()
    prompt = _HEADLINE_USER_TMPL.format(ctx=context)
    _logger.debug(
        "Requesting Ollama headline model=%s context_chars=%d",
        model,
        len(context),
    )
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
        keep_alive=OLLAMA_KEEP_ALIVE,
    )
    headline = _trim_one_line(result.response)
    _logger.debug(
        "Ollama headline response model=%s chars=%d preview=%r",
        result.model,
        len(result.response or ""),
        headline[:120],
    )
    return headline


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

    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug("OpenWebUI headline payload: %s", _format_debug_json(payload))

    try:
        response = _requests_session.post(
            _OWUI_COMPLETIONS_URL,
            json=payload,
            timeout=60,
        )
        _logger.debug(
            "OpenWebUI headline request status=%s",
            response.status_code,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Open WebUI completion request failed: {exc}") from exc

    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug(
            "OpenWebUI headline raw response: %s",
            _format_debug_text(response.text),
        )

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

    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug("OpenWebUI headline parsed JSON: %s", _format_debug_json(data))

    headline = _trim_one_line(text)
    _logger.debug(
        "OpenWebUI headline response chars=%d preview=%r",
        len(text),
        headline[:120],
    )
    return headline


def _headline_with_llm(context: str) -> str:
    """Attempt summarization via Ollama, falling back to Open WebUI if needed."""
    _logger.debug("Summarizing headline via LLM for context_chars=%d", len(context))
    try:
        headline = _headline_with_ollama(context)
        if headline:
            _logger.debug("Headline obtained via Ollama")
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
                    _logger.debug("Headline obtained via Ollama fallback model %s", fallback_model)
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

    if not OWUI_FALLBACK_ENABLED:
        _logger.debug("Open WebUI fallback disabled; returning empty headline")
        return ""

    try:
        headline = _headline_with_openwebui(context)
        if headline:
            _logger.debug("Headline obtained via Open WebUI fallback")
        return headline
    except Exception as exc:  # pragma: no cover - logged for observability.
        _logger.warning("Open WebUI fallback failed: %s", exc, exc_info=True)
        _disable_openwebui_fallback(str(exc))
        return ""


def _estimate_token_length(text: str) -> int:
    """Approximate the token length of a prompt using a simple character heuristic."""
    if not text:
        return 0
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


def _summarize_context(context: str) -> str:
    """Generate a summary via the LLM with a deterministic fallback.

    Args:
        context (str): Concatenated salient conversation lines.
    Returns:
        str: Single-line summary, either model generated or clipped context.
    """
    if not context:
        return ""
    _logger.debug("Summarizing single chat context_chars=%d", len(context))
    try:
        headline = _headline_with_llm(context)
        if headline:
            _logger.debug("Summary obtained via LLM preview=%r", headline[:120])
            return headline
        _logger.warning("LLM returned an empty summary; using fallback snippet.")
    except Exception as exc:
        _logger.warning("LLM summary request failed; falling back to snippet: %s", exc, exc_info=True)
    fallback = _trim_one_line(context)
    _logger.debug("Using fallback snippet preview=%r", fallback[:120])
    return fallback


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


def _summarize_with_chunks(
    lines: Sequence[str],
    *,
    fallback_context: str,
    on_progress: ProgressCallback,
    progress_index: int,
    total: int,
    chat_id: str,
) -> str:
    """Summarize a chat using a single request against the LLM."""
    context = fallback_context.strip()
    if not context:
        joined_lines = " ".join(str(line).strip() for line in lines if str(line).strip())
        context = joined_lines
    if not context:
        return ""

    _logger.debug(
        "Summarizing chat %s without chunking (context_chars=%d)",
        chat_id or "<unknown>",
        len(context),
    )

    summary_candidate = _summarize_context(context)
    if summary_candidate:
        final_summary = _trim_one_line(summary_candidate)
        _logger.debug(
            "Non-chunked summary generated for chat %s preview=%r",
            chat_id or "<unknown>",
            final_summary[:120],
        )
        return final_summary
    return ""


def summarize_chats(
    chats: Iterable[Dict[str, object]],
    messages: Sequence[Mapping[str, object]],
    on_progress: ProgressCallback = None,
    *,
    replace_existing: bool = False,
    on_chat_complete: Optional[Callable[[Dict[str, str]], None]] = None,
) -> tuple[Dict[str, str], Dict[str, int]]:
    """Summarize chats individually and return a map of chat_id -> summary plus stats.

    Args:
        chats (Iterable[Dict[str, object]]): Chat metadata dictionaries.
        messages (Sequence[Mapping[str, object]]): All messages in the dataset.
        on_progress (ProgressCallback, optional): Callback invoked after each chat.
        replace_existing (bool, optional): When True, regenerate summaries even if one already exists.
        on_chat_complete (Callable[[Dict[str, str]], None], optional): Callback invoked after each chat
            is processed with a single-entry dict {chat_id: summary} for immediate persistence.
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

        summary_text = _summarize_with_chunks(
            lines,
            fallback_context=context,
            on_progress=on_progress,
            progress_index=idx,
            total=total,
            chat_id=chat_id,
        )

        if summary_text:
            generated += 1
            # preserve full summary text; no enforced cap
            summary_map[chat_id] = summary_text
            _set_chat_summary(chat, summary_text)
            outcome = "generated"
            if on_chat_complete:
                try:
                    on_chat_complete({chat_id: summary_text})
                    _logger.info("Persisted summary for chat %s", chat_id)
                except Exception as exc:  # pragma: no cover
                    _logger.warning("Failed to persist summary for chat %s: %s", chat_id, exc, exc_info=True)
        else:
            failures += 1
            _set_chat_summary(chat, previous_summary if previous_summary else "")
            outcome = "failed"

        if on_progress:
            details = {
                "type": "chat",
                "chat_id": chat_id,
                "completed": idx,
                "total": total,
                "outcome": outcome,
                "event_id": f"chat-{chat_id or 'unknown'}-{idx}",
            }
            try:
                on_progress(idx, total, chat_id, outcome, details)
            except Exception:  # pragma: no cover
                pass

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
