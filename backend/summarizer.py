"""Local two-stage summarizer for chat headlines."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Runtime-configurable Ollama model and temperature settings
# ---------------------------------------------------------------------------
_SUMMARY_MODEL = (OLLAMA_SUMMARY_MODEL or "").strip()
_SUMMARY_FALLBACK_MODEL = (OLLAMA_SUMMARY_FALLBACK_MODEL or "").strip()
_SUMMARY_TEMPERATURE = OLLAMA_DEFAULT_TEMPERATURE
_SUMMARIZER_ENABLED = True  # Summarizer enabled by default


def get_summary_model() -> str:
    """Return the active Ollama model used for summarization."""
    return _SUMMARY_MODEL or OLLAMA_SUMMARY_MODEL


def get_summary_fallback_model() -> str:
    """Return the configured fallback model, ensuring it differs from the primary."""
    fallback = _SUMMARY_FALLBACK_MODEL or OLLAMA_SUMMARY_FALLBACK_MODEL or ""
    primary = get_summary_model()
    if fallback and fallback == primary:
        return ""
    return fallback


def get_summary_temperature() -> float:
    """Return the active temperature used for summarization."""
    return _SUMMARY_TEMPERATURE


def is_summarizer_enabled() -> bool:
    """Return whether the summarizer is enabled."""
    return _SUMMARIZER_ENABLED


def set_summary_model(model: Optional[str]) -> None:
    """Set the primary Ollama model used for summarization at runtime."""
    global _SUMMARY_MODEL  # noqa: PLW0603 - module-level cache is intentional
    normalized = (model or "").strip()
    _SUMMARY_MODEL = normalized


def set_summary_fallback_model(model: Optional[str]) -> None:
    """Set the fallback Ollama model used for summarization at runtime."""
    global _SUMMARY_FALLBACK_MODEL  # noqa: PLW0603 - module-level cache is intentional
    normalized = (model or "").strip()
    _SUMMARY_FALLBACK_MODEL = normalized


def set_summary_temperature(temperature: float) -> None:
    """Set the temperature used for summarization at runtime."""
    global _SUMMARY_TEMPERATURE  # noqa: PLW0603 - module-level cache is intentional
    _SUMMARY_TEMPERATURE = temperature


def set_summarizer_enabled(enabled: bool) -> None:
    """Enable or disable the summarizer at runtime."""
    global _SUMMARIZER_ENABLED  # noqa: PLW0603 - module-level cache is intentional
    _SUMMARIZER_ENABLED = bool(enabled)


@dataclass
class ConversationAnalysis:
    """Structured analysis result from LLM containing conversation metrics.

    This dataclass represents the extracted insights from analyzing a chat conversation.
    The summary field is ALWAYS plain text (never JSON or structured data).
    The outcome field is an optional integer score from 1-5 rating conversation success.

    Attributes:
        summary: Plain text summary describing the conversation topic and key points.
                 Guaranteed to be a single-line string, never JSON.
        outcome: Optional integer from 1-5 rating conversation success:
                 1 = Not Successful, 2 = Partially Successful, 3 = Moderately Successful,
                 4 = Mostly Successful, 5 = Fully Successful
    """
    summary: str
    outcome: Optional[int] = None

SALIENT_K = int(os.getenv("SALIENT_K", "10"))
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SUMMARY_FIELD = "gen_chat_summary"
OUTCOME_FIELD = "gen_chat_outcome"
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


def _validate_summary_is_plain_text(summary: str) -> str:
    """Validate and clean a summary to ensure it's plain text, never JSON.

    This is a defensive check to prevent JSON strings from being stored as summaries.
    If the summary appears to be JSON, we attempt to extract the actual text.

    Args:
        summary: Candidate summary string to validate.

    Returns:
        A cleaned plain text summary, guaranteed to not be a JSON string.
    """
    if not summary:
        return ""

    cleaned = summary.strip()

    # Check if this looks like JSON (starts with { or [)
    if cleaned.startswith("{") or cleaned.startswith("["):
        _logger.warning(
            "Summary appears to be JSON format, attempting to extract text: %s",
            cleaned[:100]
        )
        try:
            parsed = json.loads(cleaned)
            # If it's a dict with a "summary" field, extract it
            if isinstance(parsed, dict) and "summary" in parsed:
                extracted = str(parsed["summary"]).strip()
                _logger.info("Extracted plain text from JSON summary field")
                return _validate_summary_is_plain_text(extracted)  # Recursive check
            else:
                # JSON but not the expected structure - log and reject
                _logger.error(
                    "Summary is JSON but lacks expected structure. Returning empty. Content: %s",
                    cleaned[:200]
                )
                return ""
        except json.JSONDecodeError:
            # Starts with { or [ but isn't valid JSON - might be legitimate text
            # that happens to start with a bracket, so allow it
            pass

    return cleaned


def _get_chat_summary(chat: Mapping[str, object]) -> str:
    """Return the current summary value from a chat payload.

    Retrieves the summary from the chat dictionary and ensures it's plain text.

    Args:
        chat: Chat dictionary containing summary field.

    Returns:
        The chat summary as plain text, or empty string if not present.
    """
    primary = chat.get(SUMMARY_FIELD)
    raw_summary = str(primary or "").strip()
    return _validate_summary_is_plain_text(raw_summary)


def _set_chat_summary(chat: Dict[str, object], value: str) -> None:
    """Persist a summary value to the chat payload after validation.

    This function validates that the summary is plain text (never JSON) before
    storing it. This is a critical safeguard to prevent malformed data.

    Args:
        chat: Chat dictionary to update.
        value: Summary text to store (will be validated).
    """
    validated = _validate_summary_is_plain_text(value)
    chat[SUMMARY_FIELD] = validated


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
    "You analyze conversations and return structured data as valid JSON. "
    "You provide a summary and assess conversation outcome."
)

_HEADLINE_USER_TMPL = (
    "Analyze this chat interaction and return ONLY a valid JSON object with three fields:\n\n"
    "1. \"summary\": A single-line summary describing the conversation topic, what the user wanted, and what the assistant provided (no quotes, no trailing punctuation)\n"
    "2. \"outcome\": A numeric score (1-5) rating conversation success:\n"
    "   - 1 = Not Successful\n"
    "       * Irrelevant, incoherent, hallucinated, or fails to understand user intent\n"
    "       * Serious factual errors; no meaningful progress toward user goal\n"
    "   - 2 = Partially Successful\n"
    "       * Intent recognized but response incomplete, incorrect, or overly generic\n"
    "       * Noticeable factual errors; requires significant follow-up to be useful\n"
    "   - 3 = Moderately Successful\n"
    "       * Relevant and helpful but lacks depth, detail, or completeness\n"
    "       * Provides minimal value; user intent addressed only at a basic level\n"
    "   - 4 = Mostly Successful\n"
    "       * Clear, correct, helpful answer with minor omissions or limitations\n"
    "       * Actionable and well-structured; largely fulfills user intent\n"
    "   - 5 = Fully Successful\n"
    "       * Fully satisfies user intent with accurate, complete, contextual, and actionable response\n"
    "       * No follow-up needed; provides strong clarity and value\n"
    "3. \"topics\": A comma-separated list of short topical keywords describing what was discussed (e.g. \"HR, customer service, debugging, pandas\")\n\n"
    "Context:\n{ctx}\n\n"
    "Return ONLY valid JSON in this exact format:\n"
    "{{\"summary\": \"your summary here\", \"outcome\": 4, \"topics\": \"topic1, topic2\"}}"
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
    """Extract and clean the first line of text to use as a summary.

    This function normalizes whitespace, removes markdown code fences,
    and returns only the first line. It ensures that the returned summary
    is plain text without any JSON formatting or multi-line content.

    Args:
        text: Raw text that may contain newlines, excess whitespace, or formatting.

    Returns:
        A single-line string suitable for display as a chat summary.
    """
    cleaned = _clean(text)
    cleaned = cleaned.split("\n", 1)[0]
    return cleaned


def _parse_structured_response(response_text: str) -> ConversationAnalysis:
    """Parse and extract summary text from LLM response, handling various formats.

    The LLM is instructed to return JSON like: {"summary": "...", "outcome": 4}
    However, LLMs may wrap this in explanatory text, markdown code fences, or
    return it in unexpected formats. This function robustly extracts the summary
    and outcome, ensuring the summary is always plain text, never JSON.

    Parsing strategy:
    1. Remove markdown code fences if present
    2. Try parsing the entire response as JSON
    3. If that fails, search for JSON objects within the text
    4. Extract the "summary" field and validate it's plain text
    5. Fall back to using the response text directly if no JSON is found

    Args:
        response_text: Raw response text from the LLM, ideally JSON but possibly wrapped.

    Returns:
        ConversationAnalysis with plain text summary (never JSON) and optional outcome score.
    """
    if not response_text:
        return ConversationAnalysis(summary="", outcome=None)

    original_text = response_text.strip()

    # Step 1: Clean up markdown code blocks if present
    cleaned = original_text
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if len(lines) > 2:
            # Skip first line, remove last line if it's just ```
            if lines[-1].strip() == "```":
                cleaned = "\n".join(lines[1:-1])
            else:
                cleaned = "\n".join(lines[1:])
        cleaned = cleaned.strip()

    # Step 2: Try parsing the entire cleaned response as JSON
    parsed_data = None
    try:
        parsed_data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        # Step 3: JSON parsing failed, try to find JSON object within the text
        # Look for patterns like: {"summary": "...", "outcome": ...}
        match = re.search(r'\{[^{}]*"summary"[^{}]*\}', cleaned)
        if match:
            try:
                parsed_data = json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError):
                _logger.debug("Found JSON-like structure but failed to parse it")

    # Step 4: Extract summary and outcome from parsed JSON
    if parsed_data and isinstance(parsed_data, dict):
        summary_raw = parsed_data.get("summary", "")
        summary = str(summary_raw).strip() if summary_raw else ""
        outcome = parsed_data.get("outcome")

        # Validate outcome is in range 1-5
        if outcome is not None:
            try:
                outcome = int(outcome)
                if outcome < 1 or outcome > 5:
                    _logger.warning("Outcome score %d out of range, ignoring", outcome)
                    outcome = None
            except (ValueError, TypeError):
                _logger.warning("Invalid outcome value type: %s", type(outcome))
                outcome = None

        # Ensure summary is plain text, not nested JSON
        if summary:
            # Check if the summary itself looks like JSON (shouldn't happen, but be defensive)
            if summary.strip().startswith("{") or summary.strip().startswith("["):
                try:
                    # If it parses as JSON, extract text from it or reject it
                    nested = json.loads(summary)
                    if isinstance(nested, dict) and "summary" in nested:
                        summary = str(nested["summary"]).strip()
                    else:
                        # Nested JSON but no summary field, just use as-is and clean it
                        _logger.warning("Summary field contained unexpected JSON structure")
                        summary = _trim_one_line(summary)
                except json.JSONDecodeError:
                    # Not actually JSON despite looking like it, use as-is
                    pass

            # Clean and trim the summary to a single line
            summary = _trim_one_line(summary)

            if summary:
                return ConversationAnalysis(summary=summary, outcome=outcome)

    # Step 5: Fall back to using the original response text as summary
    _logger.debug("No valid JSON structure found in LLM response, using text directly")
    summary = _trim_one_line(original_text)
    return ConversationAnalysis(summary=summary, outcome=None)


def _build_context(messages: Sequence[Mapping[str, object]]) -> str:
    """Build a condensed context window from a chat's messages."""
    lines = _chat_lines(messages)
    return _build_context_from_lines(lines)


def _headline_with_ollama(context: str, *, model: Optional[str] = None) -> ConversationAnalysis:
    """Generate structured conversation analysis using the local Ollama service.

    Sends a prompt to Ollama requesting JSON output with summary and outcome fields.
    The response is parsed and validated to ensure the summary is plain text.

    Args:
        context: Condensed conversation text to analyze.
        model: Ollama model identifier to use for generation.

    Returns:
        ConversationAnalysis with plain text summary and optional outcome score.
    """
    active_model = (model or "").strip() or get_summary_model()
    client = get_ollama_client()
    prompt = _HEADLINE_USER_TMPL.format(ctx=context)
    _logger.debug(
        "Requesting Ollama analysis model=%s context_chars=%d",
        active_model,
        len(context),
    )
    options = {
        "temperature": get_summary_temperature(),
        "num_predict": 64,  # Increased for JSON response
        "num_ctx": 1024,
    }
    result = client.generate(
        prompt=prompt,
        model=active_model,
        system=_HEADLINE_SYS,
        options=options,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )
    analysis = _parse_structured_response(result.response)
    _logger.debug(
        "Ollama analysis response model=%s chars=%d summary=%r outcome=%s",
        result.model,
        len(result.response or ""),
        analysis.summary[:120],
        analysis.outcome,
    )
    return analysis


def _headline_with_openwebui(context: str) -> ConversationAnalysis:
    """Call the legacy Open WebUI completion endpoint to obtain structured analysis.

    Args:
        context (str): Condensed conversation snippet used as the LLM prompt.
    Returns:
        ConversationAnalysis: Structured analysis with summary and outcome.
    """
    payload = {
        "model": OWUI_COMPLETIONS_MODEL,
        "temperature": 0.1,
        "max_tokens": 128,  # Increased for JSON response
        "messages": [
            {"role": "system", "content": _HEADLINE_SYS},
            {"role": "user", "content": _HEADLINE_USER_TMPL.format(ctx=context)},
        ],
    }

    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug("OpenWebUI analysis payload: %s", _format_debug_json(payload))

    try:
        response = _requests_session.post(
            _OWUI_COMPLETIONS_URL,
            json=payload,
            timeout=60,
        )
        _logger.debug(
            "OpenWebUI analysis request status=%s",
            response.status_code,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Open WebUI completion request failed: {exc}") from exc

    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug(
            "OpenWebUI analysis raw response: %s",
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
        _logger.debug("OpenWebUI analysis parsed JSON: %s", _format_debug_json(data))

    analysis = _parse_structured_response(text)
    _logger.debug(
        "OpenWebUI analysis response chars=%d summary=%r outcome=%s",
        len(text),
        analysis.summary[:120],
        analysis.outcome,
    )
    return analysis


def _headline_with_llm(context: str) -> ConversationAnalysis:
    """Attempt structured conversation analysis via Ollama, falling back to Open WebUI if needed."""
    _logger.debug("Analyzing conversation via LLM for context_chars=%d", len(context))
    primary_model = get_summary_model()
    fallback_candidate = get_summary_fallback_model()
    try:
        analysis = _headline_with_ollama(context)
        if analysis.summary:
            _logger.debug("Analysis obtained via Ollama")
            return analysis
    except OllamaOutOfMemoryError as exc:  # pragma: no cover - depends on runtime model.
        fallback_model = fallback_candidate
        if fallback_model:
            _logger.warning(
                "Ollama model %s ran out of memory; retrying with fallback model %s.",
                primary_model,
                fallback_model,
                exc_info=True,
            )
            try:
                analysis = _headline_with_ollama(context, model=fallback_model)
                if analysis.summary:
                    _logger.debug("Analysis obtained via Ollama fallback model %s", fallback_model)
                    return analysis
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
        _logger.debug("Open WebUI fallback disabled; returning empty analysis")
        return ConversationAnalysis(summary="", outcome=None)

    try:
        analysis = _headline_with_openwebui(context)
        if analysis.summary:
            _logger.debug("Analysis obtained via Open WebUI fallback")
        return analysis
    except Exception as exc:  # pragma: no cover - logged for observability.
        _logger.warning("Open WebUI fallback failed: %s", exc, exc_info=True)
        _disable_openwebui_fallback(str(exc))
        return ConversationAnalysis(summary="", outcome=None)


def _estimate_token_length(text: str) -> int:
    """Approximate the token length of a prompt using a simple character heuristic."""
    if not text:
        return 0
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


def _summarize_context(context: str) -> ConversationAnalysis:
    """Generate structured analysis via the LLM with a deterministic fallback.

    Args:
        context (str): Concatenated salient conversation lines.
    Returns:
        ConversationAnalysis: Analysis with summary and outcome, or fallback summary with no outcome.
    """
    if not context:
        return ConversationAnalysis(summary="", outcome=None)
    _logger.debug("Analyzing single chat context_chars=%d", len(context))
    try:
        analysis = _headline_with_llm(context)
        if analysis.summary:
            _logger.debug("Analysis obtained via LLM summary=%r outcome=%s", analysis.summary[:120], analysis.outcome)
            return analysis
        _logger.warning("LLM returned an empty summary; using fallback snippet.")
    except Exception as exc:
        _logger.warning("LLM analysis request failed; falling back to snippet: %s", exc, exc_info=True)
    fallback_summary = _trim_one_line(context)
    _logger.debug("Using fallback snippet preview=%r", fallback_summary[:120])
    return ConversationAnalysis(summary=fallback_summary, outcome=None)


def summarize_chat(messages: Sequence[Mapping[str, object]]) -> ConversationAnalysis:
    """Analyze a single chat's message history.

    Args:
        messages (Sequence[Mapping[str, object]]): Exported messages for a single chat.
    Returns:
        ConversationAnalysis: Structured analysis with summary and outcome score.
    """
    context = _build_context(messages)
    if not context:
        return ConversationAnalysis(summary="", outcome=None)

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
) -> ConversationAnalysis:
    """Analyze a chat using a single request against the LLM."""
    context = fallback_context.strip()
    if not context:
        joined_lines = " ".join(str(line).strip() for line in lines if str(line).strip())
        context = joined_lines
    if not context:
        return ConversationAnalysis(summary="", outcome=None)

    _logger.debug(
        "Analyzing chat %s without chunking (context_chars=%d)",
        chat_id or "<unknown>",
        len(context),
    )

    analysis = _summarize_context(context)
    if analysis.summary:
        _logger.debug(
            "Analysis generated for chat %s summary=%r outcome=%s",
            chat_id or "<unknown>",
            analysis.summary[:120],
            analysis.outcome,
        )
        return analysis
    return ConversationAnalysis(summary="", outcome=None)


def summarize_chats(
    chats: Iterable[Dict[str, object]],
    messages: Sequence[Mapping[str, object]],
    on_progress: ProgressCallback = None,
    *,
    replace_existing: bool = False,
    on_chat_complete: Optional[Callable[[Dict[str, str], Dict[str, int]], None]] = None,
) -> tuple[Dict[str, str], Dict[str, int]]:
    """Summarize chats individually and return a map of chat_id -> summary plus stats.

    This function processes each chat conversation, generates a plain text summary and
    outcome score using an LLM, and ensures all summaries are validated to be plain text
    (never JSON or structured data). Summaries and outcomes are persisted both in-memory
    and optionally via callback for immediate database storage.

    Args:
        chats: Chat metadata dictionaries to summarize.
        messages: All messages in the dataset, will be grouped by chat_id.
        on_progress: Optional callback invoked after each chat with progress info.
        replace_existing: When True, regenerate summaries even if one already exists.
        on_chat_complete: Optional callback invoked after each chat is processed with
                          two dicts: {chat_id: summary} and {chat_id: outcome} for
                          immediate persistence.

    Returns:
        A tuple of (summary_map, stats) where:
        - summary_map: Dict mapping chat_id to plain text summary string
        - stats: Dict with keys 'total', 'skipped', 'failures', 'generated'
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

        analysis = _summarize_with_chunks(
            lines,
            fallback_context=context,
            on_progress=on_progress,
            progress_index=idx,
            total=total,
            chat_id=chat_id,
        )

        if analysis.summary:
            generated += 1
            # Store both summary and outcome in the chat dict
            summary_map[chat_id] = analysis.summary
            _set_chat_summary(chat, analysis.summary)
            if analysis.outcome is not None:
                chat[OUTCOME_FIELD] = analysis.outcome
            outcome = "generated"
            if on_chat_complete:
                try:
                    # Pass both summary and outcome to callback for persistence
                    outcome_map = {chat_id: analysis.outcome} if analysis.outcome is not None else {}
                    on_chat_complete({chat_id: analysis.summary}, outcome_map)
                    _logger.info("Persisted summary and outcome for chat %s", chat_id)
                except Exception as exc:  # pragma: no cover
                    _logger.warning("Failed to persist metrics for chat %s: %s", chat_id, exc, exc_info=True)
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
