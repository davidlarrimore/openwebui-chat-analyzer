"""Local two-stage summarizer for chat headlines."""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "256"))
SALIENT_K = int(os.getenv("SALIENT_K", "10"))
SUMMARY_BATCH_SIZE = max(1, int(os.getenv("SUMMARY_BATCH_SIZE", "25")))
SUMMARY_BATCH_COMPLETION_WINDOW = os.getenv("SUMMARY_BATCH_COMPLETION_WINDOW", "24h")
SUMMARY_BATCH_POLL_INTERVAL = float(os.getenv("SUMMARY_BATCH_POLL_INTERVAL", "5.0"))
SUMMARY_BATCH_TIMEOUT_SECONDS = int(os.getenv("SUMMARY_BATCH_TIMEOUT_SECONDS", "3600"))
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama3.2:3b-instruct")

_logger = logging.getLogger(__name__)
if not _logger.handlers:
    logging.basicConfig(level=logging.INFO)

_embeddings_model = SentenceTransformer(EMB_MODEL_NAME)
_client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

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
    if not text:
        return ""
    text = text.replace("```", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chat_lines(messages: Sequence[Mapping[str, object]]) -> List[str]:
    lines: List[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        if role not in {"user", "assistant"}:
            continue
        content = _clean(str(message.get("content", "")))
        if content:
            lines.append(f"{role}: {content}")
    return lines


def _cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    return embeddings @ embeddings.T


def _select_salient(lines: Sequence[str], k: int) -> List[str]:
    if not lines:
        return []

    limit = min(max(k, 8), 12)
    if len(lines) <= limit:
        return list(lines)

    embeddings = _embeddings_model.encode(
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
            scores.append(lambda_param * rel - (1 - lambda_param) * div)

        next_idx = int(np.argmax(scores))
        if scores[next_idx] < -1e8:
            break
        selected.append(next_idx)

    chosen = sorted(pre_idx[idx] for idx in selected)
    return [lines[idx] for idx in chosen]


def _trim_one_line(text: str) -> str:
    cleaned = _clean(text)
    cleaned = cleaned.split("\n", 1)[0]
    cleaned = cleaned.strip('"\'' "“”‘’ ").rstrip(".")
    if len(cleaned) > MAX_CHARS:
        return cleaned[: MAX_CHARS - 1] + "…"
    return cleaned


def _build_context(messages: Sequence[Mapping[str, object]]) -> str:
    lines = _chat_lines(messages)
    if not lines:
        return ""

    limit = min(max(SALIENT_K, 8), 12)
    try:
        salient = _select_salient(lines, limit)
    except Exception:
        salient = lines[:limit]

    context = " ".join(salient)
    return context


def _headline_with_llm(context: str) -> str:
    response = _client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.1,
        max_tokens=64,
        messages=[
            {"role": "system", "content": _HEADLINE_SYS},
            {"role": "user", "content": _HEADLINE_USER_TMPL.format(ctx=context)},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    return _trim_one_line(text)


def _chunked(sequence: Sequence[Dict[str, object]], size: int) -> Iterable[List[Dict[str, object]]]:
    for start in range(0, len(sequence), size):
        yield list(sequence[start : start + size])


def _execute_batch_requests(requests: Sequence[Tuple[str, str]]) -> Dict[str, str]:
    if not requests:
        return {}

    payload_lines = []
    for custom_id, context in requests:
        body = {
            "model": OPENAI_MODEL,
            "temperature": 0.1,
            "max_tokens": 64,
            "messages": [
                {"role": "system", "content": _HEADLINE_SYS},
                {"role": "user", "content": _HEADLINE_USER_TMPL.format(ctx=context)},
            ],
        }
        payload_lines.append(
            json.dumps(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
        )

    payload_bytes = ("\n".join(payload_lines)).encode("utf-8")
    buffer = io.BytesIO(payload_bytes)
    buffer.name = "summary-batch.jsonl"
    file_obj = None
    batch = None
    output_bytes = b""
    try:
        file_obj = _client.files.create(file=("summary-batch.jsonl", buffer), purpose="batch")
        buffer.close()

        batch = _client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window=SUMMARY_BATCH_COMPLETION_WINDOW,
        )

        start_time = time.time()
        while batch.status not in {"completed", "failed", "cancelled", "expired"}:
            if time.time() - start_time > SUMMARY_BATCH_TIMEOUT_SECONDS:
                raise TimeoutError("Batch job timed out before completion.")
            time.sleep(SUMMARY_BATCH_POLL_INTERVAL)
            batch = _client.batches.retrieve(batch.id)

        if batch.status != "completed":
            error_details = ""
            error_file_id = getattr(batch, "error_file_id", None)
            if error_file_id:
                try:
                    error_stream = _client.files.content(error_file_id)
                    error_bytes = error_stream.read()
                    if hasattr(error_stream, "close"):
                        error_stream.close()
                    error_details = error_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    error_details = ""
            raise RuntimeError(f"Batch job ended with status {batch.status}. {error_details}")

        output_stream = _client.files.content(batch.output_file_id)
        output_bytes = output_stream.read()
        if hasattr(output_stream, "close"):
            output_stream.close()
    finally:

        buffer.close()
        if file_obj is not None:
            try:
                _client.files.delete(file_obj.id)
            except Exception:
                pass
        if batch is not None:
            for file_id_attr in ("output_file_id", "error_file_id"):
                file_id = getattr(batch, file_id_attr, None)
                if file_id:
                    try:
                        _client.files.delete(file_id)
                    except Exception:
                        pass

    decoded = output_bytes.decode("utf-8")
    result_map: Dict[str, str] = {}
    for line in decoded.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        custom_id = payload.get("custom_id")
        if not custom_id:
            continue
        response = payload.get("response", {})
        status_code = response.get("status_code")
        if status_code == 200:
            body = response.get("body", {})
            choices = body.get("choices", [])
            if choices:
                message = choices[0].get("message") or {}
                content = (message.get("content") or "").strip()
                result_map[custom_id] = content
        else:
            result_map[custom_id] = ""

    return result_map


def _summarize_context(context: str) -> str:
    if not context:
        return ""
    try:
        headline = _headline_with_llm(context)
        if headline:
            return headline
        _logger.warning("LLM returned an empty summary; using fallback snippet.")
    except Exception as exc:
        _logger.warning("LLM summary request failed; falling back to snippet: %s", exc, exc_info=True)
    return _trim_one_line(context[:MAX_CHARS])


def summarize_chat(messages: Sequence[Mapping[str, object]]) -> str:
    context = _build_context(messages)
    if not context:
        return ""

    return _summarize_context(context)


ProgressCallback = Optional[Callable[[int, int, str, str], None]]


def summarize_chats(
    chats: Iterable[Dict[str, object]],
    messages: Sequence[Mapping[str, object]],
    on_progress: ProgressCallback = None,
) -> tuple[Dict[str, str], Dict[str, int]]:
    """Summarize chats and return a map of chat_id -> summary plus stats."""
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
        grouped[chat_id].append(message)

    summary_map: Dict[str, str] = {}
    skipped = 0
    failures = 0
    generated = 0

    pending: List[Dict[str, object]] = []

    for idx, chat in enumerate(chats_list, start=1):
        chat_id_raw = chat.get("chat_id")
        chat_id = str(chat_id_raw) if chat_id_raw not in (None, "") else ""
        existing_summary = str(chat.get("summary_128") or "").strip()
        if existing_summary:
            chat["summary_128"] = existing_summary
            skipped += 1
            if on_progress:
                try:
                    on_progress(idx, total, chat_id, "skipped")
                except Exception:  # pragma: no cover
                    pass
            continue

        if not chat_id:
            chat["summary_128"] = ""
            failures += 1
            if on_progress:
                try:
                    on_progress(idx, total, chat_id, "failed")
                except Exception:  # pragma: no cover
                    pass
            continue

        context = _build_context(grouped.get(chat_id, []))
        if not context:
            chat["summary_128"] = ""
            failures += 1
            if on_progress:
                try:
                    on_progress(idx, total, chat_id, "failed")
                except Exception:  # pragma: no cover
                    pass
            continue

        pending.append(
            {
                "progress_idx": idx,
                "chat": chat,
                "chat_id": chat_id,
                "context": context,
                "custom_id": f"chat-summary-{uuid4().hex}",
            }
        )

    for chunk in _chunked(pending, SUMMARY_BATCH_SIZE):
        chunk_requests = [(entry["custom_id"], entry["context"]) for entry in chunk]
        try:
            chunk_results = _execute_batch_requests(chunk_requests)
        except Exception as exc:  # pragma: no cover
            _logger.warning(
                "Batch summarization failed for %d chats; falling back to single completions: %s",
                len(chunk_requests),
                exc,
                exc_info=True,
            )
            chunk_results = {}

        for entry in chunk:
            custom_id = entry["custom_id"]
            context = entry["context"]
            summary_text = (chunk_results.get(custom_id) or "").strip()
            if not summary_text:
                summary_text = _summarize_context(context)

            summary_text = _trim_one_line(summary_text)
            chat_obj: Dict[str, object] = entry["chat"]
            chat_obj["summary_128"] = summary_text
            chat_id_value = entry["chat_id"]

            if summary_text:
                generated += 1
                summary_map[chat_id_value] = summary_text
                outcome = "generated"
            else:
                failures += 1
                outcome = "failed"

            if on_progress:
                try:
                    on_progress(entry["progress_idx"], total, chat_id_value, outcome)
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
    summarize_chats(chats, messages)
