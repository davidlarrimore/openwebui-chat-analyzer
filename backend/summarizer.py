"""Local two-stage summarizer for chat headlines."""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from uuid import uuid4

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "256"))
SALIENT_K = int(os.getenv("SALIENT_K", "10"))
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


ProgressCallback = Optional[Callable[[int, int, str, str, Optional[Dict[str, object]]], None]]


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

    for idx, chat in enumerate(chats_list, start=1):
        chat_id_raw = chat.get("chat_id")
        chat_id = str(chat_id_raw) if chat_id_raw not in (None, "") else ""
        existing_summary = str(chat.get("summary_128") or "").strip()
        if existing_summary:
            chat["summary_128"] = existing_summary
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

        if not chat_id:
            chat["summary_128"] = ""
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

        context = _build_context(grouped.get(chat_id, []))
        if not context:
            chat["summary_128"] = ""
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

        summary_text = _summarize_context(context)
        summary_text = _trim_one_line(summary_text)
        chat["summary_128"] = summary_text

        if summary_text:
            generated += 1
            summary_map[chat_id] = summary_text
            outcome = "generated"
        else:
            failures += 1
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
    summarize_chats(chats, messages)
