"""SQLite-backed persistence helpers for the Open WebUI Chat Analyzer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import delete, select, text, update, func
from sqlalchemy.exc import SQLAlchemyError

from .db import DATABASE_URL, engine, init_database, session_scope
from .db_models import (
    Account,
    AccessToken,
    ChatRecord,
    DatasetSnapshot,
    IngestLog,
    MessageRecord,
    ModelRecord,
    OpenWebUIUser,
    Setting,
)

LOGGER = logging.getLogger(__name__)


_DATACLASS_KWARGS: Dict[str, Any] = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**_DATACLASS_KWARGS)
class DatabaseState:
    """Container for hydrated database records."""

    chats: List[Dict[str, Any]]
    messages: List[Dict[str, Any]]
    users: List[Dict[str, Any]]
    models: List[Dict[str, Any]]
    accounts: List[Dict[str, Any]]
    snapshot: Optional[Dict[str, Any]]
    settings: Dict[str, Any]


@dataclass(**_DATACLASS_KWARGS)
class AccessTokenState:
    """Metadata representing a stored access token."""

    token_hash: str
    username: str
    issued_at: datetime
    expires_at: datetime
    revoked_at: Optional[datetime]


class DatabaseStorage:
    """Utility class encapsulating all database reads and writes."""

    def __init__(self) -> None:
        safe_url = DATABASE_URL if DATABASE_URL.startswith("sqlite") else "redacted"
        LOGGER.info("Initializing database storage (database_url=%s)", safe_url)
        try:
            init_database()
        except Exception:  # pylint: disable=broad-except
            LOGGER.exception("Database initialisation failed")
            raise
        LOGGER.info("Database initialisation complete")

    # ------------------------------------------------------------------
    # Hydration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _chat_to_dict(record: ChatRecord) -> Dict[str, Any]:
        return {
            "chat_id": record.chat_id,
            "user_id": record.user_id,
            "title": record.title,
            "gen_chat_summary": record.gen_chat_summary,
            "gen_chat_outcome": record.gen_chat_outcome,
            "created_at": record.created_ts,
            "updated_at": record.updated_ts,
            "timestamp": record.timestamp,
            "archived": bool(record.archived),
            "pinned": bool(record.pinned),
            "tags": list(record.tags or []),
            "files_uploaded": int(record.files_uploaded or 0),
            "files": list(record.files or []),
            "meta": dict(record.meta or {}),
            "models": list(record.models or []),
            "params": dict(record.params or {}),
            "share_id": record.share_id,
            "folder_id": record.folder_id,
            "history_current_id": record.history_current_id,
        }

    @staticmethod
    def _message_to_dict(record: MessageRecord) -> Dict[str, Any]:
        return {
            "chat_id": record.chat_id,
            "message_id": record.message_id,
            "parent_id": record.parent_id,
            "role": record.role,
            "content": record.content,
            "timestamp": record.timestamp,
            "model": record.model,
            "models": list(record.models or []),
            "children_ids": list(record.children_ids or []),
            "follow_ups": list(record.follow_ups or []),
            "status_history": list(record.status_history or []),
            "sources": list(record.sources or []),
            "files": list(record.files or []),
            "annotation": record.annotation,
            "model_name": record.model_name,
            "model_index": record.model_index,
            "last_sentence": record.last_sentence,
            "done": record.done,
            "favorite": record.favorite,
            "feedback_id": record.feedback_id,
            "error": record.error,
        }

    @staticmethod
    def _user_to_dict(record: OpenWebUIUser) -> Dict[str, Any]:
        return {
            "user_id": record.user_id,
            "name": record.name,
            "email": record.email,
            "role": record.role,
            "pseudonym": record.pseudonym,
        }

    @staticmethod
    def _model_to_dict(record: ModelRecord) -> Dict[str, Any]:
        return {
            "model_id": record.model_id,
            "name": record.name,
            "owned_by": record.owned_by,
            "connection_type": record.connection_type,
            "object": record.object_type,
            "raw": record.raw,
        }

    @staticmethod
    def _account_to_dict(record: Account) -> Dict[str, Any]:
        return {
            "username": record.username,
            "display_name": record.display_name,
            "password_hash": record.password_hash,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "is_active": bool(record.is_active),
        }

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_state(self) -> DatabaseState:
        """Load the current dataset and metadata from the database."""
        LOGGER.debug("Loading database state from persistent storage")
        with session_scope() as session:
            chats = [self._chat_to_dict(row) for row in session.execute(select(ChatRecord)).scalars()]
            messages = [
                self._message_to_dict(row) for row in session.execute(select(MessageRecord)).scalars()
            ]
            users = [self._user_to_dict(row) for row in session.execute(select(OpenWebUIUser)).scalars()]
            models = [self._model_to_dict(row) for row in session.execute(select(ModelRecord)).scalars()]
            accounts = [self._account_to_dict(row) for row in session.execute(select(Account)).scalars()]
            last_snapshot = (
                session.execute(
                    select(DatasetSnapshot).order_by(DatasetSnapshot.updated_at.desc()).limit(1)
                )
                .scalars()
                .first()
            )
            snapshot = None
            if last_snapshot is not None:
                snapshot = {
                    "dataset_source": last_snapshot.dataset_source,
                    "dataset_pulled_at": last_snapshot.dataset_pulled_at,
                    "chats_uploaded_at": last_snapshot.chats_uploaded_at,
                    "users_uploaded_at": last_snapshot.users_uploaded_at,
                    "models_uploaded_at": last_snapshot.models_uploaded_at,
                    "first_chat_day": last_snapshot.first_chat_day,
                    "last_chat_day": last_snapshot.last_chat_day,
                    "chat_count": last_snapshot.chat_count,
                    "user_count": last_snapshot.user_count,
                    "model_count": last_snapshot.model_count,
                    "message_count": last_snapshot.message_count,
                }

            settings = {
                row.key: row.value for row in session.execute(select(Setting)).scalars()
            }

        state = DatabaseState(
            chats=chats,
            messages=messages,
            users=users,
            models=models,
            accounts=accounts,
            snapshot=snapshot,
            settings=settings,
        )
        LOGGER.info(
            "Loaded database state (%d chats, %d messages, %d users, %d models, %d accounts)",
            len(state.chats),
            len(state.messages),
            len(state.users),
            len(state.models),
            len(state.accounts),
        )
        return state

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_summary(payload: Dict[str, Any]) -> Optional[str]:
        """Extract and validate the chat summary from a payload.

        This method provides a final safety check to ensure that summaries
        are always plain text and never JSON strings. If a JSON string is
        detected, it attempts to extract the actual summary text.

        Args:
            payload: Chat data dictionary potentially containing a summary.

        Returns:
            Plain text summary string, or None if not present/invalid.
        """
        import json as json_lib

        summary = payload.get("gen_chat_summary")
        if not summary:
            return None

        summary_str = str(summary).strip()
        if not summary_str:
            return None

        # Defensive check: ensure this is not a JSON string
        if summary_str.startswith("{") or summary_str.startswith("["):
            LOGGER.warning(
                "Detected JSON-formatted summary in payload for chat %s, attempting to extract plain text",
                payload.get("chat_id", "unknown")
            )
            try:
                parsed = json_lib.loads(summary_str)
                if isinstance(parsed, dict) and "summary" in parsed:
                    extracted = str(parsed["summary"]).strip()
                    LOGGER.info("Successfully extracted plain text from JSON summary")
                    return extracted if extracted else None
                else:
                    LOGGER.error(
                        "Summary is JSON but lacks expected structure for chat %s",
                        payload.get("chat_id", "unknown")
                    )
                    return None
            except json_lib.JSONDecodeError:
                # Not actually JSON, allow it
                pass

        return summary_str

    @staticmethod
    def _chat_from_dict(payload: Dict[str, Any]) -> ChatRecord:
        return ChatRecord(
            chat_id=str(payload.get("chat_id")),
            user_id=payload.get("user_id"),
            title=payload.get("title"),
            gen_chat_summary=DatabaseStorage._extract_summary(payload),
            gen_chat_outcome=payload.get("gen_chat_outcome"),
            created_ts=payload.get("created_at"),
            updated_ts=payload.get("updated_at"),
            timestamp=payload.get("timestamp"),
            archived=bool(payload.get("archived", False)),
            pinned=bool(payload.get("pinned", False)),
            tags=payload.get("tags") or [],
            files_uploaded=int(payload.get("files_uploaded") or 0),
            files=payload.get("files") or [],
            meta=payload.get("meta") or {},
            models=payload.get("models") or [],
            params=payload.get("params") or {},
            share_id=payload.get("share_id"),
            folder_id=payload.get("folder_id"),
            history_current_id=payload.get("history_current_id"),
        )

    @staticmethod
    def _message_from_dict(payload: Dict[str, Any]) -> MessageRecord:
        return MessageRecord(
            chat_id=str(payload.get("chat_id")),
            message_id=str(payload.get("message_id")),
            parent_id=payload.get("parent_id"),
            role=str(payload.get("role") or ""),
            content=str(payload.get("content") or ""),
            timestamp=payload.get("timestamp"),
            model=payload.get("model"),
            models=payload.get("models") or [],
            children_ids=payload.get("children_ids") or [],
            follow_ups=payload.get("follow_ups") or [],
            status_history=payload.get("status_history") or [],
            sources=payload.get("sources") or [],
            files=payload.get("files") or [],
            annotation=payload.get("annotation"),
            model_name=payload.get("model_name"),
            model_index=payload.get("model_index"),
            last_sentence=payload.get("last_sentence"),
            done=payload.get("done"),
            favorite=payload.get("favorite"),
            feedback_id=payload.get("feedback_id"),
            error=payload.get("error"),
        )

    @staticmethod
    def _user_from_dict(payload: Dict[str, Any]) -> OpenWebUIUser:
        return OpenWebUIUser(
            user_id=str(payload.get("user_id")),
            name=str(payload.get("name") or ""),
            email=payload.get("email"),
            role=payload.get("role"),
            pseudonym=payload.get("pseudonym"),
        )

    @staticmethod
    def _model_from_dict(payload: Dict[str, Any]) -> ModelRecord:
        return ModelRecord(
            model_id=str(payload.get("model_id")),
            name=payload.get("name"),
            owned_by=payload.get("owned_by"),
            connection_type=payload.get("connection_type"),
            object_type=payload.get("object"),
            raw=payload.get("raw") or {},
        )

    @staticmethod
    def _account_from_dict(payload: Dict[str, Any]) -> Account:
        account = Account(
            username=str(payload.get("username")).lower(),
            display_name=str(payload.get("display_name") or payload.get("username") or ""),
            password_hash=str(payload.get("password_hash") or ""),
            is_active=bool(payload.get("is_active", True)),
        )
        created_at = payload.get("created_at")
        if isinstance(created_at, datetime):
            account.created_at = created_at
        updated_at = payload.get("updated_at")
        if isinstance(updated_at, datetime):
            account.updated_at = updated_at
        return account

    def replace_dataset(
        self,
        chats: Iterable[Dict[str, Any]],
        messages: Iterable[Dict[str, Any]],
    ) -> None:
        with session_scope() as session:
            session.execute(delete(MessageRecord))
            session.execute(delete(ChatRecord))
            chat_records = [self._chat_from_dict(item) for item in chats if item.get("chat_id")]
            message_records = [
                self._message_from_dict(item) for item in messages if item.get("message_id")
            ]
            LOGGER.info(
                "Persisting dataset (%d chats, %d messages)",
                len(chat_records),
                len(message_records),
            )
            if chat_records:
                session.bulk_save_objects(chat_records)
            if message_records:
                session.bulk_save_objects(message_records)

    def replace_users(self, users: Iterable[Dict[str, Any]]) -> None:
        with session_scope() as session:
            session.execute(delete(OpenWebUIUser))
            user_records = [self._user_from_dict(item) for item in users if item.get("user_id")]
            LOGGER.info("Persisting %d user records", len(user_records))
            if user_records:
                session.bulk_save_objects(user_records)

    def update_user_pseudonyms(self, assignments: Dict[str, str]) -> None:
        """Persist pseudonym updates for a subset of users."""
        if not assignments:
            return
        with session_scope() as session:
            for user_id, alias in assignments.items():
                session.execute(
                    update(OpenWebUIUser)
                    .where(OpenWebUIUser.user_id == user_id)
                    .values(pseudonym=alias)
                )
        LOGGER.info("Updated pseudonyms for %d users", len(assignments))

    def replace_models(self, models: Iterable[Dict[str, Any]]) -> None:
        with session_scope() as session:
            session.execute(delete(ModelRecord))
            model_records = [self._model_from_dict(item) for item in models if item.get("model_id")]
            LOGGER.info("Persisting %d model records", len(model_records))
            if model_records:
                session.bulk_save_objects(model_records)

    def auth_user_count(self) -> int:
        """Return the number of stored authentication accounts."""
        try:
            with session_scope() as session:
                return int(session.execute(select(func.count()).select_from(Account)).scalar_one())
        except SQLAlchemyError:
            LOGGER.exception("Failed to count authentication accounts from database")
            return -1

    def replace_accounts(self, accounts: Iterable[Dict[str, Any]]) -> None:
        with session_scope() as session:
            session.execute(delete(Account))
            account_records = [self._account_from_dict(item) for item in accounts if item.get("username")]
            LOGGER.info("Persisting %d account records", len(account_records))
            if account_records:
                session.bulk_save_objects(account_records)

    # ------------------------------------------------------------------
    # Access token management
    # ------------------------------------------------------------------
    def create_access_token(
        self,
        *,
        token_hash: str,
        username: str,
        issued_at: datetime,
        expires_at: datetime,
    ) -> None:
        """Persist a freshly issued access token."""
        try:
            with session_scope() as session:
                session.add(
                    AccessToken(
                        token_hash=token_hash,
                        username=username,
                        issued_at=issued_at,
                        expires_at=expires_at,
                        revoked_at=None,
                    )
                )
        except SQLAlchemyError:
            LOGGER.exception("Failed to persist access token for user %s", username)
            raise

    def fetch_access_token(self, token_hash: str) -> Optional[AccessTokenState]:
        """Retrieve a stored access token by its hash."""
        try:
            with session_scope() as session:
                row = (
                    session.execute(select(AccessToken).where(AccessToken.token_hash == token_hash))
                    .scalars()
                    .first()
                )
                if row is None:
                    return None
                return AccessTokenState(
                    token_hash=row.token_hash,
                    username=row.username,
                    issued_at=row.issued_at,
                    expires_at=row.expires_at,
                    revoked_at=row.revoked_at,
                )
        except SQLAlchemyError:
            LOGGER.exception("Failed to fetch access token metadata")
            return None

    def revoke_access_token(self, token_hash: str, *, revoked_at: datetime) -> bool:
        """Mark a token as revoked."""
        try:
            with session_scope() as session:
                result = session.execute(
                    update(AccessToken)
                    .where(AccessToken.token_hash == token_hash, AccessToken.revoked_at.is_(None))
                    .values(revoked_at=revoked_at)
                )
                return result.rowcount > 0
        except SQLAlchemyError:
            LOGGER.exception("Failed to revoke access token")
            return False

    def delete_access_token(self, token_hash: str) -> bool:
        """Remove a token record outright."""
        try:
            with session_scope() as session:
                result = session.execute(delete(AccessToken).where(AccessToken.token_hash == token_hash))
                return result.rowcount > 0
        except SQLAlchemyError:
            LOGGER.exception("Failed to delete access token")
            return False

    def prune_expired_tokens(self, *, reference: datetime) -> int:
        """Drop persisted tokens past their expiry."""
        try:
            with session_scope() as session:
                result = session.execute(delete(AccessToken).where(AccessToken.expires_at <= reference))
                return int(result.rowcount or 0)
        except SQLAlchemyError:
            LOGGER.exception("Failed to prune expired access tokens")
            return 0

    def revoke_tokens_for_user(self, username: str, *, revoked_at: datetime) -> int:
        """Revoke all active tokens associated with a username."""
        try:
            with session_scope() as session:
                result = session.execute(
                    update(AccessToken)
                    .where(AccessToken.username == username, AccessToken.revoked_at.is_(None))
                    .values(revoked_at=revoked_at)
                )
                return int(result.rowcount or 0)
        except SQLAlchemyError:
            LOGGER.exception("Failed to revoke access tokens for user %s", username)
            return 0

    def record_snapshot(self, payload: Dict[str, Any]) -> None:
        snapshot = DatasetSnapshot(
            dataset_source=payload.get("dataset_source"),
            dataset_pulled_at=payload.get("dataset_pulled_at"),
            chats_uploaded_at=payload.get("chats_uploaded_at"),
            users_uploaded_at=payload.get("users_uploaded_at"),
            models_uploaded_at=payload.get("models_uploaded_at"),
            first_chat_day=payload.get("first_chat_day"),
            last_chat_day=payload.get("last_chat_day"),
            chat_count=int(payload.get("chat_count") or 0),
            user_count=int(payload.get("user_count") or 0),
            model_count=int(payload.get("model_count") or 0),
            message_count=int(payload.get("message_count") or 0),
        )
        with session_scope() as session:
            session.add(snapshot)
        LOGGER.info(
            "Recorded dataset snapshot (source=%s, chats=%s, messages=%s)",
            payload.get("dataset_source"),
            payload.get("chat_count"),
            payload.get("message_count"),
        )

    def write_settings(self, values: Dict[str, Any]) -> None:
        with session_scope() as session:
            existing = {row.key: row for row in session.execute(select(Setting)).scalars()}
            for key, value in values.items():
                current = existing.get(key)
                if current is None:
                    session.add(Setting(key=key, value=value))
                else:
                    current.value = value
        LOGGER.info("Persisted %d application settings keys", len(values))

    def record_ingest(
        self,
        *,
        operation: str,
        source: Optional[str],
        record_count: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        log = IngestLog(
            operation=operation,
            source=source,
            record_count=record_count,
            details=details or {},
        )
        try:
            with session_scope() as session:
                session.add(log)
            LOGGER.info(
                "Recorded ingest log (operation=%s, source=%s, count=%d)",
                operation,
                source,
                record_count,
            )
        except SQLAlchemyError:
            # Logging ingestion failures should never block the primary flow.
            LOGGER.exception("Failed to persist ingest log for operation %s", operation)

    def update_chat_summaries(
        self,
        summaries: Dict[str, str],
        outcomes: Optional[Dict[str, int]] = None
    ) -> None:
        """Update chat summaries and optionally outcomes in the database.

        Args:
            summaries: Dictionary mapping chat_id to summary text
            outcomes: Optional dictionary mapping chat_id to outcome score (1-5)
        """
        if not summaries:
            return
        with session_scope() as session:
            for chat_id, summary in summaries.items():
                values = {"gen_chat_summary": summary}
                # Include outcome if provided for this chat
                if outcomes and chat_id in outcomes:
                    values["gen_chat_outcome"] = outcomes[chat_id]

                session.execute(
                    update(ChatRecord)
                    .where(ChatRecord.chat_id == chat_id)
                    .values(**values)
                )

        if outcomes:
            LOGGER.info(
                "Updated summaries and outcomes for %d chats (%d with outcomes)",
                len(summaries),
                len(outcomes)
            )
        else:
            LOGGER.info("Updated summaries for %d chats", len(summaries))

    def wipe_openwebui_data(self) -> None:
        """Transactionally wipe all OpenWebUI-sourced data (chats, messages, users, models).

        This performs a clean wipe in a single transaction, respecting FK constraints.
        Messages are deleted first (or cascade from chats), then chats, users, and models.
        """
        LOGGER.info("Starting transactional wipe of OpenWebUI-sourced data")
        with session_scope() as session:
            # Enable FK enforcement for SQLite (no-op for other DBs)
            if engine.dialect.name == "sqlite":
                session.execute(text("PRAGMA foreign_keys = ON"))

            # Delete in FK-safe order: messages, chats, users, models
            # Messages have FK to chats with CASCADE, so they'll be auto-deleted,
            # but we delete explicitly for clarity and to support non-CASCADE scenarios
            msg_count = session.execute(delete(MessageRecord)).rowcount
            chat_count = session.execute(delete(ChatRecord)).rowcount
            user_count = session.execute(delete(OpenWebUIUser)).rowcount
            model_count = session.execute(delete(ModelRecord)).rowcount

            LOGGER.info(
                "Wipe complete: deleted %d messages, %d chats, %d users, %d models",
                msg_count,
                chat_count,
                user_count,
                model_count,
            )

    def fetch_ingest_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with session_scope() as session:
            logs = (
                session.execute(
                    select(IngestLog)
                    .order_by(IngestLog.created_at.desc())
                    .limit(limit)
                )
                .scalars()
                .all()
            )
        LOGGER.info("Fetched %d ingest log entries (limit=%d)", len(logs), limit)
        return [
            {
                "id": log.id,
                "operation": log.operation,
                "source": log.source,
                "record_count": log.record_count,
                "details": dict(log.details or {}),
                "created_at": log.created_at,
                "updated_at": log.updated_at,
                "started_at": log.started_at,
                "finished_at": log.finished_at,
                "notes": log.notes,
            }
            for log in logs
        ]
