"""SQLAlchemy ORM models for the Open WebUI Chat Analyzer."""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
    Index,
)

from .db import Base


class TimestampMixin:
    """Mixin that adds created_at/updated_at audit fields."""

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class ChatRecord(TimestampMixin, Base):
    """Chat metadata derived from the Open WebUI export."""

    __tablename__ = "chats"

    id = Column(Integer, primary_key=True)
    chat_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    title = Column(String(512), nullable=True)
    gen_chat_summary = Column(String(2500), nullable=True)
    created_ts = Column(DateTime(timezone=True), nullable=True)
    updated_ts = Column(DateTime(timezone=True), nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=True)
    archived = Column(Boolean, nullable=False, server_default="false")
    pinned = Column(Boolean, nullable=False, server_default="false")
    tags = Column(JSON, nullable=False, default=list)
    files_uploaded = Column(Integer, nullable=False, server_default="0")
    files = Column(JSON, nullable=False, default=list)
    meta = Column(JSON, nullable=False, default=dict)
    models = Column(JSON, nullable=False, default=list)
    params = Column(JSON, nullable=False, default=dict)
    share_id = Column(String(255), nullable=True)
    folder_id = Column(String(255), nullable=True)
    history_current_id = Column(String(255), nullable=True)


class MessageRecord(TimestampMixin, Base):
    """Flattened chat messages for analytics and summarization."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    message_id = Column(String(255), unique=True, nullable=False, index=True)
    chat_id = Column(
        String(255),
        ForeignKey("chats.chat_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parent_id = Column(String(255), nullable=True)
    role = Column(String(64), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=True)
    model = Column(String(255), nullable=True)
    models = Column(JSON, nullable=False, default=list)
    children_ids = Column(JSON, nullable=False, default=list)
    follow_ups = Column(JSON, nullable=False, default=list)
    status_history = Column(JSON, nullable=False, default=list)
    sources = Column(JSON, nullable=False, default=list)
    files = Column(JSON, nullable=False, default=list)
    annotation = Column(JSON, nullable=True)
    model_name = Column(String(255), nullable=True)
    model_index = Column(Integer, nullable=True)
    last_sentence = Column(Text, nullable=True)
    done = Column(Boolean, nullable=True)
    favorite = Column(Boolean, nullable=True)
    feedback_id = Column(String(255), nullable=True)
    error = Column(JSON, nullable=True)


Index("ix_messages_chat_parent", MessageRecord.chat_id, MessageRecord.parent_id)


class OpenWebUIUser(TimestampMixin, Base):
    """Users imported from the Open WebUI export."""

    __tablename__ = "owui_users"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(320), nullable=True)
    role = Column(String(64), nullable=True)


class ModelRecord(TimestampMixin, Base):
    """Model metadata harvested from Open WebUI."""

    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    model_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    owned_by = Column(String(255), nullable=True)
    connection_type = Column(String(255), nullable=True)
    object_type = Column(String(255), nullable=True)
    raw = Column(JSON, nullable=False, default=dict)


class Account(TimestampMixin, Base):
    """Analyzer application users that can authenticate with the backend."""

    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True)
    username = Column(String(320), nullable=False, unique=True, index=True)
    display_name = Column(String(320), nullable=False)
    password_hash = Column(Text, nullable=False)
    is_active = Column(Boolean, nullable=False, server_default="true")


class Setting(TimestampMixin, Base):
    """Key-value application settings sourced from app.json."""

    __tablename__ = "settings"

    id = Column(Integer, primary_key=True)
    key = Column(String(128), nullable=False)
    value = Column(JSON, nullable=False, default=dict)

    __table_args__ = (
        UniqueConstraint("key", name="uq_settings_key"),
    )


class IngestLog(TimestampMixin, Base):
    """Operational log describing dataset ingestion and sync events."""

    __tablename__ = "ingest_logs"

    id = Column(Integer, primary_key=True)
    operation = Column(String(128), nullable=False, index=True)
    source = Column(String(512), nullable=True)
    record_count = Column(Integer, nullable=False, server_default="0")
    details = Column(JSON, nullable=False, default=dict)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)


class DatasetSnapshot(TimestampMixin, Base):
    """Snapshot of dataset level metrics previously persisted to app.json."""

    __tablename__ = "dataset_snapshots"

    id = Column(Integer, primary_key=True)
    dataset_source = Column(String(512), nullable=True)
    dataset_pulled_at = Column(DateTime(timezone=True), nullable=True)
    chats_uploaded_at = Column(DateTime(timezone=True), nullable=True)
    users_uploaded_at = Column(DateTime(timezone=True), nullable=True)
    models_uploaded_at = Column(DateTime(timezone=True), nullable=True)
    first_chat_day = Column(Date, nullable=True)
    last_chat_day = Column(Date, nullable=True)
    chat_count = Column(Integer, nullable=False, server_default="0")
    user_count = Column(Integer, nullable=False, server_default="0")
    model_count = Column(Integer, nullable=False, server_default="0")
    message_count = Column(Integer, nullable=False, server_default="0")
