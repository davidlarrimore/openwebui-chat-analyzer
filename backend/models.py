"""Pydantic models for the Open WebUI Chat Analyzer backend."""

from datetime import date, datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class Chat(BaseModel):
    """Serialized chat metadata as exposed by the API.

    Attributes:
        chat_id: Unique identifier for the chat conversation.
        user_id: Owning user identifier, if available.
        title: User-facing conversation title.
        summary_128: Locally generated, truncated summary string.
        created_at: Creation timestamp pulled from the export.
        updated_at: Last modification timestamp pulled from the export.
        archived: Whether the chat is marked archived.
        pinned: Whether the chat is pinned in the source system.
        tags: Optional tags carried over from the export metadata.
        files_uploaded: Number of files attached to the chat.
        files: Opaque metadata about uploaded files, if any.
    """

    chat_id: str = Field(..., description="Unique identifier for the chat")
    user_id: Optional[str] = Field(
        default=None, description="Identifier for the user that owns the chat"
    )
    title: Optional[str] = Field(default=None, description="Chat title")
    summary_128: Optional[str] = Field(
        default=None,
        description="≤128 character locally generated subject line summary",
    )
    created_at: Optional[datetime] = Field(
        default=None, description="Timestamp when the chat was created"
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="Timestamp when the chat was last updated"
    )
    archived: bool = Field(default=False, description="Whether the chat is archived")
    pinned: bool = Field(default=False, description="Whether the chat is pinned")
    tags: List[str] = Field(default_factory=list, description="Tags associated with chat")
    files_uploaded: int = Field(
        default=0, description="Number of files uploaded alongside the chat"
    )
    files: List[Any] = Field(
        default_factory=list,
        description="Raw file metadata from the chat export, if present",
    )


class Message(BaseModel):
    """Serialized chat message used in analytics and summarization.

    Attributes:
        chat_id: Identifier for the parent chat.
        message_id: Stable identifier for the message.
        parent_id: Identifier of the parent message, used for threading.
        role: Role of sender (user/assistant).
        content: Plain-text message content.
        timestamp: Source timestamp from the export, if provided.
        model: Name of the model that produced the response (if assistant).
        models: Alternate model names when the export contains multiple values.
    """

    chat_id: str = Field(..., description="Identifier of the chat this message belongs to")
    message_id: str = Field(..., description="Unique identifier for the message")
    parent_id: Optional[str] = Field(
        default=None, description="Identifier of the parent message if present"
    )
    role: str = Field(..., description="Role of the sender (user or assistant)")
    content: str = Field(..., description="Message content as plain text")
    timestamp: Optional[datetime] = Field(
        default=None, description="Timestamp when the message was created"
    )
    model: str = Field(default="", description="Model name associated with the message")
    models: List[str] = Field(
        default_factory=list, description="List of alternative model names, if any"
    )


class User(BaseModel):
    """Serialized user metadata consumed by the frontend."""

    user_id: str = Field(..., description="Unique user identifier")
    name: str = Field(..., description="Display name")


class AppMetadata(BaseModel):
    """High-level metadata about the current application dataset."""

    dataset_source: str = Field(..., description="Origin of the current dataset")
    dataset_pulled_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when the dataset was retrieved from its source",
    )
    chat_uploaded_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when chat data was most recently ingested",
    )
    users_uploaded_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when user data was most recently ingested",
    )
    first_chat_day: Optional[date] = Field(
        default=None,
        description="Date of the earliest chat/message in the current dataset",
    )
    last_chat_day: Optional[date] = Field(
        default=None,
        description="Date of the latest chat/message in the current dataset",
    )
    chat_count: int = Field(..., description="Number of chat records currently available")
    user_count: int = Field(..., description="Number of user records currently available")


class DatasetMeta(BaseModel):
    """Metadata describing the currently loaded dataset.

    Attributes:
        dataset_id: Synthetic identifier that changes when data updates.
        source: Human-readable description of the dataset origin.
        last_updated: Last modification timestamp for the dataset.
        chat_count: Number of chats currently stored.
        message_count: Number of messages currently stored.
        user_count: Number of user records currently stored.
        app_metadata: Aggregate analytics metadata for the frontend.
    """

    dataset_id: str = Field(..., description="Opaque identifier that changes when data updates")
    source: str = Field(..., description="Human-friendly description of the data source")
    last_updated: datetime = Field(..., description="UTC timestamp of last modification")
    chat_count: int = Field(..., description="Number of chats available")
    message_count: int = Field(..., description="Number of messages available")
    user_count: int = Field(..., description="Number of user records available")
    app_metadata: Optional[AppMetadata] = Field(
        default=None,
        description="Aggregated metadata persisted to app.json for frontend display",
    )


class UploadResponse(BaseModel):
    """Response payload returned after successful uploads."""

    detail: str = Field(..., description="Human readable status message")
    dataset: DatasetMeta = Field(..., description="Dataset metadata after the upload finishes")


class OpenWebUISyncRequest(BaseModel):
    """Payload required to sync data directly from an Open WebUI instance."""

    hostname: str = Field(..., description="Base URL of the Open WebUI instance")
    api_key: Optional[str] = Field(
        default=None, description="Bearer token used to authenticate with Open WebUI"
    )
