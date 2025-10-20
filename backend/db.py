"""Database engine and session management for the Open WebUI Chat Analyzer."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from dotenv import load_dotenv

# Load environment from .env if available so database configuration is discoverable.
load_dotenv()


class Base(DeclarativeBase):
    """Base declarative class used by all ORM models."""


def _build_database_url() -> str:
    """Construct a SQLAlchemy connection string from environment variables."""
    url = os.getenv("OWUI_DB_URL")
    if url:
        return url

    host = os.getenv("OWUI_DB_HOST", "localhost")
    port = os.getenv("OWUI_DB_PORT", "5432")
    name = os.getenv("OWUI_DB_NAME", "openwebui_chat_analyzer")
    user = os.getenv("OWUI_DB_USER", "owui")
    password = os.getenv("OWUI_DB_PASSWORD", "owui_password")

    # Allow passwordless connections by omitting the colon segment entirely.
    if password:
        credentials = f"{quote_plus(user)}:{quote_plus(password)}"
    else:
        credentials = quote_plus(user)
    return f"postgresql+psycopg2://{credentials}@{host}:{port}/{name}"


DATABASE_URL = _build_database_url()
engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
    future=True,
)


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database() -> None:
    """Ensure all ORM tables are created in the configured database."""
    # Import models within the function to avoid circular imports.
    from . import db_models  # noqa: F401  # pylint: disable=unused-import

    Base.metadata.create_all(bind=engine)
