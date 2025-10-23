"""Database engine and session management for the Open WebUI Chat Analyzer."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from dotenv import load_dotenv

# Load environment from .env if available so database configuration is discoverable.
load_dotenv()


class Base(DeclarativeBase):
    """Base declarative class used by all ORM models."""


def _build_sqlite_url() -> str:
    """Construct the default SQLite connection string."""
    sqlite_path_env = os.getenv("OWUI_SQLITE_PATH", "data/openwebui_chat_analyzer.db")
    if sqlite_path_env == ":memory:":
        return "sqlite:///:memory:"

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sqlite_path = os.path.expanduser(sqlite_path_env)
    if not os.path.isabs(sqlite_path):
        sqlite_path = os.path.normpath(os.path.join(project_root, sqlite_path))

    directory = os.path.dirname(sqlite_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return f"sqlite:///{sqlite_path}"


def _build_database_url() -> str:
    """Construct a SQLite connection string from environment variables."""
    url = os.getenv("OWUI_DB_URL", "").strip()
    if url.startswith("sqlite://"):
        return url
    return _build_sqlite_url()


DATABASE_URL = _build_database_url()
engine_kwargs = {
    "future": True,
    "pool_pre_ping": True,
}

if DATABASE_URL.startswith("sqlite"):
    # SQLite requires disabling same-thread checks for multi-threaded FastAPI workers.
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    **engine_kwargs,
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


def _run_migrations() -> None:
    """Run schema migrations for existing databases."""
    inspector = inspect(engine)

    # Migration: Rename summary column and widen capacity.
    if inspector.has_table("chats"):
        chat_columns = {col["name"]: col for col in inspector.get_columns("chats")}
        with engine.begin() as conn:
            if "gen_chat_summary" not in chat_columns and "summary_128" in chat_columns:
                conn.execute(text("ALTER TABLE chats RENAME COLUMN summary_128 TO gen_chat_summary"))
            elif "gen_chat_summary" not in chat_columns:
                conn.execute(text("ALTER TABLE chats ADD COLUMN gen_chat_summary VARCHAR(2500)"))

        if engine.dialect.name != "sqlite":
            try:
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE chats ALTER COLUMN gen_chat_summary TYPE VARCHAR(2500)"))
            except Exception:
                # Some engines may not support altering the type; ignore failures to keep initialization resilient.
                pass

    # Migration 1: Add missing columns to owui_users table
    if inspector.has_table("owui_users"):
        columns = [col["name"] for col in inspector.get_columns("owui_users")]
        with engine.begin() as conn:
            if "email" not in columns:
                conn.execute(text("ALTER TABLE owui_users ADD COLUMN email VARCHAR(320)"))
            if "role" not in columns:
                conn.execute(text("ALTER TABLE owui_users ADD COLUMN role VARCHAR(64)"))


def init_database() -> None:
    """Ensure all ORM tables are created in the configured database."""
    # Import models within the function to avoid circular imports.
    from . import db_models  # noqa: F401  # pylint: disable=unused-import

    Base.metadata.create_all(bind=engine)
    _run_migrations()
