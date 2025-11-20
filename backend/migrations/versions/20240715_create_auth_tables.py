"""create auth tables

Revision ID: 20240715_create_auth
Revises: 
Create Date: 2024-07-15 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20240715_create_auth"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "auth_users",
        sa.Column("id", sa.String(length=40), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("1"), nullable=False),
        sa.Column("is_admin", sa.Boolean(), server_default=sa.text("0"), nullable=False),
        sa.Column("provider", sa.String(length=32), server_default="local", nullable=False),
        sa.Column("provider_subject", sa.String(length=255), nullable=True),
        sa.Column("tenant", sa.String(length=255), nullable=True),
        sa.Column("display_name", sa.String(length=320), nullable=True),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "auth_sessions",
        sa.Column("id", sa.String(length=72), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("user_id", sa.String(length=40), sa.ForeignKey("auth_users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("refresh_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("refresh_token_hash", sa.String(length=128), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rotated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("user_agent_hash", sa.String(length=128), nullable=True),
        sa.Column("ip_prefix", sa.String(length=64), nullable=True),
    )
    op.create_index("ix_auth_users_email", "auth_users", ["email"], unique=True)
    op.create_index("ix_auth_users_provider_subject", "auth_users", ["provider_subject"])
    op.create_index("ix_auth_sessions_user", "auth_sessions", ["user_id"])
    op.create_index("ix_auth_sessions_expiry", "auth_sessions", ["expires_at"])
    op.create_index("ix_auth_sessions_refresh_expiry", "auth_sessions", ["refresh_expires_at"])


def downgrade() -> None:
    op.drop_index("ix_auth_sessions_refresh_expiry", table_name="auth_sessions")
    op.drop_index("ix_auth_sessions_expiry", table_name="auth_sessions")
    op.drop_index("ix_auth_sessions_user", table_name="auth_sessions")
    op.drop_table("auth_sessions")
    op.drop_index("ix_auth_users_provider_subject", table_name="auth_users")
    op.drop_index("ix_auth_users_email", table_name="auth_users")
    op.drop_table("auth_users")
