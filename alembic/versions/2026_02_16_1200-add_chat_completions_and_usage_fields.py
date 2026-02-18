"""add chat_completions table and usage fields to messages

Revision ID: a1b2c3d4e5f6
Revises: 9efd2d71c54d
Create Date: 2026-02-16 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "9efd2d71c54d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    # Create the chat_completions table
    op.create_table(
        "chat_completions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_name", sa.String(), nullable=True),
        sa.Column("request_id", sa.String(), nullable=True),
        sa.Column("langfuse_trace_id", sa.String(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("modified_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("ix_chat_completions_organization_id", "chat_completions", ["organization_id"])
    op.create_index("ix_chat_completions_user_id", "chat_completions", ["user_id"])
    op.create_index("ix_chat_completions_created_at", "chat_completions", ["created_at"])

    # Reuse the existing update_modified_at_column function for the new table
    op.execute("""
        CREATE TRIGGER update_chat_completions_modified_at
        BEFORE UPDATE ON chat_completions
        FOR EACH ROW EXECUTE FUNCTION update_modified_at_column();
    """)

    # Add usage tracking columns to the messages table
    with op.batch_alter_table("messages") as batch_op:
        batch_op.add_column(sa.Column("model_name", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("request_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("langfuse_trace_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    """Downgrade schema."""

    # Remove usage tracking columns from messages
    with op.batch_alter_table("messages") as batch_op:
        batch_op.drop_column("output_tokens")
        batch_op.drop_column("input_tokens")
        batch_op.drop_column("langfuse_trace_id")
        batch_op.drop_column("request_id")
        batch_op.drop_column("model_name")

    # Drop trigger
    op.execute("DROP TRIGGER IF EXISTS update_chat_completions_modified_at ON chat_completions;")

    # Drop indexes
    op.drop_index("ix_chat_completions_created_at", table_name="chat_completions")
    op.drop_index("ix_chat_completions_user_id", table_name="chat_completions")
    op.drop_index("ix_chat_completions_organization_id", table_name="chat_completions")

    # Drop table
    op.drop_table("chat_completions")
