"""add_events_table

Revision ID: 7827eb761c8e
Revises: a1b2c3d4e5f6
Create Date: 2026-03-17 18:45:38.203473

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '7827eb761c8e'
down_revision: Union[str, None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "message_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("message_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("sequence_number", sa.Integer(), nullable=False),
        sa.Column("event_data", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("modified_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"]),
    )

    # Create a function to handle soft deletes for message events from messages
    op.execute("""
        CREATE OR REPLACE FUNCTION soft_delete_message_events_from_messages()
        RETURNS TRIGGER AS $$
        BEGIN
            UPDATE message_events
            SET deleted_at = NEW.deleted_at
            WHERE message_id = NEW.id
            AND deleted_at IS NULL;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Create a trigger to handle soft deletes for message events from messages
    op.execute("""
        CREATE TRIGGER trigger_soft_delete_message_events_from_messages
        AFTER UPDATE OF deleted_at ON messages
        FOR EACH ROW
        WHEN (NEW.deleted_at IS NOT NULL AND OLD.deleted_at IS NULL)
        EXECUTE FUNCTION soft_delete_message_events_from_messages();
    """)

    # Create indexes for better performance
    op.create_index(
        "ix_message_events_message_id_sequence_number",
        "message_events",
        ["message_id", "sequence_number"],
        postgresql_where=sa.text("deleted_at IS NULL"),
        unique=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop triggers
    op.execute("DROP TRIGGER IF EXISTS trigger_soft_delete_message_events_from_messages ON messages;")
    op.execute("DROP FUNCTION IF EXISTS soft_delete_message_events_from_messages();")

    # Drop indexes
    op.drop_index("ix_message_events_message_id_sequence_number", table_name="message_events")

    # Drop table
    op.drop_table("message_events")
