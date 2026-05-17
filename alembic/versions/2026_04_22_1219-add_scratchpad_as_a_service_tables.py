"""add scratchpad as a service tables


Revision ID: 1ccaab8f45da
Revises: f56327b2c585
Create Date: 2026-04-22 12:19:56.755028

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1ccaab8f45da"
down_revision: str | None = "f56327b2c585"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "scratchpads",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("backend", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for better performance
    op.create_index(
        "ix_scratchpads_organization_id_user_id_name",
        "scratchpads",
        ["organization_id", "user_id", "name"],
        postgresql_where=sa.text("deleted_at IS NULL"),
        unique=True,
    )

    op.create_table(
        "cells",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("scratchpad_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("code", sa.Text(), nullable=False),
        sa.Column("stdout", sa.Text(), nullable=False),
        sa.Column("stderr", sa.Text(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("estimated_time", sa.Text(), nullable=False),
        sa.Column("logs", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["scratchpad_id"], ["scratchpads.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for better performance
    op.create_index(
        "ix_cells_scratchpad_id",
        "cells",
        ["scratchpad_id"],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    # Create a function to handle soft deletes for cells from scratchpads
    op.execute("""
        CREATE OR REPLACE FUNCTION soft_delete_cells_from_scratchpads()
        RETURNS TRIGGER AS $$
        BEGIN
            UPDATE cells
            SET deleted_at = NEW.deleted_at
            WHERE scratchpad_id = NEW.id
            AND deleted_at IS NULL;

            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Create a trigger to handle soft deletes for cells from scratchpads
    op.execute("""
        CREATE TRIGGER trigger_soft_delete_cells_from_scratchpads
        AFTER UPDATE OF deleted_at ON scratchpads
        FOR EACH ROW
        WHEN (NEW.deleted_at IS NOT NULL AND OLD.deleted_at IS NULL)
        EXECUTE FUNCTION soft_delete_cells_from_scratchpads();
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("cells")
    op.drop_table("scratchpads")
