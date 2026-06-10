"""make conversations.mind_id nullable

Inference-only no longer creates Mind rows, so the FK + NOT NULL constraint
on conversations.mind_id blocks creation of new conversations via /v1/responses.
Drop the FK and make the column nullable. Column itself is left in place so
legacy rows survive without backfill.

Revision ID: c8d4f2a91b3e
Revises: 1ccaab8f45da
Create Date: 2026-06-09 15:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8d4f2a91b3e"
down_revision: str | None = "1ccaab8f45da"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop FK to minds(id) and make mind_id nullable."""
    op.drop_constraint("conversations_mind_id_fkey", "conversations", type_="foreignkey")
    op.alter_column("conversations", "mind_id", nullable=True)


def downgrade() -> None:
    """Restore FK + NOT NULL on conversations.mind_id."""
    op.alter_column("conversations", "mind_id", nullable=False)
    op.create_foreign_key(
        "conversations_mind_id_fkey",
        "conversations",
        "minds",
        ["mind_id"],
        ["id"],
    )
