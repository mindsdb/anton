"""make model selection cols nullable

Revision ID: 014_make_model_sel_cols_null
Revises: 013_fix_negative_row_counts
Create Date: 2026-01-05 21:04:04.677678

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "014_make_model_sel_cols_null"
down_revision: str | None = "013_fix_negative_row_counts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("minds") as batch_op:
        batch_op.alter_column("model_name", nullable=True)
        batch_op.alter_column("provider", nullable=True)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("minds") as batch_op:
        batch_op.alter_column("model_name", nullable=False)
        batch_op.alter_column("provider", nullable=False)
