"""add_description_to_ds

Revision ID: 005_add_description_to_ds
Revises: 004_drop_legacy_mind_ds_cols
Create Date: 2025-10-02 15:14:58.346627

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "005_add_description_to_ds"
down_revision: str | None = "004_drop_legacy_mind_ds_cols"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("datasources") as batch_op:
        batch_op.add_column(sa.Column("description", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("datasources") as batch_op:
        batch_op.drop_column("description")
