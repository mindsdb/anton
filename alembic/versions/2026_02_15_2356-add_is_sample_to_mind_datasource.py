"""add is_sample to mind/datasource

Revision ID: 9efd2d71c54d
Revises: f84b03880e58
Create Date: 2026-02-15 23:56:06.063761

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9efd2d71c54d"
down_revision: str | None = "f84b03880e58"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    with op.batch_alter_table("datasources") as batch_op:
        batch_op.add_column(sa.Column("is_sample", sa.Boolean(), nullable=False, server_default=sa.false()))

    with op.batch_alter_table("minds") as batch_op:
        batch_op.add_column(sa.Column("is_sample", sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade() -> None:
    """Downgrade schema."""

    with op.batch_alter_table("minds", schema=None) as batch_op:
        batch_op.drop_column("is_sample")

    with op.batch_alter_table("datasources", schema=None) as batch_op:
        batch_op.drop_column("is_sample")
