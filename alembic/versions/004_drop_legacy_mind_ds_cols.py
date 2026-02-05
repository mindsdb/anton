"""drop_legacy_mind_ds_cols

Revision ID: 004_drop_legacy_mind_ds_cols
Revises: 003_add_prefect_flow_run_id
Create Date: 2025-09-30 13:40:01.110882

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "004_drop_legacy_mind_ds_cols"
down_revision: str | None = "003_add_prefect_flow_run_id"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("mind_datasources") as batch_op:
        batch_op.drop_column("tables")
        batch_op.drop_column("purpose")


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("mind_datasources") as batch_op:
        batch_op.add_column(sa.Column("tables", postgresql.JSON(astext_type=sa.Text()), nullable=True))
        batch_op.add_column(sa.Column("purpose", sa.String(length=255), nullable=True))
