"""fix_min_max_val_types

Revision ID: 006_fix_min_max_val_types
Revises: 005_add_description_to_ds
Create Date: 2025-10-14 19:46:25.483496

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "006_fix_min_max_val_types"
down_revision: str | None = "005_add_description_to_ds"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("column_statistics") as batch_op:
        batch_op.alter_column("min_value", type_=sa.Text(), existing_type=sa.String(length=255), nullable=True)
        batch_op.alter_column("max_value", type_=sa.Text(), existing_type=sa.String(length=255), nullable=True)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("column_statistics") as batch_op:
        batch_op.alter_column("min_value", type_=sa.String(length=255), existing_type=sa.Text(), nullable=True)
        batch_op.alter_column("max_value", type_=sa.String(length=255), existing_type=sa.Text(), nullable=True)
