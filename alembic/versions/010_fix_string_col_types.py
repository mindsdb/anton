"""fix_string_col_types

Revision ID: 010_fix_string_col_types
Revises: 009_drop_connection_data_col
Create Date: 2025-11-28 18:47:09.598365

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "010_fix_string_col_types"
down_revision: str | None = "009_drop_connection_data_col"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("tables") as batch_op:
        batch_op.alter_column("name", type_=sa.Text(), existing_type=sa.String(length=255), nullable=False)
        batch_op.alter_column("schema", type_=sa.Text(), existing_type=sa.String(length=255), nullable=True)

    with op.batch_alter_table("columns") as batch_op:
        batch_op.alter_column("name", type_=sa.Text(), existing_type=sa.String(length=255), nullable=False)
        batch_op.alter_column("data_type", type_=sa.Text(), existing_type=sa.String(length=100), nullable=False)
        batch_op.alter_column("default_value", type_=sa.Text(), existing_type=sa.String(length=500), nullable=True)

    with op.batch_alter_table("primary_key_constraints") as batch_op:
        batch_op.alter_column("constraint_name", type_=sa.Text(), existing_type=sa.String(length=255), nullable=True)

    with op.batch_alter_table("foreign_key_constraints") as batch_op:
        batch_op.alter_column("constraint_name", type_=sa.Text(), existing_type=sa.String(length=255), nullable=True)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("tables") as batch_op:
        batch_op.alter_column("name", type_=sa.String(length=255), existing_type=sa.Text(), nullable=False)
        batch_op.alter_column("schema", type_=sa.String(length=255), existing_type=sa.Text(), nullable=True)

    with op.batch_alter_table("columns") as batch_op:
        batch_op.alter_column("name", type_=sa.String(length=255), existing_type=sa.Text(), nullable=False)
        batch_op.alter_column("data_type", type_=sa.String(length=100), existing_type=sa.Text(), nullable=False)
        batch_op.alter_column("default_value", type_=sa.String(length=500), existing_type=sa.Text(), nullable=True)

    with op.batch_alter_table("primary_key_constraints") as batch_op:
        batch_op.alter_column("constraint_name", type_=sa.String(length=255), existing_type=sa.Text(), nullable=True)
