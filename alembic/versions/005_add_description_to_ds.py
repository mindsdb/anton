"""add_description_to_ds

Revision ID: 005_add_description_to_ds
Revises: 004_drop_legacy_mind_ds_cols
Create Date: 2025-10-02 15:14:58.346627

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '005_add_description_to_ds'
down_revision: Union[str, None] = '004_drop_legacy_mind_ds_cols'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('datasources') as batch_op:
        batch_op.add_column(sa.Column('description', sa.String(length=255), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('datasources') as batch_op:
        batch_op.drop_column('description')
