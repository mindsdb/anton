"""drop_connection_data_col

Revision ID: 009_drop_connection_data_col
Revises: 008_drop_mind_ds_status_col
Create Date: 2025-10-23 11:22:43.896830

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '009_drop_connection_data_col'
down_revision: Union[str, None] = '008_drop_mind_ds_status_col'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column("datasources", "connection_data")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column("datasources", sa.Column("connection_data", sa.JSON(), nullable=True))
