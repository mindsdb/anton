"""rename_data_to_connection_data

Revision ID: c617e589a92e
Revises: 003_add_datasource_table
Create Date: 2025-08-27 17:33:34.281613

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c617e589a92e'
down_revision: Union[str, None] = '003_add_datasource_table'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Rename data column to connection_data in datasources table."""
    op.alter_column('datasources', 'data', new_column_name='connection_data')


def downgrade() -> None:
    """Rename connection_data column back to data in datasources table."""
    op.alter_column('datasources', 'connection_data', new_column_name='data')
