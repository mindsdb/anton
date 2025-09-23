"""add_prefect_flow_run_id

Revision ID: 003_add_prefect_flow_run_id
Revises: 004_add_data_catalog_tables
Create Date: 2025-09-23 20:52:31.195811

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003_add_prefect_flow_run_id'
down_revision: Union[str, None] = '004_add_data_catalog_tables'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('mind_datasources', sa.Column('flow_run_id', sa.String(length=255), nullable=True))

def downgrade() -> None:
    op.drop_column('mind_datasources', 'flow_run_id')
