"""drop_mind_ds_status_col

Revision ID: 008_drop_mind_ds_status_col
Revises: 007_change_tenant_id_to_uuid
Create Date: 2025-10-30 22:41:29.061816

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from minds.model.mind_datasource import DataCatalogStatus

# revision identifiers, used by Alembic.
revision: str = '008_drop_mind_ds_status_col'
down_revision: Union[str, None] = '007_change_tenant_id_to_uuid'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("mind_datasources") as batch_op:
        batch_op.drop_column("status")


def downgrade() -> None:
    """Downgrade schema."""
    status_enum = postgresql.ENUM(
        *[e.value for e in DataCatalogStatus],
        name='data_catalog_status'
    )
    with op.batch_alter_table("mind_datasources") as batch_op:
        batch_op.add_column(sa.Column("status", status_enum, nullable=False, index=True))
