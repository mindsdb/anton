"""change tenant_id to UUID

Revision ID: 1b265e0bc513
Revises: 006_fix_min_max_val_types
Create Date: 2025-10-20 09:37:00.072193

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1b265e0bc513'
down_revision: Union[str, None] = '006_fix_min_max_val_types'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    
    with op.batch_alter_table('minds') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.UUID(), existing_type=sa.String(length=255))
        batch_op.alter_column('user_id', type_=sa.UUID(), existing_type=sa.String(length=255))
        
    with op.batch_alter_table('datasources') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.UUID(), existing_type=sa.String(length=255))
        batch_op.alter_column('user_id', type_=sa.UUID(), existing_type=sa.String(length=255))
        
    with op.batch_alter_table('mind_datasources') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.UUID(), existing_type=sa.String(length=255))
        
    with op.batch_alter_table('tables') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.UUID(), existing_type=sa.String(length=255))
        
    with op.batch_alter_table('columns') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.UUID(), existing_type=sa.String(length=255))
        
    with op.batch_alter_table('column_statistics') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.UUID(), existing_type=sa.String(length=255))
        
    with op.batch_alter_table('primary_key_constraints') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.UUID(), existing_type=sa.String(length=255))
        
    with op.batch_alter_table('foreign_key_constraints') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.UUID(), existing_type=sa.String(length=255))

    with op.batch_alter_table('mind_datasource_tables') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.UUID(), existing_type=sa.String(length=255))
    
        

def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('minds') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.String(length=255), existing_type=sa.UUID())
        batch_op.alter_column('user_id', type_=sa.String(length=255), existing_type=sa.UUID())
        
    with op.batch_alter_table('datasources') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.String(length=255), existing_type=sa.UUID())
        batch_op.alter_column('user_id', type_=sa.String(length=255), existing_type=sa.UUID())
        
    with op.batch_alter_table('mind_datasources') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.String(length=255), existing_type=sa.UUID())
        
    with op.batch_alter_table('tables') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.String(length=255), existing_type=sa.UUID())
        
    with op.batch_alter_table('columns') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.String(length=255), existing_type=sa.UUID())
        
    with op.batch_alter_table('column_statistics') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.String(length=255), existing_type=sa.UUID())
        
    with op.batch_alter_table('primary_key_constraints') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.String(length=255), existing_type=sa.UUID())
        
    with op.batch_alter_table('foreign_key_constraints') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.String(length=255), existing_type=sa.UUID())
        
    with op.batch_alter_table('mind_datasource_tables') as batch_op:
        batch_op.alter_column('tenant_id', type_=sa.String(length=255), existing_type=sa.UUID())
