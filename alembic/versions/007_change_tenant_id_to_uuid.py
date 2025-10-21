"""change tenant_id to UUID

Revision ID: 1b265e0bc513
Revises: 006_fix_min_max_val_types
Create Date: 2025-10-20 09:37:00.072193

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '007_change_tenant_id_to_uuid'
down_revision: Union[str, None] = '006_fix_min_max_val_types'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    
    # Add a temporary UUID column, populate it with new UUIDs, then swap
    tables_columns = [
        ('minds', ['tenant_id', 'user_id']),
        ('datasources', ['tenant_id', 'user_id']),
        ('mind_datasources', ['tenant_id']),
        ('tables', ['tenant_id']),
        ('columns', ['tenant_id']),
        ('column_statistics', ['tenant_id']),
        ('primary_key_constraints', ['tenant_id']),
        ('foreign_key_constraints', ['tenant_id']),
        ('mind_datasource_tables', ['tenant_id'])
    ]
    
    # For each table and column, create a temp column, populate with UUIDs, then swap
    for table, columns in tables_columns:
        for column in columns:
            # Create temporary UUID column
            op.execute(f'ALTER TABLE {table} ADD COLUMN {column}_new UUID')
            # Set all rows to nil UUID (00000000-0000-0000-0000-000000000000)
            op.execute(f"UPDATE {table} SET {column}_new = '00000000-0000-0000-0000-000000000000'::uuid")
            # Drop the old column and rename the new one
            op.execute(f'ALTER TABLE {table} DROP COLUMN {column}')
            op.execute(f'ALTER TABLE {table} RENAME COLUMN {column}_new TO {column}')
    
    # Add NOT NULL constraints back
    with op.batch_alter_table('minds') as batch_op:
        batch_op.alter_column('tenant_id', nullable=False)
        batch_op.alter_column('user_id', nullable=False)
        
    with op.batch_alter_table('datasources') as batch_op:
        batch_op.alter_column('tenant_id', nullable=False)
        batch_op.alter_column('user_id', nullable=False)
        
    with op.batch_alter_table('mind_datasources') as batch_op:
        batch_op.alter_column('tenant_id', nullable=False)
        
    with op.batch_alter_table('tables') as batch_op:
        batch_op.alter_column('tenant_id', nullable=False)
        
    with op.batch_alter_table('columns') as batch_op:
        batch_op.alter_column('tenant_id', nullable=False)
        
    with op.batch_alter_table('column_statistics') as batch_op:
        batch_op.alter_column('tenant_id', nullable=False)
        
    with op.batch_alter_table('primary_key_constraints') as batch_op:
        batch_op.alter_column('tenant_id', nullable=False)
        
    with op.batch_alter_table('foreign_key_constraints') as batch_op:
        batch_op.alter_column('tenant_id', nullable=False)

    with op.batch_alter_table('mind_datasource_tables') as batch_op:
        batch_op.alter_column('tenant_id', nullable=False)
    
        

def downgrade() -> None:
    """Downgrade schema."""
    # For downgrade, we'll convert UUIDs to their string representation
    tables_columns = [
        ('minds', ['tenant_id', 'user_id']),
        ('datasources', ['tenant_id', 'user_id']),
        ('mind_datasources', ['tenant_id']),
        ('tables', ['tenant_id']),
        ('columns', ['tenant_id']),
        ('column_statistics', ['tenant_id']),
        ('primary_key_constraints', ['tenant_id']),
        ('foreign_key_constraints', ['tenant_id']),
        ('mind_datasource_tables', ['tenant_id'])
    ]
    
    # For each table and column, create a temp column with string type, 
    # populate with UUID string representation, then swap
    for table, columns in tables_columns:
        for column in columns:
            # Create temporary string column
            op.execute(f'ALTER TABLE {table} ADD COLUMN {column}_new VARCHAR(255)')
            # Convert UUID to string
            op.execute(f'UPDATE {table} SET {column}_new = {column}::text')
            # Drop the old column and rename the new one
            op.execute(f'ALTER TABLE {table} DROP COLUMN {column}')
            op.execute(f'ALTER TABLE {table} RENAME COLUMN {column}_new TO {column}')
            # Add NOT NULL constraint back
            op.execute(f'ALTER TABLE {table} ALTER COLUMN {column} SET NOT NULL')
