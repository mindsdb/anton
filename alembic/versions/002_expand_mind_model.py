"""Expand Mind model for internal storage

Revision ID: 002_expand_mind_model
Revises: 001_add_minds_table_with_uuid
Create Date: 2025-01-20 15:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '002_expand_mind_model'
down_revision: Union[str, None] = '001_add_minds_table_with_uuid'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add new columns to minds table for internal storage."""
    
    op.add_column('minds', sa.Column('provider', sa.String(length=50), nullable=False, server_default='openai'))
    op.add_column('minds', sa.Column('model_name', sa.String(length=256), nullable=False, server_default='gpt-4o'))
    
    op.add_column('minds', sa.Column('user_id', sa.String(length=256), nullable=False, server_default='default'))
    op.add_column('minds', sa.Column('company_id', sa.String(length=256), nullable=False, server_default='default'))
    
    op.add_column('minds', sa.Column('parameters', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    op.add_column('minds', sa.Column('datasources', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    
    # Add optional metadata
    op.add_column('minds', sa.Column('description', sa.Text(), nullable=True))
    
    # Add status tracking
    op.add_column('minds', sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'))
    
    # Create indexes for efficient queries
    op.create_index('ix_minds_name', 'minds', ['name'])
    op.create_index('ix_minds_user_id', 'minds', ['user_id'])
    op.create_index('ix_minds_company_id', 'minds', ['company_id'])
    
    # Remove server defaults after adding columns (they were just for migration)
    op.alter_column('minds', 'provider', server_default=None)
    op.alter_column('minds', 'model_name', server_default=None)
    op.alter_column('minds', 'user_id', server_default=None)
    op.alter_column('minds', 'company_id', server_default=None)
    op.alter_column('minds', 'is_active', server_default=None)


def downgrade() -> None:
    """Remove the added columns from minds table."""
    
    # Drop indexes
    op.drop_index('ix_minds_company_id', table_name='minds')
    op.drop_index('ix_minds_user_id', table_name='minds')
    op.drop_index('ix_minds_name', table_name='minds')
    
    # Drop added columns
    op.drop_column('minds', 'is_active')
    op.drop_column('minds', 'description')
    op.drop_column('minds', 'datasources')
    op.drop_column('minds', 'parameters')
    op.drop_column('minds', 'company_id')
    op.drop_column('minds', 'user_id')
    op.drop_column('minds', 'model_name')
    op.drop_column('minds', 'provider')
