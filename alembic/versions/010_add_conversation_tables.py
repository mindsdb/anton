"""add_conversation_tables

Revision ID: 010_add_conversation_tables
Revises: 009_drop_connection_data_col
Create Date: 2025-12-11 15:28:20.750356

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from minds.schemas.chat import Role


# revision identifiers, used by Alembic.
revision: str = '010_add_conversation_tables'
down_revision: Union[str, None] = '009_drop_connection_data_col'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('topic', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('modified_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )

    role_enum = postgresql.ENUM(
        *[e.value for e in Role],
        name='message_role'
    )
    role_enum.create(op.get_bind())

    op.create_table('messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role', role_enum, nullable=False),
        sa.Column('content', postgresql.JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('modified_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create indexes for better performance
    op.create_index('ix_conversations_user_id', 'conversations', ['user_id'])
    op.create_index('ix_messages_conversation_id', 'messages', ['conversation_id'])

    # Create triggers for automatically updating modified_at timestamps
    # Using the function update_modified_at_column from the initial schema
    op.execute('''
        CREATE TRIGGER update_conversations_modified_at
        BEFORE UPDATE ON conversations
        FOR EACH ROW EXECUTE FUNCTION update_modified_at_column();
    ''')

    op.execute('''
        CREATE TRIGGER update_messages_modified_at
        BEFORE UPDATE ON messages
        FOR EACH ROW EXECUTE FUNCTION update_modified_at_column();
    ''')

    # Create function to handle soft deletes for messages
    op.execute('''
        CREATE OR REPLACE FUNCTION soft_delete_messages()
        RETURNS TRIGGER AS $$
        BEGIN
            UPDATE messages
            SET deleted_at = NEW.deleted_at
            WHERE conversation_id = NEW.id
            AND deleted_at IS NULL;
            
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    ''')

    # Create trigger to handle soft deletes for messages
    op.execute('''
        CREATE TRIGGER trigger_soft_delete_messages
        AFTER UPDATE OF deleted_at ON conversations
        FOR EACH ROW
        WHEN (NEW.deleted_at IS NOT NULL AND OLD.deleted_at IS NULL)
        EXECUTE FUNCTION soft_delete_messages();
    ''')


def downgrade() -> None:
    """Downgrade schema."""
    # Drop triggers
    op.execute('DROP TRIGGER IF EXISTS trigger_soft_delete_messages ON messages;')
    op.execute('DROP TRIGGER IF EXISTS trigger_soft_delete_conversations ON conversations;')

    # Drop functions
    op.execute('DROP FUNCTION IF EXISTS soft_delete_messages();')
    op.execute('DROP FUNCTION IF EXISTS soft_delete_conversations();')

    # Drop indexes
    op.drop_index('ix_messages_conversation_id', 'messages')
    op.drop_index('ix_conversations_user_id', 'conversations')

    # Drop tables
    op.drop_table('messages')
    op.drop_table('conversations')

    # Drop enum
    op.execute('DROP TYPE IF EXISTS message_role;')