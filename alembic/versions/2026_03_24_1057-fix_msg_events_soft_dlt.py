from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f56327b2c585'
down_revision: Union[str, None] = '7827eb761c8e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION soft_delete_message_events_from_messages()
        RETURNS TRIGGER AS $$
        BEGIN
            UPDATE message_events
            SET deleted_at = NEW.deleted_at
            WHERE message_id = NEW.id
              AND deleted_at IS NULL;

            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )


def downgrade() -> None:
    # Restores the pre-fix function body (note: this reintroduces the runtime error).
    op.execute(
        """
        CREATE OR REPLACE FUNCTION soft_delete_message_events_from_messages()
        RETURNS TRIGGER AS $$
        BEGIN
            UPDATE message_events
            SET deleted_at = NEW.deleted_at
            WHERE message_id = NEW.id
              AND deleted_at IS NULL;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
