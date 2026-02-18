"""convert sql_query col to JSONB

Revision ID: f84b03880e58
Revises: 07ff94dbe23a
Create Date: 2026-02-10 21:28:06.410595

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f84b03880e58"
down_revision: str | None = "07ff94dbe23a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        "messages",
        "sql_query",
        existing_type=sa.Text(),
        type_=postgresql.JSONB(),
        postgresql_using="sql_query::jsonb",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        "messages",
        "sql_query",
        existing_type=postgresql.JSONB(),
        type_=sa.Text(),
        postgresql_using="sql_query::text",
    )
