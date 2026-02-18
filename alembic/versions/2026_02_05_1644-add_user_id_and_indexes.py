"""add user id and indexes

Revision ID: 07ff94dbe23a
Revises: 5027aa7bbdb4
Create Date: 2026-02-05 16:44:28.173955

"""

from collections.abc import Sequence
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "07ff94dbe23a"
down_revision: str | None = "5027aa7bbdb4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Default user_id value for existing records (will be addressed later elsewhere)
DEFAULT_USER_ID = UUID("00000000-0000-0000-0000-000000000000")

# Tables that already have user_id column AND index:
# - minds: has user_id column and ix_minds_user_id (migration 001)
# - conversations: has user_id column and ix_conversations_user_id (migration 011)
TABLES_WITH_USER_ID_AND_INDEX = [
    "minds",
    "conversations",
]

# Tables that have user_id column but NO index
TABLES_WITH_USER_ID_NO_INDEX = [
    "datasources",
]

# Tables that need user_id column added (no column, no index)
TABLES_WITHOUT_USER_ID = [
    "messages",
    "mind_datasources",
    "mind_datasource_tables",
    "tables",
    "columns",
    "column_statistics",
    "primary_key_constraints",
    "foreign_key_constraints",
]

# All tables (for organization_id index - none have it yet)
ALL_TABLES = TABLES_WITH_USER_ID_AND_INDEX + TABLES_WITH_USER_ID_NO_INDEX + TABLES_WITHOUT_USER_ID

# Tables that need new user_id index (excludes minds and conversations which already have it)
TABLES_NEEDING_USER_ID_INDEX = TABLES_WITH_USER_ID_NO_INDEX + TABLES_WITHOUT_USER_ID


def upgrade() -> None:
    """Add user_id column to tables that don't have it and create indexes."""
    # Add user_id column to tables that don't have it
    for table in TABLES_WITHOUT_USER_ID:
        op.add_column(
            table,
            sa.Column(
                "user_id",
                PostgreSQLUUID(as_uuid=True),
                nullable=False,
                server_default=sa.text(f"'{DEFAULT_USER_ID}'::uuid"),
            ),
        )

    # Create index on organization_id for all tables (none exist yet)
    for table in ALL_TABLES:
        op.create_index(
            f"ix_{table}_organization_id",
            table,
            ["organization_id"],
        )

    # Create index on user_id for tables that don't have it yet
    # (minds and conversations already have user_id indexes from migrations 001 and 011)
    for table in TABLES_NEEDING_USER_ID_INDEX:
        op.create_index(
            f"ix_{table}_user_id",
            table,
            ["user_id"],
        )


def downgrade() -> None:
    """Remove user_id column and indexes."""
    # Drop user_id indexes (only the ones we created)
    for table in TABLES_NEEDING_USER_ID_INDEX:
        op.drop_index(f"ix_{table}_user_id", table_name=table)

    # Drop organization_id indexes (we created all of these)
    for table in ALL_TABLES:
        op.drop_index(f"ix_{table}_organization_id", table_name=table)

    # Remove user_id column from tables that didn't have it before
    for table in TABLES_WITHOUT_USER_ID:
        op.drop_column(table, "user_id")
