"""rename tenant id to org id

Revision ID: 5027aa7bbdb4
Revises: 10622ac23126
Create Date: 2026-02-05 16:43:48.794807

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5027aa7bbdb4"
down_revision: str | None = "10622ac23126"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Tables that have tenant_id column to rename
TABLES_WITH_TENANT_ID = [
    "minds",
    "datasources",
    "mind_datasources",
    "tables",
    "columns",
    "column_statistics",
    "primary_key_constraints",
    "foreign_key_constraints",
    "mind_datasource_tables",
    "conversations",
    "messages",
]


def upgrade() -> None:
    """Rename tenant_id column to organization_id in all tables."""
    for table in TABLES_WITH_TENANT_ID:
        op.alter_column(table, "tenant_id", new_column_name="organization_id")


def downgrade() -> None:
    """Rename organization_id column back to tenant_id in all tables."""
    for table in TABLES_WITH_TENANT_ID:
        op.alter_column(table, "organization_id", new_column_name="tenant_id")
