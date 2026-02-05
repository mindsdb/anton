"""fix_negative_row_counts

Revision ID: 013_fix_negative_row_counts
Revises: 012_add_unique_constraints
Create Date: 2026-01-20 00:00:00.000000

"""

from collections.abc import Sequence

from sqlalchemy import text
from sqlalchemy.engine import Connection

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "013_fix_negative_row_counts"
down_revision: str | None = "012_add_unique_constraints"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """
    Fix row_count = -1 by setting them to NULL.

    The -1 values are invalid and should be treated as unknown (NULL).
    The actual row counts will be populated when the catalog is refreshed.
    """
    connection: Connection = op.get_bind()

    # First, count how many records will be updated
    count_result = connection.execute(
        text("""
        SELECT COUNT(*) as count
        FROM tables
        WHERE row_count < 0
        AND deleted_at IS NULL;
    """)
    )

    count = count_result.fetchone()[0]

    # Set all negative row_count values (including -1) to NULL
    # This treats them as "unknown" which is the correct semantic meaning
    op.execute(
        text("""
        UPDATE tables
        SET row_count = NULL
        WHERE row_count < 0
        AND deleted_at IS NULL;
    """)
    )

    if count > 0:
        print(f"✅ Fixed row_count: Set {count} tables with negative row_count to NULL")
    else:
        print("✅ No tables with negative row_count found")


def downgrade() -> None:
    """
    Downgrade: Cannot restore the original -1 values as we don't know which ones had -1.
    This is a data migration, so downgrade is a no-op.
    """
    # This is a data migration, so we can't really downgrade it
    # The -1 values were invalid anyway, so leaving them as NULL is fine
    pass
