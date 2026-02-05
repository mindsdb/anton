"""add_unique_constraints_for_tables

Revision ID: 012_add_unique_constraints
Revises: 011_add_conversation_tables
Create Date: 2026-01-20 00:00:00.000000

"""

from collections.abc import Sequence

from sqlalchemy import text

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "012_add_unique_constraints"
down_revision: str | None = "011_add_conversation_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    # Step 1: Remove duplicate tables
    # Keep the oldest record (lowest id) for each (datasource_id, name, tenant_id, deleted_at) combination
    # Update foreign keys in related tables to point to the kept record

    # Update columns to point to the kept table record
    op.execute(
        text("""
        UPDATE columns c
        SET table_id = (
            SELECT t2.id
            FROM tables t2
            WHERE t2.datasource_id = (
                SELECT datasource_id FROM tables t1 WHERE t1.id = c.table_id
            )
            AND t2.name = (SELECT name FROM tables t1 WHERE t1.id = c.table_id)
            AND t2.tenant_id = (SELECT tenant_id FROM tables t1 WHERE t1.id = c.table_id)
            AND (t2.deleted_at IS NULL AND (SELECT deleted_at FROM tables t1 WHERE t1.id = c.table_id) IS NULL
                 OR t2.deleted_at = (SELECT deleted_at FROM tables t1 WHERE t1.id = c.table_id))
            ORDER BY t2.id
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1
            FROM tables t
            WHERE t.id = c.table_id
            AND EXISTS (
                SELECT 1
                FROM tables t2
                WHERE t2.datasource_id = t.datasource_id
                AND t2.name = t.name
                AND t2.tenant_id = t.tenant_id
                AND (t2.deleted_at IS NULL AND t.deleted_at IS NULL OR t2.deleted_at = t.deleted_at)
                AND t2.id < t.id
            )
        );
    """)
    )

    # Update primary_key_constraints
    op.execute(
        text("""
        UPDATE primary_key_constraints pkc
        SET table_id = (
            SELECT t2.id
            FROM tables t2
            WHERE t2.datasource_id = (SELECT datasource_id FROM tables t1 WHERE t1.id = pkc.table_id)
            AND t2.name = (SELECT name FROM tables t1 WHERE t1.id = pkc.table_id)
            AND t2.tenant_id = (SELECT tenant_id FROM tables t1 WHERE t1.id = pkc.table_id)
            AND (t2.deleted_at IS NULL AND (SELECT deleted_at FROM tables t1 WHERE t1.id = pkc.table_id) IS NULL
                 OR t2.deleted_at = (SELECT deleted_at FROM tables t1 WHERE t1.id = pkc.table_id))
            ORDER BY t2.id
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1 FROM tables t
            WHERE t.id = pkc.table_id
            AND EXISTS (
                SELECT 1 FROM tables t2
                WHERE t2.datasource_id = t.datasource_id
                AND t2.name = t.name
                AND t2.tenant_id = t.tenant_id
                AND (t2.deleted_at IS NULL AND t.deleted_at IS NULL OR t2.deleted_at = t.deleted_at)
                AND t2.id < t.id
            )
        );
    """)
    )

    # Update foreign_key_constraints (table_id)
    op.execute(
        text("""
        UPDATE foreign_key_constraints fkc
        SET table_id = (
            SELECT t2.id
            FROM tables t2
            WHERE t2.datasource_id = (SELECT datasource_id FROM tables t1 WHERE t1.id = fkc.table_id)
            AND t2.name = (SELECT name FROM tables t1 WHERE t1.id = fkc.table_id)
            AND t2.tenant_id = (SELECT tenant_id FROM tables t1 WHERE t1.id = fkc.table_id)
            AND (t2.deleted_at IS NULL AND (SELECT deleted_at FROM tables t1 WHERE t1.id = fkc.table_id) IS NULL
                 OR t2.deleted_at = (SELECT deleted_at FROM tables t1 WHERE t1.id = fkc.table_id))
            ORDER BY t2.id
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1 FROM tables t
            WHERE t.id = fkc.table_id
            AND EXISTS (
                SELECT 1 FROM tables t2
                WHERE t2.datasource_id = t.datasource_id
                AND t2.name = t.name
                AND t2.tenant_id = t.tenant_id
                AND (t2.deleted_at IS NULL AND t.deleted_at IS NULL OR t2.deleted_at = t.deleted_at)
                AND t2.id < t.id
            )
        );
    """)
    )

    # Update foreign_key_constraints (referenced_table_id)
    op.execute(
        text("""
        UPDATE foreign_key_constraints fkc
        SET referenced_table_id = (
            SELECT t2.id
            FROM tables t2
            WHERE t2.datasource_id = (SELECT datasource_id FROM tables t1 WHERE t1.id = fkc.referenced_table_id)
            AND t2.name = (SELECT name FROM tables t1 WHERE t1.id = fkc.referenced_table_id)
            AND t2.tenant_id = (SELECT tenant_id FROM tables t1 WHERE t1.id = fkc.referenced_table_id)
            AND (t2.deleted_at IS NULL AND (SELECT deleted_at FROM tables t1 WHERE t1.id = fkc.referenced_table_id) IS NULL
                 OR t2.deleted_at = (SELECT deleted_at FROM tables t1 WHERE t1.id = fkc.referenced_table_id))
            ORDER BY t2.id
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1 FROM tables t
            WHERE t.id = fkc.referenced_table_id
            AND EXISTS (
                SELECT 1 FROM tables t2
                WHERE t2.datasource_id = t.datasource_id
                AND t2.name = t.name
                AND t2.tenant_id = t.tenant_id
                AND (t2.deleted_at IS NULL AND t.deleted_at IS NULL OR t2.deleted_at = t.deleted_at)
                AND t2.id < t.id
            )
        );
    """)
    )

    # Update mind_datasource_tables to point to kept table
    op.execute(
        text("""
        UPDATE mind_datasource_tables mdt
        SET table_id = (
            SELECT t2.id
            FROM tables t2
            WHERE t2.datasource_id = (SELECT datasource_id FROM tables t1 WHERE t1.id = mdt.table_id)
            AND t2.name = (SELECT name FROM tables t1 WHERE t1.id = mdt.table_id)
            AND t2.tenant_id = (SELECT tenant_id FROM tables t1 WHERE t1.id = mdt.table_id)
            AND (t2.deleted_at IS NULL AND (SELECT deleted_at FROM tables t1 WHERE t1.id = mdt.table_id) IS NULL
                 OR t2.deleted_at = (SELECT deleted_at FROM tables t1 WHERE t1.id = mdt.table_id))
            ORDER BY t2.id
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1 FROM tables t
            WHERE t.id = mdt.table_id
            AND EXISTS (
                SELECT 1 FROM tables t2
                WHERE t2.datasource_id = t.datasource_id
                AND t2.name = t.name
                AND t2.tenant_id = t.tenant_id
                AND (t2.deleted_at IS NULL AND t.deleted_at IS NULL OR t2.deleted_at = t.deleted_at)
                AND t2.id < t.id
            )
        );
    """)
    )

    # Now delete duplicate tables (keep the one with minimum id)
    op.execute(
        text("""
        DELETE FROM tables t1
        WHERE EXISTS (
            SELECT 1 FROM tables t2
            WHERE t2.datasource_id = t1.datasource_id
            AND t2.name = t1.name
            AND t2.tenant_id = t1.tenant_id
            AND (t2.deleted_at IS NULL AND t1.deleted_at IS NULL OR t2.deleted_at = t1.deleted_at)
            AND t2.id < t1.id
        );
    """)
    )

    # Step 2: Remove duplicate mind_datasource_tables
    # Keep the oldest record (lowest id) for each (mind_datasource_id, table_id, tenant_id, deleted_at) combination
    op.execute(
        text("""
        DELETE FROM mind_datasource_tables mdt1
        WHERE EXISTS (
            SELECT 1 FROM mind_datasource_tables mdt2
            WHERE mdt2.mind_datasource_id = mdt1.mind_datasource_id
            AND mdt2.table_id = mdt1.table_id
            AND mdt2.tenant_id = mdt1.tenant_id
            AND (mdt2.deleted_at IS NULL AND mdt1.deleted_at IS NULL OR mdt2.deleted_at = mdt1.deleted_at)
            AND mdt2.id < mdt1.id
        );
    """)
    )

    # Step 3: Now add the unique constraints
    op.create_unique_constraint(
        "unique_table_per_datasource", "tables", ["datasource_id", "name", "tenant_id", "deleted_at"]
    )
    op.create_unique_constraint(
        "unique_mind_datasource_table",
        "mind_datasource_tables",
        ["mind_datasource_id", "table_id", "tenant_id", "deleted_at"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("unique_table_per_datasource", "tables")
    op.drop_constraint("unique_mind_datasource_table", "mind_datasource_tables")
