"""fix tables constraint

Revision ID: bd5033662da9
Revises: 014_make_model_sel_cols_null
Create Date: 2026-02-03 17:16:33.153258

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bd5033662da9'
down_revision: Union[str, None] = '014_make_model_sel_cols_null'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # The original unique constraint included nullable deleted_at, which (in Postgres)
    # does NOT prevent duplicate active rows because UNIQUE treats NULLs as distinct.
    # Fix this by enforcing uniqueness only for non-deleted rows via partial unique indexes.
    op.drop_constraint("unique_table_per_datasource", "tables", type_="unique")
    op.drop_constraint("unique_mind_datasource_table", "mind_datasource_tables", type_="unique")

    # Cleanup: de-duplicate any active (deleted_at IS NULL) rows before creating indexes.
    # Keep the oldest record (lowest created_at; id as tie-breaker) for each
    # (datasource_id, name, tenant_id) among active rows.
    op.execute(
        sa.text("""
        UPDATE columns c
        SET table_id = (
            SELECT t2.id
            FROM tables t2
            WHERE t2.datasource_id = (SELECT datasource_id FROM tables t1 WHERE t1.id = c.table_id)
            AND t2.name = (SELECT name FROM tables t1 WHERE t1.id = c.table_id)
            AND t2.tenant_id = (SELECT tenant_id FROM tables t1 WHERE t1.id = c.table_id)
            AND t2.deleted_at IS NULL
            ORDER BY COALESCE(t2.created_at, '-infinity'::timestamptz) ASC, t2.id ASC
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1
            FROM tables t
            WHERE t.id = c.table_id
            AND t.deleted_at IS NULL
            AND EXISTS (
                SELECT 1
                FROM tables t2
                WHERE t2.datasource_id = t.datasource_id
                AND t2.name = t.name
                AND t2.tenant_id = t.tenant_id
                AND t2.deleted_at IS NULL
                AND (
                    COALESCE(t2.created_at, '-infinity'::timestamptz) < COALESCE(t.created_at, '-infinity'::timestamptz)
                    OR (
                        COALESCE(t2.created_at, '-infinity'::timestamptz) = COALESCE(t.created_at, '-infinity'::timestamptz)
                        AND t2.id < t.id
                    )
                )
            )
        );
    """)
    )

    op.execute(
        sa.text("""
        UPDATE primary_key_constraints pkc
        SET table_id = (
            SELECT t2.id
            FROM tables t2
            WHERE t2.datasource_id = (SELECT datasource_id FROM tables t1 WHERE t1.id = pkc.table_id)
            AND t2.name = (SELECT name FROM tables t1 WHERE t1.id = pkc.table_id)
            AND t2.tenant_id = (SELECT tenant_id FROM tables t1 WHERE t1.id = pkc.table_id)
            AND t2.deleted_at IS NULL
            ORDER BY COALESCE(t2.created_at, '-infinity'::timestamptz) ASC, t2.id ASC
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1 FROM tables t
            WHERE t.id = pkc.table_id
            AND t.deleted_at IS NULL
            AND EXISTS (
                SELECT 1 FROM tables t2
                WHERE t2.datasource_id = t.datasource_id
                AND t2.name = t.name
                AND t2.tenant_id = t.tenant_id
                AND t2.deleted_at IS NULL
                AND (
                    COALESCE(t2.created_at, '-infinity'::timestamptz) < COALESCE(t.created_at, '-infinity'::timestamptz)
                    OR (
                        COALESCE(t2.created_at, '-infinity'::timestamptz) = COALESCE(t.created_at, '-infinity'::timestamptz)
                        AND t2.id < t.id
                    )
                )
            )
        );
    """)
    )

    op.execute(
        sa.text("""
        UPDATE foreign_key_constraints fkc
        SET table_id = (
            SELECT t2.id
            FROM tables t2
            WHERE t2.datasource_id = (SELECT datasource_id FROM tables t1 WHERE t1.id = fkc.table_id)
            AND t2.name = (SELECT name FROM tables t1 WHERE t1.id = fkc.table_id)
            AND t2.tenant_id = (SELECT tenant_id FROM tables t1 WHERE t1.id = fkc.table_id)
            AND t2.deleted_at IS NULL
            ORDER BY COALESCE(t2.created_at, '-infinity'::timestamptz) ASC, t2.id ASC
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1 FROM tables t
            WHERE t.id = fkc.table_id
            AND t.deleted_at IS NULL
            AND EXISTS (
                SELECT 1 FROM tables t2
                WHERE t2.datasource_id = t.datasource_id
                AND t2.name = t.name
                AND t2.tenant_id = t.tenant_id
                AND t2.deleted_at IS NULL
                AND (
                    COALESCE(t2.created_at, '-infinity'::timestamptz) < COALESCE(t.created_at, '-infinity'::timestamptz)
                    OR (
                        COALESCE(t2.created_at, '-infinity'::timestamptz) = COALESCE(t.created_at, '-infinity'::timestamptz)
                        AND t2.id < t.id
                    )
                )
            )
        );
    """)
    )

    op.execute(
        sa.text("""
        UPDATE foreign_key_constraints fkc
        SET referenced_table_id = (
            SELECT t2.id
            FROM tables t2
            WHERE t2.datasource_id = (SELECT datasource_id FROM tables t1 WHERE t1.id = fkc.referenced_table_id)
            AND t2.name = (SELECT name FROM tables t1 WHERE t1.id = fkc.referenced_table_id)
            AND t2.tenant_id = (SELECT tenant_id FROM tables t1 WHERE t1.id = fkc.referenced_table_id)
            AND t2.deleted_at IS NULL
            ORDER BY COALESCE(t2.created_at, '-infinity'::timestamptz) ASC, t2.id ASC
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1 FROM tables t
            WHERE t.id = fkc.referenced_table_id
            AND t.deleted_at IS NULL
            AND EXISTS (
                SELECT 1 FROM tables t2
                WHERE t2.datasource_id = t.datasource_id
                AND t2.name = t.name
                AND t2.tenant_id = t.tenant_id
                AND t2.deleted_at IS NULL
                AND (
                    COALESCE(t2.created_at, '-infinity'::timestamptz) < COALESCE(t.created_at, '-infinity'::timestamptz)
                    OR (
                        COALESCE(t2.created_at, '-infinity'::timestamptz) = COALESCE(t.created_at, '-infinity'::timestamptz)
                        AND t2.id < t.id
                    )
                )
            )
        );
    """)
    )

    op.execute(
        sa.text("""
        UPDATE mind_datasource_tables mdt
        SET table_id = (
            SELECT t2.id
            FROM tables t2
            WHERE t2.datasource_id = (SELECT datasource_id FROM tables t1 WHERE t1.id = mdt.table_id)
            AND t2.name = (SELECT name FROM tables t1 WHERE t1.id = mdt.table_id)
            AND t2.tenant_id = (SELECT tenant_id FROM tables t1 WHERE t1.id = mdt.table_id)
            AND t2.deleted_at IS NULL
            ORDER BY COALESCE(t2.created_at, '-infinity'::timestamptz) ASC, t2.id ASC
            LIMIT 1
        )
        WHERE EXISTS (
            SELECT 1 FROM tables t
            WHERE t.id = mdt.table_id
            AND t.deleted_at IS NULL
            AND EXISTS (
                SELECT 1
                FROM tables t2
                WHERE t2.datasource_id = t.datasource_id
                AND t2.name = t.name
                AND t2.tenant_id = t.tenant_id
                AND t2.deleted_at IS NULL
                AND (
                    COALESCE(t2.created_at, '-infinity'::timestamptz) < COALESCE(t.created_at, '-infinity'::timestamptz)
                    OR (
                        COALESCE(t2.created_at, '-infinity'::timestamptz) = COALESCE(t.created_at, '-infinity'::timestamptz)
                        AND t2.id < t.id
                    )
                )
            )
        );
    """)
    )

    op.execute(
        sa.text("""
        DELETE FROM tables t1
        WHERE t1.deleted_at IS NULL
        AND EXISTS (
            SELECT 1 FROM tables t2
            WHERE t2.datasource_id = t1.datasource_id
            AND t2.name = t1.name
            AND t2.tenant_id = t1.tenant_id
            AND t2.deleted_at IS NULL
            AND (
                COALESCE(t2.created_at, '-infinity'::timestamptz) < COALESCE(t1.created_at, '-infinity'::timestamptz)
                OR (
                    COALESCE(t2.created_at, '-infinity'::timestamptz) = COALESCE(t1.created_at, '-infinity'::timestamptz)
                    AND t2.id < t1.id
                )
            )
        );
    """)
    )

    op.execute(
        sa.text("""
        DELETE FROM mind_datasource_tables mdt1
        WHERE mdt1.deleted_at IS NULL
        AND EXISTS (
            SELECT 1 FROM mind_datasource_tables mdt2
            WHERE mdt2.mind_datasource_id = mdt1.mind_datasource_id
            AND mdt2.table_id = mdt1.table_id
            AND mdt2.tenant_id = mdt1.tenant_id
            AND mdt2.deleted_at IS NULL
            AND (
                COALESCE(mdt2.created_at, '-infinity'::timestamptz) < COALESCE(mdt1.created_at, '-infinity'::timestamptz)
                OR (
                    COALESCE(mdt2.created_at, '-infinity'::timestamptz) = COALESCE(mdt1.created_at, '-infinity'::timestamptz)
                    AND mdt2.id < mdt1.id
                )
            )
        );
    """)
    )

    op.create_index(
        "unique_table_per_datasource",
        "tables",
        ["datasource_id", "name", "tenant_id"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index(
        "unique_mind_datasource_table",
        "mind_datasource_tables",
        ["mind_datasource_id", "table_id", "tenant_id"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("unique_table_per_datasource", table_name="tables")
    op.drop_index("unique_mind_datasource_table", table_name="mind_datasource_tables")

    # Readd constraint with deleted_at
    op.create_unique_constraint(
        "unique_table_per_datasource", "tables", ["datasource_id", "name", "tenant_id", "deleted_at"]
    )
    op.create_unique_constraint(
        "unique_mind_datasource_table",
        "mind_datasource_tables",
        ["mind_datasource_id", "table_id", "tenant_id", "deleted_at"],
    )
