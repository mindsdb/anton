"""encrypt msg content

Revision ID: 10622ac23126
Revises: bd5033662da9
Create Date: 2026-02-05 11:44:42.884746

"""

from collections.abc import Sequence

import sqlalchemy as sa
from mind_castle.sqlalchemy_type import SecretData
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op
from minds.common.settings.app_settings import get_app_settings

# revision identifiers, used by Alembic.
revision: str = "10622ac23126"
down_revision: str | None = "bd5033662da9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


class SecretDataJSONB(SecretData):
    impl = JSONB


def upgrade() -> None:
    settings = get_app_settings()
    conn = op.get_bind()

    # IMPORTANT: declare a "virtual table" whose content column uses SecretDataJSONB
    messages = sa.table(
        "messages",
        sa.column("id", postgresql.UUID(as_uuid=True)),
        sa.column("content", SecretDataJSONB(settings.mind_castle.encryption_type)),
        sa.column("sql_query", SecretData(settings.mind_castle.encryption_type)),
    )

    batch_size = 1000
    last_id = None

    while True:
        # Read RAW data (no SecretData type here)
        rows = conn.execute(
            sa.text(
                """
                SELECT id, content, sql_query
                FROM messages
                WHERE (content IS NOT NULL OR sql_query IS NOT NULL)
                  AND (:last_id IS NULL OR id > :last_id)
                ORDER BY id
                LIMIT :limit
                """
            ),
            {"last_id": last_id, "limit": batch_size},
        ).fetchall()

        if not rows:
            break

        for row in rows:
            # This UPDATE binds through SecretData => encrypts on write
            conn.execute(
                messages.update().where(messages.c.id == row.id).values(content=row.content, sql_query=row.sql_query)
            )

        last_id = rows[-1].id


def downgrade() -> None:
    # This is a data migration, so we can't really downgrade it
    pass
