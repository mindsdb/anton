"""add memory_rules and memory_topics tables

Revision ID: d4e5f6a7b8c9
Revises: 571e802fcd6d
Create Date: 2026-03-16 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    # Create enum type for rule_type
    rule_type_enum = postgresql.ENUM("always", "never", "when", name="rule_type_enum", create_type=False)
    rule_type_enum.create(op.get_bind(), checkfirst=True)

    # Create memory_rules table
    op.create_table(
        "memory_rules",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("mind_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("rule_type", rule_type_enum, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("modified_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("ix_memory_rules_organization_id", "memory_rules", ["organization_id"])
    op.create_index("ix_memory_rules_user_id", "memory_rules", ["user_id"])
    op.create_index("ix_memory_rules_mind_id", "memory_rules", ["mind_id"])
    op.create_foreign_key("fk_memory_rules_mind_id", "memory_rules", "minds", ["mind_id"], ["id"])

    op.execute("""
        CREATE TRIGGER update_memory_rules_modified_at
        BEFORE UPDATE ON memory_rules
        FOR EACH ROW EXECUTE FUNCTION update_modified_at_column();
    """)

    # Create memory_topics table
    op.create_table(
        "memory_topics",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("mind_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("tags", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("modified_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("ix_memory_topics_organization_id", "memory_topics", ["organization_id"])
    op.create_index("ix_memory_topics_user_id", "memory_topics", ["user_id"])
    op.create_index("ix_memory_topics_mind_id", "memory_topics", ["mind_id"])
    op.create_foreign_key("fk_memory_topics_mind_id", "memory_topics", "minds", ["mind_id"], ["id"])
    op.execute(
        "CREATE UNIQUE INDEX unique_memory_topic_title_per_mind "
        "ON memory_topics (mind_id, title) WHERE deleted_at IS NULL"
    )

    op.execute("""
        CREATE TRIGGER update_memory_topics_modified_at
        BEFORE UPDATE ON memory_topics
        FOR EACH ROW EXECUTE FUNCTION update_modified_at_column();
    """)


def downgrade() -> None:
    """Downgrade schema."""

    # Drop memory_topics
    op.execute("DROP TRIGGER IF EXISTS update_memory_topics_modified_at ON memory_topics;")
    op.execute("ALTER TABLE memory_topics DROP CONSTRAINT IF EXISTS unique_memory_topic_title_per_mind")
    op.execute("DROP INDEX IF EXISTS unique_memory_topic_title_per_mind")
    op.drop_constraint("fk_memory_topics_mind_id", "memory_topics", type_="foreignkey")
    op.drop_index("ix_memory_topics_mind_id", table_name="memory_topics")
    op.drop_index("ix_memory_topics_user_id", table_name="memory_topics")
    op.drop_index("ix_memory_topics_organization_id", table_name="memory_topics")
    op.drop_table("memory_topics")

    # Drop memory_rules
    op.execute("DROP TRIGGER IF EXISTS update_memory_rules_modified_at ON memory_rules;")
    op.drop_constraint("fk_memory_rules_mind_id", "memory_rules", type_="foreignkey")
    op.drop_index("ix_memory_rules_mind_id", table_name="memory_rules")
    op.drop_index("ix_memory_rules_user_id", table_name="memory_rules")
    op.drop_index("ix_memory_rules_organization_id", table_name="memory_rules")
    op.drop_table("memory_rules")

    # Drop enum type
    rule_type_enum = postgresql.ENUM(name="rule_type_enum")
    rule_type_enum.drop(op.get_bind(), checkfirst=True)
