"""
Schema Linker for Text-to-SQL Pipeline.

Identifies relevant tables and columns from the data catalog before SQL generation.
This reduces context size and focuses the LLM on relevant schema elements.

Based on research from:
- SQL-of-Thought (EMNLP 2025)
- CHASE-SQL (2024)
- LinkAlign (EMNLP 2025)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from minds.agents.candidate_sql_agent.linker_agent.instructions_template import SCHEMA_LINKING_PROMPT
from minds.agents.helpers import current_date_time_layer, mind_layer, model_for
from minds.common.logger import get_logger
from minds.common.settings.app_settings import get_app_settings

if TYPE_CHECKING:
    from minds.model.data_catalog import DataCatalog
    from minds.model.mind import Mind

logger = get_logger(__name__)
settings = get_app_settings()


class LinkedSchema(BaseModel):
    """
    Output of schema linking - the relevant subset of the database schema.
    """

    tables: list[str] = Field(description="Fully-qualified table names (datasource.table) relevant to the question.")
    columns: dict[str, list[str]] = Field(description="Mapping of table names to relevant column names.")
    joins: list[list[str]] = Field(
        default_factory=list, description="Join relationships as [table1, col1, table2, col2] tuples."
    )
    reasoning: str = Field(default="", description="Brief explanation of schema selection reasoning.")

    def to_context_str(self) -> str:
        """Convert linked schema to a context string for SQL generation."""
        lines = []
        lines.append("## Relevant Schema")
        lines.append("")

        for table in self.tables:
            cols = self.columns.get(table, [])
            if cols:
                cols_str = ", ".join(cols)
                lines.append(f"Table: {table}")
                lines.append(f"  Columns: {cols_str}")
                lines.append("")

        if self.joins:
            lines.append("## Join Relationships")
            for join in self.joins:
                if len(join) == 4:
                    lines.append(f"  {join[0]}.{join[1]} -> {join[2]}.{join[3]}")
            lines.append("")

        return "\n".join(lines)

    def get_table_list(self) -> list[str]:
        """Get list of table names without datasource prefix."""
        return [t.split(".")[-1] if "." in t else t for t in self.tables]


@dataclass
class SchemaLinkerDeps:
    """Dependencies for schema linking agent."""

    mind: "Mind"
    schema_context: str


schema_linker_agent: Agent[SchemaLinkerDeps, LinkedSchema] = Agent(
    model=None,
    output_type=LinkedSchema,
)


@schema_linker_agent.instructions
async def schema_linker_instructions(ctx: RunContext[SchemaLinkerDeps]) -> str:
    """Build instructions for schema linking."""
    p = current_date_time_layer()
    mp = mind_layer(ctx.deps.mind)
    if mp:
        p += "\n" + mp
    return p


class SchemaLinker:
    """
    Schema Linker identifies relevant tables and columns from the data catalog.

    This is the first step in the SOTA text-to-SQL pipeline, reducing context
    size before SQL generation.
    """

    def __init__(self, mind: "Mind"):
        self.mind = mind

    async def link(
        self,
        question: str,
        data_catalogs: list["DataCatalog"],
    ) -> LinkedSchema:
        """
        Identify relevant schema elements for a question.

        Args:
            question: Natural language question from user
            data_catalogs: List of data catalogs containing full schema

        Returns:
            LinkedSchema with relevant tables, columns, and joins
        """
        schema_context = self._build_schema_summary(data_catalogs)

        prompt = SCHEMA_LINKING_PROMPT.format(
            question=question,
            schema=schema_context,
        )

        deps = SchemaLinkerDeps(
            mind=self.mind,
            schema_context=schema_context,
        )

        model = model_for(self.mind)
        try:
            result = await schema_linker_agent.run(
                prompt,
                deps=deps,
                model=model,
            )
            linked_schema = result.output
            logger.info(
                f"Schema linking complete: {len(linked_schema.tables)} tables, "
                f"{sum(len(cols) for cols in linked_schema.columns.values())} columns"
            )
            return linked_schema
        except Exception as exc:
            logger.warning(f"Schema linking failed: {exc}")
            return LinkedSchema(
                tables=[],
                columns={},
                joins=[],
                reasoning="Schema linking failed; returned empty schema.",
            )

    def _build_schema_summary(self, data_catalogs: list["DataCatalog"]) -> str:
        """
        Build a compact schema summary for the LLM.

        Uses table list format (no column details) to keep context small.
        """
        lines = []

        for catalog in data_catalogs:
            try:
                datasource_name = catalog.mind_datasource.datasource.name
            except Exception:
                datasource_name = "unknown"

            lines.append(f"## Datasource: {datasource_name}")
            lines.append("")

            for mdt in catalog.mind_datasource.mind_datasource_tables:
                table = mdt.table
                qualified_name = f"{datasource_name}.{table.name}"

                # Table header
                table_line = f"Table: {qualified_name}"
                if table.description:
                    table_line += f" - {table.description}"
                lines.append(table_line)
                lines.append("")

        return "\n".join(lines)

    def filter_catalogs_by_linked_schema(
        self,
        data_catalogs: list["DataCatalog"],
        linked_schema: LinkedSchema,
    ) -> str:
        """
        Filter data catalogs to only include tables from linked schema.

        Returns a context string with detailed schema for linked tables only.
        """
        linked_tables_set = set(linked_schema.tables)
        # Also match by table name only (without datasource prefix)
        linked_table_names = {t.split(".")[-1] for t in linked_schema.tables}

        lines = []
        lines.append("## Schema for Query Generation")
        lines.append("")

        for catalog in data_catalogs:
            try:
                datasource_name = catalog.mind_datasource.datasource.name
            except Exception:
                datasource_name = "unknown"

            for mdt in catalog.mind_datasource.mind_datasource_tables:
                table = mdt.table
                qualified_name = f"{datasource_name}.{table.name}"

                # Check if table is in linked schema
                if qualified_name not in linked_tables_set and table.name not in linked_table_names:
                    continue

                # Include full table details
                lines.append(f"Table: {qualified_name}")
                if table.description:
                    lines.append(f"  Description: {table.description}")

                # Primary keys
                if table.primary_key_constraints:
                    pk_cols = [pk.column.name for pk in table.primary_key_constraints]
                    lines.append(f"  Primary Key: {', '.join(pk_cols)}")

                # Columns with details
                lines.append("  Columns:")
                for col in table.columns:
                    col_info = f"    - {col.name} ({col.data_type})"
                    if not col.is_nullable:
                        col_info += " NOT NULL"
                    if col.description:
                        col_info += f" - {col.description}"
                    lines.append(col_info)

                # Foreign keys
                if table.foreign_key_constraints:
                    fk_lines = []
                    for fk in table.foreign_key_constraints:
                        if fk.referenced_table.name != table.name:
                            fk_lines.append(
                                f"    - {fk.column.name} -> "
                                f"{datasource_name}.{fk.referenced_table.name}({fk.referenced_column.name})"
                            )
                    if fk_lines:
                        lines.append("  Foreign Keys:")
                        lines.extend(fk_lines)

                lines.append("")

        return "\n".join(lines)
