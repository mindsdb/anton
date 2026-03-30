from datetime import datetime
from typing import TYPE_CHECKING

from sqlmodel import Field, SQLModel

if TYPE_CHECKING:
    from minds.model.data_catalog.column import Column
    from minds.model.data_catalog.table import Table
    from minds.model.mind_datasource import MindDatasource


class DataCatalog(SQLModel, table=False):
    """Data catalog metadata - a helper model for accessing datasource metadata."""

    mind_datasource: "MindDatasource" = Field(..., description="MindDatasource")
    modified_at: datetime = Field(..., description="Last modified timestamp")

    @classmethod
    def from_mind_datasource(cls, mind_datasource: "MindDatasource") -> "DataCatalog":
        """Create a DataCatalog instance from a datasource and its tables."""
        return cls(mind_datasource=mind_datasource, modified_at=datetime.now())

    def _format_header(self) -> list[str]:
        """Format the header section with data source information."""
        lines = []
        lines.append(f"MindsDB Data Source: {self.mind_datasource.datasource.name}")
        lines.append(f"Engine: {self.mind_datasource.datasource.engine}")
        if self.mind_datasource.datasource.description:
            lines.append(f"Description: {self.mind_datasource.datasource.description}")

        if self.mind_datasource.datasource.engine_info:
            lines.append(f"Engine Info: {self.mind_datasource.datasource.engine_info}")
        lines.append("")

        lines.append(f"Number of Tables: {len(self.mind_datasource.mind_datasource_tables)}")
        lines.append("")
        return lines

    def _format_table(self, table: "Table", include_datasource_name: bool = True) -> list[str]:
        """Format a single table's complete information."""
        lines = []

        # Table header.
        qualified_table_name = table.qualified_name
        if include_datasource_name:
            qualified_table_name = f"{self.mind_datasource.datasource.name}.{qualified_table_name}"
        table_header = f"Table: {qualified_table_name}"
        if table.description:
            table_header += f" - {table.description}"
        lines.append(table_header)

        lines.extend(self._format_table_constraints(table, include_datasource_name))

        lines.append("  Columns:")
        for column in table.columns:
            lines.extend(self._format_column(column))

        lines.append("")  # Blank line between tables.
        return lines

    def _format_table_without_stats(self, table: "Table", include_datasource_name: bool = True) -> list[str]:
        """Format a single table's information without statistics (compact version)."""
        lines = []

        qualified_table_name = table.qualified_name
        if include_datasource_name:
            qualified_table_name = f"{self.mind_datasource.datasource.name}.{qualified_table_name}"
        table_header = f"Table: {qualified_table_name}"
        if table.description:
            table_header += f" - {table.description}"
        lines.append(table_header)

        lines.extend(self._format_table_constraints(table, include_datasource_name))

        lines.append("  Columns:")
        for column in table.columns:
            column_info = f"    - {column.name} ({column.data_type})"
            if not column.is_nullable:
                column_info += " NOT NULL"
            if column.description:
                column_info += f" - {column.description}"
            lines.append(column_info)

        lines.append("")
        return lines

    def _format_column(self, column: "Column") -> list[str]:
        """Format a single column's information including statistics and samples."""
        lines = []

        column_info = f"    - {column.name} ({column.data_type})"
        if not column.is_nullable:
            column_info += " NOT NULL"
        if column.default_value and column.default_value != "[NULL]":
            column_info += f" DEFAULT {column.default_value}"
        if column.description:
            column_info += f" - {column.description}"
        lines.append(column_info)

        lines.extend(self._format_column_statistics(column))

        # Sample values.
        # if column.sample_values and len(column.sample_values) > 0:
        #     sample_str = ", ".join([str(val) for val in column.sample_values[:5]])
        #     lines.append(f"      Samples: {sample_str}")

        return lines

    def _format_column_statistics(self, column: "Column") -> list[str]:
        """Format column statistics for MindsDB-specific data."""
        lines = []
        if not column.statistics:
            return lines

        stats = column.statistics
        if stats.distinct_values_count is not None:
            lines.append(f"      Distinct Values: {stats.distinct_values_count}")

        if stats.null_percentage is not None:
            lines.append(f"      Null %: {stats.null_percentage:.1f}%")

        if stats.min_value is not None and stats.max_value is not None:
            lines.append(f"      Range: {stats.min_value} to {stats.max_value}")

        # Handle MindsDB's empty array format for most common values.
        if (
            stats.most_common_values
            and stats.most_common_values != [""]
            and stats.most_common_frequencies
            and stats.most_common_frequencies != [""]
        ):
            common_values = list(zip(stats.most_common_values[:3], stats.most_common_frequencies[:3], strict=False))
            if common_values:
                common_str = ", ".join([f"{val} ({freq:.1%})" for val, freq in common_values])
                lines.append(f"      Most Common: {common_str}")

        return lines

    def _format_table_constraints(self, table: "Table", include_datasource_name: bool = True) -> list[str]:
        """Format primary keys and foreign keys for a table."""
        lines = []

        # Primary keys.
        if table.primary_key_constraints:
            pk_columns = ", ".join([pk.column.name for pk in table.primary_key_constraints])
            lines.append(f"  Primary Key: {pk_columns}")

        # Foreign keys.
        if table.foreign_key_constraints:
            lines.append("  Foreign Keys:")
            for fk in table.foreign_key_constraints:
                # TODO: Validate these conditions are correct.
                # Foreign keys will also contain instances where this table is the referenced table.
                # i.e., the relationship is represented both ways.
                if fk.referenced_table.name == table.name:
                    continue

                # Further, foreign keys will also contains instances where the referenced table is
                # not included in the relationship.
                valid_tables = [
                    mind_datasource_table.table.name
                    for mind_datasource_table in self.mind_datasource.mind_datasource_tables
                ]
                if fk.referenced_table.name not in valid_tables:
                    continue

                fk_column_name = fk.column.name
                ref_column_name = fk.referenced_column.name
                qualified_fk_table = fk.referenced_table.qualified_name
                if include_datasource_name:
                    qualified_fk_table = f"{self.mind_datasource.datasource.name}.{qualified_fk_table}"
                fk_info = f"    - {fk_column_name} → {qualified_fk_table}({ref_column_name})"
                lines.append(fk_info)

        return lines

    def _format_relationships(self, table: "Table", include_datasource_name: bool = True) -> list[str]:
        """Format the relationships section for each table."""
        lines = []

        related_tables = []
        for fk in table.foreign_key_constraints:
            # Same guardrails as the table section.
            if fk.referenced_table.name == table.name:
                continue

            valid_tables = [
                mind_datasource_table.table.name
                for mind_datasource_table in self.mind_datasource.mind_datasource_tables
            ]
            if fk.referenced_table.name not in valid_tables:
                continue

            related_tables.append(fk.referenced_table.qualified_name)

        if related_tables:
            qualified_table_name = table.qualified_name
            if include_datasource_name:
                qualified_table_name = f"{self.mind_datasource.datasource.name}.{qualified_table_name}"
            lines.append(f"{qualified_table_name} is related to: {', '.join(related_tables)}")

        return lines

    def to_context_str(
        self,
        max_tokens: int | None = None,
        include_statistics: bool = True,
        table_names: list[str] | None = None,
        include_datasource_name: bool = True,
    ) -> str:
        """
        Convert the MindsDB data catalog to a formatted string for LLM context.

        The format is optimized for LLMs to understand MindsDB integration structure.
        for SQL query generation against the underlying data source.

        Args:
            max_tokens: Maximum number of tokens to include (approximate, uses char count / 4).
                       If None, no limit is applied.
            include_statistics: Whether to include column statistics. Set to False to reduce size.
            table_names: Optional list of specific table names to include. If None, includes all tables.
            include_datasource_name: Whether to include the data source as part of the tables names.

        Returns:
            A formatted string representation of the MindsDB data catalog.
        """
        lines = []
        relationship_lines = []

        lines.extend(self._format_header())

        # Filter tables if table_names is provided
        tables_to_include = self.mind_datasource.mind_datasource_tables
        if table_names:
            tables_to_include = [mdt for mdt in tables_to_include if mdt.table.name in table_names]

        for mind_datasource_table in tables_to_include:
            table = mind_datasource_table.table

            if include_statistics:
                lines.extend(self._format_table(table, include_datasource_name))
            else:
                lines.extend(self._format_table_without_stats(table, include_datasource_name))

            relationship_lines.extend(self._format_relationships(table, include_datasource_name))

        if relationship_lines:
            lines.append("Relationships:")
            lines.extend([f"  - {line}" for line in relationship_lines])
            lines.append("")

        result = "\n".join(lines)

        if max_tokens:
            max_chars = max_tokens * 4
            if len(result) > max_chars:
                # Instead of hard truncation, provide a warning
                result = (
                    result[:max_chars] + "\n\n[... catalog truncated due to size limits. "
                    "Consider using catalog pruning to focus on specific tables ...]"
                )

        return result

    def _get_table_summary(self, table: "Table", include_datasource_name: bool = True) -> str:
        """Get a one-line summary of a table for compact overview."""
        qualified_name = table.qualified_name
        if include_datasource_name:
            qualified_name = f"{self.mind_datasource.datasource.name}.{qualified_name}"
        col_count = len(table.columns)
        summary = f"{qualified_name} ({col_count} columns)"
        if table.description:
            summary += f" - {table.description}"
        return summary

    def get_table_list_summary(self, include_datasource_name: bool = True) -> str:
        """Get a compact list of all tables with basic info for initial context."""
        lines = []

        if include_datasource_name:
            lines.append(f"Data Source: {self.mind_datasource.datasource.name}")
        lines.append(f"Total Tables: {len(self.mind_datasource.mind_datasource_tables)}")
        lines.append("")
        lines.append("Available Tables:")

        for mind_datasource_table in self.mind_datasource.mind_datasource_tables:
            table = mind_datasource_table.table
            lines.append(f"  - {self._get_table_summary(table, include_datasource_name)}")

        lines.append("")
        lines.append("Note: Use catalog pruning to focus on specific tables for detailed schema information.")
        return "\n".join(lines)


# Resolve forward references after all models are defined
if not TYPE_CHECKING:
    from minds.model.mind_datasource import MindDatasource

    DataCatalog.model_rebuild()
