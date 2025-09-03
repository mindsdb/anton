from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlmodel import JSON, Field, Relationship
from sqlmodel import Column as SQLModelColumn

from minds.model.base import BaseSQLModel

if TYPE_CHECKING:
    from minds.model.datasource import Datasource


class Table(BaseSQLModel, table=True):
    """Data source table metadata."""

    __tablename__ = "tables"

    datasource_id: UUID = Field(..., description="Datasource ID", foreign_key="datasources.id")
    name: str = Field(..., description="Table name")
    schema: str | None = Field(default=None, description="Schema name")
    description: str | None = Field(default=None, description="Table description/comment")
    type: str | None = Field(default=None, description="Table type")
    row_count: int | None = Field(default=None, description="Row count")

    columns: list["Column"] = Relationship(cascade_delete=True)
    primary_key_constraints: list["PrimaryKeyConstraint"] = Relationship(cascade_delete=True)
    foreign_key_constraints: list["ForeignKeyConstraint"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "ForeignKeyConstraint.table_id",
        },
        cascade_delete=True,
    )


class Column(BaseSQLModel, table=True):
    """Data source column metadata."""

    __tablename__ = "columns"

    table_id: UUID = Field(..., description="Table ID", foreign_key="tables.id")
    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Column data type")
    description: str | None = Field(default=None, description="Column description/comment")
    default_value: str | None = Field(default=None, description="Column default value")
    is_nullable: bool = Field(default=True, description="Whether the column is nullable")

    statistics: "ColumnStatistics" = Relationship(cascade_delete=True)


class ColumnStatistics(BaseSQLModel, table=True):
    """Data source column statistics."""

    __tablename__ = "column_statistics"

    column_id: UUID = Field(..., description="Column ID", foreign_key="columns.id")
    most_common_values: list[Any] | None = Field(
        default_factory=list, sa_column=SQLModelColumn(JSON), description="List of most common values"
    )
    most_common_frequencies: list[float] | None = Field(
        default_factory=list, sa_column=SQLModelColumn(JSON), description="List of most common frequencies"
    )
    null_percentage: float | None = Field(default=None, description="Null percentage")
    distinct_values_count: int | None = Field(default=None, description="Distinct values count")
    min_value: str | None = Field(default=None, description="Minimum value")
    max_value: str | None = Field(default=None, description="Maximum value")


class PrimaryKeyConstraint(BaseSQLModel, table=True):
    """Data source primary key metadata."""

    __tablename__ = "primary_key_constraints"

    table_id: UUID = Field(..., description="Table ID", foreign_key="tables.id")
    column_id: UUID = Field(..., description="Column ID", foreign_key="columns.id")
    ordinal_position: int | None = Field(default=None, description="Ordinal position")
    constraint_name: str | None = Field(default=None, description="Constraint name")

    column: "Column" = Relationship()


class ForeignKeyConstraint(BaseSQLModel, table=True):
    """Data source foreign key metadata."""

    __tablename__ = "foreign_key_constraints"

    table_id: UUID = Field(..., description="Table ID", foreign_key="tables.id")
    column_id: UUID = Field(..., description="Column ID", foreign_key="columns.id")
    referenced_table_id: UUID = Field(..., description="Referenced table ID", foreign_key="tables.id")
    referenced_column_id: UUID = Field(..., description="Referenced column ID", foreign_key="columns.id")
    constraint_name: str | None = Field(default=None, description="Constraint name")
    ordinal_position: int | None = Field(default=None, description="Ordinal position")

    column: "Column" = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "ForeignKeyConstraint.column_id",
        }
    )
    referenced_table: "Table" = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "ForeignKeyConstraint.referenced_table_id",
        }
    )
    referenced_column: "Column" = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "ForeignKeyConstraint.referenced_column_id",
        }
    )


class DataCatalog(BaseSQLModel, table=False):
    """Data catalog metadata - a helper model for accessing datasource metadata."""

    datasource: "Datasource" = Field(..., description="Datasource")

    @classmethod
    def from_datasource(cls, datasource: "Datasource") -> "DataCatalog":
        """Create a DataCatalog instance from a datasource and its tables."""
        return cls(datasource=datasource)

    def _format_header(self) -> list[str]:
        """Format the header section with data source information."""
        lines = []
        lines.append(f"MindsDB Data Source: {self.datasource.name}")
        lines.append(f"Engine: {self.datasource.engine}")

        if self.datasource.handler_info:
            lines.append(f"Handler Info: {self.datasource.handler_info}")
        lines.append("")

        lines.append(f"Tables: {len(self.datasource.tables)}")
        lines.append("")
        return lines

    def _format_table(self, table: Table) -> list[str]:
        """Format a single table's complete information."""
        lines = []

        # Table header.
        qualified_table_name = f"{self.datasource.name}.{table.name}"
        table_header = f"Table: {qualified_table_name}"
        if table.description:
            table_header += f" - {table.description}"
        lines.append(table_header)

        lines.extend(self._format_table_constraints(table))

        lines.append("  Columns:")
        for column in table.columns:
            lines.extend(self._format_column(column))

        lines.append("")  # Blank line between tables.
        return lines

    def _format_column(self, column: Column) -> list[str]:
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

    def _format_column_statistics(self, column: Column) -> list[str]:
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

    def _format_table_constraints(self, table: Table) -> list[str]:
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
                fk_column_name = fk.column.name
                ref_column_name = fk.referenced_column.name
                qualified_fk_table = f"{self.datasource.name}.{fk.referenced_table.name}"
                fk_info = f"    - {fk_column_name} → {qualified_fk_table}({ref_column_name})"
                # if fk.is_implicit:
                #     confidence = fk.confidence or 0.0
                #     fk_info += f" [implicit, confidence: {confidence:.1%}]"
                lines.append(fk_info)

        return lines

    def _format_relationships(self) -> list[str]:
        """Format the relationships section."""
        lines = []
        relationship_lines = []

        for table in self.datasource.tables:
            related = self._get_related_tables(table)
            related = [f"{self.datasource.name}.{n}" for n in related]
            if related:
                qualified_table_name = f"{self.datasource.name}.{table.name}"
                relationship_lines.append(f"{qualified_table_name} is related to: {', '.join(related)}")

        if relationship_lines:
            lines.append("Relationships:")
            lines.extend([f"  - {line}" for line in relationship_lines])
            lines.append("")

        return lines

    def _get_related_tables(self, table: Table) -> list[str]:
        """Get tables related to the given table through foreign keys."""
        related_tables = []

        # Tables referenced by this table's foreign keys.
        for fk in table.foreign_key_constraints:
            related_tables.append(fk.referenced_table)

        # Tables that reference this table.
        for other_table in self.datasource.tables:
            if other_table.name == table.name:
                continue

            for fk in other_table.foreign_key_constraints:
                if fk.referenced_table == table.name:
                    related_tables.append(other_table.name)

        return list(set(related_tables))

    def to_context_str(self) -> str:
        """
        Convert the MindsDB data catalog to a formatted string for LLM context.

        The format is optimized for LLMs to understand MindsDB integration structure.
        for SQL query generation against the underlying data source.

        Returns:
            A formatted string representation of the MindsDB data catalog.
        """
        lines = []

        lines.extend(self._format_header())

        for table in self.datasource.tables:
            lines.extend(self._format_table(table))

        lines.extend(self._format_relationships())

        return "\n".join(lines)


# Resolve forward references after all models are defined
if not TYPE_CHECKING:
    from minds.model.datasource import Datasource

    DataCatalog.model_rebuild()
