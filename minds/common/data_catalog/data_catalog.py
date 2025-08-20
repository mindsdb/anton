from datetime import datetime
from typing import Dict, List, Optional, Any
import math

from pydantic import BaseModel, Field, field_validator


class DataCatalog(BaseModel):
    """Base class for data catalogs."""

    created_at: datetime = Field(
        default_factory=datetime.now, description="When this catalog was created"
    )
    last_updated: datetime = Field(
        default_factory=datetime.now, description="When this catalog was last updated"
    )

    def to_context_str(self) -> str:
        """
        Convert the data catalog to a formatted string for LLM context.

        This method should be implemented by subclasses to provide a
        human-readable representation of the catalog that is optimized
        for LLM understanding.

        Returns:
            A formatted string representation of the data catalog
        """
        raise NotImplementedError("Subclasses must implement to_context_str method")


class ColumnStatistics(BaseModel):
    """Statistics for a database column from pg_stats."""

    most_common_values: Optional[List[Any]] = Field(
        default=None, description="Most common values in the column"
    )
    most_common_frequencies: Optional[List[float]] = Field(
        default=None, description="Frequencies of the most common values"
    )
    null_percentage: Optional[float] = Field(
        default=None, description="Percentage of NULL values in the column"
    )
    distinct_values_count: Optional[int] = Field(
        default=None, description="Number of distinct values in the column"
    )
    min_value: Optional[Any] = Field(
        default=None, description="Minimum value in the column"
    )
    max_value: Optional[Any] = Field(
        default=None, description="Maximum value in the column"
    )

    @field_validator("distinct_values_count", mode="before")
    @classmethod
    def validate_distinct_values_count(cls, v):
        """Convert NaN or infinite values to None for distinct_values_count."""
        if isinstance(v, (int, float)):
            if math.isnan(v) or math.isinf(v):
                return None
            return int(v)
        return v

    @field_validator("null_percentage", mode="before")
    @classmethod
    def validate_null_percentage(cls, v):
        """Convert NaN or infinite values to None for null_percentage."""
        if isinstance(v, (int, float)):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        return v


class Column(BaseModel):
    """Database column information."""

    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Data type of the column")
    is_nullable: bool = Field(
        ..., description="Whether the column can contain NULL values"
    )
    column_default: Optional[str] = Field(
        default=None, description="Default value for the column"
    )
    description: Optional[str] = Field(
        default=None, description="Column description/comment"
    )
    statistics: Optional[ColumnStatistics] = Field(
        default=None, description="Column statistics from pg_stats"
    )
    sample_values: Optional[List[Any]] = Field(
        default=None, description="Sample values from the column"
    )


class PrimaryKey(BaseModel):
    """Primary key information."""

    constraint_name: str = Field(..., description="Name of the primary key constraint")
    column_names: List[str] = Field(
        ..., description="Column(s) that make up the primary key"
    )


class ForeignKey(BaseModel):
    """Foreign key information."""

    constraint_name: str = Field(..., description="Name of the foreign key constraint")
    column_names: List[str] = Field(
        ..., description="Column(s) that make up the foreign key"
    )
    referenced_table: str = Field(
        ..., description="Table referenced by the foreign key"
    )
    referenced_columns: List[str] = Field(
        ..., description="Column(s) referenced by the foreign key"
    )
    is_implicit: bool = Field(
        default=False,
        description="Whether this is an implicit relationship detected from column names",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="For implicit FKs, confidence level of this being valid (0-1)",
    )


class Table(BaseModel):
    """Database table information."""

    name: str = Field(..., description="Table name")
    description: Optional[str] = Field(
        default=None, description="Table description/comment"
    )
    columns: List[Column] = Field(
        default_factory=list, description="Columns in the table"
    )
    primary_key: Optional[PrimaryKey] = Field(
        default=None, description="Primary key information"
    )
    foreign_keys: List[ForeignKey] = Field(
        default_factory=list, description="Foreign key constraints (including implicit)"
    )
    created_at: Optional[datetime] = Field(
        default=None, description="When the table was created"
    )


class RelationalDataCatalog(DataCatalog):
    """Data catalog for relational databases like PostgreSQL."""

    datasource_name: str = Field(
        ..., description="Name of MindsDB datasource for data catalog"
    )
    database_name: str = Field(..., description="Name of database")
    engine: str = Field(
        ..., description="Database engine (e.g., 'PostgreSQL', 'MySQL')"
    )
    version: Optional[str] = Field(default=None, description="Database version")
    tables: Dict[str, Table] = Field(
        default_factory=dict, description="Tables in the database, keyed by table name"
    )

    def get_table(self, table_name: str) -> Optional[Table]:
        """Get a table by name."""
        return self.tables.get(table_name)

    def get_related_tables(self, table_name: str) -> List[str]:
        """Get tables related to the given table through foreign keys."""
        table = self.get_table(table_name)
        if not table:
            return []

        related_tables = []
        # Tables referenced by this table's foreign keys
        for fk in table.foreign_keys:
            related_tables.append(fk.referenced_table)

        # Tables that reference this table
        for other_table_name, other_table in self.tables.items():
            if other_table_name == table_name:
                continue

            for fk in other_table.foreign_keys:
                if fk.referenced_table == table_name:
                    related_tables.append(other_table_name)

        return list(set(related_tables))  # Remove duplicates

    def to_context_str(self) -> str:
        """
        Convert the relational data catalog to a formatted string for LLM context.

        The format is optimized for LLMs to understand database structure for SQL query generation.

        Returns:
            A formatted string representation of the relational data catalog
        """
        lines = []

        # Database information header
        lines.append(f"Database: {self.database_name}")
        lines.append(f"Engine: {self.engine}")
        if self.version:
            lines.append(f"Version: {self.version}")
        lines.append("")

        # Table information
        lines.append(f"Tables: {len(self.tables)}")
        lines.append("")

        # Format each table
        for table_name, table in self.tables.items():
            # Table header with description if available
            qualified_table_name = f"{self.datasource_name}.{table_name}"
            table_header = f"Table: {qualified_table_name}"
            if table.description:
                table_header += f" - {table.description}"
            lines.append(table_header)

            # Primary key
            if table.primary_key:
                pk_columns = ", ".join(table.primary_key.column_names)
                lines.append(f"  Primary Key: {pk_columns}")

            # Columns
            lines.append("  Columns:")
            for column in table.columns:
                column_info = f"    - {column.name} ({column.data_type})"
                if not column.is_nullable:
                    column_info += " NOT NULL"
                if column.column_default:
                    column_info += f" DEFAULT {column.column_default}"
                if column.description:
                    column_info += f" - {column.description}"
                lines.append(column_info)

                # Include column statistics if available
                if column.statistics:
                    stats = column.statistics
                    if stats.distinct_values_count is not None:
                        lines.append(
                            f"      Distinct Values: {stats.distinct_values_count}"
                        )
                    if stats.null_percentage is not None:
                        lines.append(f"      Null %: {stats.null_percentage:.1f}%")
                    if stats.min_value is not None and stats.max_value is not None:
                        lines.append(
                            f"      Range: {stats.min_value} to {stats.max_value}"
                        )
                    if stats.most_common_values and stats.most_common_frequencies:
                        common_values = list(
                            zip(
                                stats.most_common_values[:3],
                                stats.most_common_frequencies[:3],
                            )
                        )
                        if common_values:
                            common_str = ", ".join(
                                [f"{val} ({freq:.1%})" for val, freq in common_values]
                            )
                            lines.append(f"      Most Common: {common_str}")

                # Include sample values if available (limited to 5)
                if column.sample_values and len(column.sample_values) > 0:
                    sample_str = ", ".join(
                        [str(val) for val in column.sample_values[:5]]
                    )
                    lines.append(f"      Samples: {sample_str}")

            # Foreign keys
            if table.foreign_keys:
                lines.append("  Foreign Keys:")
                for fk in table.foreign_keys:
                    fk_columns = ", ".join(fk.column_names)
                    ref_columns = ", ".join(fk.referenced_columns)
                    qualified_fk_table = f"{self.datasource_name}.{fk.referenced_table}"
                    fk_info = (
                        f"    - {fk_columns} → {qualified_fk_table}({ref_columns})"
                    )
                    if fk.is_implicit:
                        confidence = fk.confidence or 0.0
                        fk_info += f" [implicit, confidence: {confidence:.1%}]"
                    lines.append(fk_info)

            # Add a blank line between tables
            lines.append("")

        # Add a section for relationships if there are any foreign keys
        relationship_lines = []
        for table_name, table in self.tables.items():
            related = self.get_related_tables(table_name)
            related = [f"{self.datasource_name}.{n}" for n in related]
            if related:
                qualified_table_name = f"{self.datasource_name}.{table_name}"
                relationship_lines.append(
                    f"{qualified_table_name} is related to: {', '.join(related)}"
                )

        if relationship_lines:
            lines.append("Relationships:")
            lines.extend([f"  - {line}" for line in relationship_lines])
            lines.append("")

        return "\n".join(lines)


class MindsDBDataCatalog(DataCatalog):
    """Data catalog for MindsDB integrations with comprehensive metadata support."""

    integration_name: str = Field(
        ..., description="Name of MindsDB integration (TABLE_SCHEMA)"
    )
    engine: str = Field(default="MindsDB", description="Database engine type")
    handler_info: Optional[str] = Field(
        default=None, description="Handler information from META_HANDLER_INFO"
    )
    tables: Dict[str, Table] = Field(
        default_factory=dict,
        description="Tables in the integration, keyed by table name",
    )

    def get_table(self, table_name: str) -> Optional[Table]:
        """Get a table by name."""
        return self.tables.get(table_name)

    def get_related_tables(self, table_name: str) -> List[str]:
        """Get tables related to the given table through foreign keys."""
        table = self.get_table(table_name)
        if not table:
            return []

        related_tables = []
        # Tables referenced by this table's foreign keys
        for fk in table.foreign_keys:
            related_tables.append(fk.referenced_table)

        # Tables that reference this table
        for other_table_name, other_table in self.tables.items():
            if other_table_name == table_name:
                continue

            for fk in other_table.foreign_keys:
                if fk.referenced_table == table_name:
                    related_tables.append(other_table_name)

        return list(set(related_tables))  # Remove duplicates

    def _format_header(self) -> List[str]:
        """Format the header section with integration information."""
        lines = []
        lines.append(f"MindsDB Integration: {self.integration_name}")
        lines.append(f"Engine: {self.engine}")
        if self.handler_info:
            lines.append(f"Handler Info: {self.handler_info}")
        lines.append("")
        lines.append(f"Tables: {len(self.tables)}")
        lines.append("")
        return lines

    def _format_column_statistics(self, column: Column) -> List[str]:
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

        # Handle MindsDB's empty array format for most common values
        if (
            stats.most_common_values
            and stats.most_common_values != [""]
            and stats.most_common_frequencies
            and stats.most_common_frequencies != [""]
        ):
            common_values = list(
                zip(stats.most_common_values[:3], stats.most_common_frequencies[:3])
            )
            if common_values:
                common_str = ", ".join(
                    [f"{val} ({freq:.1%})" for val, freq in common_values]
                )
                lines.append(f"      Most Common: {common_str}")

        return lines

    def _format_column_info(self, column: Column) -> List[str]:
        """Format a single column's information including statistics and samples."""
        lines = []

        # Basic column info
        column_info = f"    - {column.name} ({column.data_type})"
        if not column.is_nullable:
            column_info += " NOT NULL"
        if column.column_default and column.column_default != "[NULL]":
            column_info += f" DEFAULT {column.column_default}"
        if column.description:
            column_info += f" - {column.description}"
        lines.append(column_info)

        # Statistics
        lines.extend(self._format_column_statistics(column))

        # Sample values
        if column.sample_values and len(column.sample_values) > 0:
            sample_str = ", ".join([str(val) for val in column.sample_values[:5]])
            lines.append(f"      Samples: {sample_str}")

        return lines

    def _format_table_constraints(self, table: Table) -> List[str]:
        """Format primary keys and foreign keys for a table."""
        lines = []

        # Primary key
        if table.primary_key:
            pk_columns = ", ".join(table.primary_key.column_names)
            lines.append(f"  Primary Key: {pk_columns}")

        # Foreign keys
        if table.foreign_keys:
            lines.append("  Foreign Keys:")
            for fk in table.foreign_keys:
                fk_columns = ", ".join(fk.column_names)
                ref_columns = ", ".join(fk.referenced_columns)
                qualified_fk_table = f"{self.integration_name}.{fk.referenced_table}"
                fk_info = f"    - {fk_columns} → {qualified_fk_table}({ref_columns})"
                if fk.is_implicit:
                    confidence = fk.confidence or 0.0
                    fk_info += f" [implicit, confidence: {confidence:.1%}]"
                lines.append(fk_info)

        return lines

    def _format_single_table(self, table_name: str, table: Table) -> List[str]:
        """Format a single table's complete information."""
        lines = []

        # Table header
        qualified_table_name = f"{self.integration_name}.{table_name}"
        table_header = f"Table: {qualified_table_name}"
        if table.description:
            table_header += f" - {table.description}"
        lines.append(table_header)

        # Constraints
        lines.extend(self._format_table_constraints(table))

        # Columns
        lines.append("  Columns:")
        for column in table.columns:
            lines.extend(self._format_column_info(column))

        lines.append("")  # Blank line between tables
        return lines

    def _format_relationships(self) -> List[str]:
        """Format the relationships section."""
        lines = []
        relationship_lines = []

        for table_name, table in self.tables.items():
            related = self.get_related_tables(table_name)
            related = [f"{self.integration_name}.{n}" for n in related]
            if related:
                qualified_table_name = f"{self.integration_name}.{table_name}"
                relationship_lines.append(
                    f"{qualified_table_name} is related to: {', '.join(related)}"
                )

        if relationship_lines:
            lines.append("Relationships:")
            lines.extend([f"  - {line}" for line in relationship_lines])
            lines.append("")

        return lines

    def to_context_str(self) -> str:
        """
        Convert the MindsDB data catalog to a formatted string for LLM context.

        The format is optimized for LLMs to understand MindsDB integration structure
        for SQL query generation against the underlying data source.

        Returns:
            A formatted string representation of the MindsDB data catalog
        """
        lines = []

        # Header section
        lines.extend(self._format_header())

        # Tables section
        for table_name, table in self.tables.items():
            lines.extend(self._format_single_table(table_name, table))

        # Relationships section
        lines.extend(self._format_relationships())

        return "\n".join(lines)
