from typing import Any, Optional, List
from uuid import UUID

from sqlmodel import Field, Relationship, SQLModel

from minds.model.base import BaseSQLModel


class Table(BaseSQLModel, table=True):
    """Data source table metadata."""
    __tablename__ = "tables"

    # TODO: Connect to datasource table.
    name: str = Field(..., description="Table name")
    schema: str = Field(default=None, description="Schema name")
    description: Optional[str] = Field(default=None, description="Table description/comment")
    type: str = Field(default=None, description="Table type")
    row_count: Optional[int] = Field(default=None, description="Row count")

    columns: List["Column"] = Relationship(cascade_delete=True)
    primary_key_constraints: List["PrimaryKeyConstraint"] = Relationship(cascade_delete=True)
    foreign_key_constraints: List["ForeignKeyConstraint"] = Relationship(cascade_delete=True)


class Column(BaseSQLModel, table=True):
    """Data source column metadata."""
    __tablename__ = "columns"
    
    table_id: UUID = Field(..., description="Table ID", foreign_key="tables.id")
    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Column data type")
    description: Optional[str] = Field(default=None, description="Column description/comment")
    default_value: Optional[str] = Field(default=None, description="Column default value")
    is_nullable: bool = Field(default=True, description="Whether the column is nullable")

    statistics: "ColumnStatistics" = Relationship(cascade_delete=True)


class ColumnStatistics(BaseSQLModel, table=True):
    """Data source column statistics."""
    __tablename__ = "column_statistics"
    
    column_id: UUID = Field(..., description="Column ID", foreign_key="columns.id")
    most_common_values: Optional[List[Any]] = Field(default=None, description="Most common values")
    most_common_frequencies: Optional[List[float]] = Field(default=None, description="Most common frequencies")
    null_percentage: Optional[float] = Field(default=None, description="Null percentage")
    distinct_values_count: Optional[int] = Field(default=None, description="Distinct values count")
    min_value: Optional[str] = Field(default=None, description="Minimum value")
    max_value: Optional[str] = Field(default=None, description="Maximum value")


class PrimaryKeyConstraint(BaseSQLModel, table=True):
    """Data source primary key metadata."""
    __tablename__ = "primary_key_constraints"
    
    table_id: UUID = Field(..., description="Table ID", foreign_key="tables.id")
    column_id: UUID = Field(..., description="Column ID", foreign_key="columns.id")
    ordinal_position: Optional[int] = Field(default=None, description="Ordinal position")
    constraint_name: Optional[str] = Field(default=None, description="Constraint name")

    column: "Column" = Relationship()


class ForeignKeyConstraint(BaseSQLModel, table=True):
    """Data source foreign key metadata."""
    __tablename__ = "foreign_key_constraints"
    
    table_id: UUID = Field(..., description="Table ID", foreign_key="tables.id")
    column_id: UUID = Field(..., description="Column ID", foreign_key="columns.id")
    referenced_table_id: UUID = Field(..., description="Referenced table ID", foreign_key="tables.id")
    referenced_column_id: UUID = Field(..., description="Referenced column ID", foreign_key="columns.id")
    constraint_name: Optional[str] = Field(default=None, description="Constraint name")
    ordinal_position: Optional[int] = Field(default=None, description="Ordinal position")

    column: "Column" = Relationship()
    referenced_table: "Table" = Relationship()
    referenced_column: "Column" = Relationship()


class DataCatalog(BaseSQLModel, table=False):
    """Data catalog metadata."""
    # TODO: Connect to datasource table.
    tables: List["Table"] = Relationship()

    def _format_header(self) -> List[str]:
        """Format the header section with data source information."""
        # TODO: Fix reference to data source.
        lines = []
        lines.append(f"MindsDB Data Source: {self.data_source.name}")
        lines.append(f"Engine: {self.data_source.engine}")
        if self.data_source.handler_info:
            lines.append(f"Handler Info: {self.data_source.handler_info}")
        lines.append("")
        lines.append(f"Tables: {len(self.tables)}")
        lines.append("")
        return lines
    
    def _format_table(self, table: Table) -> List[str]:
        """Format a single table's complete information."""
        lines = []
        
        # Table header
        qualified_table_name = f"{self.data_source.name}.{table.name}"
        table_header = f"Table: {qualified_table_name}"
        if table.description:
            table_header += f" - {table.description}"
        lines.append(table_header)

        lines.extend(self._format_table_constraints(table))

        lines.append("  Columns:")
        for column in table.columns:
            lines.extend(self._format_column(column))

        lines.append("")  # Blank line between tables
        return lines
    
    def _format_column(self, column: Column) -> List[str]:
        """Format a single column's information including statistics and samples."""
        lines = []

        column_info = f"    - {column.name} ({column.data_type})"
        if not column.is_nullable:
            column_info += " NOT NULL"
        if column.column_default and column.column_default != "[NULL]":
            column_info += f" DEFAULT {column.column_default}"
        if column.description:
            column_info += f" - {column.description}"
        lines.append(column_info)

        lines.extend(self._format_column_statistics(column))

        # Sample values
        # if column.sample_values and len(column.sample_values) > 0:
        #     sample_str = ", ".join([str(val) for val in column.sample_values[:5]])
        #     lines.append(f"      Samples: {sample_str}")

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
        
        # Handle MindsDB's empty array format for most common values.
        if (stats.most_common_values and 
            stats.most_common_values != [""] and 
            stats.most_common_frequencies and
            stats.most_common_frequencies != [""]):
            common_values = list(zip(
                stats.most_common_values[:3],
                stats.most_common_frequencies[:3]
            ))
            if common_values:
                common_str = ", ".join(
                    [f"{val} ({freq:.1%})" for val, freq in common_values]
                )
                lines.append(f"      Most Common: {common_str}")

        return lines
    
    def _format_table_constraints(self, table: Table) -> List[str]:
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
                fk_columns = ", ".join([fk.column.name for fk in fk.column])
                ref_columns = ", ".join([fk.referenced_column.name for fk in fk.referenced_column])
                qualified_fk_table = f"{self.data_source.name}.{fk.referenced_table}"
                fk_info = f"    - {fk_columns} → {qualified_fk_table}({ref_columns})"
                # if fk.is_implicit:
                #     confidence = fk.confidence or 0.0
                #     fk_info += f" [implicit, confidence: {confidence:.1%}]"
                lines.append(fk_info)

        return lines
    
    def _format_relationships(self) -> List[str]:
        """Format the relationships section."""
        lines = []
        relationship_lines = []

        for table in self.tables:
            related = self._get_related_tables(table)
            related = [f"{self.data_source.name}.{n}" for n in related]
            if related:
                qualified_table_name = f"{self.data_source.name}.{table.name}"
                relationship_lines.append(
                    f"{qualified_table_name} is related to: {', '.join(related)}"
                )

        if relationship_lines:
            lines.append("Relationships:")
            lines.extend([f"  - {line}" for line in relationship_lines])
            lines.append("")

        return lines

    def _get_related_tables(self, table: Table) -> List[str]:
        """Get tables related to the given table through foreign keys."""
        related_tables = []

        # Tables referenced by this table's foreign keys.
        for fk in table.foreign_key_constraints:
            related_tables.append(fk.referenced_table)

        # Tables that reference this table.
        for other_table in self.tables:
            if other_table.name == table.name:
                continue

            for fk in other_table.foreign_key_constraints:
                if fk.referenced_table == table.name:
                    related_tables.append(other_table.name)

        return list(set(related_tables))  # Remove duplicates