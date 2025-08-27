from typing import Any, Optional, List
from uuid import UUID

from sqlmodel import Field, Relationship

from minds.model.base import BaseSQLModel


class Table(BaseSQLModel, table=True):
    """Data source table metadata."""
    __tablename__ = "tables"
    
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