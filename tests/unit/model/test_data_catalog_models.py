from datetime import datetime
from typing import Any, Optional, Union
from unittest.mock import Mock
from uuid import UUID

import pytest

from minds.model.data_catalog import (
    Table,
    Column,
    ColumnStatistics,
    PrimaryKeyConstraint,
    ForeignKeyConstraint,
    DataCatalog,
)
# Import related models to ensure SQLAlchemy can resolve all relationships
from minds.model.datasource import Datasource
from minds.model.mind_datasource import MindDatasource
from minds.model.mind import Mind


class TestTable:
    """Test cases for Table model class."""

    def test_table_initialization_with_required_fields(self):
        """Test that Table can be instantiated with required fields."""
        datasource_id = UUID("12345678-1234-5678-1234-567812345678")
        table_name = "test_table"
        table = Table(datasource_id=datasource_id, name=table_name)

        assert table.datasource_id == datasource_id
        assert table.name == table_name
        # Inherited from BaseSQLModel
        assert table.id is None
        assert table.created_at is None
        assert table.modified_at is None

    def test_table_inherits_from_base_sql_model(self):
        """Test that Table inherits all properties from BaseSQLModel."""
        datasource_id = UUID("12345678-1234-5678-1234-567812345678")
        table = Table(datasource_id=datasource_id, name="test_table")

        # Should have all BaseSQLModel attributes
        assert hasattr(table, "id")
        assert hasattr(table, "created_at")
        assert hasattr(table, "modified_at")

        # Should have the count class method
        assert hasattr(Table, "count")

    def test_table_name(self):
        """Test that Table has correct table name."""
        assert Table.__tablename__ == "tables"

    def test_table_with_all_fields(self):
        """Test Table instantiation with all fields including inherited ones."""
        test_id = UUID("12345678-1234-5678-1234-567812345678")
        datasource_id = UUID("87654321-4321-8765-4321-876543210987")
        test_datetime = datetime(2023, 1, 1, 12, 0, 0)
        table_name = "complete_table"
        schema = "public"
        description = "A complete test table"
        table_type = "BASE TABLE"
        row_count = 1000

        table = Table(
            id=test_id,
            datasource_id=datasource_id,
            name=table_name,
            schema=schema,
            description=description,
            type=table_type,
            row_count=row_count,
            created_at=test_datetime,
            modified_at=test_datetime,
        )

        assert table.id == test_id
        assert table.datasource_id == datasource_id
        assert table.name == table_name
        assert table.schema == schema
        assert table.description == description
        assert table.type == table_type
        assert table.row_count == row_count
        assert table.created_at == test_datetime
        assert table.modified_at == test_datetime

    def test_table_field_descriptions(self):
        """Test that fields have proper descriptions."""
        fields = Table.model_fields
        
        assert fields["datasource_id"].description == "Datasource ID"
        assert fields["name"].description == "Table name"
        assert fields["schema"].description == "Schema name"
        assert fields["description"].description == "Table description/comment"
        assert fields["type"].description == "Table type"
        assert fields["row_count"].description == "Row count"

    def test_table_field_types(self):
        """Test that fields have correct types."""
        fields = Table.model_fields
        
        assert fields["datasource_id"].annotation is UUID
        assert fields["name"].annotation is str
        assert fields["schema"].annotation is Optional[str]
        # Handle both Optional and Union syntax
        assert (fields["description"].annotation == Optional[str] or 
                fields["description"].annotation == Union[str, None] or
                str(fields["description"].annotation) == "str | None")
        assert fields["type"].annotation is Optional[str]
        # Handle both Optional and Union syntax
        assert (fields["row_count"].annotation == Optional[int] or 
                fields["row_count"].annotation == Union[int, None] or
                str(fields["row_count"].annotation) == "int | None")

    def test_table_is_table_model(self):
        """Test that Table is configured as a table model."""
        assert hasattr(Table, "__table__")
        assert Table.__table__.name == "tables"

    def test_table_relationships(self):
        """Test that Table has proper relationships."""
        datasource_id = UUID("12345678-1234-5678-1234-567812345678")
        table = Table(datasource_id=datasource_id, name="test_table")

        # Should have relationship attributes
        assert hasattr(table, "columns")
        assert hasattr(table, "primary_key_constraints")
        assert hasattr(table, "foreign_key_constraints")

    def test_table_string_representation(self):
        """Test string representation of Table model."""
        datasource_id = UUID("12345678-1234-5678-1234-567812345678")
        table = Table(datasource_id=datasource_id, name="test_table")

        str_repr = str(table)
        assert "test_table" in str_repr


class TestColumn:
    """Test cases for Column model class."""

    def test_column_initialization_with_required_fields(self):
        """Test that Column can be instantiated with required fields."""
        table_id = UUID("12345678-1234-5678-1234-567812345678")
        column_name = "test_column"
        data_type = "VARCHAR(255)"
        column = Column(table_id=table_id, name=column_name, data_type=data_type)

        assert column.table_id == table_id
        assert column.name == column_name
        assert column.data_type == data_type
        # Inherited from BaseSQLModel
        assert column.id is None
        assert column.created_at is None
        assert column.modified_at is None

    def test_column_inherits_from_base_sql_model(self):
        """Test that Column inherits all properties from BaseSQLModel."""
        table_id = UUID("12345678-1234-5678-1234-567812345678")
        column = Column(table_id=table_id, name="test_column", data_type="VARCHAR")

        assert hasattr(column, "id")
        assert hasattr(column, "created_at")
        assert hasattr(column, "modified_at")
        assert hasattr(Column, "count")

    def test_column_name(self):
        """Test that Column has correct table name."""
        assert Column.__tablename__ == "columns"

    def test_column_with_all_fields(self):
        """Test Column instantiation with all fields including inherited ones."""
        test_id = UUID("12345678-1234-5678-1234-567812345678")
        table_id = UUID("87654321-4321-8765-4321-876543210987")
        test_datetime = datetime(2023, 1, 1, 12, 0, 0)
        column_name = "complete_column"
        data_type = "INTEGER"
        description = "A complete test column"
        default_value = "0"
        is_nullable = False

        column = Column(
            id=test_id,
            table_id=table_id,
            name=column_name,
            data_type=data_type,
            description=description,
            default_value=default_value,
            is_nullable=is_nullable,
            created_at=test_datetime,
            modified_at=test_datetime,
        )

        assert column.id == test_id
        assert column.table_id == table_id
        assert column.name == column_name
        assert column.data_type == data_type
        assert column.description == description
        assert column.default_value == default_value
        assert column.is_nullable == is_nullable
        assert column.created_at == test_datetime
        assert column.modified_at == test_datetime

    def test_column_field_descriptions(self):
        """Test that fields have proper descriptions."""
        fields = Column.model_fields
        
        assert fields["table_id"].description == "Table ID"
        assert fields["name"].description == "Column name"
        assert fields["data_type"].description == "Column data type"
        assert fields["description"].description == "Column description/comment"
        assert fields["default_value"].description == "Column default value"
        assert fields["is_nullable"].description == "Whether the column is nullable"

    def test_column_field_types(self):
        """Test that fields have correct types."""
        fields = Column.model_fields
        
        assert fields["table_id"].annotation is UUID
        assert fields["name"].annotation is str
        assert fields["data_type"].annotation is str
        # Handle both Optional and Union syntax
        assert (fields["description"].annotation == Optional[str] or 
                fields["description"].annotation == Union[str, None] or
                str(fields["description"].annotation) == "str | None")
        assert (fields["default_value"].annotation == Optional[str] or 
                fields["default_value"].annotation == Union[str, None] or
                str(fields["default_value"].annotation) == "str | None")
        assert fields["is_nullable"].annotation is bool

    def test_column_is_table_model(self):
        """Test that Column is configured as a table model."""
        assert hasattr(Column, "__table__")
        assert Column.__table__.name == "columns"

    def test_column_relationships(self):
        """Test that Column has proper relationships."""
        table_id = UUID("12345678-1234-5678-1234-567812345678")
        column = Column(table_id=table_id, name="test_column", data_type="VARCHAR")

        assert hasattr(column, "statistics")

    def test_column_default_values(self):
        """Test that Column has proper default values."""
        table_id = UUID("12345678-1234-5678-1234-567812345678")
        column = Column(table_id=table_id, name="test_column", data_type="VARCHAR")

        # is_nullable should default to True
        assert column.is_nullable is True

    def test_column_string_representation(self):
        """Test string representation of Column model."""
        table_id = UUID("12345678-1234-5678-1234-567812345678")
        column = Column(table_id=table_id, name="test_column", data_type="VARCHAR")

        str_repr = str(column)
        assert "test_column" in str_repr


class TestColumnStatistics:
    """Test cases for ColumnStatistics model class."""

    def test_column_statistics_initialization_with_required_fields(self):
        """Test that ColumnStatistics can be instantiated with required fields."""
        column_id = UUID("12345678-1234-5678-1234-567812345678")
        stats = ColumnStatistics(column_id=column_id)

        assert stats.column_id == column_id
        # Inherited from BaseSQLModel
        assert stats.id is None
        assert stats.created_at is None
        assert stats.modified_at is None

    def test_column_statistics_inherits_from_base_sql_model(self):
        """Test that ColumnStatistics inherits all properties from BaseSQLModel."""
        column_id = UUID("12345678-1234-5678-1234-567812345678")
        stats = ColumnStatistics(column_id=column_id)

        assert hasattr(stats, "id")
        assert hasattr(stats, "created_at")
        assert hasattr(stats, "modified_at")
        assert hasattr(ColumnStatistics, "count")

    def test_column_statistics_name(self):
        """Test that ColumnStatistics has correct table name."""
        assert ColumnStatistics.__tablename__ == "column_statistics"

    def test_column_statistics_with_all_fields(self):
        """Test ColumnStatistics instantiation with all fields including inherited ones."""
        test_id = UUID("12345678-1234-5678-1234-567812345678")
        column_id = UUID("87654321-4321-8765-4321-876543210987")
        test_datetime = datetime(2023, 1, 1, 12, 0, 0)
        most_common_values = ["value1", "value2", "value3"]
        most_common_frequencies = [0.3, 0.2, 0.1]
        null_percentage = 5.5
        distinct_values_count = 100
        min_value = "0"
        max_value = "999"

        stats = ColumnStatistics(
            id=test_id,
            column_id=column_id,
            most_common_values=most_common_values,
            most_common_frequencies=most_common_frequencies,
            null_percentage=null_percentage,
            distinct_values_count=distinct_values_count,
            min_value=min_value,
            max_value=max_value,
            created_at=test_datetime,
            modified_at=test_datetime,
        )

        assert stats.id == test_id
        assert stats.column_id == column_id
        assert stats.most_common_values == most_common_values
        assert stats.most_common_frequencies == most_common_frequencies
        assert stats.null_percentage == null_percentage
        assert stats.distinct_values_count == distinct_values_count
        assert stats.min_value == min_value
        assert stats.max_value == max_value
        assert stats.created_at == test_datetime
        assert stats.modified_at == test_datetime

    def test_column_statistics_field_descriptions(self):
        """Test that fields have proper descriptions."""
        fields = ColumnStatistics.model_fields
        
        assert fields["column_id"].description == "Column ID"
        assert fields["most_common_values"].description == "List of most common values"
        assert fields["most_common_frequencies"].description == "List of most common frequencies"
        assert fields["null_percentage"].description == "Null percentage"
        assert fields["distinct_values_count"].description == "Distinct values count"
        assert fields["min_value"].description == "Minimum value"
        assert fields["max_value"].description == "Maximum value"

    def test_column_statistics_field_types(self):
        """Test that fields have correct types."""
        fields = ColumnStatistics.model_fields
        
        assert fields["column_id"].annotation is UUID
        # Handle both list[Any] | None and Optional[list[Any]] syntax
        most_common_values_type = fields["most_common_values"].annotation
        assert (str(most_common_values_type) == "list[Any] | None" or
                most_common_values_type == Optional[list[Any]] or
                most_common_values_type == Union[list[Any], None])
        
        most_common_frequencies_type = fields["most_common_frequencies"].annotation
        assert (str(most_common_frequencies_type) == "list[float] | None" or
                most_common_frequencies_type == Optional[list[float]] or
                most_common_frequencies_type == Union[list[float], None])
        
        # Handle both Optional and Union syntax
        assert (fields["null_percentage"].annotation == Optional[float] or 
                fields["null_percentage"].annotation == Union[float, None] or
                str(fields["null_percentage"].annotation) == "float | None")
        assert (fields["distinct_values_count"].annotation == Optional[int] or 
                fields["distinct_values_count"].annotation == Union[int, None] or
                str(fields["distinct_values_count"].annotation) == "int | None")
        assert (fields["min_value"].annotation == Optional[str] or 
                fields["min_value"].annotation == Union[str, None] or
                str(fields["min_value"].annotation) == "str | None")
        assert (fields["max_value"].annotation == Optional[str] or 
                fields["max_value"].annotation == Union[str, None] or
                str(fields["max_value"].annotation) == "str | None")

    def test_column_statistics_is_table_model(self):
        """Test that ColumnStatistics is configured as a table model."""
        assert hasattr(ColumnStatistics, "__table__")
        assert ColumnStatistics.__table__.name == "column_statistics"

    def test_column_statistics_default_values(self):
        """Test that ColumnStatistics has proper default values."""
        column_id = UUID("12345678-1234-5678-1234-567812345678")
        stats = ColumnStatistics(column_id=column_id)

        # JSON fields should default to empty lists
        assert stats.most_common_values == []
        assert stats.most_common_frequencies == []

    def test_column_statistics_string_representation(self):
        """Test string representation of ColumnStatistics model."""
        column_id = UUID("12345678-1234-5678-1234-567812345678")
        stats = ColumnStatistics(column_id=column_id)

        str_repr = str(stats)
        assert str_repr is not None


class TestPrimaryKeyConstraint:
    """Test cases for PrimaryKeyConstraint model class."""

    def test_primary_key_constraint_initialization_with_required_fields(self):
        """Test that PrimaryKeyConstraint can be instantiated with required fields."""
        table_id = UUID("12345678-1234-5678-1234-567812345678")
        column_id = UUID("87654321-4321-8765-4321-876543210987")
        constraint = PrimaryKeyConstraint(table_id=table_id, column_id=column_id)

        assert constraint.table_id == table_id
        assert constraint.column_id == column_id
        # Inherited from BaseSQLModel
        assert constraint.id is None
        assert constraint.created_at is None
        assert constraint.modified_at is None

    def test_primary_key_constraint_inherits_from_base_sql_model(self):
        """Test that PrimaryKeyConstraint inherits all properties from BaseSQLModel."""
        table_id = UUID("12345678-1234-5678-1234-567812345678")
        column_id = UUID("87654321-4321-8765-4321-876543210987")
        constraint = PrimaryKeyConstraint(table_id=table_id, column_id=column_id)

        assert hasattr(constraint, "id")
        assert hasattr(constraint, "created_at")
        assert hasattr(constraint, "modified_at")
        assert hasattr(PrimaryKeyConstraint, "count")

    def test_primary_key_constraint_name(self):
        """Test that PrimaryKeyConstraint has correct table name."""
        assert PrimaryKeyConstraint.__tablename__ == "primary_key_constraints"

    def test_primary_key_constraint_with_all_fields(self):
        """Test PrimaryKeyConstraint instantiation with all fields including inherited ones."""
        test_id = UUID("12345678-1234-5678-1234-567812345678")
        table_id = UUID("87654321-4321-8765-4321-876543210987")
        column_id = UUID("11111111-2222-3333-4444-555555555555")
        test_datetime = datetime(2023, 1, 1, 12, 0, 0)
        ordinal_position = 1
        constraint_name = "pk_test_table"

        constraint = PrimaryKeyConstraint(
            id=test_id,
            table_id=table_id,
            column_id=column_id,
            ordinal_position=ordinal_position,
            constraint_name=constraint_name,
            created_at=test_datetime,
            modified_at=test_datetime,
        )

        assert constraint.id == test_id
        assert constraint.table_id == table_id
        assert constraint.column_id == column_id
        assert constraint.ordinal_position == ordinal_position
        assert constraint.constraint_name == constraint_name
        assert constraint.created_at == test_datetime
        assert constraint.modified_at == test_datetime

    def test_primary_key_constraint_field_descriptions(self):
        """Test that fields have proper descriptions."""
        fields = PrimaryKeyConstraint.model_fields
        
        assert fields["table_id"].description == "Table ID"
        assert fields["column_id"].description == "Column ID"
        assert fields["ordinal_position"].description == "Ordinal position"
        assert fields["constraint_name"].description == "Constraint name"

    def test_primary_key_constraint_field_types(self):
        """Test that fields have correct types."""
        fields = PrimaryKeyConstraint.model_fields
        
        assert fields["table_id"].annotation is UUID
        assert fields["column_id"].annotation is UUID
        # Handle both Optional and Union syntax
        assert (fields["ordinal_position"].annotation == Optional[int] or 
                fields["ordinal_position"].annotation == Union[int, None] or
                str(fields["ordinal_position"].annotation) == "int | None")
        assert (fields["constraint_name"].annotation == Optional[str] or 
                fields["constraint_name"].annotation == Union[str, None] or
                str(fields["constraint_name"].annotation) == "str | None")

    def test_primary_key_constraint_is_table_model(self):
        """Test that PrimaryKeyConstraint is configured as a table model."""
        assert hasattr(PrimaryKeyConstraint, "__table__")
        assert PrimaryKeyConstraint.__table__.name == "primary_key_constraints"

    def test_primary_key_constraint_relationships(self):
        """Test that PrimaryKeyConstraint has proper relationships."""
        table_id = UUID("12345678-1234-5678-1234-567812345678")
        column_id = UUID("87654321-4321-8765-4321-876543210987")
        constraint = PrimaryKeyConstraint(table_id=table_id, column_id=column_id)

        assert hasattr(constraint, "column")

    def test_primary_key_constraint_string_representation(self):
        """Test string representation of PrimaryKeyConstraint model."""
        table_id = UUID("12345678-1234-5678-1234-567812345678")
        column_id = UUID("87654321-4321-8765-4321-876543210987")
        constraint = PrimaryKeyConstraint(table_id=table_id, column_id=column_id)

        str_repr = str(constraint)
        assert str_repr is not None


class TestForeignKeyConstraint:
    """Test cases for ForeignKeyConstraint model class."""

    def test_foreign_key_constraint_initialization_with_required_fields(self):
        """Test that ForeignKeyConstraint can be instantiated with required fields."""
        table_id = UUID("12345678-1234-5678-1234-567812345678")
        column_id = UUID("87654321-4321-8765-4321-876543210987")
        referenced_table_id = UUID("11111111-2222-3333-4444-555555555555")
        referenced_column_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        constraint = ForeignKeyConstraint(
            table_id=table_id,
            column_id=column_id,
            referenced_table_id=referenced_table_id,
            referenced_column_id=referenced_column_id
        )

        assert constraint.table_id == table_id
        assert constraint.column_id == column_id
        assert constraint.referenced_table_id == referenced_table_id
        assert constraint.referenced_column_id == referenced_column_id
        # Inherited from BaseSQLModel
        assert constraint.id is None
        assert constraint.created_at is None
        assert constraint.modified_at is None

    def test_foreign_key_constraint_inherits_from_base_sql_model(self):
        """Test that ForeignKeyConstraint inherits all properties from BaseSQLModel."""
        table_id = UUID("12345678-1234-5678-1234-567812345678")
        column_id = UUID("87654321-4321-8765-4321-876543210987")
        referenced_table_id = UUID("11111111-2222-3333-4444-555555555555")
        referenced_column_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        constraint = ForeignKeyConstraint(
            table_id=table_id,
            column_id=column_id,
            referenced_table_id=referenced_table_id,
            referenced_column_id=referenced_column_id
        )

        assert hasattr(constraint, "id")
        assert hasattr(constraint, "created_at")
        assert hasattr(constraint, "modified_at")
        assert hasattr(ForeignKeyConstraint, "count")

    def test_foreign_key_constraint_name(self):
        """Test that ForeignKeyConstraint has correct table name."""
        assert ForeignKeyConstraint.__tablename__ == "foreign_key_constraints"

    def test_foreign_key_constraint_with_all_fields(self):
        """Test ForeignKeyConstraint instantiation with all fields including inherited ones."""
        test_id = UUID("12345678-1234-5678-1234-567812345678")
        table_id = UUID("87654321-4321-8765-4321-876543210987")
        column_id = UUID("11111111-2222-3333-4444-555555555555")
        referenced_table_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        referenced_column_id = UUID("bbbbbbbb-cccc-dddd-eeee-ffffffffffff")
        test_datetime = datetime(2023, 1, 1, 12, 0, 0)
        constraint_name = "fk_test_table"
        ordinal_position = 1

        constraint = ForeignKeyConstraint(
            id=test_id,
            table_id=table_id,
            column_id=column_id,
            referenced_table_id=referenced_table_id,
            referenced_column_id=referenced_column_id,
            constraint_name=constraint_name,
            ordinal_position=ordinal_position,
            created_at=test_datetime,
            modified_at=test_datetime,
        )

        assert constraint.id == test_id
        assert constraint.table_id == table_id
        assert constraint.column_id == column_id
        assert constraint.referenced_table_id == referenced_table_id
        assert constraint.referenced_column_id == referenced_column_id
        assert constraint.constraint_name == constraint_name
        assert constraint.ordinal_position == ordinal_position
        assert constraint.created_at == test_datetime
        assert constraint.modified_at == test_datetime

    def test_foreign_key_constraint_field_descriptions(self):
        """Test that fields have proper descriptions."""
        fields = ForeignKeyConstraint.model_fields
        
        assert fields["table_id"].description == "Table ID"
        assert fields["column_id"].description == "Column ID"
        assert fields["referenced_table_id"].description == "Referenced table ID"
        assert fields["referenced_column_id"].description == "Referenced column ID"
        assert fields["constraint_name"].description == "Constraint name"
        assert fields["ordinal_position"].description == "Ordinal position"

    def test_foreign_key_constraint_field_types(self):
        """Test that fields have correct types."""
        fields = ForeignKeyConstraint.model_fields
        
        assert fields["table_id"].annotation is UUID
        assert fields["column_id"].annotation is UUID
        assert fields["referenced_table_id"].annotation is UUID
        assert fields["referenced_column_id"].annotation is UUID
        # Handle both Optional and Union syntax
        assert (fields["constraint_name"].annotation == Optional[str] or 
                fields["constraint_name"].annotation == Union[str, None] or
                str(fields["constraint_name"].annotation) == "str | None")
        assert (fields["ordinal_position"].annotation == Optional[int] or 
                fields["ordinal_position"].annotation == Union[int, None] or
                str(fields["ordinal_position"].annotation) == "int | None")

    def test_foreign_key_constraint_is_table_model(self):
        """Test that ForeignKeyConstraint is configured as a table model."""
        assert hasattr(ForeignKeyConstraint, "__table__")
        assert ForeignKeyConstraint.__table__.name == "foreign_key_constraints"

    def test_foreign_key_constraint_relationships(self):
        """Test that ForeignKeyConstraint has proper relationships."""
        table_id = UUID("12345678-1234-5678-1234-567812345678")
        column_id = UUID("87654321-4321-8765-4321-876543210987")
        referenced_table_id = UUID("11111111-2222-3333-4444-555555555555")
        referenced_column_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        constraint = ForeignKeyConstraint(
            table_id=table_id,
            column_id=column_id,
            referenced_table_id=referenced_table_id,
            referenced_column_id=referenced_column_id
        )

        assert hasattr(constraint, "column")
        assert hasattr(constraint, "referenced_table")
        assert hasattr(constraint, "referenced_column")

    def test_foreign_key_constraint_string_representation(self):
        """Test string representation of ForeignKeyConstraint model."""
        table_id = UUID("12345678-1234-5678-1234-567812345678")
        column_id = UUID("87654321-4321-8765-4321-876543210987")
        referenced_table_id = UUID("11111111-2222-3333-4444-555555555555")
        referenced_column_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        constraint = ForeignKeyConstraint(
            table_id=table_id,
            column_id=column_id,
            referenced_table_id=referenced_table_id,
            referenced_column_id=referenced_column_id
        )

        str_repr = str(constraint)
        assert str_repr is not None


class TestDataCatalog:
    """Test cases for DataCatalog model class."""

    @pytest.fixture
    def mock_datasource(self):
        """Mock datasource for testing."""
        datasource = Mock(spec=Datasource)
        datasource.name = "test_datasource"
        datasource.engine = "postgresql"
        datasource.tables = []
        return datasource

    @pytest.fixture
    def sample_table(self):
        """Sample table for testing."""
        table = Mock(spec=Table)
        table.name = "users"
        table.description = "User information table"
        table.columns = []
        table.primary_key_constraints = []
        table.foreign_key_constraints = []
        return table

    @pytest.fixture
    def sample_column(self):
        """Sample column for testing."""
        column = Mock(spec=Column)
        column.name = "id"
        column.data_type = "INTEGER"
        column.is_nullable = False
        column.default_value = None
        column.description = "Primary key"
        column.statistics = None
        return column

    @pytest.fixture
    def sample_column_with_stats(self):
        """Sample column with statistics for testing."""
        column = Mock(spec=Column)
        column.name = "age"
        column.data_type = "INTEGER"
        column.is_nullable = True
        column.default_value = None
        column.description = "User age"
        
        # Mock statistics
        stats = Mock(spec=ColumnStatistics)
        stats.distinct_values_count = 50
        stats.null_percentage = 5.0
        stats.min_value = "18"
        stats.max_value = "65"
        stats.most_common_values = ["25", "30", "35"]
        stats.most_common_frequencies = [0.15, 0.12, 0.10]
        column.statistics = stats
        
        return column

    def test_data_catalog_initialization(self, mock_datasource):
        """Test DataCatalog initialization."""
        catalog = DataCatalog(datasource=mock_datasource)
        
        assert catalog.datasource == mock_datasource
        assert catalog.created_at is not None
        assert catalog.modified_at is not None

    def test_data_catalog_from_datasource_class_method(self, mock_datasource):
        """Test DataCatalog.from_datasource class method."""
        catalog = DataCatalog.from_datasource(mock_datasource)
        
        assert catalog.datasource == mock_datasource
        assert catalog.created_at is not None
        assert catalog.modified_at is not None

    def test_data_catalog_is_not_table_model(self):
        """Test that DataCatalog is not configured as a table model."""
        assert not hasattr(DataCatalog, "__table__")

    def test_format_header(self, mock_datasource):
        """Test _format_header method."""
        mock_datasource.tables = [Mock(), Mock()]  # 2 tables
        catalog = DataCatalog(datasource=mock_datasource)
        
        header_lines = catalog._format_header()
        
        assert len(header_lines) == 5
        assert "MindsDB Data Source: test_datasource" in header_lines[0]
        assert "Engine: postgresql" in header_lines[1]
        assert header_lines[2] == ""  # Empty line
        assert "Tables: 2" in header_lines[3]
        assert header_lines[4] == ""  # Empty line

    def test_format_table_basic(self, mock_datasource, sample_table):
        """Test _format_table method with basic table."""
        sample_table.columns = []
        sample_table.primary_key_constraints = []
        sample_table.foreign_key_constraints = []
        catalog = DataCatalog(datasource=mock_datasource)
        
        table_lines = catalog._format_table(sample_table)
        
        assert len(table_lines) >= 3
        assert "Table: test_datasource.users - User information table" in table_lines[0]
        assert "  Columns:" in table_lines[1]
        assert table_lines[-1] == ""  # Blank line at end

    def test_format_table_without_description(self, mock_datasource):
        """Test _format_table method with table without description."""
        table = Mock(spec=Table)
        table.name = "orders"
        table.description = None
        table.columns = []
        table.primary_key_constraints = []
        table.foreign_key_constraints = []
        catalog = DataCatalog(datasource=mock_datasource)
        
        table_lines = catalog._format_table(table)
        
        assert "Table: test_datasource.orders" in table_lines[0]
        assert " - " not in table_lines[0]  # No description separator

    def test_format_column_basic(self, sample_column, mock_datasource):
        """Test _format_column method with basic column."""
        catalog = DataCatalog(datasource=mock_datasource)
        
        column_lines = catalog._format_column(sample_column)
        
        assert len(column_lines) >= 1
        assert "    - id (INTEGER) NOT NULL - Primary key" in column_lines[0]

    def test_format_column_with_default_value(self, mock_datasource):
        """Test _format_column method with default value."""
        column = Mock(spec=Column)
        column.name = "status"
        column.data_type = "VARCHAR"
        column.is_nullable = True
        column.default_value = "active"
        column.description = "User status"
        column.statistics = None
        
        catalog = DataCatalog(datasource=mock_datasource)
        column_lines = catalog._format_column(column)
        
        assert "    - status (VARCHAR) DEFAULT active - User status" in column_lines[0]

    def test_format_column_with_null_default(self, mock_datasource):
        """Test _format_column method with [NULL] default value."""
        column = Mock(spec=Column)
        column.name = "notes"
        column.data_type = "TEXT"
        column.is_nullable = True
        column.default_value = "[NULL]"
        column.description = "User notes"
        column.statistics = None
        
        catalog = DataCatalog(datasource=mock_datasource)
        column_lines = catalog._format_column(column)
        
        # Should not include DEFAULT [NULL] in output
        assert "    - notes (TEXT) - User notes" in column_lines[0]
        assert "DEFAULT" not in column_lines[0]

    def test_format_column_without_description(self, mock_datasource):
        """Test _format_column method without description."""
        column = Mock(spec=Column)
        column.name = "email"
        column.data_type = "VARCHAR"
        column.is_nullable = True
        column.default_value = None
        column.description = None
        column.statistics = None
        
        catalog = DataCatalog(datasource=mock_datasource)
        column_lines = catalog._format_column(column)
        
        assert "    - email (VARCHAR)" in column_lines[0]
        # Should not have description after the data type
        assert "VARCHAR) - " not in column_lines[0]  # No description separator

    def test_format_column_statistics(self, sample_column_with_stats, mock_datasource):
        """Test _format_column_statistics method."""
        catalog = DataCatalog(datasource=mock_datasource)
        
        stats_lines = catalog._format_column_statistics(sample_column_with_stats)
        
        assert len(stats_lines) >= 3
        assert "      Distinct Values: 50" in stats_lines
        assert "      Null %: 5.0%" in stats_lines
        assert "      Range: 18 to 65" in stats_lines
        assert "      Most Common: 25 (15.0%), 30 (12.0%), 35 (10.0%)" in stats_lines

    def test_format_column_statistics_no_stats(self, sample_column, mock_datasource):
        """Test _format_column_statistics method with no statistics."""
        catalog = DataCatalog(datasource=mock_datasource)
        
        stats_lines = catalog._format_column_statistics(sample_column)
        
        assert len(stats_lines) == 0

    def test_format_column_statistics_empty_common_values(self, mock_datasource):
        """Test _format_column_statistics method with empty common values."""
        column = Mock(spec=Column)
        column.statistics = Mock(spec=ColumnStatistics)
        column.statistics.distinct_values_count = 10
        column.statistics.null_percentage = 0.0
        column.statistics.min_value = "1"
        column.statistics.max_value = "10"
        column.statistics.most_common_values = [""]  # Empty string
        column.statistics.most_common_frequencies = [""]  # Empty string
        
        catalog = DataCatalog(datasource=mock_datasource)
        stats_lines = catalog._format_column_statistics(column)
        
        # Should not include Most Common line for empty values
        assert len(stats_lines) == 3
        assert "Most Common:" not in "".join(stats_lines)

    def test_format_table_constraints_with_primary_key(self, mock_datasource):
        """Test _format_table_constraints method with primary key."""
        table = Mock(spec=Table)
        table.primary_key_constraints = []
        table.foreign_key_constraints = []
        
        # Mock primary key constraint
        pk_constraint = Mock()
        pk_constraint.column = Mock()
        pk_constraint.column.name = "id"
        table.primary_key_constraints = [pk_constraint]
        
        catalog = DataCatalog(datasource=mock_datasource)
        constraint_lines = catalog._format_table_constraints(table)
        
        assert len(constraint_lines) == 1
        assert "  Primary Key: id" in constraint_lines[0]

    def test_format_table_constraints_with_foreign_keys(self, mock_datasource):
        """Test _format_table_constraints method with foreign keys."""
        table = Mock(spec=Table)
        table.primary_key_constraints = []
        
        # Mock foreign key constraint
        fk_constraint = Mock()
        fk_constraint.column = Mock()
        fk_constraint.column.name = "user_id"
        fk_constraint.referenced_table = Mock()
        fk_constraint.referenced_table.name = "users"
        fk_constraint.referenced_column = Mock()
        fk_constraint.referenced_column.name = "id"
        table.foreign_key_constraints = [fk_constraint]
        
        catalog = DataCatalog(datasource=mock_datasource)
        constraint_lines = catalog._format_table_constraints(table)
        
        assert len(constraint_lines) == 2
        assert "  Foreign Keys:" in constraint_lines[0]
        assert "    - user_id → test_datasource.users(id)" in constraint_lines[1]

    def test_format_table_constraints_empty(self, mock_datasource):
        """Test _format_table_constraints method with no constraints."""
        table = Mock(spec=Table)
        table.primary_key_constraints = []
        table.foreign_key_constraints = []
        
        catalog = DataCatalog(datasource=mock_datasource)
        constraint_lines = catalog._format_table_constraints(table)
        
        assert len(constraint_lines) == 0

    def test_get_related_tables(self, mock_datasource):
        """Test _get_related_tables method."""
        # Create tables with foreign key relationships
        table1 = Mock(spec=Table)
        table1.name = "orders"
        table1.foreign_key_constraints = []
        
        table2 = Mock(spec=Table)
        table2.name = "users"
        table2.foreign_key_constraints = []
        
        # Mock foreign key from orders to users
        fk_constraint = Mock()
        fk_constraint.referenced_table = "users"  # This should be a string (table name)
        table1.foreign_key_constraints = [fk_constraint]
        
        # Mock foreign key from users to orders (reverse relationship)
        fk_constraint2 = Mock()
        fk_constraint2.referenced_table = "orders"  # This should be a string (table name)
        table2.foreign_key_constraints = [fk_constraint2]
        
        mock_datasource.tables = [table1, table2]
        catalog = DataCatalog(datasource=mock_datasource)
        
        related_tables = catalog._get_related_tables(table1)
        
        assert "users" in related_tables  # From table1's foreign key
        assert "orders" not in related_tables  # Self-reference should be excluded
        assert len(related_tables) == 1  # Only "users" should be returned

    def test_format_relationships(self, mock_datasource):
        """Test _format_relationships method."""
        # Create tables with relationships
        table1 = Mock(spec=Table)
        table1.name = "orders"
        table1.foreign_key_constraints = []
        
        table2 = Mock(spec=Table)
        table2.name = "users"
        table2.foreign_key_constraints = []
        
        # Mock foreign key relationship
        fk_constraint = Mock()
        fk_constraint.referenced_table = "users"
        table1.foreign_key_constraints = [fk_constraint]
        
        mock_datasource.tables = [table1, table2]
        catalog = DataCatalog(datasource=mock_datasource)
        
        relationship_lines = catalog._format_relationships()
        
        assert len(relationship_lines) == 4  # Header + 2 relationships + empty line
        assert "Relationships:" in relationship_lines[0]
        assert "test_datasource.orders is related to: test_datasource.users" in relationship_lines[1]
        assert "test_datasource.users is related to: test_datasource.orders" in relationship_lines[2]
        assert relationship_lines[3] == ""

    def test_format_relationships_no_relationships(self, mock_datasource):
        """Test _format_relationships method with no relationships."""
        table = Mock(spec=Table)
        table.name = "standalone"
        table.foreign_key_constraints = []
        mock_datasource.tables = [table]
        
        catalog = DataCatalog(datasource=mock_datasource)
        relationship_lines = catalog._format_relationships()
        
        assert len(relationship_lines) == 0

    def test_to_context_str_complete(self, mock_datasource, sample_table, sample_column_with_stats):
        """Test to_context_str method with complete data."""
        # Setup table with column and statistics
        sample_table.columns = [sample_column_with_stats]
        sample_table.primary_key_constraints = []
        sample_table.foreign_key_constraints = []
        mock_datasource.tables = [sample_table]
        
        catalog = DataCatalog(datasource=mock_datasource)
        context_str = catalog.to_context_str()
        
        lines = context_str.split('\n')
        
        # Check header
        assert "MindsDB Data Source: test_datasource" in lines[0]
        assert "Engine: postgresql" in lines[1]
        assert "Tables: 1" in lines[3]
        
        # Check table information
        assert "Table: test_datasource.users - User information table" in lines[5]
        assert "  Columns:" in lines[6]
        assert "    - age (INTEGER) - User age" in lines[7]
        assert "      Distinct Values: 50" in lines[8]
        assert "      Null %: 5.0%" in lines[9]
        assert "      Range: 18 to 65" in lines[10]
        assert "      Most Common: 25 (15.0%), 30 (12.0%), 35 (10.0%)" in lines[11]

    def test_to_context_str_empty_datasource(self, mock_datasource):
        """Test to_context_str method with empty datasource."""
        mock_datasource.tables = []
        catalog = DataCatalog(datasource=mock_datasource)
        context_str = catalog.to_context_str()
        
        lines = context_str.split('\n')
        
        # Should only have header
        assert "MindsDB Data Source: test_datasource" in lines[0]
        assert "Engine: postgresql" in lines[1]
        assert "Tables: 0" in lines[3]
        assert len(lines) == 5  # Header + empty lines only

    def test_data_catalog_field_descriptions(self):
        """Test that DataCatalog fields have proper descriptions."""
        fields = DataCatalog.model_fields
        
        assert fields["created_at"].description == "The date and time the catalog was created."
        assert fields["modified_at"].description == "The date and time the catalog was updated."
        assert fields["datasource"].description == "Datasource"

    def test_data_catalog_field_types(self):
        """Test that DataCatalog fields have correct types."""
        fields = DataCatalog.model_fields
        
        assert fields["created_at"].annotation is datetime
        assert fields["modified_at"].annotation is datetime
        # Datasource field type is complex due to forward reference
        assert "Datasource" in str(fields["datasource"].annotation)
