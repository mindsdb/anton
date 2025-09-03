from datetime import datetime
from typing import Any, Optional, Union
from uuid import UUID


from minds.model.data_catalog import (
    Table,
    Column,
    ColumnStatistics,
    PrimaryKeyConstraint,
    ForeignKeyConstraint,
)


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

