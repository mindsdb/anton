from unittest.mock import Mock, patch
from uuid import UUID

import pandas as pd
import pytest
from sqlmodel import Session

from minds.jobs.data_catalog_loader_flow import (
    _convert_row_to_column,
    _convert_row_to_column_statistics,
    _convert_row_to_foreign_key,
    _convert_row_to_primary_key,
    _convert_row_to_table,
    _execute_mindsdb_query,
    _normalize_boolean,
    _normalize_null_value,
    filter_loaded_tables,
    get_column_statistics,
    get_columns,
    get_foreign_keys,
    get_primary_keys,
    get_tables,
    load_column_statistics,
    load_columns,
    load_foreign_keys,
    load_primary_keys,
    load_tables,
    normalize_distinct_count,
)
from minds.model.data_catalog import Column, ColumnStatistics, ForeignKeyConstraint, PrimaryKeyConstraint, Table
from minds.model.datasource import Datasource
from minds.model.mind_datasource import DataCatalogStatus, MindDatasource


class TestDataCatalogLoaderFlow:
    """Test cases for data catalog loader flow functions."""

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = Mock(spec=Session)
        session.add_all = Mock()
        session.flush = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        session.exec = Mock()
        session.add = Mock()
        session.refresh = Mock()
        return session

    @pytest.fixture
    def mock_mindsdb_client(self):
        """Mock MindsDB client."""
        client = Mock()
        client.query = Mock()
        return client

    @pytest.fixture
    def mock_mind_datasource(self):
        """Mock mind datasource."""
        mind_datasource = Mock(spec=MindDatasource)
        mind_datasource.id = UUID("12345678-1234-5678-1234-567812345678")
        mind_datasource.tenant_id = "test_tenant_456"
        mind_datasource.datasource_id = UUID("87654321-4321-8765-4321-876543218765")
        mind_datasource.status = DataCatalogStatus.PENDING

        # Mock datasource
        datasource = Mock(spec=Datasource)
        datasource.id = UUID("87654321-4321-8765-4321-876543218765")
        datasource.name = "test_datasource"
        mind_datasource.datasource = datasource

        return mind_datasource

    def test_execute_mindsdb_query_success(self, mock_mindsdb_client):
        """Test successful MindsDB query execution."""
        mock_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        mock_query_result = Mock()
        mock_query_result.fetch.return_value = mock_df
        mock_mindsdb_client.query.return_value = mock_query_result

        result = _execute_mindsdb_query(mock_mindsdb_client, "SELECT * FROM test")

        mock_mindsdb_client.query.assert_called_once_with("SELECT * FROM test")
        mock_query_result.fetch.assert_called_once()
        pd.testing.assert_frame_equal(result, mock_df)

    def test_execute_mindsdb_query_failure(self, mock_mindsdb_client):
        """Test MindsDB query execution failure."""
        mock_mindsdb_client.query.side_effect = Exception("Query failed")

        with pytest.raises(Exception, match="Query failed"):
            _execute_mindsdb_query(mock_mindsdb_client, "SELECT * FROM test")

    def test_get_tables_without_filter(self, mock_mindsdb_client):
        """Test get_tables without table name filter."""
        mock_df = pd.DataFrame(
            {
                "TABLE_NAME": ["table1", "table2"],
                "TABLE_SCHEMA": ["test_datasource", "test_datasource"],
                "TABLE_DESCRIPTION": ["desc1", "desc2"],
                "TABLE_TYPE": ["BASE TABLE", "VIEW"],
                "ROW_COUNT": [100, 200],
            }
        )

        with patch("minds.jobs.data_catalog_loader_flow._execute_mindsdb_query", return_value=mock_df) as mock_execute:
            result = get_tables(mock_mindsdb_client, "test_datasource", None)

            expected_query = """
    SELECT * FROM INFORMATION_SCHEMA.META_TABLES 
    WHERE TABLE_SCHEMA = 'test_datasource'
    """
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args[0][1]
            assert call_args.strip() == expected_query.strip()
            pd.testing.assert_frame_equal(result, mock_df)

    def test_get_tables_with_filter(self, mock_mindsdb_client):
        """Test get_tables with table name filter."""
        mock_df = pd.DataFrame(
            {
                "TABLE_NAME": ["table1"],
                "TABLE_SCHEMA": ["test_datasource"],
                "TABLE_DESCRIPTION": ["desc1"],
                "TABLE_TYPE": ["BASE TABLE"],
                "ROW_COUNT": [100],
            }
        )

        with patch("minds.jobs.data_catalog_loader_flow._execute_mindsdb_query", return_value=mock_df) as mock_execute:
            result = get_tables(mock_mindsdb_client, "test_datasource", ["table1", "table2"])

            mock_execute.assert_called_once()
            call_args = mock_execute.call_args[0][1]
            assert "AND TABLE_NAME IN ('table1', 'table2')" in call_args
            pd.testing.assert_frame_equal(result, mock_df)

    def test_filter_loaded_tables(self, mock_session):
        """Test filter_loaded_tables function."""
        tables_df = pd.DataFrame(
            {
                "TABLE_NAME": ["table1", "table2", "table3"],
                "TABLE_SCHEMA": ["test_datasource", "test_datasource", "test_datasource"],
            }
        )

        # Mock existing tables
        existing_table = Mock(spec=Table)
        existing_table.name = "table2"
        mock_session.exec.return_value.all.return_value = [existing_table]

        result = filter_loaded_tables(
            mock_session, tables_df, UUID("87654321-4321-8765-4321-876543218765"), "test_tenant_456"
        )

        # Verify result excludes existing table
        assert len(result) == 2
        assert "table2" not in result["TABLE_NAME"].values
        assert "table1" in result["TABLE_NAME"].values
        assert "table3" in result["TABLE_NAME"].values

    def test_get_columns(self, mock_mindsdb_client):
        """Test get_columns function."""
        mock_df = pd.DataFrame(
            {
                "TABLE_NAME": ["table1", "table1"],
                "COLUMN_NAME": ["col1", "col2"],
                "DATA_TYPE": ["VARCHAR", "INTEGER"],
                "COLUMN_DESCRIPTION": ["desc1", "desc2"],
                "COLUMN_DEFAULT": ["default1", "[NULL]"],
                "IS_NULLABLE": ["YES", "NO"],
            }
        )

        with patch("minds.jobs.data_catalog_loader_flow._execute_mindsdb_query", return_value=mock_df) as mock_execute:
            result = get_columns(mock_mindsdb_client, "test_datasource", ["table1"])

            mock_execute.assert_called_once()
            call_args = mock_execute.call_args[0][1]
            assert "AND TABLE_NAME IN ('table1')" in call_args
            pd.testing.assert_frame_equal(result, mock_df)

    def test_get_column_statistics(self, mock_mindsdb_client):
        """Test get_column_statistics function."""
        mock_df = pd.DataFrame(
            {
                "TABLE_NAME": ["table1"],
                "COLUMN_NAME": ["col1"],
                "MOST_COMMON_VALS": [["val1", "val2"]],
                "MOST_COMMON_FREQS": [[0.5, 0.3]],
                "NULL_FRAC": [0.1],
                "N_DISTINCT": [10],
                "MIN_VALUE": ["min_val"],
                "MAX_VALUE": ["max_val"],
            }
        )

        with patch("minds.jobs.data_catalog_loader_flow._execute_mindsdb_query", return_value=mock_df) as mock_execute:
            result = get_column_statistics(mock_mindsdb_client, "test_datasource", None)

            mock_execute.assert_called_once()
            call_args = mock_execute.call_args[0][1]
            assert "WHERE TABLE_SCHEMA = 'test_datasource'" in call_args
            pd.testing.assert_frame_equal(result, mock_df)

    def test_get_primary_keys(self, mock_mindsdb_client):
        """Test get_primary_keys function."""
        mock_df = pd.DataFrame(
            {"TABLE_NAME": ["table1"], "COLUMN_NAME": ["id"], "ORDINAL_POSITION": [1], "CONSTRAINT_NAME": ["pk_table1"]}
        )

        with patch("minds.jobs.data_catalog_loader_flow._execute_mindsdb_query", return_value=mock_df) as mock_execute:
            result = get_primary_keys(mock_mindsdb_client, "test_datasource", None)

            mock_execute.assert_called_once()
            call_args = mock_execute.call_args[0][1]
            assert "CONSTRAINT_TYPE = 'PRIMARY KEY'" in call_args
            pd.testing.assert_frame_equal(result, mock_df)

    def test_get_foreign_keys(self, mock_mindsdb_client):
        """Test get_foreign_keys function."""
        mock_df = pd.DataFrame(
            {
                "TABLE_NAME": ["table1"],
                "COLUMN_NAME": ["foreign_id"],
                "ORDINAL_POSITION": [1],
                "CONSTRAINT_NAME": ["fk_table1"],
                "REFERENCED_TABLE_NAME": ["table2"],
                "REFERENCED_COLUMN_NAME": ["id"],
            }
        )

        with patch("minds.jobs.data_catalog_loader_flow._execute_mindsdb_query", return_value=mock_df) as mock_execute:
            result = get_foreign_keys(mock_mindsdb_client, "test_datasource", None)

            mock_execute.assert_called_once()
            call_args = mock_execute.call_args[0][1]
            assert "CONSTRAINT_TYPE = 'FOREIGN KEY'" in call_args
            pd.testing.assert_frame_equal(result, mock_df)

    def test_convert_row_to_table(self):
        """Test _convert_row_to_table function."""
        row = pd.Series(
            {
                "TABLE_NAME": "table1",
                "TABLE_SCHEMA": "test_datasource",
                "TABLE_DESCRIPTION": "Test table",
                "TABLE_TYPE": "BASE TABLE",
                "ROW_COUNT": 100,
            }
        )

        table = _convert_row_to_table(row, "test_tenant_456", UUID("87654321-4321-8765-4321-876543218765"))

        assert isinstance(table, Table)
        assert table.name == "table1"
        assert table.schema == "test_datasource"
        assert table.description == "Test table"
        assert table.type == "BASE TABLE"
        assert table.row_count == 100
        assert table.tenant_id == "test_tenant_456"
        assert table.datasource_id == UUID("87654321-4321-8765-4321-876543218765")

    def test_convert_row_to_table_with_nan_row_count(self):
        """Test _convert_row_to_table function with NaN row_count."""
        row = pd.Series(
            {
                "TABLE_NAME": "table1",
                "TABLE_SCHEMA": "test_datasource",
                "TABLE_DESCRIPTION": "Test table",
                "TABLE_TYPE": "BASE TABLE",
                "ROW_COUNT": pd.NA,
            }
        )

        table = _convert_row_to_table(row, "test_tenant_456", UUID("87654321-4321-8765-4321-876543218765"))

        assert table.row_count is None

    def test_load_tables(self, mock_session):
        """Test load_tables function."""
        tables_df = pd.DataFrame(
            {
                "TABLE_NAME": ["table1", "table2"],
                "TABLE_SCHEMA": ["test_datasource", "test_datasource"],
                "TABLE_DESCRIPTION": ["desc1", "desc2"],
                "TABLE_TYPE": ["BASE TABLE", "VIEW"],
                "ROW_COUNT": [100, 200],
            }
        )

        # Mock the session to assign IDs to tables after flush
        def mock_add(table):
            table.id = UUID(f"12345678-1234-5678-1234-56781234567{len(mock_session.add.call_args_list)}")

        mock_session.add.side_effect = mock_add

        result = load_tables(
            mock_session,
            tables_df,
            UUID("12345678-1234-5678-1234-567812345678"),
            UUID("87654321-4321-8765-4321-876543218765"),
            "test_tenant_456",
        )

        assert len(result) == 2
        assert mock_session.add.call_count == 2  # Two for tables
        assert mock_session.add_all.call_count == 1  # One for mind_datasource_tables
        assert mock_session.flush.call_count == 3  # Two for tables + one for add_all
        assert mock_session.refresh.call_count == 2  # One for each table

        # Verify table objects were created correctly
        added_tables = [call[0][0] for call in mock_session.add.call_args_list]
        assert len(added_tables) == 2
        assert all(isinstance(table, Table) for table in added_tables)
        assert added_tables[0].name == "table1"
        assert added_tables[0].datasource_id == UUID("87654321-4321-8765-4321-876543218765")
        assert added_tables[0].row_count == 100

    def test_convert_row_to_column(self):
        """Test _convert_row_to_column function."""
        row = pd.Series(
            {
                "TABLE_NAME": "table1",
                "COLUMN_NAME": "col1",
                "DATA_TYPE": "VARCHAR",
                "COLUMN_DESCRIPTION": "Test column",
                "COLUMN_DEFAULT": "default_value",
                "IS_NULLABLE": "YES",
            }
        )

        column = _convert_row_to_column(row, "test_tenant_456", UUID("11111111-1111-1111-1111-111111111111"))

        assert isinstance(column, Column)
        assert column.name == "col1"
        assert column.data_type == "VARCHAR"
        assert column.description == "Test column"
        assert column.default_value == "default_value"
        assert column.is_nullable is True
        assert column.tenant_id == "test_tenant_456"
        assert column.table_id == UUID("11111111-1111-1111-1111-111111111111")

    def test_convert_row_to_column_with_null_default(self):
        """Test _convert_row_to_column function with [NULL] default."""
        row = pd.Series(
            {
                "TABLE_NAME": "table1",
                "COLUMN_NAME": "col1",
                "DATA_TYPE": "VARCHAR",
                "COLUMN_DESCRIPTION": "Test column",
                "COLUMN_DEFAULT": "[NULL]",
                "IS_NULLABLE": "NO",
            }
        )

        column = _convert_row_to_column(row, "test_tenant_456", UUID("11111111-1111-1111-1111-111111111111"))

        assert column.default_value is None
        assert column.is_nullable is False

    def test_load_columns(self, mock_session):
        """Test load_columns function."""
        # Setup mock tables
        table1 = Mock(spec=Table)
        table1.id = UUID("11111111-1111-1111-1111-111111111111")
        table1.name = "table1"
        table2 = Mock(spec=Table)
        table2.id = UUID("22222222-2222-2222-2222-222222222222")
        table2.name = "table2"
        tables = [table1, table2]

        # Setup mock data
        columns_df = pd.DataFrame(
            {
                "TABLE_NAME": ["table1", "table2"],
                "COLUMN_NAME": ["col1", "col2"],
                "DATA_TYPE": ["VARCHAR", "INTEGER"],
                "COLUMN_DESCRIPTION": ["desc1", "desc2"],
                "COLUMN_DEFAULT": ["default1", "[NULL]"],
                "IS_NULLABLE": ["YES", "NO"],
            }
        )

        result = load_columns(mock_session, columns_df, tables, "test_tenant_456")

        assert len(result) == 2
        mock_session.add_all.assert_called_once()
        mock_session.flush.assert_called_once()

        # Verify column objects were created correctly
        added_columns = mock_session.add_all.call_args[0][0]
        assert len(added_columns) == 2
        assert all(isinstance(column, Column) for column in added_columns)
        assert added_columns[0].name == "col1"
        assert added_columns[0].table_id == table1.id
        assert added_columns[0].is_nullable is True
        assert added_columns[1].is_nullable is False

    def test_normalize_boolean(self):
        """Test _normalize_boolean function."""
        # Test YES string
        assert _normalize_boolean("YES") is True
        assert _normalize_boolean("yes") is True
        assert _normalize_boolean("Yes") is True

        # Test NO string
        assert _normalize_boolean("NO") is False
        assert _normalize_boolean("no") is False
        assert _normalize_boolean("No") is False

        # Test other strings
        assert _normalize_boolean("maybe") is False
        assert _normalize_boolean("") is False

        # Test non-string values
        assert _normalize_boolean(True) is True
        assert _normalize_boolean(False) is False
        assert _normalize_boolean(1) is True
        assert _normalize_boolean(0) is False
        assert _normalize_boolean(None) is False

    def test_normalize_null_value(self):
        """Test _normalize_null_value function."""
        # Test [NULL] string
        assert _normalize_null_value("[NULL]") is None
        assert _normalize_null_value(None) is None

        # Test other values
        assert _normalize_null_value("test") == "test"
        assert _normalize_null_value(123) == 123
        assert _normalize_null_value("") == ""

    def test_normalize_distinct_count(self):
        """Test normalize_distinct_count function."""
        # Test valid integer
        assert normalize_distinct_count(10) == 10
        assert normalize_distinct_count("10") == 10

        # Test NaN values
        assert normalize_distinct_count(pd.NA) is None
        assert normalize_distinct_count(None) is None

        # Test invalid values
        assert normalize_distinct_count("invalid") is None
        assert normalize_distinct_count("") is None

    def test_convert_row_to_column_statistics(self):
        """Test _convert_row_to_column_statistics function."""
        row = pd.Series(
            {
                "COLUMN_NAME": "col1",
                "MOST_COMMON_VALS": ["val1", "val2"],
                "MOST_COMMON_FREQS": [0.5, 0.3],
                "NULL_FRAC": 0.1,
                "N_DISTINCT": 10,
                "MIN_VALUE": "min_val",
                "MAX_VALUE": "max_val",
            }
        )

        stats = _convert_row_to_column_statistics(row, "test_tenant_456", UUID("11111111-1111-1111-1111-111111111111"))

        assert isinstance(stats, ColumnStatistics)
        assert stats.column_id == UUID("11111111-1111-1111-1111-111111111111")
        assert stats.most_common_values == ["val1", "val2"]
        assert stats.most_common_frequencies == [0.5, 0.3]
        assert stats.null_percentage == 0.1
        assert stats.distinct_values_count == 10
        assert stats.min_value == "min_val"
        assert stats.max_value == "max_val"
        assert stats.tenant_id == "test_tenant_456"

    def test_load_column_statistics(self, mock_session):
        """Test load_column_statistics function."""
        # Setup mock columns
        column1 = Mock(spec=Column)
        column1.id = UUID("11111111-1111-1111-1111-111111111111")
        column1.name = "col1"
        columns = [column1]

        # Setup mock data
        column_statistics_df = pd.DataFrame(
            {
                "COLUMN_NAME": ["col1"],
                "MOST_COMMON_VALS": [["val1", "val2"]],
                "MOST_COMMON_FREQS": [[0.5, 0.3]],
                "NULL_FRAC": [0.1],
                "N_DISTINCT": [10],
                "MIN_VALUE": ["min_val"],
                "MAX_VALUE": ["max_val"],
            }
        )

        load_column_statistics(mock_session, column_statistics_df, columns, "test_tenant_456")

        mock_session.add_all.assert_called_once()
        mock_session.flush.assert_called_once()

        # Verify column statistics objects were created correctly
        added_stats = mock_session.add_all.call_args[0][0]
        assert len(added_stats) == 1
        assert isinstance(added_stats[0], ColumnStatistics)
        assert added_stats[0].column_id == column1.id
        assert added_stats[0].most_common_values == ["val1", "val2"]
        assert added_stats[0].null_percentage == 0.1
        assert added_stats[0].distinct_values_count == 10

    def test_convert_row_to_primary_key(self):
        """Test _convert_row_to_primary_key function."""
        row = pd.Series(
            {"TABLE_NAME": "table1", "COLUMN_NAME": "id", "ORDINAL_POSITION": 1, "CONSTRAINT_NAME": "pk_table1"}
        )

        pk = _convert_row_to_primary_key(
            row,
            "test_tenant_456",
            UUID("11111111-1111-1111-1111-111111111111"),
            UUID("22222222-2222-2222-2222-222222222222"),
        )

        assert isinstance(pk, PrimaryKeyConstraint)
        assert pk.table_id == UUID("11111111-1111-1111-1111-111111111111")
        assert pk.column_id == UUID("22222222-2222-2222-2222-222222222222")
        assert pk.ordinal_position == 1
        assert pk.constraint_name == "pk_table1"
        assert pk.tenant_id == "test_tenant_456"

    def test_load_primary_keys(self, mock_session):
        """Test load_primary_keys function."""
        # Setup mock tables and columns
        table1 = Mock(spec=Table)
        table1.id = UUID("11111111-1111-1111-1111-111111111111")
        table1.name = "table1"
        tables = [table1]

        column1 = Mock(spec=Column)
        column1.id = UUID("22222222-2222-2222-2222-222222222222")
        column1.name = "id"
        columns = [column1]

        # Setup mock data
        primary_keys_df = pd.DataFrame(
            {"TABLE_NAME": ["table1"], "COLUMN_NAME": ["id"], "ORDINAL_POSITION": [1], "CONSTRAINT_NAME": ["pk_table1"]}
        )

        load_primary_keys(mock_session, primary_keys_df, tables, columns, "test_tenant_456")

        mock_session.add_all.assert_called_once()
        mock_session.flush.assert_called_once()

        # Verify primary key objects were created correctly
        added_pks = mock_session.add_all.call_args[0][0]
        assert len(added_pks) == 1
        assert isinstance(added_pks[0], PrimaryKeyConstraint)
        assert added_pks[0].table_id == table1.id
        assert added_pks[0].column_id == column1.id
        assert added_pks[0].constraint_name == "pk_table1"

    def test_convert_row_to_foreign_key(self):
        """Test _convert_row_to_foreign_key function."""
        row = pd.Series(
            {
                "TABLE_NAME": "table1",
                "COLUMN_NAME": "foreign_id",
                "ORDINAL_POSITION": 1,
                "CONSTRAINT_NAME": "fk_table1",
                "REFERENCED_TABLE_NAME": "table2",
                "REFERENCED_COLUMN_NAME": "id",
            }
        )

        fk = _convert_row_to_foreign_key(
            row,
            "test_tenant_456",
            UUID("11111111-1111-1111-1111-111111111111"),
            UUID("22222222-2222-2222-2222-222222222222"),
            UUID("33333333-3333-3333-3333-333333333333"),
            UUID("44444444-4444-4444-4444-444444444444"),
        )

        assert isinstance(fk, ForeignKeyConstraint)
        assert fk.table_id == UUID("11111111-1111-1111-1111-111111111111")
        assert fk.column_id == UUID("22222222-2222-2222-2222-222222222222")
        assert fk.referenced_table_id == UUID("33333333-3333-3333-3333-333333333333")
        assert fk.referenced_column_id == UUID("44444444-4444-4444-4444-444444444444")
        assert fk.constraint_name == "fk_table1"
        assert fk.ordinal_position == 1
        assert fk.tenant_id == "test_tenant_456"

    def test_load_foreign_keys(self, mock_session):
        """Test load_foreign_keys function."""
        # Setup mock tables and columns
        table1 = Mock(spec=Table)
        table1.id = UUID("11111111-1111-1111-1111-111111111111")
        table1.name = "table1"
        table2 = Mock(spec=Table)
        table2.id = UUID("33333333-3333-3333-3333-333333333333")
        table2.name = "table2"
        tables = [table1, table2]

        column1 = Mock(spec=Column)
        column1.id = UUID("22222222-2222-2222-2222-222222222222")
        column1.name = "foreign_id"
        column2 = Mock(spec=Column)
        column2.id = UUID("44444444-4444-4444-4444-444444444444")
        column2.name = "id"
        columns = [column1, column2]

        # Setup mock data
        foreign_keys_df = pd.DataFrame(
            {
                "TABLE_NAME": ["table1"],
                "COLUMN_NAME": ["foreign_id"],
                "ORDINAL_POSITION": [1],
                "CONSTRAINT_NAME": ["fk_table1"],
                "REFERENCED_TABLE_NAME": ["table2"],
                "REFERENCED_COLUMN_NAME": ["id"],
            }
        )

        load_foreign_keys(mock_session, foreign_keys_df, tables, columns, "test_tenant_456")

        mock_session.add_all.assert_called_once()
        mock_session.flush.assert_called_once()

        # Verify foreign key objects were created correctly
        added_fks = mock_session.add_all.call_args[0][0]
        assert len(added_fks) == 1
        assert isinstance(added_fks[0], ForeignKeyConstraint)
        assert added_fks[0].table_id == table1.id
        assert added_fks[0].column_id == column1.id
        assert added_fks[0].referenced_table_id == table2.id
        assert added_fks[0].referenced_column_id == column2.id
        assert added_fks[0].constraint_name == "fk_table1"

    def test_filter_loaded_tables_empty_dataframe(self, mock_session):
        """Test filter_loaded_tables with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["TABLE_NAME", "TABLE_SCHEMA"])

        # Mock session.exec to return empty list
        mock_session.exec.return_value.all.return_value = []

        result = filter_loaded_tables(
            mock_session, empty_df, UUID("87654321-4321-8765-4321-876543218765"), "test_tenant_456"
        )

        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_load_tables_empty_dataframe(self, mock_session):
        """Test load_tables with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["TABLE_NAME", "TABLE_SCHEMA", "TABLE_DESCRIPTION", "TABLE_TYPE", "ROW_COUNT"])

        result = load_tables(
            mock_session,
            empty_df,
            UUID("12345678-1234-5678-1234-567812345678"),
            UUID("87654321-4321-8765-4321-876543218765"),
            "test_tenant_456",
        )

        assert len(result) == 0
        assert isinstance(result, list)
        mock_session.add.assert_not_called()
        # add_all is called with empty list for mind_datasource_tables
        mock_session.add_all.assert_called_once_with([])

    def test_load_columns_empty_dataframe(self, mock_session):
        """Test load_columns with empty DataFrame."""
        empty_df = pd.DataFrame(
            columns=["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE", "COLUMN_DESCRIPTION", "COLUMN_DEFAULT", "IS_NULLABLE"]
        )
        tables = []

        result = load_columns(mock_session, empty_df, tables, "test_tenant_456")

        assert len(result) == 0
        assert isinstance(result, list)
        # add_all is called with empty list
        mock_session.add_all.assert_called_once_with([])

    def test_load_column_statistics_empty_dataframe(self, mock_session):
        """Test load_column_statistics with empty DataFrame."""
        empty_df = pd.DataFrame(
            columns=[
                "COLUMN_NAME",
                "MOST_COMMON_VALS",
                "MOST_COMMON_FREQS",
                "NULL_FRAC",
                "N_DISTINCT",
                "MIN_VALUE",
                "MAX_VALUE",
            ]
        )
        columns = []

        load_column_statistics(mock_session, empty_df, columns, "test_tenant_456")

        # add_all is called with empty list
        mock_session.add_all.assert_called_once_with([])

    def test_load_primary_keys_empty_dataframe(self, mock_session):
        """Test load_primary_keys with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["TABLE_NAME", "COLUMN_NAME", "ORDINAL_POSITION", "CONSTRAINT_NAME"])
        tables = []
        columns = []

        load_primary_keys(mock_session, empty_df, tables, columns, "test_tenant_456")

        # add_all is called with empty list
        mock_session.add_all.assert_called_once_with([])

    def test_load_foreign_keys_empty_dataframe(self, mock_session):
        """Test load_foreign_keys with empty DataFrame."""
        empty_df = pd.DataFrame(
            columns=[
                "TABLE_NAME",
                "COLUMN_NAME",
                "ORDINAL_POSITION",
                "CONSTRAINT_NAME",
                "REFERENCED_TABLE_NAME",
                "REFERENCED_COLUMN_NAME",
            ]
        )
        tables = []
        columns = []

        load_foreign_keys(mock_session, empty_df, tables, columns, "test_tenant_456")

        # add_all is called with empty list
        mock_session.add_all.assert_called_once_with([])

    def test_convert_row_to_column_statistics_with_none_values(self):
        """Test _convert_row_to_column_statistics with None values."""
        row = pd.Series(
            {
                "COLUMN_NAME": "col1",
                "MOST_COMMON_VALS": None,
                "MOST_COMMON_FREQS": None,
                "NULL_FRAC": None,
                "N_DISTINCT": None,
                "MIN_VALUE": None,
                "MAX_VALUE": None,
            }
        )

        stats = _convert_row_to_column_statistics(row, "test_tenant_456", UUID("11111111-1111-1111-1111-111111111111"))

        assert isinstance(stats, ColumnStatistics)
        assert stats.column_id == UUID("11111111-1111-1111-1111-111111111111")
        assert stats.most_common_values is None
        assert stats.most_common_frequencies is None
        assert stats.null_percentage is None
        assert stats.distinct_values_count is None
        assert stats.min_value is None
        assert stats.max_value is None
