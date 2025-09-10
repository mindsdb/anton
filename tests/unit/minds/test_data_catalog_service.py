from unittest.mock import Mock
from uuid import UUID

import pandas as pd
import pytest
from sqlmodel import Session

from minds.model.data_catalog import Column, ColumnStatistics, ForeignKeyConstraint, PrimaryKeyConstraint, Table
from minds.model.datasource import Datasource
from minds.services.data_catalog import DataCatalogLoader, DataCatalogLoaderError


class TestDataCatalogLoader:
    """Test cases for DataCatalogLoader class."""

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = Mock(spec=Session)
        session.add_all = Mock()
        session.flush = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        session.exec = Mock()
        return session

    @pytest.fixture
    def mock_mindsdb_client(self):
        """Mock MindsDB client."""
        client = Mock()
        client.query = Mock()
        return client

    @pytest.fixture
    def mock_datasource(self):
        """Mock datasource."""
        datasource = Mock(spec=Datasource)
        datasource.id = UUID("12345678-1234-5678-1234-567812345678")
        datasource.name = "test_datasource"
        return datasource

    @pytest.fixture
    def data_catalog_loader(self, mock_session, mock_mindsdb_client):
        """Create DataCatalogLoader instance with mocked dependencies."""
        return DataCatalogLoader(session=mock_session, mindsdb_client=mock_mindsdb_client, user_id="test_user_123", tenant_id="test_tenant_456")

    def test_initialization(self, mock_session, mock_mindsdb_client):
        """Test DataCatalogLoader initialization."""
        loader = DataCatalogLoader(session=mock_session, mindsdb_client=mock_mindsdb_client, user_id="test_user_123", tenant_id="test_tenant_456")

        assert loader.session == mock_session
        assert loader.mindsdb_client == mock_mindsdb_client
        assert loader.user_id == "test_user_123"

    def test_normalize_boolean(self, data_catalog_loader):
        """Test _normalize_boolean method."""
        # Test YES string
        assert data_catalog_loader._normalize_boolean("YES") is True
        assert data_catalog_loader._normalize_boolean("yes") is True
        assert data_catalog_loader._normalize_boolean("Yes") is True

        # Test NO string
        assert data_catalog_loader._normalize_boolean("NO") is False
        assert data_catalog_loader._normalize_boolean("no") is False
        assert data_catalog_loader._normalize_boolean("No") is False

        # Test other strings
        assert data_catalog_loader._normalize_boolean("maybe") is False
        assert data_catalog_loader._normalize_boolean("") is False

        # Test non-string values
        assert data_catalog_loader._normalize_boolean(True) is True
        assert data_catalog_loader._normalize_boolean(False) is False
        assert data_catalog_loader._normalize_boolean(1) is True
        assert data_catalog_loader._normalize_boolean(0) is False
        assert data_catalog_loader._normalize_boolean(None) is False

    def test_normalize_null_value(self, data_catalog_loader):
        """Test _normalize_null_value method."""
        # Test [NULL] string
        assert data_catalog_loader._normalize_null_value("[NULL]") is None
        assert data_catalog_loader._normalize_null_value(None) is None

        # Test other values
        assert data_catalog_loader._normalize_null_value("test") == "test"
        assert data_catalog_loader._normalize_null_value(123) == 123
        assert data_catalog_loader._normalize_null_value("") == ""

    def test_execute_query_success(self, data_catalog_loader):
        """Test _execute_query method with successful execution."""
        # Setup mock
        mock_query_result = Mock()
        mock_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        mock_query_result.fetch.return_value = mock_df
        data_catalog_loader.mindsdb_client.query.return_value = mock_query_result

        # Execute query
        result = data_catalog_loader._execute_query("SELECT * FROM test")

        # Verify
        data_catalog_loader.mindsdb_client.query.assert_called_once_with("SELECT * FROM test")
        mock_query_result.fetch.assert_called_once()
        pd.testing.assert_frame_equal(result, mock_df)

    def test_execute_query_failure(self, data_catalog_loader):
        """Test _execute_query method with failed execution."""
        # Setup mock to raise exception
        data_catalog_loader.mindsdb_client.query.side_effect = Exception("Query failed")

        # Execute query and expect exception
        with pytest.raises(Exception, match="Query failed"):
            data_catalog_loader._execute_query("SELECT * FROM test")

    async def test_load_success(self, data_catalog_loader, mock_datasource):
        """Test successful load operation."""
        # Setup mock for database query
        mock_result = Mock()
        mock_result.first.return_value = mock_datasource
        data_catalog_loader.session.exec.return_value = mock_result

        # Mock all the _get_* methods to return empty DataFrames (no data to process)
        data_catalog_loader._get_tables = Mock(return_value=pd.DataFrame())
        data_catalog_loader._get_columns = Mock(return_value=pd.DataFrame())
        data_catalog_loader._get_column_statistics = Mock(return_value=pd.DataFrame())
        data_catalog_loader._get_primary_keys = Mock(return_value=pd.DataFrame())
        data_catalog_loader._get_foreign_keys = Mock(return_value=pd.DataFrame())

        # Execute load
        mock_mind_datasource = Mock()
        mock_mind_datasource.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_mind_datasource.datasource_id = UUID("87654321-4321-8765-4321-876543218765")
        mock_datasource_config = Mock()
        mock_datasource_config.name = "test_datasource"
        mock_datasource_config.tables = ["table1", "table2"]
        
        await data_catalog_loader.load(mock_mind_datasource, mock_datasource_config)

        # Verify that the method was called and commit was called
        data_catalog_loader.session.commit.assert_called_once()

    async def test_load_with_error_and_rollback(self, data_catalog_loader):
        """Test load operation with error and rollback."""
        # Setup mock for database query that returns None (datasource not found)
        mock_result = Mock()
        mock_result.first.return_value = None
        data_catalog_loader.session.exec.return_value = mock_result

        # Mock _get_tables to raise an exception
        data_catalog_loader._get_tables = Mock(side_effect=Exception("Database connection failed"))

        # Execute load and expect exception
        mock_mind_datasource = Mock()
        mock_mind_datasource.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_mind_datasource.datasource_id = UUID("87654321-4321-8765-4321-876543218765")
        mock_datasource_config = Mock()
        mock_datasource_config.name = "nonexistent_datasource"
        mock_datasource_config.tables = ["table1", "table2"]
        
        with pytest.raises(DataCatalogLoaderError, match="Error loading data catalog: Database connection failed"):
            await data_catalog_loader.load(mock_mind_datasource, mock_datasource_config)

        # Verify rollback was called
        data_catalog_loader.session.rollback.assert_called_once()
        data_catalog_loader.session.commit.assert_not_called()

    def test_get_tables_without_filter(self, data_catalog_loader, mock_datasource):
        """Test _get_tables method without table name filter."""
        # Setup mock
        mock_df = pd.DataFrame(
            {
                "TABLE_NAME": ["table1", "table2"],
                "TABLE_SCHEMA": ["schema1", "schema2"],
                "TABLE_DESCRIPTION": ["desc1", "desc2"],
                "TABLE_TYPE": ["BASE TABLE", "VIEW"],
                "ROW_COUNT": [100, 200],
            }
        )
        data_catalog_loader._execute_query = Mock(return_value=mock_df)
        data_catalog_loader.session.exec.return_value.all.return_value = []

        # Execute
        result = data_catalog_loader._get_tables(mock_datasource.id, "test_datasource")

        # Verify query was constructed correctly
        expected_query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_TABLES 
        WHERE TABLE_SCHEMA = '{mock_datasource.name}'
        """
        data_catalog_loader._execute_query.assert_called_once()
        call_args = data_catalog_loader._execute_query.call_args[0][0]
        assert call_args == expected_query

        pd.testing.assert_frame_equal(result, mock_df)

    def test_get_tables_with_filter(self, data_catalog_loader, mock_datasource):
        """Test _get_tables method with table name filter."""
        # Setup mock
        mock_df = pd.DataFrame(
            {
                "TABLE_NAME": ["table1"],
                "TABLE_SCHEMA": ["schema1"],
                "TABLE_DESCRIPTION": ["desc1"],
                "TABLE_TYPE": ["BASE TABLE"],
                "ROW_COUNT": [100],
            }
        )
        data_catalog_loader._execute_query = Mock(return_value=mock_df)
        data_catalog_loader.session.exec.return_value.all.return_value = []

        # Execute with filter
        result = data_catalog_loader._get_tables(mock_datasource.id, mock_datasource.name, ["table1", "table2"])

        # Verify query includes filter
        expected_query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_TABLES 
        WHERE TABLE_SCHEMA = '{mock_datasource.name}'
         AND TABLE_NAME IN ('table1', 'table2')"""
        data_catalog_loader._execute_query.assert_called_once()
        call_args = data_catalog_loader._execute_query.call_args[0][0]
        assert call_args == expected_query

        pd.testing.assert_frame_equal(result, mock_df)

    def test_get_tables_excludes_existing(self, data_catalog_loader, mock_datasource):
        """Test _get_tables method excludes already existing tables."""
        # Setup mock
        mock_df = pd.DataFrame(
            {
                "TABLE_NAME": ["table1", "table2", "table3"],
                "TABLE_SCHEMA": ["schema1", "schema2", "schema3"],
                "TABLE_DESCRIPTION": ["desc1", "desc2", "desc3"],
                "TABLE_TYPE": ["BASE TABLE", "BASE TABLE", "VIEW"],
                "ROW_COUNT": [100, 200, 300],
            }
        )
        data_catalog_loader._execute_query = Mock(return_value=mock_df)

        # Mock existing tables
        existing_table = Mock(spec=Table)
        existing_table.name = "table2"
        data_catalog_loader.session.exec.return_value.all.return_value = [existing_table]

        # Execute
        result = data_catalog_loader._get_tables(mock_datasource.id, "test_datasource")

        # Verify result excludes existing table
        assert len(result) == 2
        assert "table2" not in result["TABLE_NAME"].values
        assert "table1" in result["TABLE_NAME"].values
        assert "table3" in result["TABLE_NAME"].values

    def test_get_columns(self, data_catalog_loader, mock_datasource):
        """Test _get_columns method."""
        # Setup mock
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
        data_catalog_loader._execute_query = Mock(return_value=mock_df)

        # Execute
        result = data_catalog_loader._get_columns(mock_datasource.name, ["table1"])

        expected_query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_COLUMNS 
        WHERE TABLE_SCHEMA = '{mock_datasource.name}'
         AND TABLE_NAME IN ('table1')"""
        data_catalog_loader._execute_query.assert_called_once()
        call_args = data_catalog_loader._execute_query.call_args[0][0]
        assert call_args == expected_query

        pd.testing.assert_frame_equal(result, mock_df)

    def test_get_column_statistics(self, data_catalog_loader, mock_datasource):
        """Test _get_column_statistics method."""
        # Setup mock
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
        data_catalog_loader._execute_query = Mock(return_value=mock_df)

        # Execute
        result = data_catalog_loader._get_column_statistics(mock_datasource.name, None)

        expected_query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_COLUMN_STATISTICS 
        WHERE TABLE_SCHEMA = '{mock_datasource.name}'
        """
        data_catalog_loader._execute_query.assert_called_once()
        call_args = data_catalog_loader._execute_query.call_args[0][0]
        assert call_args == expected_query

        pd.testing.assert_frame_equal(result, mock_df)

    def test_get_primary_keys(self, data_catalog_loader, mock_datasource):
        """Test _get_primary_keys method."""
        # Setup mock
        mock_df = pd.DataFrame(
            {"TABLE_NAME": ["table1"], "COLUMN_NAME": ["id"], "ORDINAL_POSITION": [1], "CONSTRAINT_NAME": ["pk_table1"]}
        )
        data_catalog_loader._execute_query = Mock(return_value=mock_df)

        # Execute
        result = data_catalog_loader._get_primary_keys(mock_datasource.name, None)

        expected_query = f"""
        SELECT 
            kcu.TABLE_NAME,
            kcu.COLUMN_NAME,
            kcu.ORDINAL_POSITION,
            kcu.CONSTRAINT_NAME
        FROM information_schema.META_KEY_COLUMN_USAGE kcu
        INNER JOIN information_schema.META_TABLE_CONSTRAINTS tc 
            ON kcu.CONSTRAINT_SCHEMA = tc.CONSTRAINT_SCHEMA
            AND kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
            AND kcu.TABLE_NAME = tc.TABLE_NAME
        WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
            AND kcu.TABLE_SCHEMA = '{mock_datasource.name}'
            AND tc.TABLE_SCHEMA = '{mock_datasource.name}'
        """
        data_catalog_loader._execute_query.assert_called_once()
        call_args = data_catalog_loader._execute_query.call_args[0][0]
        assert call_args == expected_query

        pd.testing.assert_frame_equal(result, mock_df)

    def test_get_foreign_keys(self, data_catalog_loader, mock_datasource):
        """Test _get_foreign_keys method."""
        # Setup mock
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
        data_catalog_loader._execute_query = Mock(return_value=mock_df)

        # Execute
        result = data_catalog_loader._get_foreign_keys(mock_datasource.name, None)

        expected_query = f"""
        SELECT 
            kcu.TABLE_NAME,
            kcu.COLUMN_NAME,
            kcu.ORDINAL_POSITION,
            kcu.CONSTRAINT_NAME,
            kcu.REFERENCED_TABLE_NAME,
            kcu.REFERENCED_COLUMN_NAME
        FROM information_schema.META_KEY_COLUMN_USAGE kcu
        INNER JOIN information_schema.META_TABLE_CONSTRAINTS tc 
            ON kcu.CONSTRAINT_SCHEMA = tc.CONSTRAINT_SCHEMA
            AND kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
            AND kcu.TABLE_NAME = tc.TABLE_NAME
        WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
            AND kcu.TABLE_SCHEMA = '{mock_datasource.name}'
            AND tc.TABLE_SCHEMA = '{mock_datasource.name}'
        """
        data_catalog_loader._execute_query.assert_called_once()
        call_args = data_catalog_loader._execute_query.call_args[0][0]
        assert call_args == expected_query

        pd.testing.assert_frame_equal(result, mock_df)

    def test_load_tables(self, data_catalog_loader, mock_datasource):
        """Test _load_tables method."""
        # Setup mock data
        tables_df = pd.DataFrame(
            {
                "TABLE_NAME": ["table1", "table2"],
                "TABLE_SCHEMA": ["schema1", "schema2"],
                "TABLE_DESCRIPTION": ["desc1", "desc2"],
                "TABLE_TYPE": ["BASE TABLE", "VIEW"],
                "ROW_COUNT": [100, 200],
            }
        )

        # Mock the session to assign IDs to tables after flush
        def mock_add(table):
            table.id = UUID(f"12345678-1234-5678-1234-56781234567{len(data_catalog_loader.session.add.call_args_list)}")
        data_catalog_loader.session.add.side_effect = mock_add

        # Execute
        result = data_catalog_loader._load_tables(mock_datasource.id, mock_datasource.id, tables_df)

        # Verify
        assert len(result) == 2
        assert data_catalog_loader.session.add.call_count == 2  # Two for tables
        assert data_catalog_loader.session.add_all.call_count == 1  # One for mind_datasource_tables
        assert data_catalog_loader.session.flush.call_count == 3  # Two for tables + one for add_all
        assert data_catalog_loader.session.refresh.call_count == 2  # One for each table

        # Verify table objects were created correctly
        added_tables = [call[0][0] for call in data_catalog_loader.session.add.call_args_list]
        assert len(added_tables) == 2
        assert all(isinstance(table, Table) for table in added_tables)
        assert added_tables[0].name == "table1"
        assert added_tables[0].datasource_id == mock_datasource.id
        assert added_tables[0].row_count == 100

    def test_load_columns(self, data_catalog_loader):
        """Test _load_columns method."""
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

        # Execute
        result = data_catalog_loader._load_columns(columns_df, tables)

        # Verify
        assert len(result) == 2
        data_catalog_loader.session.add_all.assert_called_once()
        data_catalog_loader.session.flush.assert_called_once()

        # Verify column objects were created correctly
        added_columns = data_catalog_loader.session.add_all.call_args[0][0]
        assert len(added_columns) == 2
        assert all(isinstance(column, Column) for column in added_columns)
        assert added_columns[0].name == "col1"
        assert added_columns[0].table_id == table1.id
        assert added_columns[0].is_nullable is True
        assert added_columns[1].is_nullable is False

    def test_load_column_statistics(self, data_catalog_loader):
        """Test _load_column_statistics method."""
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

        # Execute
        data_catalog_loader._load_column_statistics(column_statistics_df, columns)

        # Verify
        data_catalog_loader.session.add_all.assert_called_once()
        data_catalog_loader.session.flush.assert_called_once()

        # Verify column statistics objects were created correctly
        added_stats = data_catalog_loader.session.add_all.call_args[0][0]
        assert len(added_stats) == 1
        assert isinstance(added_stats[0], ColumnStatistics)
        assert added_stats[0].column_id == column1.id
        assert added_stats[0].most_common_values == ["val1", "val2"]
        assert added_stats[0].null_percentage == 0.1
        assert added_stats[0].distinct_values_count == 10

    def test_load_primary_keys(self, data_catalog_loader):
        """Test _load_primary_keys method."""
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

        # Execute
        data_catalog_loader._load_primary_keys(primary_keys_df, tables, columns)

        # Verify
        data_catalog_loader.session.add_all.assert_called_once()
        data_catalog_loader.session.flush.assert_called_once()

        # Verify primary key objects were created correctly
        added_pks = data_catalog_loader.session.add_all.call_args[0][0]
        assert len(added_pks) == 1
        assert isinstance(added_pks[0], PrimaryKeyConstraint)
        assert added_pks[0].table_id == table1.id
        assert added_pks[0].column_id == column1.id
        assert added_pks[0].constraint_name == "pk_table1"

    def test_load_foreign_keys(self, data_catalog_loader):
        """Test _load_foreign_keys method."""
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

        # Execute
        data_catalog_loader._load_foreign_keys(foreign_keys_df, tables, columns)

        # Verify
        data_catalog_loader.session.add_all.assert_called_once()
        data_catalog_loader.session.flush.assert_called_once()

        # Verify foreign key objects were created correctly
        added_fks = data_catalog_loader.session.add_all.call_args[0][0]
        assert len(added_fks) == 1
        assert isinstance(added_fks[0], ForeignKeyConstraint)
        assert added_fks[0].table_id == table1.id
        assert added_fks[0].column_id == column1.id
        assert added_fks[0].referenced_table_id == table2.id
        assert added_fks[0].referenced_column_id == column2.id
        assert added_fks[0].constraint_name == "fk_table1"

    def test_load_tables_with_nan_row_count(self, data_catalog_loader, mock_datasource):
        """Test _load_tables method handles NaN row_count values."""
        # Setup mock data with NaN row_count
        tables_df = pd.DataFrame(
            {
                "TABLE_NAME": ["table1"],
                "TABLE_SCHEMA": ["schema1"],
                "TABLE_DESCRIPTION": ["desc1"],
                "TABLE_TYPE": ["BASE TABLE"],
                "ROW_COUNT": [pd.NA],  # NaN value
            }
        )

        # Mock the session to assign IDs to tables after flush
        def mock_add(table):
            table.id = UUID(f"12345678-1234-5678-1234-56781234567{len(data_catalog_loader.session.add.call_args_list)}")
        data_catalog_loader.session.add.side_effect = mock_add

        # Execute
        data_catalog_loader._load_tables(mock_datasource.id, mock_datasource.id, tables_df)

        # Verify
        added_tables = [call[0][0] for call in data_catalog_loader.session.add.call_args_list]
        assert added_tables[0].row_count is None
        assert len(added_tables) == 1

    def test_load_column_statistics_with_invalid_distinct_count(self, data_catalog_loader):
        """Test _load_column_statistics method handles invalid distinct count values."""
        # Setup mock columns
        column1 = Mock(spec=Column)
        column1.id = UUID("11111111-1111-1111-1111-111111111111")
        column1.name = "col1"
        columns = [column1]

        # Setup mock data with invalid distinct count
        column_statistics_df = pd.DataFrame(
            {
                "COLUMN_NAME": ["col1"],
                "MOST_COMMON_VALS": [["val1"]],
                "MOST_COMMON_FREQS": [[0.5]],
                "NULL_FRAC": [0.1],
                "N_DISTINCT": ["invalid"],  # Invalid value
                "MIN_VALUE": ["min_val"],
                "MAX_VALUE": ["max_val"],
            }
        )

        # Execute
        data_catalog_loader._load_column_statistics(column_statistics_df, columns)

        # Verify
        added_stats = data_catalog_loader.session.add_all.call_args[0][0]
        assert added_stats[0].distinct_values_count is None
