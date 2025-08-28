from typing import Any, Optional, List

from mindsdb_sdk.server import Server
import pandas as pd

from minds.common.logger import setup_logging
from minds.model.data_catalog import Column, ColumnStatistics, Table, PrimaryKeyConstraint, ForeignKeyConstraint


logger = setup_logging()


class DataCatalogLoader:
    """A class for loading data catalogs."""

    def __init__(
        self,
        datasource_name: str,
        server: Server,
        tables_filter: Optional[List[str]] = None,
        engine: Optional[str] = None,
    ):
        """
        Initialize the MindsDB data catalog loader.

        Args:
            datasource_name: Name of the MindsDB datasource.
            server: MindsDB server instance from mindsdb_sdk.connect().
            tables_filter: Optional list of table names to filter (empty list = no filtering).
            engine: Optional engine type (e.g., 'salesforce', 'mysql', 'snowflake').
        """
        self.datasource_name = datasource_name.strip()
        self.server = server
        self.project = server.get_project()
        self.tables_filter = tables_filter or []
        self.engine = engine

    def _execute_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query against MindsDB and return DataFrame."""
        try:
            logger.info(f"Executing MindsDB query for datasource '{self.datasource_name}': {query}")
            query_result = self.project.query(query)
            df = query_result.fetch()
            logger.info(f"MindsDB query returned {len(df)} rows for datasource '{self.datasource_name}'")
            return df
        except Exception as e:
            logger.error(f"Failed to execute MindsDB query for datasource '{self.datasource_name}': {query}. Error: {str(e)}")
            raise

    def _normalize_boolean(self, value: Any) -> bool:
        """Convert MindsDB YES/NO strings to boolean."""
        if isinstance(value, str):
            return value.upper() == "YES"
        return bool(value)

    def _normalize_null_value(self, value: Any) -> Optional[Any]:
        """Convert MindsDB [NULL] strings to None."""
        if value == "[NULL]" or value is None:
            return None
        return value
    
    def _get_tables_metadata(self) -> pd.DataFrame:
        """Get table metadata from META_TABLES with optional filtering."""
        logger.info(
            f"Getting table metadata for datasource '{self.datasource_name}' with filter: {self.tables_filter}"
        )

        query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_TABLES 
        WHERE TABLE_CATALOG = '{self.datasource_name}'
        """

        if self.tables_filter:
            query += f" AND TABLE_NAME IN ({', '.join(f"'{name}'" for name in self.tables_filter)})"

        df = self._execute_query(query)
        logger.info(f"Found {len(df)} tables before filtering")

        return df
    
    def _get_columns_metadata(self) -> pd.DataFrame:
        """Get column metadata from META_COLUMNS with optional filtering."""
        logger.info(
            f"Getting column metadata for integration '{self.integration_name}' with filter: {self.tables_filter}"
        )

        query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_COLUMNS 
        WHERE TABLE_CATALOG = '{self.integration_name}'
        """

        if self.tables_filter:
            query += f" AND TABLE_NAME IN ({', '.join(f"'{name}'" for name in self.tables_filter)})"

        df = self._execute_query(query)
        logger.info(f"Found {len(df)} columns before filtering")

        return df
    
    def _get_column_statistics(self) -> pd.DataFrame:
        """Get column statistics from META_COLUMN_STATISTICS with optional filtering."""
        logger.info(
            f"Getting column statistics for datasource '{self.datasource_name}' with filter: {self.tables_filter}"
        )

        query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_COLUMN_STATISTICS 
        WHERE TABLE_CATALOG = '{self.datasource_name}'
        """

        if self.tables_filter:
            query += f" AND TABLE_NAME IN ({', '.join(f"'{name}'" for name in self.tables_filter)})"

        df = self._execute_query(query)
        logger.info(f"Found {len(df)} column statistics before filtering")

        return df
    
    def _get_primary_keys(self) -> pd.DataFrame:
        """Get primary key information from META_KEY_COLUMN_USAGE with optional filtering."""
        query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_KEY_COLUMN_USAGE 
        WHERE TABLE_CATALOG = '{self.datasource_name}'
        """

        if self.tables_filter:
            query += f" AND TABLE_NAME IN ({', '.join(f"'{name}'" for name in self.tables_filter)})"

        df = self._execute_query(query)
        logger.info(f"Found {len(df)} primary keys before filtering")

        return df

    def _get_table_constraints(self) -> pd.DataFrame:
        """Get table constraints from META_TABLE_CONSTRAINTS with optional filtering."""
        query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_TABLE_CONSTRAINTS 
        WHERE TABLE_CATALOG = '{self.datasource_name}'
        """

        if self.tables_filter:
            query += f" AND TABLE_NAME IN ({', '.join(f"'{name}'" for name in self.tables_filter)})"

        df = self._execute_query(query)
        logger.info(f"Found {len(df)} table constraints before filtering")

        return df
