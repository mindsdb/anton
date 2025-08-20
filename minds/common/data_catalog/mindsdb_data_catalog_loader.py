from typing import Dict, Any, Optional, List
import pandas as pd
import mindsdb_sdk
import logging

from minds.common.data_catalog.data_catalog import (
    MindsDBDataCatalog,
    Table,
    Column,
    PrimaryKey,
    ForeignKey,
    ColumnStatistics,
)
from minds.common.data_catalog.data_catalog_loader import DataCatalogLoader

logger = logging.getLogger(__name__)


class MindsDBDataCatalogLoader(DataCatalogLoader):
    """Data catalog loader for MindsDB integrations."""

    def __init__(
        self,
        integration_name: str,
        server: mindsdb_sdk.server.Server,
        tables_filter: Optional[List[str]] = None,
        engine: Optional[str] = None,
    ):
        """
        Initialize the MindsDB data catalog loader.

        Args:
            integration_name: Name of the MindsDB integration (TABLE_SCHEMA)
            server: MindsDB server instance from mindsdb_sdk.connect()
            tables_filter: Optional list of table names to filter (empty list = no filtering)
            engine: Optional engine type (e.g., 'salesforce', 'mysql', 'snowflake')

        Raises:
            ValueError: If server is None or integration_name is empty
        """
        if not server:
            raise ValueError("MindsDB server instance is required")
        if not integration_name or not integration_name.strip():
            raise ValueError("Integration name is required")

        self.integration_name = integration_name.strip()
        self.server = server
        self.project = server.get_project()
        self.tables_filter = tables_filter or []  # Store tables filter
        self.engine = engine  # Store the original engine type

    def _execute_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query against MindsDB and return DataFrame."""
        try:
            logger.info(
                f"Executing MindsDB query for integration '{self.integration_name}': {query}"
            )
            query_result = self.project.query(query)
            df = query_result.fetch()
            logger.info(
                f"MindsDB query returned {len(df)} rows for integration '{self.integration_name}'"
            )
            return df
        except Exception as e:
            logger.error(
                f"Failed to execute MindsDB query for integration '{self.integration_name}': {query}. Error: {str(e)}"
            )
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
            f"Getting table metadata for integration '{self.integration_name}' with filter: {self.tables_filter}"
        )

        query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_TABLES 
        WHERE TABLE_SCHEMA = '{self.integration_name}'
        """

        df = self._execute_query(query)
        logger.info(f"Found {len(df)} tables before filtering")

        # Apply table filtering after fetching if specified
        if self.tables_filter and not df.empty:
            # Case-insensitive filtering
            tables_filter_lower = [table.lower() for table in self.tables_filter]
            df = df[df["TABLE_NAME"].str.lower().isin(tables_filter_lower)]
            logger.info(
                f"Applied table filter {self.tables_filter}, found {len(df)} tables after filtering"
            )
        else:
            logger.info("No table filter applied - loading all tables")

        return df

    def _get_columns_metadata(self) -> pd.DataFrame:
        """Get column metadata from META_COLUMNS with optional filtering."""
        logger.info(
            f"Getting column metadata for integration '{self.integration_name}' with filter: {self.tables_filter}"
        )

        query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_COLUMNS 
        WHERE TABLE_SCHEMA = '{self.integration_name}'
        """

        df = self._execute_query(query)
        logger.info(f"Found {len(df)} columns before filtering")

        # Apply table filtering after fetching if specified
        if self.tables_filter and not df.empty:
            # Case-insensitive filtering
            tables_filter_lower = [table.lower() for table in self.tables_filter]
            df = df[df["TABLE_NAME"].str.lower().isin(tables_filter_lower)]
            logger.info(f"Found {len(df)} columns after filtering")
        else:
            logger.info(f"No table filter applied - keeping all {len(df)} columns")

        return df

    def _get_column_statistics(self) -> pd.DataFrame:
        """Get column statistics from META_COLUMN_STATISTICS with optional filtering."""
        logger.info(
            f"Getting column statistics for integration '{self.integration_name}' with filter: {self.tables_filter}"
        )

        query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_COLUMN_STATISTICS 
        WHERE TABLE_SCHEMA = '{self.integration_name}'
        """

        try:
            df = self._execute_query(query)
            logger.info(f"Found statistics for {len(df)} columns before filtering")

            # Apply table filtering after fetching if specified
            if self.tables_filter and not df.empty:
                # Case-insensitive filtering
                tables_filter_lower = [table.lower() for table in self.tables_filter]
                df = df[df["TABLE_NAME"].str.lower().isin(tables_filter_lower)]
                logger.info(f"Found statistics for {len(df)} columns after filtering")
            else:
                logger.info(
                    f"No table filter applied - keeping all {len(df)} column statistics"
                )

            return df
        except Exception as e:
            logger.warning(
                f"Failed to get column statistics for integration '{self.integration_name}': {str(e)}"
            )
            return pd.DataFrame()

    def _get_primary_keys(self) -> pd.DataFrame:
        """Get primary key information from META_KEY_COLUMN_USAGE with optional filtering."""
        query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_KEY_COLUMN_USAGE 
        WHERE TABLE_SCHEMA = '{self.integration_name}'
        """

        try:
            df = self._execute_query(query)
            logger.info(f"Found {len(df)} primary key entries before filtering")

            # Apply table filtering after fetching if specified
            if self.tables_filter and not df.empty:
                # Case-insensitive filtering
                tables_filter_lower = [table.lower() for table in self.tables_filter]
                df = df[df["TABLE_NAME"].str.lower().isin(tables_filter_lower)]
                logger.info(f"Found {len(df)} primary key entries after filtering")
            else:
                logger.info(
                    f"No table filter applied - keeping all {len(df)} primary key entries"
                )

            return df
        except Exception as e:
            logger.warning(f"Failed to get primary keys: {str(e)}")
            return pd.DataFrame()

    def _get_table_constraints(self) -> pd.DataFrame:
        """Get table constraints from META_TABLE_CONSTRAINTS with optional filtering."""
        query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_TABLE_CONSTRAINTS 
        WHERE TABLE_SCHEMA = '{self.integration_name}'
        """

        try:
            df = self._execute_query(query)
            logger.info(f"Found {len(df)} table constraints before filtering")

            # Apply table filtering after fetching if specified
            if self.tables_filter and not df.empty:
                # Case-insensitive filtering
                tables_filter_lower = [table.lower() for table in self.tables_filter]
                df = df[df["TABLE_NAME"].str.lower().isin(tables_filter_lower)]
                logger.info(f"Found {len(df)} table constraints after filtering")
            else:
                logger.info(
                    f"No table filter applied - keeping all {len(df)} table constraints"
                )

            return df
        except Exception as e:
            logger.warning(f"Failed to get table constraints: {str(e)}")
            return pd.DataFrame()

    def _get_handler_info(self) -> Optional[str]:
        """Get handler information from META_HANDLER_INFO."""
        query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_HANDLER_INFO 
        WHERE TABLE_SCHEMA = '{self.integration_name}'
        """
        try:
            df = self._execute_query(query)
            if not df.empty and "HANDLER_INFO" in df.columns:
                return df.iloc[0]["HANDLER_INFO"]
        except Exception as e:
            logger.warning(f"Failed to get handler info: {str(e)}")
        return None

    def _build_column_statistics(
        self, stats_df: pd.DataFrame, table_name: str, column_name: str
    ) -> Optional[ColumnStatistics]:
        """Build ColumnStatistics from META_COLUMN_STATISTICS data."""
        if stats_df.empty:
            return None

        # Find the row for this table/column
        matching_rows = stats_df[
            (stats_df["TABLE_NAME"] == table_name)
            & (stats_df["COLUMN_NAME"] == column_name)
        ]

        if matching_rows.empty:
            return None

        row = matching_rows.iloc[0]

        # Handle MindsDB's array format - convert empty arrays [""] to None
        def normalize_array(val):
            if isinstance(val, list) and val == [""]:
                return None
            return val if isinstance(val, list) else None

        # Helper function to handle NaN values
        def normalize_numeric(val):
            """Convert pandas NaN to None, keep valid numbers as is."""
            if pd.isna(val):
                return None
            return val

        # Helper function to handle distinct_values_count specifically
        def normalize_distinct_count(val):
            """Convert distinct values count to proper integer or None."""
            if pd.isna(val):
                return None
            try:
                return int(val) if val is not None else None
            except (ValueError, TypeError):
                return None

        return ColumnStatistics(
            most_common_values=normalize_array(row.get("MOST_COMMON_VALS")),
            most_common_frequencies=normalize_array(row.get("MOST_COMMON_FREQS")),
            null_percentage=normalize_numeric(row.get("NULL_FRAC")),
            distinct_values_count=normalize_distinct_count(row.get("N_DISTINCT")),
            min_value=normalize_numeric(row.get("MIN_VALUE")),
            max_value=normalize_numeric(row.get("MAX_VALUE")),
        )

    def _build_columns(
        self, columns_df: pd.DataFrame, stats_df: pd.DataFrame, table_name: str
    ) -> List[Column]:
        """Build Column objects from META_COLUMNS data."""
        columns = []

        # Filter columns for this table
        table_columns = columns_df[columns_df["TABLE_NAME"] == table_name]

        for _, col_row in table_columns.iterrows():
            # Build column statistics
            statistics = self._build_column_statistics(
                stats_df, table_name, col_row["COLUMN_NAME"]
            )

            column = Column(
                name=col_row["COLUMN_NAME"],
                data_type=col_row["DATA_TYPE"],
                is_nullable=self._normalize_boolean(col_row["IS_NULLABLE"]),
                column_default=self._normalize_null_value(
                    col_row.get("COLUMN_DEFAULT")
                ),
                description=col_row.get("COLUMN_DESCRIPTION"),
                statistics=statistics,
            )
            columns.append(column)

        return columns

    def _build_primary_keys(
        self, pk_df: pd.DataFrame, table_name: str
    ) -> Optional[PrimaryKey]:
        """Build PrimaryKey from META_KEY_COLUMN_USAGE data."""
        if pk_df.empty:
            return None

        # Filter primary keys for this table
        table_pks = pk_df[pk_df["TABLE_NAME"] == table_name]

        if table_pks.empty:
            return None

        # Group by constraint name and collect column names
        constraint_groups = table_pks.groupby("CONSTRAINT_NAME")

        for constraint_name, group in constraint_groups:
            column_names = group["COLUMN_NAME"].tolist()
            return PrimaryKey(
                constraint_name=constraint_name, column_names=column_names
            )

        return None

    def _build_tables(
        self,
        tables_df: pd.DataFrame,
        columns_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        pk_df: pd.DataFrame,
    ) -> Dict[str, Table]:
        """Build Table objects from metadata DataFrames."""
        logger.info(f"Building table objects for integration '{self.integration_name}'")
        tables = {}

        for _, table_row in tables_df.iterrows():
            table_name = table_row["TABLE_NAME"]
            logger.info(
                f"Building table '{table_name}' for integration '{self.integration_name}'"
            )

            # Build columns for this table
            columns = self._build_columns(columns_df, stats_df, table_name)
            logger.info(
                f"Built {len(columns)} columns for table '{self.integration_name}.{table_name}'"
            )

            # Build primary key for this table
            primary_key = self._build_primary_keys(pk_df, table_name)
            if primary_key:
                logger.info(
                    f"Found primary key for table '{self.integration_name}.{table_name}': {primary_key.column_names}"
                )
            else:
                logger.info(
                    f"No primary key found for table '{self.integration_name}.{table_name}'"
                )

            # Build table description with row count if available
            description = table_row.get("TABLE_DESCRIPTION")
            row_count = table_row.get("ROW_COUNT")
            if row_count and not description:
                description = f"Table with {row_count:,} rows"
            elif row_count and description:
                description += f" ({row_count:,} rows)"

            table = Table(
                name=table_name,
                description=description,
                columns=columns,
                primary_key=primary_key,
                foreign_keys=[],  # MindsDB typically doesn't have explicit foreign keys
            )

            tables[table_name] = table
            logger.info(
                f"Successfully built table '{self.integration_name}.{table_name}' with {len(columns)} columns"
            )

        logger.info(
            f"Successfully built {len(tables)} tables for integration '{self.integration_name}'"
        )
        return tables

    def load(self) -> MindsDBDataCatalog:
        """
        Load a MindsDB data catalog by querying INFORMATION_SCHEMA tables.

        Returns:
            A MindsDBDataCatalog instance with comprehensive metadata

        Raises:
            Exception: For MindsDB connection or query errors
        """
        try:
            logger.info(
                f"Starting MindsDB data catalog loading for integration: '{self.integration_name}'"
            )

            # Get all metadata in parallel-safe manner
            logger.info(
                f"Fetching metadata for integration '{self.integration_name}'..."
            )
            tables_df = self._get_tables_metadata()
            columns_df = self._get_columns_metadata()
            stats_df = self._get_column_statistics()
            pk_df = self._get_primary_keys()
            handler_info = self._get_handler_info()
            logger.info(
                f"Completed metadata fetching for integration '{self.integration_name}'"
            )

            # Build tables with all metadata
            logger.info(
                f"Building data catalog structure for integration '{self.integration_name}'"
            )
            tables = self._build_tables(tables_df, columns_df, stats_df, pk_df)

            # Create the catalog
            catalog = MindsDBDataCatalog(
                integration_name=self.integration_name,
                engine=self.engine
                or "MindsDB",  # Use original engine if available, fallback to "MindsDB"
                handler_info=handler_info,
                tables=tables,
            )

            logger.info(
                f"Successfully loaded MindsDB data catalog for integration '{self.integration_name}' with {len(tables)} tables"
            )
            if handler_info:
                logger.info(
                    f"Handler info for integration '{self.integration_name}': {handler_info[:100]}..."
                )

            # Log table names for debugging
            table_names = list(tables.keys())
            logger.info(
                f"Tables in integration '{self.integration_name}': {table_names}"
            )

            return catalog

        except Exception as e:
            logger.error(
                f"Failed to load MindsDB data catalog for integration '{self.integration_name}': {str(e)}",
                exc_info=True,
            )
            raise
