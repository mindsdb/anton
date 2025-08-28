from typing import Any, Optional, List

from mindsdb_sdk.server import Server
import pandas as pd
from sqlmodel import Session

from minds.common.logger import setup_logging
from minds.model.data_catalog import Column, ColumnStatistics, Table, PrimaryKeyConstraint, ForeignKeyConstraint
from minds.services.datasources import DatasourcesService


logger = setup_logging()


class DataCatalogLoader:
    """A class for loading data catalogs."""

    def __init__(
        self,
        datasource_name: str,
        mindsdb_client: Server,
        session: Session,
        user_id: str,
        company_id: str,
        tables_filter: Optional[List[str]] = None,
    ):
        """
        Initialize the MindsDB data catalog loader.

        Args:
            datasource_name: Name of the MindsDB datasource.
            mindsdb_client: MindsDB server instance from mindsdb_sdk.connect().
            session: Database session for internal storage.
            user_id: Current user ID.
            company_id: Current company ID.
            tables_filter: Optional list of table names to filter (empty list = no filtering).
        """
        self.datasource_name = datasource_name.strip()
        self.mindsdb_client = mindsdb_client
        self.tables_filter = tables_filter or []
        self.session = session
        self.user_id = user_id
        self.company_id = company_id

    def _execute_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query against MindsDB and return DataFrame."""
        try:
            logger.info(f"Executing MindsDB query for datasource '{self.datasource_name}': {query}")
            query_result = self.mindsdb_client.query(query)
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
        query = """
        SELECT 
            kcu.TABLE_NAME,
            kcu.COLUMN_NAME,
            kcu.ORDINAL_POSITION,
            kcu.CONSTRAINT_NAME
        FROM INFORMATION_SCHEMA.META_KEY_COLUMN_USAGE kcu
        INNER JOIN INFORMATION_SCHEMA.META_TABLE_CONSTRAINTS tc 
            ON kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
            AND kcu.TABLE_NAME = tc.TABLE_NAME
            AND kcu.TABLE_SCHEMA = tc.TABLE_SCHEMA
            AND kcu.CONSTRAINT_CATALOG = tc.CONSTRAINT_CATALOG
        WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
            AND kcu.CONSTRAINT_CATALOG = '{self.datasource_name}'
        ORDER BY kcu.TABLE_NAME, kcu.ORDINAL_POSITION;
        """

        if self.tables_filter:
            query += f" AND TABLE_NAME IN ({', '.join(f"'{name}'" for name in self.tables_filter)})"

        df = self._execute_query(query)
        logger.info(f"Found {len(df)} primary keys before filtering")

        return df

    def _get_table_constraints(self) -> pd.DataFrame:
        """Get table constraints from META_TABLE_CONSTRAINTS with optional filtering."""
        query = f"""
        SELECT 
            kcu.TABLE_NAME,
            kcu.COLUMN_NAME,
            kcu.ORDINAL_POSITION,
            kcu.CONSTRAINT_NAME,
            kcu.REFERENCED_TABLE_NAME,
            kcu.REFERENCED_COLUMN_NAME
        FROM INFORMATION_SCHEMA.META_KEY_COLUMN_USAGE kcu
        INNER JOIN INFORMATION_SCHEMA.META_TABLE_CONSTRAINTS tc 
            ON kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
            AND kcu.TABLE_NAME = tc.TABLE_NAME
            AND kcu.TABLE_SCHEMA = tc.TABLE_SCHEMA
            AND kcu.CONSTRAINT_CATALOG = tc.CONSTRAINT_CATALOG
        WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
            AND kcu.CONSTRAINT_CATALOG = '{self.datasource_name}'
        ORDER BY kcu.TABLE_NAME, kcu.ORDINAL_POSITION;
        """

        if self.tables_filter:
            query += f" AND TABLE_NAME IN ({', '.join(f"'{name}'" for name in self.tables_filter)})"

        df = self._execute_query(query)
        logger.info(f"Found {len(df)} table constraints before filtering")

        return df

    def _load_tables(self, tables_df: pd.DataFrame) -> List[Table]:
        """Load tables from metadata."""
        datasource_service = DatasourcesService(
            session=self.session,
            mindsdb_client=self.mindsdb_client,
            user_id=self.user_id,
            company_id=self.company_id
        )
        datasource = datasource_service.get_datasource(self.datasource_name)

        tables = []
        for _, row in tables_df.iterrows():
            row_count = row.get("ROW_COUNT")
            row_count = int(row_count) if pd.notna(row_count) else None

            table = Table(
                datasource_id=datasource.id,
                name=row['TABLE_NAME'],
                schema=row['TABLE_SCHEMA'],
                description=row['TABLE_DESCRIPTION'],
                type=row['TABLE_TYPE'],
                row_count=row_count,
            )

            tables.append(table)

        self.session.add_all(tables)
        self.session.commit()

        return tables

    def _load_columns(self, columns_df: pd.DataFrame, tables: List[Table]) -> List[Column]:
        """Load columns from metadata."""
        # Add table ids to columns_df.
        columns_df['TABLE_ID'] = columns_df['TABLE_NAME'].map(lambda name: next((table.id for table in tables if table.name == name), None))

        columns = []
        for _, row in columns_df.iterrows():
            column = Column(
                table_id=row['TABLE_ID'],
                name=row['COLUMN_NAME'],
                data_type=row['DATA_TYPE'],
                description=row['COLUMN_DESCRIPTION'],
                default_value=self._normalize_null_value(row['COLUMN_DEFAULT']),
                is_nullable=self._normalize_boolean(row['IS_NULLABLE']),
            )

            columns.append(column)

        self.session.add_all(columns)
        self.session.commit()

        return columns
    
    def _load_column_statistics(self, column_statistics_df: pd.DataFrame, columns: List[Column]) -> None:
        """Load column statistics from metadata."""
        # Add column ids to column_statistics_df.
        column_statistics_df['COLUMN_ID'] = column_statistics_df['COLUMN_NAME'].map(lambda name: next((column.id for column in columns if column.name == name), None))

        column_statistics = []
        for _, row in column_statistics_df.iterrows():
            column_statistics.append(ColumnStatistics(
                column_id=row['COLUMN_ID'],
                most_common_values=row['MOST_COMMON_VALS'],
                most_common_frequencies=row['MOST_COMMON_FREQS'],
                null_percentage=row['NULL_FRAC'],
                distinct_values_count=row['N_DISTINCT'],
                min_value=row['MIN_VALUE'],
                max_value=row['MAX_VALUE'],
            ))

        self.session.add_all(column_statistics)
        self.session.commit()

    def _load_primary_keys(self, primary_keys_df: pd.DataFrame, tables: List[Table], columns: List[Column]) -> None:
        """Load primary keys from metadata."""
        # Add table and column ids to primary_keys_df.
        primary_keys_df['TABLE_ID'] = primary_keys_df['TABLE_NAME'].map(lambda name: next((table.id for table in tables if table.name == name), None))
        primary_keys_df['COLUMN_ID'] = primary_keys_df['COLUMN_NAME'].map(lambda name: next((column.id for column in columns if column.name == name), None))

        primary_keys = []
        for _, row in primary_keys_df.iterrows():
            primary_keys.append(PrimaryKeyConstraint(
                table_id=row['TABLE_ID'],
                column_id=row['COLUMN_ID'],
                ordinal_position=row['ORDINAL_POSITION'],
                constraint_name=row['CONSTRAINT_NAME'],
            ))

        self.session.add_all(primary_keys)
        self.session.commit()

    def _load_foreign_keys(self, foreign_keys_df: pd.DataFrame, tables: List[Table], columns: List[Column]) -> None:
        """Load foreign keys from metadata."""
        # Add table and column ids to foreign_keys_df.
        foreign_keys_df['TABLE_ID'] = foreign_keys_df['TABLE_NAME'].map(lambda name: next((table.id for table in tables if table.name == name), None))
        foreign_keys_df['COLUMN_ID'] = foreign_keys_df['COLUMN_NAME'].map(lambda name: next((column.id for column in columns if column.name == name), None))
        foreign_keys_df['REFERENCED_TABLE_ID'] = foreign_keys_df['REFERENCED_TABLE_NAME'].map(lambda name: next((table.id for table in tables if table.name == name), None))
        foreign_keys_df['REFERENCED_COLUMN_ID'] = foreign_keys_df['REFERENCED_COLUMN_NAME'].map(lambda name: next((column.id for column in columns if column.name == name), None))

        foreign_keys = []
        for _, row in foreign_keys_df.iterrows():
            foreign_keys.append(ForeignKeyConstraint(
                table_id=row['TABLE_ID'],
                column_id=row['COLUMN_ID'],
                referenced_table_id=row['REFERENCED_TABLE_ID'],
                referenced_column_id=row['REFERENCED_COLUMN_ID'],
                constraint_name=row['CONSTRAINT_NAME'],
                ordinal_position=row['ORDINAL_POSITION'],
            ))

        self.session.add_all(foreign_keys)
        self.session.commit()
