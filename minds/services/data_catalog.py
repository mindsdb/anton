from typing import Any

import pandas as pd
from mindsdb_sdk.server import Server
from sqlmodel import Session, and_, select

from minds.common.logger import setup_logging
from minds.model.data_catalog import Column, ColumnStatistics, ForeignKeyConstraint, PrimaryKeyConstraint, Table
from minds.model.datasource import Datasource

logger = setup_logging()


class DataCatalogLoader:
    """A class for loading data catalogs."""

    def __init__(self, session: Session, mindsdb_client: Server, user_id: str):
        """
        Initialize the MindsDB data catalog loader.

        Args:
            session: Database session for internal storage.
            mindsdb_client: MindsDB server instance from mindsdb_sdk.connect().
            user_id: Current user ID.
        """
        self.session = session
        self.mindsdb_client = mindsdb_client
        self.user_id = user_id

    async def load(self, datasource_name: str, table_names: list[str] | None = None) -> None:
        """Load the data catalog."""
        try:
            statement = select(Datasource).where(
                and_(
                    Datasource.name == datasource_name,
                    Datasource.user_id == self.user_id,
                    Datasource.deleted_at.is_(None),
                )
            )
            result = self.session.exec(statement)
            datasource = result.first()

            tables_df = self._get_tables(datasource, table_names)

            if len(tables_df) > 0:
                tables = self._load_tables(datasource, tables_df)

                engine_info = self._get_engine_info(datasource)
                if engine_info:
                    datasource.engine_info = engine_info

                columns_df = self._get_columns(datasource, table_names)
                columns = self._load_columns(columns_df, tables)

                column_statistics_df = self._get_column_statistics(datasource, table_names)
                if len(column_statistics_df) > 0:
                    self._load_column_statistics(column_statistics_df, columns)

                primary_keys_df = self._get_primary_keys(datasource, table_names)
                if len(primary_keys_df) > 0:
                    self._load_primary_keys(primary_keys_df, tables, columns)

                foreign_keys_df = self._get_foreign_keys(datasource, table_names)
                if len(foreign_keys_df) > 0:
                    self._load_foreign_keys(foreign_keys_df, tables, columns)

            # Only commit if everything succeeded
            self.session.commit()
            logger.info("Successfully committed all data catalog information to database")

        except Exception as e:
            # Rollback all changes if any error occurs
            self.session.rollback()
            logger.error(f"Error loading data catalog: {str(e)}. All changes have been rolled back.")
            raise

    def _execute_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query against MindsDB and return DataFrame."""
        try:
            logger.info(f"Executing MindsDB query: {query}")
            query_result = self.mindsdb_client.query(query)
            df = query_result.fetch()
            logger.info(f"MindsDB query returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to execute MindsDB query: {query}. Error: {str(e)}")
            raise

    def _normalize_boolean(self, value: Any) -> bool:
        """Convert MindsDB YES/NO strings to boolean."""
        if isinstance(value, str):
            return value.upper() == "YES"
        return bool(value)

    def _normalize_null_value(self, value: Any) -> Any | None:
        """Convert MindsDB [NULL] strings to None."""
        if value == "[NULL]" or value is None:
            return None
        return value

    def _get_engine_info(self, datasource: Datasource) -> str:
        """Get engine info from datasource."""
        logger.info(f"Getting engine info for datasource '{datasource.name}'")

        query = f"""
        SELECT HANDLER_INFO FROM INFORMATION_SCHEMA.META_HANDLER_INFO
        WHERE TABLE_SCHEMA = '{datasource.name}'
        """

        df = self._execute_query(query)
        logger.info(f"Found {len(df)} engine info")

        return df["HANDLER_INFO"].iloc[0]

    def _get_tables(self, datasource: Datasource, table_names: list[str] | None = None) -> pd.DataFrame:
        """Get table metadata from META_TABLES with optional filtering."""
        logger.info(f"Getting table metadata for datasource '{datasource.name}' with filter: {table_names}")

        query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_TABLES 
        WHERE TABLE_SCHEMA = '{datasource.name}'
        """

        if table_names:
            query += f" AND TABLE_NAME IN ({', '.join(f'{name!r}' for name in table_names)})"

        df = self._execute_query(query)
        logger.info(f"Found {len(df)} tables")

        # Check if the returned tables are already in the database.
        existing_tables = self.session.exec(
            select(Table).where(and_(Table.datasource_id == datasource.id, Table.name.in_(df["TABLE_NAME"])))
        ).all()
        logger.info(f"{len(existing_tables)} have already been loaded")

        existing_tables_names = [table.name for table in existing_tables]
        df = df[~df["TABLE_NAME"].isin(existing_tables_names)]

        logger.info(f"Found {len(df)} tables to load")

        return df

    def _get_columns(self, datasource: Datasource, table_names: list[str] | None = None) -> pd.DataFrame:
        """Get column metadata from META_COLUMNS with optional filtering."""
        logger.info(f"Getting column metadata for datasource '{datasource.name}' with filter: {table_names}")

        query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_COLUMNS 
        WHERE TABLE_SCHEMA = '{datasource.name}'
        """

        if table_names:
            query += f" AND TABLE_NAME IN ({', '.join(f'{name!r}' for name in table_names)})"

        df = self._execute_query(query)
        logger.info(f"Found {len(df)} columns")

        return df

    def _get_column_statistics(self, datasource: Datasource, table_names: list[str] | None = None) -> pd.DataFrame:
        """Get column statistics from META_COLUMN_STATISTICS with optional filtering."""
        logger.info(f"Getting column statistics for datasource '{datasource.name}' with filter: {table_names}")

        query = f"""
        SELECT * FROM INFORMATION_SCHEMA.META_COLUMN_STATISTICS 
        WHERE TABLE_SCHEMA = '{datasource.name}'
        """

        if table_names:
            query += f" AND TABLE_NAME IN ({', '.join(f'{name!r}' for name in table_names)})"

        df = self._execute_query(query)
        logger.info(f"Found {len(df)} column statistics")

        return df

    def _get_primary_keys(self, datasource: Datasource, table_names: list[str] | None = None) -> pd.DataFrame:
        """Get primary key information from META_KEY_COLUMN_USAGE with optional filtering."""
        logger.info(f"Getting primary key information for datasource '{datasource.name}' with filter: {table_names}")

        # TODO: This query is hacky. It is written to allow it to run on MindsDB.
        query = f"""
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
            AND kcu.TABLE_SCHEMA = '{datasource.name}'
            AND tc.TABLE_SCHEMA = '{datasource.name}'
        """

        if table_names:
            query += f" AND kcu.TABLE_NAME IN ({', '.join(f'{name!r}' for name in table_names)})"

        df = self._execute_query(query)
        logger.info(f"Found {len(df)} primary keys")

        return df

    def _get_foreign_keys(self, datasource: Datasource, table_names: list[str] | None = None) -> pd.DataFrame:
        """Get table constraints from META_TABLE_CONSTRAINTS with optional filtering."""
        logger.info(f"Getting foreign key information for datasource '{datasource.name}' with filter: {table_names}")

        # TODO: This query is hacky. It is written to allow it to run on MindsDB.
        query = f"""
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
            AND kcu.TABLE_SCHEMA = '{datasource.name}'
            AND tc.TABLE_SCHEMA = '{datasource.name}'
        """

        if table_names:
            query += f" AND kcu.TABLE_NAME IN ({', '.join(f'{name!r}' for name in table_names)})"

        df = self._execute_query(query)
        logger.info(f"Found {len(df)} foreign keys")

        return df

    def _load_tables(self, datasource: Datasource, tables_df: pd.DataFrame) -> list[Table]:
        """Load tables from metadata."""
        logger.info(f"Loading {len(tables_df)} tables into the database")

        tables = []
        for _, row in tables_df.iterrows():
            row_count = row.get("ROW_COUNT")
            row_count = int(row_count) if pd.notna(row_count) else None

            table = Table(
                datasource_id=datasource.id,
                name=row["TABLE_NAME"],
                schema=row["TABLE_SCHEMA"],
                description=row["TABLE_DESCRIPTION"],
                type=row["TABLE_TYPE"],
                row_count=row_count,
            )

            tables.append(table)

        self.session.add_all(tables)
        self.session.flush()  # Generate IDs but don't commit yet
        logger.info(f"Loaded {len(tables)} tables into the database")

        return tables

    def _load_columns(self, columns_df: pd.DataFrame, tables: list[Table]) -> list[Column]:
        """Load columns from metadata."""
        logger.info(f"Loading {len(columns_df)} columns into the database")

        # Add table ids to columns_df.
        columns_df["TABLE_ID"] = columns_df["TABLE_NAME"].map(
            lambda name: next((table.id for table in tables if table.name == name), None)
        )

        columns = []
        for _, row in columns_df.iterrows():
            column = Column(
                table_id=row["TABLE_ID"],
                name=row["COLUMN_NAME"],
                data_type=row["DATA_TYPE"],
                description=row["COLUMN_DESCRIPTION"],
                default_value=self._normalize_null_value(row["COLUMN_DEFAULT"]),
                is_nullable=self._normalize_boolean(row["IS_NULLABLE"]),
            )

            columns.append(column)

        self.session.add_all(columns)
        self.session.flush()  # Generate IDs but don't commit yet
        logger.info(f"Loaded {len(columns)} columns into the database")
        return columns

    def _load_column_statistics(self, column_statistics_df: pd.DataFrame, columns: list[Column]) -> None:
        """Load column statistics from metadata."""
        logger.info(f"Loading {len(column_statistics_df)} column statistics into the database")

        # Add column ids to column_statistics_df.
        column_statistics_df["COLUMN_ID"] = column_statistics_df["COLUMN_NAME"].map(
            lambda name: next((column.id for column in columns if column.name == name), None)
        )

        def normalize_distinct_count(val: Any) -> int | None:
            """Convert distinct values count to proper integer or None."""
            if pd.isna(val):
                return None
            try:
                return int(val) if val is not None else None
            except (ValueError, TypeError):
                return None

        column_statistics = []
        for _, row in column_statistics_df.iterrows():
            column_statistics.append(
                ColumnStatistics(
                    column_id=row["COLUMN_ID"],
                    most_common_values=row["MOST_COMMON_VALS"],
                    most_common_frequencies=row["MOST_COMMON_FREQS"],
                    null_percentage=row["NULL_FRAC"],
                    distinct_values_count=normalize_distinct_count(row["N_DISTINCT"]),
                    min_value=row["MIN_VALUE"],
                    max_value=row["MAX_VALUE"],
                )
            )

        self.session.add_all(column_statistics)
        self.session.flush()  # Generate IDs but don't commit yet
        logger.info(f"Loaded {len(column_statistics)} column statistics into the database")

    def _load_primary_keys(self, primary_keys_df: pd.DataFrame, tables: list[Table], columns: list[Column]) -> None:
        """Load primary keys from metadata."""
        logger.info(f"Loading {len(primary_keys_df)} primary keys into the database")

        # Add table and column ids to primary_keys_df.
        primary_keys_df["TABLE_ID"] = primary_keys_df["TABLE_NAME"].map(
            lambda name: next((table.id for table in tables if table.name == name), None)
        )
        primary_keys_df["COLUMN_ID"] = primary_keys_df["COLUMN_NAME"].map(
            lambda name: next((column.id for column in columns if column.name == name), None)
        )

        primary_keys = []
        for _, row in primary_keys_df.iterrows():
            primary_keys.append(
                PrimaryKeyConstraint(
                    table_id=row["TABLE_ID"],
                    column_id=row["COLUMN_ID"],
                    ordinal_position=row["ORDINAL_POSITION"],
                    constraint_name=row["CONSTRAINT_NAME"],
                )
            )

        self.session.add_all(primary_keys)
        self.session.flush()  # Generate IDs but don't commit yet
        logger.info(f"Loaded {len(primary_keys)} primary keys into the database")

    def _load_foreign_keys(self, foreign_keys_df: pd.DataFrame, tables: list[Table], columns: list[Column]) -> None:
        """Load foreign keys from metadata."""
        logger.info(f"Loading {len(foreign_keys_df)} foreign keys into the database")

        # Add table and column ids to foreign_keys_df.
        foreign_keys_df["TABLE_ID"] = foreign_keys_df["TABLE_NAME"].map(
            lambda name: next((table.id for table in tables if table.name == name), None)
        )
        foreign_keys_df["COLUMN_ID"] = foreign_keys_df["COLUMN_NAME"].map(
            lambda name: next((column.id for column in columns if column.name == name), None)
        )
        foreign_keys_df["REFERENCED_TABLE_ID"] = foreign_keys_df["REFERENCED_TABLE_NAME"].map(
            lambda name: next((table.id for table in tables if table.name == name), None)
        )
        foreign_keys_df["REFERENCED_COLUMN_ID"] = foreign_keys_df["REFERENCED_COLUMN_NAME"].map(
            lambda name: next((column.id for column in columns if column.name == name), None)
        )

        foreign_keys = []
        for _, row in foreign_keys_df.iterrows():
            foreign_keys.append(
                ForeignKeyConstraint(
                    table_id=row["TABLE_ID"],
                    column_id=row["COLUMN_ID"],
                    referenced_table_id=row["REFERENCED_TABLE_ID"],
                    referenced_column_id=row["REFERENCED_COLUMN_ID"],
                    constraint_name=row["CONSTRAINT_NAME"],
                    ordinal_position=row["ORDINAL_POSITION"],
                )
            )

        self.session.add_all(foreign_keys)
        self.session.flush()  # Generate IDs but don't commit yet
        logger.info(f"Loaded {len(foreign_keys)} foreign keys into the database")
