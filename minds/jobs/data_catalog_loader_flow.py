"""
Data catalog loader service.
This service is implemented in a function style to support both
synchronous and asynchronous execution modes.

Asynchronous executions are run via Prefect.
"""

import pandas as pd
from typing import Any
from uuid import UUID

from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from sqlalchemy.orm import selectinload
from sqlmodel import select, and_, Session
from mindsdb_sdk.server import Server

from minds.client.mindsdb import create_mindsdb_client_from_env
from minds.common.logger import setup_logging
from minds.db.pg_session import get_session
from minds.model.data_catalog import Column, ColumnStatistics, ForeignKeyConstraint, PrimaryKeyConstraint, Table
from minds.model.mind_datasource import DataCatalogStatus, MindDatasource
from minds.model.mind_datasource_table import MindDatasourceTable


logger = setup_logging()


class DataCatalogLoaderError(Exception):
    """Base exception for data catalog loader errors."""

    pass


@flow
def load_data_catalog(
    mind_datasource_id: UUID,
    tenant_id: str,
    table_names: list[str] | None = None,
) -> None:
    """
    Load data catalog information from MindsDB into the local database.

    Args:
        mind_datasource_id (UUID): The ID of the mind datasource.
        tenant_id (str): The ID of the tenant.
        table_names (list[str] | None): Optional list of table names to filter by. If None, load all tables.

    Raises:
        DataCatalogLoaderError: If there is an error during the loading process.
    """
    try:
        # Create a database session
        session_generator = get_session()
        session = next(session_generator)

        # Create a MindsDB client
        mindsdb_client = create_mindsdb_client_from_env()

        # Get the MindDatasource object from the database
        statement = (
            select(MindDatasource)
            .where(
                and_(
                    MindDatasource.id == mind_datasource_id,
                    MindDatasource.tenant_id == tenant_id,
                )
            )
            .options(
                selectinload(MindDatasource.datasource),
            )
        )
        mind_datasource = session.exec(statement).first()

        datasource_id = mind_datasource.datasource_id
        datasource_name = mind_datasource.datasource.name

        mind_datasource.status = DataCatalogStatus.LOADING
        session.add(mind_datasource)
        session.commit()

        tables_df = get_tables(mindsdb_client, datasource_name, table_names)
        tables_df = filter_loaded_tables(session, tables_df, datasource_id, tenant_id)

        if len(tables_df) > 0:
            tables = load_tables(session, tables_df, mind_datasource_id, datasource_id, tenant_id)

            columns_df = get_columns(mindsdb_client, datasource_name, table_names)
            columns = load_columns(session, columns_df, tables, tenant_id)

            column_statistics_df = get_column_statistics(mindsdb_client, datasource_name, table_names)
            if len(column_statistics_df) > 0:
                load_column_statistics(session, column_statistics_df, columns, tenant_id)

            primary_keys_df = get_primary_keys(mindsdb_client, datasource_name, table_names)
            if len(primary_keys_df) > 0:
                load_primary_keys(session, primary_keys_df, tables, columns, tenant_id)

            foreign_keys_df = get_foreign_keys(mindsdb_client, datasource_name, table_names)
            if len(foreign_keys_df) > 0:
                load_foreign_keys(session, foreign_keys_df, tables, columns, tenant_id)

        # Only commit if everything succeeded
        session.commit()
        logger.info("Successfully committed all data catalog information to database")

        mind_datasource.status = DataCatalogStatus.COMPLETED
        session.add(mind_datasource)
        session.commit()
    except Exception as e:
        session.rollback()
        mind_datasource.status = DataCatalogStatus.FAILED
        session.add(mind_datasource)
        session.commit()
        err_message = f"Failed to load data catalog for MindDatasource ID {mind_datasource_id}: {str(e)}"
        logger.error(f"Failed to load data catalog: {err_message}")
        raise DataCatalogLoaderError(f"Failed to load data catalog: {err_message}")


def _execute_mindsdb_query(mindsdb_client: Server, query: str) -> pd.DataFrame:
    """
    Execute a SQL query against MindsDB and return DataFrame.

    Args:
        mindsdb_client (Server): MindsDB client for executing queries.
        query (str): The SQL query to execute.

    Returns:
        pd.DataFrame: The result of the query as a DataFrame.
    """
    try:
        logger.info(f"Executing MindsDB query: {query}")
        query_result = mindsdb_client.query(query)
        df = query_result.fetch()
        logger.info(f"MindsDB query returned {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to execute MindsDB query: {query}. Error: {str(e)}")
        raise


@task
def get_tables(mindsdb_client: Server, datasource_name: str, table_names: list[str] | None = None) -> pd.DataFrame:
    """
    Get table metadata from MindsDB (INFORMATION_SCHEMA.META_TABLES) for a given datasource.

    Args:
        mindsdb_client (Server): MindsDB client for querying metadata.
        datasource_name (str): The name of the datasource (schema).
        table_names (list[str] | None): Optional list of table names to filter by.

    Returns:
        pd.DataFrame: DataFrame containing table metadata.
    """
    logger.info(f"Getting table metadata for datasource '{datasource_name}' with filter: {table_names}")

    query = f"""
    SELECT * FROM INFORMATION_SCHEMA.META_TABLES 
    WHERE TABLE_SCHEMA = '{datasource_name}'
    """

    if table_names:
        query += f" AND TABLE_NAME IN ({', '.join(f'{name!r}' for name in table_names)})"

    df = _execute_mindsdb_query(mindsdb_client, query)
    logger.info(f"Found {len(df)} tables")

    return df


@task(cache_policy=NO_CACHE)
def filter_loaded_tables(
    session: Session, tables_df: pd.DataFrame, datasource_id: UUID, tenant_id: UUID
) -> pd.DataFrame:
    """
    Filter out tables that have already been loaded.

    Args:
        session (Session): Database session for querying existing tables.
        tables_df (pd.DataFrame): DataFrame containing all tables from MindsDB.
        datasource_id (UUID): The ID of the datasource.
        tenant_id (UUID): The ID of the tenant.

    Returns:
        pd.DataFrame: DataFrame containing only tables that haven't been loaded yet.
    """
    existing_tables = session.exec(
        select(Table).where(
            and_(
                Table.datasource_id == datasource_id,
                Table.name.in_(tables_df["TABLE_NAME"]),
                Table.tenant_id == tenant_id,
                Table.deleted_at.is_(None),
            )
        )
    ).all()
    logger.info(f"{len(existing_tables)} have already been loaded")
    existing_tables_names = [table.name for table in existing_tables]
    return tables_df[~tables_df["TABLE_NAME"].isin(existing_tables_names)]


@task
def get_columns(mindsdb_client: Server, datasource_name: str, table_names: list[str] | None = None) -> pd.DataFrame:
    """
    Get column metadata from MindsDB (INFORMATION_SCHEMA.META_COLUMNS) for a given datasource.

    Args:
        mindsdb_client (Server): MindsDB client for querying metadata.
        datasource_name (str): The name of the datasource (schema).
        table_names (list[str] | None): Optional list of table names to filter by.

    Returns:
        pd.DataFrame: DataFrame containing column metadata.
    """
    logger.info(f"Getting column metadata for datasource '{datasource_name}' with filter: {table_names}")

    query = f"""
    SELECT * FROM INFORMATION_SCHEMA.META_COLUMNS 
    WHERE TABLE_SCHEMA = '{datasource_name}'
    """

    if table_names:
        query += f" AND TABLE_NAME IN ({', '.join(f'{name!r}' for name in table_names)})"

    df = _execute_mindsdb_query(mindsdb_client, query)
    logger.info(f"Found {len(df)} columns")

    return df


@task
def get_column_statistics(
    mindsdb_client: Server, datasource_name: str, table_names: list[str] | None = None
) -> pd.DataFrame:
    """
    Get column statistics from MindsDB (INFORMATION_SCHEMA.META_COLUMN_STATISTICS) for a given datasource.

    Args:
        mindsdb_client (Server): MindsDB client for querying metadata.
        datasource_name (str): The name of the datasource (schema).
        table_names (list[str] | None): Optional list of table names to filter by.

    Returns:
        pd.DataFrame: DataFrame containing column statistics.
    """
    logger.info(f"Getting column statistics for datasource '{datasource_name}' with filter: {table_names}")

    query = f"""
    SELECT * FROM INFORMATION_SCHEMA.META_COLUMN_STATISTICS 
    WHERE TABLE_SCHEMA = '{datasource_name}'
    """

    if table_names:
        query += f" AND TABLE_NAME IN ({', '.join(f'{name!r}' for name in table_names)})"

    df = _execute_mindsdb_query(mindsdb_client, query)
    logger.info(f"Found {len(df)} column statistics")

    return df


@task
def get_primary_keys(
    mindsdb_client: Server, datasource_name: str, table_names: list[str] | None = None
) -> pd.DataFrame:
    """
    Get primary keys from MindsDB (INFORMATION_SCHEMA.META_KEY_COLUMN_USAGE) for a given datasource.

    Args:
        mindsdb_client (Server): MindsDB client for querying metadata.
        datasource_name (str): The name of the datasource (schema).
        table_names (list[str] | None): Optional list of table names to filter by.

    Returns:
        pd.DataFrame: DataFrame containing primary key information.
    """
    logger.info(f"Getting primary keys for datasource '{datasource_name}' with filter: {table_names}")

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
        AND kcu.TABLE_SCHEMA = '{datasource_name}'
        AND tc.TABLE_SCHEMA = '{datasource_name}'
    """

    if table_names:
        query += f" AND kcu.TABLE_NAME IN ({', '.join(f'{name!r}' for name in table_names)})"

    df = _execute_mindsdb_query(mindsdb_client, query)
    logger.info(f"Found {len(df)} primary keys")

    return df


@task
def get_foreign_keys(
    mindsdb_client: Server, datasource_name: str, table_names: list[str] | None = None
) -> pd.DataFrame:
    """
    Get foreign keys from MindsDB (INFORMATION_SCHEMA.META_KEY_COLUMN_USAGE) for a given datasource.

    Args:
        mindsdb_client (Server): MindsDB client for querying metadata.
        datasource_name (str): The name of the datasource (schema).
        table_names (list[str] | None): Optional list of table names to filter by.

    Returns:
        pd.DataFrame: DataFrame containing foreign key information.
    """
    logger.info(f"Getting foreign key information for datasource '{datasource_name}' with filter: {table_names}")

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
        AND kcu.TABLE_SCHEMA = '{datasource_name}'
        AND tc.TABLE_SCHEMA = '{datasource_name}'
    """

    if table_names:
        query += f" AND kcu.TABLE_NAME IN ({', '.join(f'{name!r}' for name in table_names)})"

    df = _execute_mindsdb_query(mindsdb_client, query)
    logger.info(f"Found {len(df)} foreign keys")

    return df


@task(cache_policy=NO_CACHE)
def load_tables(
    session: Session, tables_df: pd.DataFrame, mind_datasource_id: UUID, datasource_id: UUID, tenant_id: UUID
) -> list[Table]:
    """
    Load tables into the database.

    Args:
        session (Session): Database session for database operations.
        tables_df (pd.DataFrame): The dataframe containing the tables to load.
        mind_datasource_id (UUID): The ID of the mind datasource.
        datasource_id (UUID): The ID of the datasource.
        tenant_id (UUID): The ID of the tenant.

    Returns:
        list[Table]: The list of tables that were loaded.
    """
    logger.info(f"Loading {len(tables_df)} tables into the database")

    tables = []
    mind_datasource_tables = []
    for _, row in tables_df.iterrows():
        table = _convert_row_to_table(row, tenant_id, datasource_id)
        session.add(table)
        session.flush()
        session.refresh(table)

        mind_datasource_table = MindDatasourceTable(
            tenant_id=tenant_id,
            mind_datasource_id=mind_datasource_id,
            table_id=table.id,
        )
        mind_datasource_tables.append(mind_datasource_table)
        tables.append(table)

    session.add_all(mind_datasource_tables)
    session.flush()  # Generate IDs but don't commit yet
    logger.info(f"Loaded {len(tables)} tables into the database")

    return tables


def _convert_row_to_table(row: pd.Series, tenant_id: UUID, datasource_id: UUID) -> Table:
    """
    Convert a metadata row to a Table object.

    Args:
        row (pd.Series): The metadata row to convert.
        tenant_id (UUID): The ID of the tenant.
        datasource_id (UUID): The ID of the datasource.

    Returns:
        Table: The Table object.
    """
    row_count = row.get("ROW_COUNT")
    row_count = int(row_count) if pd.notna(row_count) else None

    table = Table(
        tenant_id=tenant_id,
        datasource_id=datasource_id,
        name=row["TABLE_NAME"],
        schema=row["TABLE_SCHEMA"],
        description=row["TABLE_DESCRIPTION"],
        type=row["TABLE_TYPE"],
        row_count=row_count,
    )

    return table


@task(cache_policy=NO_CACHE)
def load_columns(session: Session, columns_df: pd.DataFrame, tables: list[Table], tenant_id: UUID) -> list[Column]:
    """
    Load columns into the database.

    Args:
        session (Session): Database session for database operations.
        columns_df (pd.DataFrame): The dataframe containing the columns to load.
        tables (list[Table]): The list of tables that the columns belong to.
        tenant_id (UUID): The ID of the tenant.

    Returns:
        list[Column]: The list of columns that were loaded.
    """
    logger.info(f"Loading {len(columns_df)} columns into the database")

    # Create a lookup dictionary for table name to table ID
    table_name_to_id = {table.name: table.id for table in tables}

    columns = []
    for _, row in columns_df.iterrows():
        table_id = table_name_to_id.get(row["TABLE_NAME"])
        column = _convert_row_to_column(row, tenant_id, table_id)
        columns.append(column)

    session.add_all(columns)
    session.flush()  # Generate IDs but don't commit yet
    logger.info(f"Loaded {len(columns)} columns into the database")
    return columns


def _normalize_boolean(value: Any) -> bool:
    """Convert MindsDB YES/NO strings to boolean."""
    if isinstance(value, str):
        return value.upper() == "YES"
    return bool(value)


def _normalize_null_value(value: Any) -> Any | None:
    """Convert MindsDB [NULL] strings to None."""
    if value == "[NULL]" or value is None:
        return None
    return value


def _convert_row_to_column(row: pd.Series, tenant_id: UUID, table_id: UUID) -> Column:
    """
    Convert a metadata row to a Column object.

    Args:
        row (pd.Series): The metadata row to convert.
        tenant_id (UUID): The ID of the tenant.
        table_id (UUID): The ID of the table.

    Returns:
        Column: The Column object.
    """
    return Column(
        tenant_id=tenant_id,
        table_id=table_id,
        name=row["COLUMN_NAME"],
        data_type=row["DATA_TYPE"],
        description=row["COLUMN_DESCRIPTION"],
        default_value=_normalize_null_value(row["COLUMN_DEFAULT"]),
        is_nullable=_normalize_boolean(row["IS_NULLABLE"]),
    )


@task(cache_policy=NO_CACHE)
def load_column_statistics(
    session: Session, column_statistics_df: pd.DataFrame, columns: list[Column], tenant_id: UUID
) -> None:
    """
    Load column statistics into the database.

    Args:
        session (Session): Database session for database operations.
        column_statistics_df (pd.DataFrame): The dataframe containing the column statistics to load.
        columns (list[Column]): The list of columns that the statistics belong to.
        tenant_id (UUID): The ID of the tenant.

    Returns:
        None
    """
    logger.info(f"Loading {len(column_statistics_df)} column statistics into the database")

    # Create lookup dictionary for column name to ID
    column_name_to_id = {column.name: column.id for column in columns}

    column_statistics = []
    for _, row in column_statistics_df.iterrows():
        column_id = column_name_to_id.get(row["COLUMN_NAME"])
        column_statistics_obj = _convert_row_to_column_statistics(row, tenant_id, column_id)
        column_statistics.append(column_statistics_obj)

    session.add_all(column_statistics)
    session.flush()  # Generate IDs but don't commit yet
    logger.info(f"Loaded {len(column_statistics)} column statistics into the database")


def normalize_distinct_count(val: Any) -> int | None:
    """Convert distinct values count to proper integer or None."""
    if pd.isna(val):
        return None
    try:
        return int(val) if val is not None else None
    except (ValueError, TypeError):
        return None


def _convert_row_to_column_statistics(row: pd.Series, tenant_id: UUID, column_id: UUID) -> ColumnStatistics:
    """
    Convert a metadata row to a ColumnStatistics object.

    Args:
        row (pd.Series): The metadata row to convert.
        tenant_id (UUID): The ID of the tenant.
        column_id (UUID): The ID of the column.

    Returns:
        ColumnStatistics: The ColumnStatistics object.
    """
    return ColumnStatistics(
        tenant_id=tenant_id,
        column_id=column_id,
        most_common_values=row["MOST_COMMON_VALS"],
        most_common_frequencies=row["MOST_COMMON_FREQS"],
        null_percentage=row["NULL_FRAC"],
        distinct_values_count=normalize_distinct_count(row["N_DISTINCT"]),
        min_value=row["MIN_VALUE"],
        max_value=row["MAX_VALUE"],
    )


@task(cache_policy=NO_CACHE)
def load_primary_keys(
    session: Session, primary_keys_df: pd.DataFrame, tables: list[Table], columns: list[Column], tenant_id: UUID
) -> None:
    """
    Load primary keys into the database.

    Args:
        session (Session): Database session for database operations.
        primary_keys_df (pd.DataFrame): The dataframe containing the primary keys to load.
        tables (list[Table]): The list of tables that the primary keys belong to.
        columns (list[Column]): The list of columns that the primary keys belong to.
        tenant_id (UUID): The ID of the tenant.

    Returns:
        None
    """
    logger.info(f"Loading {len(primary_keys_df)} primary keys into the database")

    # Create lookup dictionaries for table and column name to ID
    table_name_to_id = {table.name: table.id for table in tables}
    column_name_to_id = {column.name: column.id for column in columns}

    primary_keys = []
    for _, row in primary_keys_df.iterrows():
        table_id = table_name_to_id.get(row["TABLE_NAME"])
        column_id = column_name_to_id.get(row["COLUMN_NAME"])
        primary_key = _convert_row_to_primary_key(row, tenant_id, table_id, column_id)
        primary_keys.append(primary_key)

    session.add_all(primary_keys)
    session.flush()  # Generate IDs but don't commit yet
    logger.info(f"Loaded {len(primary_keys)} primary keys into the database")


def _convert_row_to_primary_key(row: pd.Series, tenant_id: UUID, table_id: UUID, column_id: UUID) -> PrimaryKeyConstraint:
    """
    Convert a metadata row to a PrimaryKeyConstraint object.

    Args:
        row (pd.Series): The metadata row to convert.
        tenant_id (UUID): The ID of the tenant.
        table_id (UUID): The ID of the table.
        column_id (UUID): The ID of the column.

    Returns:
        PrimaryKeyConstraint: The PrimaryKeyConstraint object.
    """
    return PrimaryKeyConstraint(
        tenant_id=tenant_id,
        table_id=table_id,
        column_id=column_id,
        ordinal_position=row["ORDINAL_POSITION"],
        constraint_name=row["CONSTRAINT_NAME"],
    )


@task(cache_policy=NO_CACHE)
def load_foreign_keys(
    session: Session, foreign_keys_df: pd.DataFrame, tables: list[Table], columns: list[Column], tenant_id: UUID
) -> None:
    """
    Load foreign keys into the database.

    Args:
        session (Session): Database session for database operations.
        foreign_keys_df (pd.DataFrame): The dataframe containing the foreign keys to load.
        tables (list[Table]): The list of tables that the foreign keys belong to.
        columns (list[Column]): The list of columns that the foreign keys belong to.
        tenant_id (UUID): The ID of the tenant.

    Returns:
        None
    """
    logger.info(f"Loading {len(foreign_keys_df)} foreign keys into the database")

    # Create lookup dictionaries for table and column name to ID
    table_name_to_id = {table.name: table.id for table in tables}
    column_name_to_id = {column.name: column.id for column in columns}

    foreign_keys = []
    for _, row in foreign_keys_df.iterrows():
        table_id = table_name_to_id.get(row["TABLE_NAME"])
        column_id = column_name_to_id.get(row["COLUMN_NAME"])
        referenced_table_id = table_name_to_id.get(row["REFERENCED_TABLE_NAME"])
        referenced_column_id = column_name_to_id.get(row["REFERENCED_COLUMN_NAME"])
        foreign_key = _convert_row_to_foreign_key(row, tenant_id, table_id, column_id, referenced_table_id, referenced_column_id)
        foreign_keys.append(foreign_key)

    session.add_all(foreign_keys)
    session.flush()
    logger.info(f"Loaded {len(foreign_keys)} foreign keys into the database")


def _convert_row_to_foreign_key(row: pd.Series, tenant_id: UUID, table_id: UUID, column_id: UUID, referenced_table_id: UUID, referenced_column_id: UUID) -> ForeignKeyConstraint:
    """
    Convert a metadata row to a ForeignKeyConstraint object.

    Args:
        row (pd.Series): The metadata row to convert.
        tenant_id (UUID): The ID of the tenant.
        table_id (UUID): The ID of the table.
        column_id (UUID): The ID of the column.
        referenced_table_id (UUID): The ID of the referenced table.
        referenced_column_id (UUID): The ID of the referenced column.

    Returns:
        ForeignKeyConstraint: The ForeignKeyConstraint object.
    """
    return ForeignKeyConstraint(
        tenant_id=tenant_id,
        table_id=table_id,
        column_id=column_id,
        referenced_table_id=referenced_table_id,
        referenced_column_id=referenced_column_id,
        constraint_name=row["CONSTRAINT_NAME"],
        ordinal_position=row["ORDINAL_POSITION"],
    )
