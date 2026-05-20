"""
Data catalog loader service.
This service is implemented in a function style to support both
synchronous and asynchronous execution modes.

Asynchronous executions are run via Prefect.
"""

import re
from typing import Any
from uuid import UUID

import pandas as pd
from mindsdb_sdk.server import Server
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from sqlalchemy.orm import selectinload
from sqlmodel import Session, and_, select

from minds.client.mindsdb import create_mindsdb_client_with_credentials
from minds.common.logger import get_logger
from minds.common.utilities import safe_parse
from minds.db.pg_session import get_session
from minds.jobs.settings import get_prefect_settings
from minds.model.conversation import Conversation  # noqa: F401
from minds.model.data_catalog import Column, ColumnStatistics, ForeignKeyConstraint, PrimaryKeyConstraint, Table
from minds.model.message import Message  # noqa: F401
from minds.model.message_event import MessageEvent  # noqa: F401
from minds.model.mind import Mind  # noqa: F401
from minds.model.mind_datasource import MindDatasource
from minds.model.mind_datasource_table import MindDatasourceTable

logger = get_logger(__name__)


class DataCatalogLoaderError(Exception):
    """Base exception for data catalog loader errors."""

    pass


@flow
def load_data_catalog(
    mind_datasource_id: UUID,
    organization_id: UUID,
    user_id: UUID,
    table_names: list[str] | None = None,
) -> None:
    """
    Load data catalog information from MindsDB into the local database.

    Args:
        mind_datasource_id (UUID): The ID of the mind datasource.
        organization_id (UUID): The ID of the organization.
        user_id (UUID): The ID of the user.
        table_names (list[str] | None): Optional list of table names to filter by. If None, load all tables.

    Raises:
        DataCatalogLoaderError: If there is an error during the loading process.
    """
    prefect_settings = get_prefect_settings()
    logger = get_run_logger()

    # Debug logging to trace parameter values
    logger.info(
        f"load_data_catalog called with: mind_datasource_id={mind_datasource_id} (type={type(mind_datasource_id)}), "
        f"organization_id={organization_id} (type={type(organization_id)}), "
        f"user_id={user_id} (type={type(user_id)}), "
        f"table_names={table_names}"
    )

    # Validate required parameters
    if user_id is None:
        raise DataCatalogLoaderError("user_id is required but was None - check Prefect parameter serialization")
    if organization_id is None:
        raise DataCatalogLoaderError("organization_id is required but was None - check Prefect parameter serialization")

    # Create a database session
    session_generator = get_session(prefect_settings.database_uri)
    session = next(session_generator)

    # Get the MindDatasource object from the database
    statement = (
        select(MindDatasource)
        .where(
            and_(
                MindDatasource.id == mind_datasource_id,
                MindDatasource.organization_id == organization_id,
            )
        )
        .options(
            selectinload(MindDatasource.datasource),
        )
    )
    mind_datasource = session.exec(statement).first()

    try:
        # Create a MindsDB client
        mindsdb_client = create_mindsdb_client_with_credentials(
            url=prefect_settings.mindsdb_url,
            api_key=prefect_settings.mindsdb_api_key,
            login=prefect_settings.mindsdb_login,
            password=prefect_settings.mindsdb_password,
            organization_id=organization_id,
            user_id=user_id,
        )

        datasource_id = mind_datasource.datasource_id
        datasource_name = mind_datasource.datasource.name

        tables_df = get_tables(mindsdb_client=mindsdb_client, datasource_name=datasource_name, table_names=table_names)

        tables_df = filter_loaded_tables(
            session=session,
            tables_df=tables_df,
            mind_datasource_id=mind_datasource_id,
            datasource_id=datasource_id,
            organization_id=organization_id,
            user_id=user_id,
        )

        if len(tables_df) > 0:
            tables = load_tables(
                session=session,
                tables_df=tables_df,
                mind_datasource_id=mind_datasource_id,
                datasource_id=datasource_id,
                organization_id=organization_id,
                user_id=user_id,
            )
            loaded_table_names = [table.name for table in tables]

            columns_df = get_columns(mindsdb_client, datasource_name, loaded_table_names)
            columns = load_columns(
                session=session, columns_df=columns_df, tables=tables, organization_id=organization_id, user_id=user_id
            )

            column_statistics_df = get_column_statistics(mindsdb_client, datasource_name, loaded_table_names)
            if len(column_statistics_df) > 0:
                load_column_statistics(
                    session=session,
                    column_statistics_df=column_statistics_df,
                    columns=columns,
                    organization_id=organization_id,
                    user_id=user_id,
                )

            primary_keys_df = get_primary_keys(mindsdb_client, datasource_name, loaded_table_names)
            if len(primary_keys_df) > 0:
                load_primary_keys(
                    session=session,
                    primary_keys_df=primary_keys_df,
                    tables=tables,
                    columns=columns,
                    organization_id=organization_id,
                    user_id=user_id,
                )

            foreign_keys_df = get_foreign_keys(mindsdb_client, datasource_name, loaded_table_names)
            if len(foreign_keys_df) > 0:
                load_foreign_keys(
                    session=session,
                    foreign_keys_df=foreign_keys_df,
                    tables=tables,
                    columns=columns,
                    organization_id=organization_id,
                    user_id=user_id,
                )

        # Only commit if everything succeeded
        session.commit()
        logger.info("Successfully committed all data catalog information to database")
    except Exception as e:
        session.rollback()
        err_message = f"Failed to load data catalog for MindDatasource ID {mind_datasource_id}: {str(e)}"
        logger.error(f"Failed to load data catalog: {err_message}")
        raise DataCatalogLoaderError(f"Failed to load data catalog: {err_message}") from e


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

    # At the moment, the schema that is being returned by the above query is the same as the datasource name.
    # This is because of the way the META_ tables have been implemented
    # (based on the MYSQL INFORMATION_SCHEMA tables).
    # Notice how we are filtering by the TABLE_SCHEMA above.
    # As a result, the schema will have to be retrieved separately via the connection parameters
    # of the MindsDB database.
    # This only applies when storing this field in the database.
    # For all other intents and purposes, TABLE_SCHEMA should be the name of the data source.
    params = mindsdb_client.databases.get(datasource_name).params or {}
    params = params if isinstance(params, dict) else safe_parse(params)

    schema = params.get("schema")

    if table_names:
        query += f" AND TABLE_NAME IN ({', '.join(f'{name!r}' for name in table_names)})"

    df = _execute_mindsdb_query(mindsdb_client, query)
    df["TABLE_SCHEMA"] = schema
    logger.info(f"Found {len(df)} tables")

    return df


@task(cache_policy=NO_CACHE)
def filter_loaded_tables(
    session: Session,
    tables_df: pd.DataFrame,
    mind_datasource_id: UUID,
    datasource_id: UUID,
    organization_id: UUID,
    user_id: UUID,
) -> pd.DataFrame:
    """
    Filter out tables that have already been loaded.
    If the loaded tables exist but are not associated with the mind_datasource, associate them.

    Args:
        session (Session): Database session for querying existing tables.
        tables_df (pd.DataFrame): DataFrame containing all tables from MindsDB.
        mind_datasource_id (UUID): The ID of the mind datasource.
        datasource_id (UUID): The ID of the datasource.
        organization_id (UUID): The ID of the organization.
        user_id (UUID): The ID of the user.
    Returns:
        pd.DataFrame: DataFrame containing only tables that haven't been loaded yet.
    """
    existing_tables = session.exec(
        select(Table).where(
            and_(
                Table.datasource_id == datasource_id,
                Table.name.in_(tables_df["TABLE_NAME"]),
                Table.organization_id == organization_id,
                Table.deleted_at.is_(None),
            )
        )
    ).all()
    logger.info(f"{len(existing_tables)} have already been loaded")

    # Check if the existing tables are associated with the mind_datasource
    if existing_tables:
        mind_datasource_tables = []
        for table in existing_tables:
            mind_datasource_table = session.exec(
                select(MindDatasourceTable).where(
                    and_(
                        MindDatasourceTable.mind_datasource_id == mind_datasource_id,
                        MindDatasourceTable.table_id == table.id,
                        MindDatasourceTable.organization_id == organization_id,
                        MindDatasourceTable.deleted_at.is_(None),
                    )
                )
            ).first()
            if mind_datasource_table:
                logger.info(f"Table '{table.name}' is already associated with the mind datasource")
            else:
                logger.info(f"Table '{table.name}' is not associated with the mind datasource. Associating now.")
                # Associate the existing table with the mind_datasource
                mind_datasource_table = MindDatasourceTable(
                    organization_id=organization_id,
                    user_id=user_id,
                    mind_datasource_id=mind_datasource_id,
                    table_id=table.id,
                )
                mind_datasource_tables.append(mind_datasource_table)

        if mind_datasource_tables:
            session.add_all(mind_datasource_tables)
            session.commit()

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
    session: Session,
    tables_df: pd.DataFrame,
    mind_datasource_id: UUID,
    datasource_id: UUID,
    organization_id: UUID,
    user_id: UUID,
) -> list[Table]:
    """
    Load tables into the database.

    Args:
        session (Session): Database session for database operations.
        tables_df (pd.DataFrame): The dataframe containing the tables to load.
        mind_datasource_id (UUID): The ID of the mind datasource.
        datasource_id (UUID): The ID of the datasource.
        organization_id (UUID): The ID of the organization.
        user_id (UUID): The ID of the user.

    Returns:
        list[Table]: The list of tables that were loaded.
    """
    logger.info(f"Loading {len(tables_df)} tables into the database")

    tables = []
    mind_datasource_tables = []
    for _, row in tables_df.iterrows():
        table = _convert_row_to_table(
            row=row, organization_id=organization_id, datasource_id=datasource_id, user_id=user_id
        )
        session.add(table)
        session.flush()
        session.refresh(table)

        mind_datasource_table = MindDatasourceTable(
            organization_id=organization_id,
            user_id=user_id,
            mind_datasource_id=mind_datasource_id,
            table_id=table.id,
        )
        mind_datasource_tables.append(mind_datasource_table)
        tables.append(table)

    session.add_all(mind_datasource_tables)
    session.flush()  # Generate IDs but don't commit yet
    logger.info(f"Loaded {len(tables)} tables into the database")

    return tables


def _convert_row_to_table(row: pd.Series, organization_id: UUID, datasource_id: UUID, user_id: UUID) -> Table:
    """
    Convert a metadata row to a Table object.

    Args:
        row (pd.Series): The metadata row to convert.
        datasource_id (UUID): The ID of the datasource.
        organization_id (UUID): The ID of the organization.
        user_id (UUID): The ID of the user.

    Returns:
        Table: The Table object.
    """
    row_count = row.get("ROW_COUNT")
    row_count = int(row_count) if pd.notna(row_count) else None

    table = Table(
        organization_id=organization_id,
        user_id=user_id,
        datasource_id=datasource_id,
        name=row["TABLE_NAME"],
        schema=row["TABLE_SCHEMA"],
        description=row["TABLE_DESCRIPTION"],
        type=row["TABLE_TYPE"],
        row_count=row_count,
    )

    return table


@task(cache_policy=NO_CACHE)
def load_columns(
    session: Session, columns_df: pd.DataFrame, tables: list[Table], organization_id: UUID, user_id: UUID
) -> list[Column]:
    """
    Load columns into the database.

    Args:
        session (Session): Database session for database operations.
        columns_df (pd.DataFrame): The dataframe containing the columns to load.
        tables (list[Table]): The list of tables that the columns belong to.
        organization_id (UUID): The ID of the organization.
        user_id (UUID): The ID of the user.

    Returns:
        list[Column]: The list of columns that were loaded.
    """
    logger.info(f"Loading {len(columns_df)} columns into the database")

    # Create a lookup dictionary for table name to table ID
    table_name_to_id = {table.name: table.id for table in tables}

    columns = []
    for _, row in columns_df.iterrows():
        table_id = table_name_to_id.get(row["TABLE_NAME"])
        column = _convert_row_to_column(row=row, organization_id=organization_id, table_id=table_id, user_id=user_id)
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


def _convert_row_to_column(row: pd.Series, organization_id: UUID, table_id: UUID, user_id: UUID) -> Column:
    """
    Convert a metadata row to a Column object.

    Args:
        row (pd.Series): The metadata row to convert.
        table_id (UUID): The ID of the table.
        organization_id (UUID): The ID of the organization.
        user_id (UUID): The ID of the user.

    Returns:
        Column: The Column object.
    """
    return Column(
        organization_id=organization_id,
        user_id=user_id,
        table_id=table_id,
        name=row["COLUMN_NAME"],
        data_type=row["DATA_TYPE"],
        description=row["COLUMN_DESCRIPTION"],
        default_value=_normalize_null_value(row["COLUMN_DEFAULT"]),
        is_nullable=_normalize_boolean(row["IS_NULLABLE"]),
    )


@task(cache_policy=NO_CACHE)
def load_column_statistics(
    session: Session, column_statistics_df: pd.DataFrame, columns: list[Column], organization_id: UUID, user_id: UUID
) -> None:
    """
    Load column statistics into the database.

    Args:
        session (Session): Database session for database operations.
        column_statistics_df (pd.DataFrame): The dataframe containing the column statistics to load.
        columns (list[Column]): The list of columns that the statistics belong to.
        organization_id (UUID): The ID of the organization.
        user_id (UUID): The ID of the user.

    Returns:
        None
    """
    logger.info(f"Loading {len(column_statistics_df)} column statistics into the database")

    # Create lookup dictionary for column name to ID
    column_name_to_id = {column.name: column.id for column in columns}

    column_statistics = []
    for _, row in column_statistics_df.iterrows():
        column_id = column_name_to_id.get(row["COLUMN_NAME"])
        column_statistics_obj = _convert_row_to_column_statistics(
            row=row, organization_id=organization_id, column_id=column_id, user_id=user_id
        )
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


def _clean_string(val: str) -> str:
    """
    Remove problematic control characters (like NUL and other non-printables)
    but keep legitimate whitespace such as tabs, newlines, and spaces.
    """
    if not isinstance(val, str):
        return val
    # Matches ASCII control characters except \t (tab), \n (newline), and \r (carriage return)
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", val)


def _convert_row_to_column_statistics(
    row: pd.Series, organization_id: UUID, column_id: UUID, user_id: UUID
) -> ColumnStatistics:
    """
    Convert a metadata row to a ColumnStatistics object.

    Args:
        row (pd.Series): The metadata row to convert.
        column_id (UUID): The ID of the column.
        organization_id (UUID): The ID of the organization.
        user_id (UUID): The ID of the user.

    Returns:
        ColumnStatistics: The ColumnStatistics object.
    """
    return ColumnStatistics(
        organization_id=organization_id,
        user_id=user_id,
        column_id=column_id,
        most_common_values=row["MOST_COMMON_VALS"],
        most_common_frequencies=row["MOST_COMMON_FREQS"],
        null_percentage=row["NULL_FRAC"],
        distinct_values_count=normalize_distinct_count(row["N_DISTINCT"]),
        min_value=_clean_string(row["MIN_VALUE"]),
        max_value=_clean_string(row["MAX_VALUE"]),
    )


@task(cache_policy=NO_CACHE)
def load_primary_keys(
    session: Session,
    primary_keys_df: pd.DataFrame,
    tables: list[Table],
    columns: list[Column],
    organization_id: UUID,
    user_id: UUID,
) -> None:
    """
    Load primary keys into the database.

    Args:
        session (Session): Database session for database operations.
        primary_keys_df (pd.DataFrame): The dataframe containing the primary keys to load.
        tables (list[Table]): The list of tables that the primary keys belong to.
        columns (list[Column]): The list of columns that the primary keys belong to.
        organization_id (UUID): The ID of the organization.
        user_id (UUID): The ID of the user.

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
        primary_key = _convert_row_to_primary_key(
            row=row, organization_id=organization_id, table_id=table_id, column_id=column_id, user_id=user_id
        )
        primary_keys.append(primary_key)

    session.add_all(primary_keys)
    session.flush()  # Generate IDs but don't commit yet
    logger.info(f"Loaded {len(primary_keys)} primary keys into the database")


def _convert_row_to_primary_key(
    row: pd.Series, organization_id: UUID, table_id: UUID, column_id: UUID, user_id: UUID
) -> PrimaryKeyConstraint:
    """
    Convert a metadata row to a PrimaryKeyConstraint object.

    Args:
        row (pd.Series): The metadata row to convert.
        organization_id (UUID): The ID of the organization.
        table_id (UUID): The ID of the table.
        column_id (UUID): The ID of the column.
        user_id (UUID): The ID of the user.

    Returns:
        PrimaryKeyConstraint: The PrimaryKeyConstraint object.
    """
    return PrimaryKeyConstraint(
        organization_id=organization_id,
        user_id=user_id,
        table_id=table_id,
        column_id=column_id,
        ordinal_position=row["ORDINAL_POSITION"],
        constraint_name=row["CONSTRAINT_NAME"],
    )


@task(cache_policy=NO_CACHE)
def load_foreign_keys(
    session: Session,
    foreign_keys_df: pd.DataFrame,
    tables: list[Table],
    columns: list[Column],
    organization_id: UUID,
    user_id: UUID,
) -> None:
    """
    Load foreign keys into the database.

    Args:
        session (Session): Database session for database operations.
        foreign_keys_df (pd.DataFrame): The dataframe containing the foreign keys to load.
        tables (list[Table]): The list of tables that the foreign keys belong to.
        columns (list[Column]): The list of columns that the foreign keys belong to.
        organization_id (UUID): The ID of the organization.
        user_id (UUID): The ID of the user.
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
        foreign_key = _convert_row_to_foreign_key(
            row=row,
            table_id=table_id,
            column_id=column_id,
            referenced_table_id=referenced_table_id,
            referenced_column_id=referenced_column_id,
            organization_id=organization_id,
            user_id=user_id,
        )
        foreign_keys.append(foreign_key)

    session.add_all(foreign_keys)
    session.flush()
    logger.info(f"Loaded {len(foreign_keys)} foreign keys into the database")


def _convert_row_to_foreign_key(
    row: pd.Series,
    organization_id: UUID,
    user_id: UUID,
    table_id: UUID,
    column_id: UUID,
    referenced_table_id: UUID,
    referenced_column_id: UUID,
) -> ForeignKeyConstraint:
    """
    Convert a metadata row to a ForeignKeyConstraint object.

    Args:
        row (pd.Series): The metadata row to convert.
        organization_id (UUID): The ID of the organization.
        user_id (UUID): The ID of the user.
        table_id (UUID): The ID of the table.
        column_id (UUID): The ID of the column.
        referenced_table_id (UUID): The ID of the referenced table.
        referenced_column_id (UUID): The ID of the referenced column.


    Returns:
        ForeignKeyConstraint: The ForeignKeyConstraint object.
    """
    return ForeignKeyConstraint(
        organization_id=organization_id,
        user_id=user_id,
        table_id=table_id,
        column_id=column_id,
        referenced_table_id=referenced_table_id,
        referenced_column_id=referenced_column_id,
        constraint_name=row["CONSTRAINT_NAME"],
        ordinal_position=row["ORDINAL_POSITION"],
    )
