from typing import Any, Dict
from uuid import UUID

from prefect import flow, task
from sqlalchemy.orm import selectinload
from sqlmodel import select, and_

from minds.client.mindsdb import create_mindsdb_client_from_env
from minds.common.logger import setup_logging
from minds.db.pg_session import get_session
from minds.model.mind_datasource import DataCatalogStatus, MindDatasource
from minds.services.data_catalog.data_catalog_loader import (
    DataCatalogLoaderError,
    filter_loaded_tables,
    get_tables,
    get_columns,
    get_column_statistics,
    get_primary_keys,
    get_foreign_keys,
    load_tables,
    load_columns,
    load_column_statistics,
    load_primary_keys,
    load_foreign_keys,
)


logger = setup_logging()


# Convert data catalog component functions to Prefect tasks.
get_tables = task(get_tables)
get_columns = task(get_columns)
get_column_statistics = task(get_column_statistics)
get_primary_keys = task(get_primary_keys)
get_foreign_keys = task(get_foreign_keys)

load_tables = task(load_tables)
load_columns = task(load_columns)
load_column_statistics = task(load_column_statistics)
load_primary_keys = task(load_primary_keys)
load_foreign_keys = task(load_foreign_keys)


@flow
def load_data_catalog(
    mind_datasource_id: UUID,
    tenant_id: str,
    table_names: list[str] | None = None,
) -> None:
    """
    Prefect flow version of load_data_catalog that uses task-wrapped functions.

    Args:
        mind_datasource (Dict[str, Any]): The MindDatasource object representing the datasource to load in dictionary format. A dictionary is accepted here because Prefect requires serializable objects.
        table_names (list[str] | None): Optional list of table names to filter by. If None, load all tables.
    """
    try:
        # Create a database session
        session_generator = get_session()
        session = next(session_generator)

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

        mindsdb_client = create_mindsdb_client_from_env()
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
    finally:
        session.close()
