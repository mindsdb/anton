"""
Data catalog loader service layer for loading metadata from MindsDB into the local database.

This module contains the DataCatalogLoader class that handles all business logic
related to loading the data catalog, including asynchronous and synchronous execution modes.
Both execution modes are handled by Prefect.
"""

from enum import Enum

from prefect.deployments import run_deployment
from prefect.exceptions import PrefectException
from sqlmodel import Session

from minds.common.logger import setup_logging
from minds.common.vars import DATA_CATALOG_EXECUTION_MODE, DATA_CATALOG_JOB_DEPLOYMENT_NAME, DATA_CATALOG_JOB_NAME
from minds.jobs.data_catalog_loader_flow import load_data_catalog
from minds.model.mind_datasource import MindDatasource

logger = setup_logging()


class DataCatalogExecutionMode(str, Enum):
    ASYNC = "asynchronous"
    SYNC = "synchronous"


class DataCatalogLoader:
    """
    Service class for loading the data catalog.
    """

    def __init__(self, session: Session, tenant_id: str):
        self.session = session
        self.tenant_id = tenant_id

    async def load(
        self,
        mind_datasource: MindDatasource,
        table_names: list[str] | None = None,
    ) -> None:
        """
        Load the data catalog for a given mind-datasource relationship.

        Args:
            mind_datasource_id (UUID): The ID of the mind-datasource relationship.
            tenant_id (str): The ID of the tenant.
            table_names (list[str] | None): Optional list of table names to filter by. If None, load all tables.
        """
        if DATA_CATALOG_EXECUTION_MODE not in [mode.value for mode in DataCatalogExecutionMode]:
            raise ValueError(f"Invalid data catalog execution mode: {DATA_CATALOG_EXECUTION_MODE}")

        logger.debug(f"Running the data catalog loader flow in {DATA_CATALOG_EXECUTION_MODE} mode")
        if DataCatalogExecutionMode.ASYNC.value == DATA_CATALOG_EXECUTION_MODE:
            flow_run = await run_deployment(
                name=f"{DATA_CATALOG_JOB_NAME}/{DATA_CATALOG_JOB_DEPLOYMENT_NAME}",
                parameters={
                    "mind_datasource_id": mind_datasource.id,
                    "tenant_id": self.tenant_id,
                    "table_names": table_names,
                },
                timeout=0
            )
            if not (flow_run.state.is_scheduled() or flow_run.state.is_pending() or flow_run.state.is_running()):
                error_msg = (
                    f"Failed to start data catalog loader flow run. "
                    f"Flow run is in '{flow_run.state.type}' state"
                    + (f" with error: {flow_run.state.message}" if flow_run.state.message else "")
                )
                logger.error(error_msg)
                raise PrefectException(error_msg)

            logger.debug(f"Data catalog loader flow run started: {flow_run.id}")
            mind_datasource.flow_run_id = flow_run.id
            self.session.add(mind_datasource)
            self.session.commit()

        else:
            load_data_catalog(
                mind_datasource_id=mind_datasource.id,
                tenant_id=self.tenant_id,
                table_names=table_names,
            )
