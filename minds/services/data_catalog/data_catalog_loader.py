"""
Data catalog loader service layer for loading metadata from MindsDB into the local database.

This module contains the DataCatalogLoader class that handles all business logic
related to loading the data catalog, including asynchronous and synchronous execution modes.
Both execution modes are handled by Prefect.
"""

from enum import Enum
from uuid import UUID

from prefect.deployments import run_deployment
from prefect.exceptions import PrefectException
from sqlmodel import Session

from minds.common.logger import setup_logging
from minds.common.settings.app_settings import get_app_settings
from minds.jobs.data_catalog_loader_flow import load_data_catalog
from minds.model.mind_datasource import MindDatasource

logger = setup_logging()
settings = get_app_settings()


class DataCatalogExecutionMode(str, Enum):
    ASYNC = "asynchronous"
    SYNC = "synchronous"


class DataCatalogLoader:
    """
    Service class for loading the data catalog.
    """

    def __init__(self, session: Session, tenant_id: UUID, user_id: UUID):
        self.session = session
        self.tenant_id = tenant_id
        self.user_id = user_id

    async def load(
        self,
        mind_datasource: MindDatasource,
        table_names: list[str] | None = None,
    ) -> None:
        """
        Load the data catalog for a given mind-datasource relationship.

        Args:
            mind_datasource (MindDatasource): The mind-datasource relationship to load the catalog for.
            table_names (list[str] | None): Optional list of table names to filter by. If None, load all tables.
        """

        if settings.data_catalog.execution_mode not in [mode.value for mode in DataCatalogExecutionMode]:
            raise ValueError(f"Invalid data catalog execution mode: {settings.data_catalog.execution_mode}")

        logger.debug(f"Running the data catalog loader flow in {settings.data_catalog.execution_mode} mode")
        if DataCatalogExecutionMode.ASYNC.value == settings.data_catalog.execution_mode:
            flow_run = await run_deployment(
                name=f"{settings.data_catalog.job_name}/{settings.data_catalog.job_deployment_name}",
                parameters={
                    "mind_datasource_id": mind_datasource.id,
                    "tenant_id": self.tenant_id,
                    "user_id": self.user_id,
                    "table_names": table_names,
                },
                timeout=0,
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
                user_id=self.user_id,
                table_names=table_names,
            )
