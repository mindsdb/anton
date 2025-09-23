"""
Data catalog loader service.
"""

from uuid import UUID

from prefect.deployments import run_deployment

from minds.common.logger import setup_logging
from minds.common.vars import DATA_CATALOG_EXECUTION_MODE, DATA_CATALOG_JOB_NAME, DATA_CATALOG_JOB_DEPLOYMENT_NAME
from minds.jobs.data_catalog_loader_flow import load_data_catalog


logger = setup_logging()


class DataCatalogLoader:
    async def load(
        self,
        mind_datasource_id: UUID,
        tenant_id: str,
        table_names: list[str] | None = None,
    ) -> None:
        logger.debug(
            f"Running the data catalog loader flow in {DATA_CATALOG_EXECUTION_MODE} mode"
        )
        if DATA_CATALOG_EXECUTION_MODE == "asynchronous":
            flow_run = await run_deployment(
                name=f"{DATA_CATALOG_JOB_NAME}/{DATA_CATALOG_JOB_DEPLOYMENT_NAME}",
                timeout=0,
                parameters={
                    "mind_datasource_id": mind_datasource_id,
                    "tenant_id": tenant_id,
                    "table_names": table_names,
                },
            )
            logger.debug(
                f"Data catalog loader flow run started: {flow_run.id}"
            )
            # TODO: Store the flow run id in the database.
        else:
            load_data_catalog(
                mind_datasource_id=mind_datasource_id,
                tenant_id=tenant_id,
                table_names=table_names,
            )
            return None
