from typing import Dict, Any, Optional, List
import mindsdb_sdk
import logging

from minds.common.data_catalog.data_catalog_loader import DataCatalogLoader
from minds.common.data_catalog.mindsdb_data_catalog_loader import (
    MindsDBDataCatalogLoader,
)

logger = logging.getLogger(__name__)


class DataCatalogLoaderFactory:
    """Factory for creating data catalog loaders."""

    @staticmethod
    def create(
        datasource_name: str,
        engine: str,
        server: Optional[mindsdb_sdk.server.Server] = None,
        tables_filter: Optional[List[str]] = None,
    ) -> DataCatalogLoader:
        """
        Create a data catalog loader for the specified database engine.

        Args:
            datasource_name: Name of the datasource/integration
            engine: The database engine (e.g., 'postgres', 'mysql', 'snowflake')
            server: MindsDB server instance (required for non-postgres engines)
            tables_filter: Optional list of table names to filter (empty list = no filtering)

        Returns:
            A DataCatalogLoader instance

        Raises:
            ValueError: If required parameters are missing for specific engines
        """
        engine = engine.lower()
        logger.info(
            f"Creating data catalog loader for datasource '{datasource_name}' with engine '{engine}' and tables filter: {tables_filter}"
        )

        logger.info(f"Using MindsDBDataCatalogLoader for: '{engine}'")
        if not server:
            logger.error(
                f"MindsDB server instance is required for engine '{engine}' but was not provided"
            )
            raise ValueError(
                "MindsDB server instance is required for non-postgres integrations"
            )
        logger.info(
            f"Successfully created MindsDBDataCatalogLoader for integration '{datasource_name}'"
        )
        return MindsDBDataCatalogLoader(datasource_name, server, tables_filter, engine)
