from abc import ABC, abstractmethod
import logging

from minds.common.data_catalog.data_catalog import DataCatalog

logger = logging.getLogger(__name__)


class DataCatalogLoader(ABC):
    """Abstract base class for data catalog loaders."""

    @abstractmethod
    def load(self) -> DataCatalog:
        """
        Load a data catalog from the database.

        Returns:
            A DataCatalog instance

        Raises:
            Exception: For database connection or query errors
        """
        pass
