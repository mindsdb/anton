from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime

from minds.common.logger import setup_logging
from minds.common.settings.app_settings import get_app_settings
from minds.model.data_catalog import DataCatalog
from minds.model.mind import Mind
from minds.model.mind_datasource import DataCatalogStatus

logger = setup_logging()
settings = get_app_settings()


class DataCatalogCache(ABC):
    """Base class for storing and retrieving data catalogs."""

    class MindKey:
        """Key class for storing and retrieving data catalogs."""

        def __init__(self, mind_name: str, modified_at: datetime):
            """
            Initialize a MindKey.

            Args:
                mind_name: The name of the mind.
                modified_at: The last updated timestamp of the mind.
            """
            self.mind_name = mind_name
            self.modified_at = modified_at

        def to_string(self) -> str:
            """
            Convert the MindKey to a string representation.

            Returns:
                A string representation of the MindKey.
            """
            return f"{self.mind_name}:{str(self.modified_at)}"

    @abstractmethod
    async def load(self, mind: Mind) -> list[DataCatalog]:
        """
        Load data catalogs for the given mind.

        Args:
            mind: The mind requesting the catalog.

        Returns:
            A list of data catalogs if found or created, empty list otherwise.
        """
        pass

    @abstractmethod
    async def save(self, key: MindKey, catalogs: list[DataCatalog]) -> None:
        """
        Save data catalogs for the given key.

        Args:
            key: A MindKey identifier for the data catalogs.
            catalogs: The data catalogs to save.
        """
        pass

    @abstractmethod
    async def invalidate(self, key: MindKey) -> None:
        """
        Invalidate cached data catalogs.

        Args:
            key: The MindKey of the catalogs to invalidate.
        """
        pass


class DataCatalogInMemoryCache(DataCatalogCache):
    """An LRU cache implementation of DataCatalogStore."""

    def __init__(self, max_size: int = 100):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of catalogs to keep in the cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, list[DataCatalog]] = OrderedDict()

    async def load(self, mind: Mind) -> list[DataCatalog]:
        """
        Load data catalogs from the cache or create new ones.

        Args:
            mind: The Mind requesting the data catalogs.

        Returns:
            A list of cached data catalogs if found, or newly created ones.
        """
        key = DataCatalogCache.MindKey(mind_name=mind.name, modified_at=mind.modified_at)
        key_str = key.to_string()

        # Check if we have it in the cache.
        if key_str in self.cache:
            # Move to the end to mark as most recently used.
            catalogs = self.cache.pop(key_str)
            self.cache[key_str] = catalogs
            logger.info(f"Cache hit for data catalogs with key: {key_str}")
            return catalogs

        logger.info(f"Cache miss for data catalogs with key: {key_str}")

        data_catalogs = []
        for mind_datasource in mind.mind_datasources:
            logger.info(f"Loading data catalog for mind datasource {mind_datasource.id} for mind {mind.name}")
            status = await mind_datasource.status
            overall_status = status.overall_status
            logger.info(f"Mind datasource overall status: {overall_status}")
            if overall_status != DataCatalogStatus.COMPLETED:
                logger.info(
                    f"Skipping data catalog for mind datasource {mind_datasource.id} "
                    f"for mind {mind.name} because it is in status {overall_status}"
                )
                continue

            data_catalog = DataCatalog.from_mind_datasource(mind_datasource)
            data_catalogs.append(data_catalog)

        # Cache the catalogs if any were created.
        if data_catalogs:
            await self.save(key, data_catalogs)

        return data_catalogs

    async def save(self, key: DataCatalogCache.MindKey, catalogs: list[DataCatalog]) -> None:
        """
        Save data catalogs to the cache.

        Args:
            key: A MindKey identifier for the data catalogs.
            catalogs: The data catalogs to cache.
        """
        key_str = key.to_string()

        # If we're at capacity, remove the least recently used item.
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        # Update the last_updated field for all catalogs.
        for catalog in catalogs:
            catalog.modified_at = datetime.now()

        # Add to the cache.
        self.cache[key_str] = catalogs
        logger.info(f"Saved {len(catalogs)} data catalogs with key: {key_str}")

    async def invalidate(self, key: DataCatalogCache.MindKey) -> None:
        """
        Invalidate cached data catalogs.

        Args:
            key: The MindKey of the catalogs to invalidate.
        """
        key_str = key.to_string()

        if key_str in self.cache:
            catalog_count = len(self.cache[key_str])
            self.cache.pop(key_str)
            logger.info(f"Invalidated {catalog_count} data catalogs with key: {key_str}")

    async def clear(self) -> None:
        """Clear the entire cache."""
        self.cache.clear()
        logger.info("Cleared data catalog cache")

    async def size(self) -> int:
        """Get the current size of the cache."""
        return len(self.cache)


# TODO: Implement a Redis cache implementation of DataCatalogCache.
class DataCatalogRedisCache(DataCatalogCache):
    """A Redis cache implementation of DataCatalogCache."""


class DataCatalogCacheFactory:
    """Factory class for creating DataCatalogCache instances."""

    @staticmethod
    def create_cache() -> DataCatalogCache:
        """Create a DataCatalogCache instance based on the cache type."""
        if settings.data_catalog.cache_type == "in_memory":
            return DataCatalogInMemoryCache(max_size=settings.data_catalog.cache_max_size)
        else:
            raise ValueError(f"Unknown cache type '{settings.data_catalog.cache_type}'")
