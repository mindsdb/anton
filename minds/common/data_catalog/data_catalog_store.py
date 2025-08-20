from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime
from typing import Optional, List
import logging

import mindsdb_sdk

from minds.common.data_catalog.data_catalog import DataCatalog
from minds.common.data_catalog.data_catalog_loader_factory import (
    DataCatalogLoaderFactory,
)

logger = logging.getLogger(__name__)


class DataCatalogStore(ABC):
    """Base class for storing and retrieving data catalogs."""

    class MindKey:
        """Key class for storing and retrieving data catalogs."""

        def __init__(self, mind_name: str, updated_at: str):
            """
            Initialize a MindKey.

            Args:
                mind_name: The name of the mind
                updated_at: The last updated timestamp of the mind
            """
            self.mind_name = mind_name
            self.updated_at = updated_at

        @classmethod
        def from_agent(
            cls, agent: mindsdb_sdk.agents.Agent
        ) -> "DataCatalogStore.MindKey":
            """
            Create a MindKey from a MindsDB agent.

            Args:
                agent: The MindsDB agent

            Returns:
                A MindKey instance
            """
            return cls(mind_name=agent.name, updated_at=str(agent.updated_at))

        def to_string(self) -> str:
            """
            Convert the MindKey to a string representation.

            Returns:
                A string representation of the MindKey
            """
            return f"{self.mind_name}:{self.updated_at}"

    @abstractmethod
    def load(
        self, con: mindsdb_sdk.server.Server, agent: mindsdb_sdk.agents.Agent
    ) -> List[DataCatalog]:
        """
        Load data catalogs for the given agent.

        Args:
            con: The MindsDB connection to use
            agent: The agent requesting the catalog

        Returns:
            A list of data catalogs if found or created, empty list otherwise
        """
        pass

    @abstractmethod
    def save(self, key: MindKey, catalogs: List[DataCatalog]) -> None:
        """
        Save data catalogs for the given key.

        Args:
            key: A MindKey identifier for the data catalogs
            catalogs: The data catalogs to save
        """
        pass

    @abstractmethod
    def invalidate(self, key: MindKey) -> None:
        """
        Invalidate cached data catalogs.

        Args:
            key: The MindKey of the catalogs to invalidate
        """
        pass


class DataCatalogInMemoryCache(DataCatalogStore):
    """An LRU cache implementation of DataCatalogStore."""

    def __init__(self, max_size: int = 100):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of catalogs to keep in the cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, List[DataCatalog]] = OrderedDict()

    def load(
        self, con: mindsdb_sdk.server.Server, agent: mindsdb_sdk.agents.Agent
    ) -> List[DataCatalog]:
        """
        Load data catalogs from the cache or create new ones.

        Args:
            con: The MindsDB connection to use
            agent: The agent requesting the catalogs

        Returns:
            A list of cached data catalogs if found, or newly created ones
        """
        # Create a MindKey from the agent
        key = DataCatalogStore.MindKey.from_agent(agent)
        key_str = key.to_string()

        # Check if we have it in the cache
        if key_str in self.cache:
            # Move to the end to mark as most recently used
            catalogs = self.cache.pop(key_str)
            self.cache[key_str] = catalogs
            logger.info(f"Cache hit for data catalogs with key: {key_str}")
            return catalogs

        logger.info(f"Cache miss for data catalogs with key: {key_str}")

        # Build catalogs for all SQL skills in the agent
        data_catalogs = []
        for skill in agent.skills:
            if "sql" not in skill.type:
                continue
            database_name = skill.params.get("database")

            skills_extra_params = agent.skills_extra_parameters.get(skill.name, {})

            # When table filters are specified via the UI (while creating the agent),
            # they are stored as part of agent (agent-skill association)
            if skills_extra_params.get("tables"):
                tables_filter = skills_extra_params["tables"]
            # When table filters are specified via the SDK, they are stored as part of the skill
            else:
                tables_filter = skill.params.get("tables", [])

            database_to_load: Optional[mindsdb_sdk.databases.Database] = None
            try:
                database_to_load = con.databases.get(database_name)
            except AttributeError:
                logger.warning(
                    f"Database {database_name} does not exist. Skipping data catalog"
                )
                continue
            if not database_to_load:
                logger.warning(
                    f"Database {database_name} does not exist. Skipping data catalog"
                )
                continue

            logger.info(
                f"Loading data catalog for database '{database_name}' with tables filter: {tables_filter}"
            )
            loader = DataCatalogLoaderFactory.create(
                database_name,
                database_to_load.engine,
                server=con,  # Pass MindsDB server for dependency injection
                tables_filter=tables_filter,  # Pass tables filter to loader
            )
            catalog = loader.load()
            data_catalogs.append(catalog)

        # Cache the catalogs if any were created
        if data_catalogs:
            self.save(key, data_catalogs)

        return data_catalogs

    def save(self, key: DataCatalogStore.MindKey, catalogs: List[DataCatalog]) -> None:
        """
        Save data catalogs to the cache.

        Args:
            key: A MindKey identifier for the data catalogs
            catalogs: The data catalogs to cache
        """
        key_str = key.to_string()

        # If we're at capacity, remove the least recently used item
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        # Update the last_updated field for all catalogs
        for catalog in catalogs:
            catalog.last_updated = datetime.now()

        # Add to the cache
        self.cache[key_str] = catalogs
        logger.info(f"Saved {len(catalogs)} data catalogs with key: {key_str}")

    def invalidate(self, key: DataCatalogStore.MindKey) -> None:
        """
        Invalidate cached data catalogs.

        Args:
            key: The MindKey of the catalogs to invalidate
        """
        key_str = key.to_string()

        if key_str in self.cache:
            catalog_count = len(self.cache[key_str])
            self.cache.pop(key_str)
            logger.info(
                f"Invalidated {catalog_count} data catalogs with key: {key_str}"
            )

    def clear(self) -> None:
        """Clear the entire cache."""
        self.cache.clear()
        logger.info("Cleared data catalog cache")

    def get_keys(self) -> List[str]:
        """Get a list of all keys in the cache."""
        return list(self.cache.keys())

    def size(self) -> int:
        """Get the current size of the cache."""
        return len(self.cache)


# Global instance for convenience
global_data_catalog_cache = DataCatalogInMemoryCache()
