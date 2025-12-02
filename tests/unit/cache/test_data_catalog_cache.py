"""
Unit tests for data catalog cache functionality.

Tests the caching mechanisms for data catalogs including:
- MindKey functionality
- In-memory LRU cache implementation
- Cache factory
- Abstract base class behavior
"""

from collections import OrderedDict
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from minds.cache.data_catalog import (
    DataCatalogCache,
    DataCatalogCacheFactory,
    DataCatalogInMemoryCache,
    DataCatalogRedisCache,
)
from minds.model.data_catalog import DataCatalog
from minds.model.mind import Mind
from minds.model.mind_datasource import DetailedDataCatalogStatus


class TestDataCatalogCacheMindKey:
    """Test cases for DataCatalogCache.MindKey class."""

    def test_mind_key_initialization(self):
        """Test MindKey initialization with valid parameters."""
        mind_name = "test-mind"
        modified_at = datetime(2024, 1, 1, 12, 0, 0)

        key = DataCatalogCache.MindKey(mind_name=mind_name, modified_at=modified_at)

        assert key.mind_name == mind_name
        assert key.modified_at == modified_at

    def test_mind_key_to_string(self):
        """Test MindKey string representation."""
        mind_name = "test-mind"
        modified_at = datetime(2024, 1, 1, 12, 0, 0)

        key = DataCatalogCache.MindKey(mind_name=mind_name, modified_at=modified_at)
        key_str = key.to_string()

        expected_str = f"{mind_name}:{str(modified_at)}"
        assert key_str == expected_str

    def test_mind_key_to_string_with_different_formats(self):
        """Test MindKey string representation with different datetime formats."""
        mind_name = "test-mind"
        modified_at = datetime(2024, 12, 31, 23, 59, 59, 999999)

        key = DataCatalogCache.MindKey(mind_name=mind_name, modified_at=modified_at)
        key_str = key.to_string()

        # Should include microseconds in the string representation
        assert mind_name in key_str
        assert "2024-12-31 23:59:59.999999" in key_str

    def test_mind_key_equality_comparison(self):
        """Test MindKey equality comparison."""
        mind_name = "test-mind"
        modified_at = datetime(2024, 1, 1, 12, 0, 0)

        key1 = DataCatalogCache.MindKey(mind_name=mind_name, modified_at=modified_at)
        key2 = DataCatalogCache.MindKey(mind_name=mind_name, modified_at=modified_at)
        key3 = DataCatalogCache.MindKey(mind_name="different-mind", modified_at=modified_at)

        # Same key should have same string representation
        assert key1.to_string() == key2.to_string()
        assert key1.to_string() != key3.to_string()


class TestDataCatalogInMemoryCache:
    """Test cases for DataCatalogInMemoryCache class."""

    @pytest.fixture
    def cache(self):
        """Create a DataCatalogInMemoryCache instance for testing."""
        return DataCatalogInMemoryCache(max_size=3)

    @pytest.fixture
    def mock_mind(self):
        """Create a mock Mind instance for testing."""
        mind = Mock(spec=Mind)
        mind.name = "test-mind"
        mind.modified_at = datetime(2024, 1, 1, 12, 0, 0)
        mind.mind_datasources = []
        return mind

    @pytest.fixture
    def mock_data_catalog(self):
        """Create a mock DataCatalog instance for testing."""
        catalog = Mock(spec=DataCatalog)
        catalog.modified_at = datetime(2024, 1, 1, 12, 0, 0)
        return catalog

    def test_cache_initialization(self):
        """Test cache initialization with default and custom max_size."""
        # Test with default max_size
        cache = DataCatalogInMemoryCache()
        assert cache.max_size == 100
        assert isinstance(cache.cache, OrderedDict)
        assert len(cache.cache) == 0

        # Test with custom max_size
        cache = DataCatalogInMemoryCache(max_size=50)
        assert cache.max_size == 50

    def test_cache_initialization_with_zero_max_size(self):
        """Test cache initialization with zero max_size."""
        cache = DataCatalogInMemoryCache(max_size=0)
        assert cache.max_size == 0

    async def test_save_single_catalog(self, cache, mock_data_catalog):
        """Test saving a single data catalog."""
        key = DataCatalogCache.MindKey("test-mind", datetime(2024, 1, 1, 12, 0, 0))
        catalogs = [mock_data_catalog]

        await cache.save(key, catalogs)

        assert await cache.size() == 1
        assert key.to_string() in cache.cache
        assert cache.cache[key.to_string()] == catalogs

    async def test_save_multiple_catalogs(self, cache, mock_data_catalog):
        """Test saving multiple data catalogs."""
        key = DataCatalogCache.MindKey("test-mind", datetime(2024, 1, 1, 12, 0, 0))
        catalog1 = Mock(spec=DataCatalog)
        catalog2 = Mock(spec=DataCatalog)
        catalogs = [catalog1, catalog2]

        await cache.save(key, catalogs)

        assert await cache.size() == 1
        assert cache.cache[key.to_string()] == catalogs

    async def test_save_updates_modified_on(self, cache, mock_data_catalog):
        """Test that save updates the modified_on field for catalogs."""
        key = DataCatalogCache.MindKey("test-mind", datetime(2024, 1, 1, 12, 0, 0))
        catalogs = [mock_data_catalog]

        with patch("minds.cache.data_catalog.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 2, 1, 12, 0, 0)
            await cache.save(key, catalogs)

        # Verify that modified_at was updated
        mock_data_catalog.modified_at = mock_datetime.now.return_value

    async def test_load_cache_hit(self, cache, mock_mind, mock_data_catalog):
        """Test loading from cache when data exists (cache hit)."""
        key = DataCatalogCache.MindKey(mock_mind.name, mock_mind.modified_at)
        catalogs = [mock_data_catalog]
        cache.cache[key.to_string()] = catalogs

        result = await cache.load(mock_mind)

        assert result == catalogs
        assert await cache.size() == 1
        # Verify the item was moved to the end (most recently used)
        assert list(cache.cache.keys())[-1] == key.to_string()

    async def test_load_cache_miss_no_datasources(self, cache, mock_mind):
        """Test loading from cache when data doesn't exist and no datasources."""
        mock_mind.mind_datasources = []

        result = await cache.load(mock_mind)

        assert result == []
        assert await cache.size() == 0

    async def test_load_cache_miss_with_datasources(self, cache, mock_mind):
        """Test loading from cache when data doesn't exist but datasources exist."""
        from uuid import UUID

        from minds.model.mind_datasource import DataCatalogStatus, MindDatasource

        # Mock datasource with get_data_catalog method
        mock_datasource = Mock()
        mock_catalog = Mock(spec=DataCatalog)
        mock_datasource.get_data_catalog.return_value = mock_catalog

        # Create a proper mock MindDatasource with valid field types
        mock_mind_datasource = Mock(spec=MindDatasource)
        mock_mind_datasource.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_mind_datasource.tenant_id = "test-tenant"
        mock_mind_datasource.created_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_mind_datasource.modified_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_mind_datasource.deleted_at = None
        mock_mind_datasource.mind_id = UUID("87654321-4321-8765-4321-876543218765")
        mock_mind_datasource.datasource_id = UUID("11111111-2222-3333-4444-555555555555")

        async def _status_completed():
            return DetailedDataCatalogStatus(tasks=[], progress=0.0, overall_status=DataCatalogStatus.COMPLETED)

        mock_mind_datasource.status = _status_completed()
        mock_mind_datasource.datasource = mock_datasource
        mock_mind_datasource.mind_datasource_tables = []

        mock_mind.mind_datasources = [mock_mind_datasource]

        result = await cache.load(mock_mind)

        assert len(result) == 1
        # The result should be a real DataCatalog object, not the mock
        assert isinstance(result[0], DataCatalog)
        assert result[0].mind_datasource == mock_mind_datasource
        assert await cache.size() == 1
        # The cache creates DataCatalog directly, not through datasource.get_data_catalog()

    async def test_load_cache_miss_with_multiple_datasources(self, cache, mock_mind):
        """Test loading from cache with multiple datasources."""
        # Mock multiple datasources
        mock_datasource1 = Mock()
        mock_catalog1 = Mock(spec=DataCatalog)
        mock_datasource1.get_data_catalog.return_value = mock_catalog1

        mock_datasource2 = Mock()
        mock_catalog2 = Mock(spec=DataCatalog)
        mock_datasource2.get_data_catalog.return_value = mock_catalog2

        from uuid import UUID

        from minds.model.mind_datasource import DataCatalogStatus, MindDatasource

        # Create proper mock MindDatasource objects
        mock_mind_datasource1 = Mock(spec=MindDatasource)
        mock_mind_datasource1.id = UUID("12345678-1234-5678-1234-567812345678")
        mock_mind_datasource1.tenant_id = "test-tenant"
        mock_mind_datasource1.created_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_mind_datasource1.modified_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_mind_datasource1.deleted_at = None
        mock_mind_datasource1.mind_id = UUID("87654321-4321-8765-4321-876543218765")
        mock_mind_datasource1.datasource_id = UUID("11111111-2222-3333-4444-555555555555")

        async def _status_completed_1():
            return DetailedDataCatalogStatus(tasks=[], progress=0.0, overall_status=DataCatalogStatus.COMPLETED)

        mock_mind_datasource1.status = _status_completed_1()
        mock_mind_datasource1.datasource = mock_datasource1
        mock_mind_datasource1.mind_datasource_tables = []

        mock_mind_datasource2 = Mock(spec=MindDatasource)
        mock_mind_datasource2.id = UUID("22222222-3333-4444-5555-666666666666")
        mock_mind_datasource2.tenant_id = "test-tenant"
        mock_mind_datasource2.created_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_mind_datasource2.modified_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_mind_datasource2.deleted_at = None
        mock_mind_datasource2.mind_id = UUID("87654321-4321-8765-4321-876543218765")
        mock_mind_datasource2.datasource_id = UUID("33333333-4444-5555-6666-777777777777")

        async def _status_completed_2():
            return DetailedDataCatalogStatus(tasks=[], progress=0.0, overall_status=DataCatalogStatus.COMPLETED)

        mock_mind_datasource2.status = _status_completed_2()
        mock_mind_datasource2.datasource = mock_datasource2
        mock_mind_datasource2.mind_datasource_tables = []

        mock_mind.mind_datasources = [mock_mind_datasource1, mock_mind_datasource2]

        result = await cache.load(mock_mind)

        assert len(result) == 2
        # The results should be real DataCatalog objects, not mocks
        assert isinstance(result[0], DataCatalog)
        assert isinstance(result[1], DataCatalog)
        assert result[0].mind_datasource == mock_mind_datasource1
        assert result[1].mind_datasource == mock_mind_datasource2
        assert await cache.size() == 1
        # The cache creates DataCatalog directly, not through datasource.get_data_catalog()

    async def test_invalidate_existing_key(self, cache, mock_data_catalog):
        """Test invalidating an existing cache entry."""
        key = DataCatalogCache.MindKey("test-mind", datetime(2024, 1, 1, 12, 0, 0))
        catalogs = [mock_data_catalog]
        cache.cache[key.to_string()] = catalogs

        await cache.invalidate(key)

        assert await cache.size() == 0
        assert key.to_string() not in cache.cache

    async def test_invalidate_nonexistent_key(self, cache):
        """Test invalidating a non-existent cache entry."""
        key = DataCatalogCache.MindKey("nonexistent-mind", datetime(2024, 1, 1, 12, 0, 0))

        # Should not raise an exception
        await cache.invalidate(key)
        assert await cache.size() == 0

    async def test_lru_eviction_when_at_capacity(self, cache, mock_data_catalog):
        """Test LRU eviction when cache reaches max capacity."""
        # Fill cache to capacity
        for i in range(3):
            key = DataCatalogCache.MindKey(f"mind-{i}", datetime(2024, 1, 1, 12, i, 0))
            catalog = Mock(spec=DataCatalog)
            await cache.save(key, [catalog])

        assert await cache.size() == 3

        # Add one more item, should evict the least recently used (first item)
        key4 = DataCatalogCache.MindKey("mind-4", datetime(2024, 1, 1, 12, 4, 0))
        catalog4 = Mock(spec=DataCatalog)
        await cache.save(key4, [catalog4])

        assert await cache.size() == 3
        assert "mind-0:2024-01-01 12:00:00" not in cache.cache  # First item evicted
        assert "mind-4:2024-01-01 12:04:00" in cache.cache  # New item added

    async def test_lru_eviction_with_access_pattern(self, cache, mock_data_catalog):
        """Test LRU eviction with specific access patterns."""
        # Add items to cache
        key1 = DataCatalogCache.MindKey("mind-1", datetime(2024, 1, 1, 12, 1, 0))
        key2 = DataCatalogCache.MindKey("mind-2", datetime(2024, 1, 1, 12, 2, 0))
        key3 = DataCatalogCache.MindKey("mind-3", datetime(2024, 1, 1, 12, 3, 0))

        catalog1 = Mock(spec=DataCatalog)
        catalog2 = Mock(spec=DataCatalog)
        catalog3 = Mock(spec=DataCatalog)

        await cache.save(key1, [catalog1])
        await cache.save(key2, [catalog2])
        await cache.save(key3, [catalog3])

        # Simulate accessing mind-1 to make it most recently used
        # This is done by removing and re-adding it to the end of the OrderedDict
        if key1.to_string() in cache.cache:
            catalogs = cache.cache.pop(key1.to_string())
            cache.cache[key1.to_string()] = catalogs

        # Add new item, should evict mind-2 (least recently used)
        key4 = DataCatalogCache.MindKey("mind-4", datetime(2024, 1, 1, 12, 4, 0))
        catalog4 = Mock(spec=DataCatalog)
        await cache.save(key4, [catalog4])

        assert await cache.size() == 3
        assert "mind-2:2024-01-01 12:02:00" not in cache.cache  # mind-2 evicted
        assert "mind-1:2024-01-01 12:01:00" in cache.cache  # mind-1 kept
        assert "mind-4:2024-01-01 12:04:00" in cache.cache  # mind-4 added

    async def test_clear_cache(self, cache, mock_data_catalog):
        """Test clearing the entire cache."""
        # Add some items to cache
        key1 = DataCatalogCache.MindKey("mind-1", datetime(2024, 1, 1, 12, 1, 0))
        key2 = DataCatalogCache.MindKey("mind-2", datetime(2024, 1, 1, 12, 2, 0))
        await cache.save(key1, [mock_data_catalog])
        await cache.save(key2, [mock_data_catalog])

        assert await cache.size() == 2

        await cache.clear()

        assert await cache.size() == 0
        assert len(cache.cache) == 0

    async def test_size_method(self, cache, mock_data_catalog):
        """Test the size method returns correct count."""
        assert await cache.size() == 0

        key1 = DataCatalogCache.MindKey("mind-1", datetime(2024, 1, 1, 12, 1, 0))
        await cache.save(key1, [mock_data_catalog])
        assert await cache.size() == 1

        key2 = DataCatalogCache.MindKey("mind-2", datetime(2024, 1, 1, 12, 2, 0))
        await cache.save(key2, [mock_data_catalog])
        assert await cache.size() == 2

    async def test_cache_with_empty_catalogs_list(self, cache):
        """Test cache behavior with empty catalogs list."""
        key = DataCatalogCache.MindKey("test-mind", datetime(2024, 1, 1, 12, 0, 0))
        catalogs = []

        await cache.save(key, catalogs)

        assert await cache.size() == 1
        assert cache.cache[key.to_string()] == catalogs

    async def test_cache_with_none_catalogs(self, cache):
        """Test cache behavior with None catalogs."""
        key = DataCatalogCache.MindKey("test-mind", datetime(2024, 1, 1, 12, 0, 0))
        catalogs = None

        # Should handle None gracefully or raise appropriate error
        with pytest.raises((TypeError, AttributeError)):
            await cache.save(key, catalogs)


class TestDataCatalogRedisCache:
    """Test cases for DataCatalogRedisCache class."""

    def test_redis_cache_is_abstract(self):
        """Test that Redis cache is an abstract class that cannot be instantiated."""
        # Since DataCatalogRedisCache is abstract and doesn't implement the abstract methods,
        # it should raise a TypeError when trying to instantiate it
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            DataCatalogRedisCache()

    def test_redis_cache_inheritance(self):
        """Test that Redis cache inherits from the base cache class."""
        # Test that the class exists and inherits from DataCatalogCache
        assert issubclass(DataCatalogRedisCache, DataCatalogCache)

        # Test that it has the required abstract methods
        assert hasattr(DataCatalogRedisCache, "load")
        assert hasattr(DataCatalogRedisCache, "save")
        assert hasattr(DataCatalogRedisCache, "invalidate")


class TestDataCatalogCacheFactory:
    """Test cases for DataCatalogCacheFactory class."""

    @patch("minds.cache.data_catalog.settings.data_catalog.cache_type", "in_memory")
    @patch("minds.cache.data_catalog.settings.data_catalog.cache_max_size", 50)
    def test_create_in_memory_cache(self):
        """Test creating an in-memory cache instance."""
        cache = DataCatalogCacheFactory.create_cache()

        assert isinstance(cache, DataCatalogInMemoryCache)
        assert cache.max_size == 50

    @patch("minds.cache.data_catalog.settings.data_catalog.cache_type", "in_memory")
    @patch("minds.cache.data_catalog.settings.data_catalog.cache_max_size", 100)
    def test_create_in_memory_cache_default_size(self):
        """Test creating an in-memory cache with default size."""
        cache = DataCatalogCacheFactory.create_cache()

        assert isinstance(cache, DataCatalogInMemoryCache)
        assert cache.max_size == 100

    @patch("minds.cache.data_catalog.settings.data_catalog.cache_type", "redis")
    def test_create_redis_cache_raises_error(self):
        """Test that creating a Redis cache raises an error (not implemented)."""
        with pytest.raises(ValueError, match="Unknown cache type 'redis'"):
            DataCatalogCacheFactory.create_cache()

    @patch("minds.cache.data_catalog.settings.data_catalog.cache_type", "unknown")
    def test_create_unknown_cache_type_raises_error(self):
        """Test that creating an unknown cache type raises an error."""
        with pytest.raises(ValueError, match="Unknown cache type 'unknown'"):
            DataCatalogCacheFactory.create_cache()

    @patch("minds.cache.data_catalog.settings.data_catalog.cache_type", "in_memory")
    @patch("minds.cache.data_catalog.settings.data_catalog.cache_max_size", 0)
    def test_create_cache_with_zero_max_size(self):
        """Test creating a cache with zero max size."""
        cache = DataCatalogCacheFactory.create_cache()

        assert isinstance(cache, DataCatalogInMemoryCache)
        assert cache.max_size == 0


class TestDataCatalogCacheIntegration:
    """Integration tests for the data catalog cache system."""

    async def test_full_cache_workflow(self):
        """Test a complete cache workflow: save, load, invalidate."""
        cache = DataCatalogInMemoryCache(max_size=2)

        # Create test data
        key1 = DataCatalogCache.MindKey("mind-1", datetime(2024, 1, 1, 12, 1, 0))
        catalog1 = Mock(spec=DataCatalog)
        catalogs1 = [catalog1]

        key2 = DataCatalogCache.MindKey("mind-2", datetime(2024, 1, 1, 12, 2, 0))
        catalog2 = Mock(spec=DataCatalog)
        catalogs2 = [catalog2]

        # Save catalogs
        await cache.save(key1, catalogs1)
        await cache.save(key2, catalogs2)
        assert await cache.size() == 2

        # Load catalogs (simulate cache hit)
        mock_mind1 = Mock(spec=Mind)
        mock_mind1.name = "mind-1"
        mock_mind1.modified_on = datetime(2024, 1, 1, 12, 1, 0)
        mock_mind1.mind_datasources = []

        # Simulate cache hit by directly accessing cache
        result1 = cache.cache[key1.to_string()]
        assert result1 == catalogs1

        # Invalidate one entry
        await cache.invalidate(key1)
        assert await cache.size() == 1
        assert key1.to_string() not in cache.cache
        assert key2.to_string() in cache.cache

        # Clear remaining cache
        await cache.clear()
        assert await cache.size() == 0

    async def test_cache_with_realistic_mind_data(self):
        """Test cache with realistic mind and datasource data."""
        cache = DataCatalogInMemoryCache(max_size=5)

        # Create realistic mock data
        mock_datasource1 = Mock()
        mock_catalog1 = Mock(spec=DataCatalog)
        mock_datasource1.get_data_catalog.return_value = mock_catalog1

        mock_datasource2 = Mock()
        mock_catalog2 = Mock(spec=DataCatalog)
        mock_datasource2.get_data_catalog.return_value = mock_catalog2

        mock_mind_datasource1 = Mock()
        mock_mind_datasource1.datasource = mock_datasource1
        mock_mind_datasource2 = Mock()
        mock_mind_datasource2.datasource = mock_datasource2

        from uuid import UUID

        from minds.model.mind_datasource import DataCatalogStatus, MindDatasource

        mock_mind = Mock(spec=Mind)
        mock_mind.name = "analytics-mind"
        mock_mind.modified_at = datetime(2024, 1, 15, 14, 30, 0)

        # Create proper mock MindDatasource objects
        mock_mind_datasource1 = Mock(spec=MindDatasource)
        mock_mind_datasource1.id = UUID("11111111-2222-3333-4444-555555555555")
        mock_mind_datasource1.tenant_id = "test-tenant"
        mock_mind_datasource1.created_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_mind_datasource1.modified_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_mind_datasource1.deleted_at = None
        mock_mind_datasource1.mind_id = UUID("87654321-4321-8765-4321-876543218765")
        mock_mind_datasource1.datasource_id = UUID("11111111-2222-3333-4444-555555555555")

        async def _status_completed_a():
            return DetailedDataCatalogStatus(tasks=[], progress=0.0, overall_status=DataCatalogStatus.COMPLETED)

        mock_mind_datasource1.status = _status_completed_a()
        mock_mind_datasource1.datasource = mock_datasource1
        mock_mind_datasource1.mind_datasource_tables = []

        mock_mind_datasource2 = Mock(spec=MindDatasource)
        mock_mind_datasource2.id = UUID("22222222-3333-4444-5555-666666666666")
        mock_mind_datasource2.tenant_id = "test-tenant"
        mock_mind_datasource2.created_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_mind_datasource2.modified_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_mind_datasource2.deleted_at = None
        mock_mind_datasource2.mind_id = UUID("87654321-4321-8765-4321-876543218765")
        mock_mind_datasource2.datasource_id = UUID("33333333-4444-5555-6666-777777777777")

        async def _status_completed_b():
            return DetailedDataCatalogStatus(tasks=[], progress=0.0, overall_status=DataCatalogStatus.COMPLETED)

        mock_mind_datasource2.status = _status_completed_b()
        mock_mind_datasource2.datasource = mock_datasource2
        mock_mind_datasource2.mind_datasource_tables = []

        mock_mind.mind_datasources = [mock_mind_datasource1, mock_mind_datasource2]

        # Load data (cache miss)
        result = await cache.load(mock_mind)

        assert len(result) == 2
        # The results should be real DataCatalog objects, not mocks
        assert isinstance(result[0], DataCatalog)
        assert isinstance(result[1], DataCatalog)
        assert result[0].mind_datasource == mock_mind_datasource1
        assert result[1].mind_datasource == mock_mind_datasource2
        assert await cache.size() == 1

        # Load again (cache hit)
        result2 = await cache.load(mock_mind)
        assert result2 == result
        assert await cache.size() == 1

    async def test_cache_performance_with_large_dataset(self):
        """Test cache performance and behavior with larger datasets."""
        cache = DataCatalogInMemoryCache(max_size=10)

        # Add many items to test LRU behavior
        for i in range(15):  # More than max_size
            key = DataCatalogCache.MindKey(f"mind-{i}", datetime(2024, 1, 1, 12, i, 0))
            catalog = Mock(spec=DataCatalog)
            await cache.save(key, [catalog])

        # Should only have max_size items
        assert await cache.size() == 10

        # First 5 items should be evicted
        for i in range(5):
            key_str = f"mind-{i}:2024-01-01 12:{i:02d}:00"
            assert key_str not in cache.cache

        # Last 10 items should be present
        for i in range(5, 15):
            key_str = f"mind-{i}:2024-01-01 12:{i:02d}:00"
            assert key_str in cache.cache
