"""
Unit tests for Minds schemas.

Tests the Pydantic models used for minds API including:
- Request validation
- Response serialization
- Field constraints
- Default values
"""

import pytest
from pydantic import ValidationError

from minds.model.mind_datasource import DataCatalogStatus, DetailedDataCatalogStatus
from minds.schemas.minds import (
    DatasourceConfig,
    DetailedDatasourceConfig,
    MindCreateRequest,
    MindResponse,
    MindUpdateRequest,
)


class TestMindCreateRequest:
    """Test suite for MindCreateRequest schema."""

    def test_valid_create_request(self):
        """Test valid mind creation request."""
        data = {
            "name": "test-mind",
            "provider": "openai",
            "model_name": "gpt-4o",
            "parameters": {"temperature": 0.7, "max_tokens": 100},
            "datasources": [
                DatasourceConfig(name="datasource1", tables=["table1", "table2"]),
                DatasourceConfig(name="datasource2", tables=None),
            ],
        }

        request = MindCreateRequest(**data)

        assert request.name == "test-mind"
        assert request.provider == "openai"
        assert request.model_name == "gpt-4o"
        assert request.parameters == {"temperature": 0.7, "max_tokens": 100}
        assert len(request.datasources) == 2
        assert request.datasources[0].name == "datasource1"
        assert request.datasources[0].tables == ["table1", "table2"]
        assert request.datasources[1].name == "datasource2"
        assert request.datasources[1].tables is None

    def test_minimal_create_request(self):
        """Test minimal valid creation request."""
        data = {
            "name": "minimal-mind"
            # Provider/model_name are optional unless explicitly provided
        }

        request = MindCreateRequest(**data)

        assert request.name == "minimal-mind"
        assert request.provider is None
        assert request.model_name is None  # Optional field
        assert request.parameters == {}  # Default factory
        assert request.datasources == []  # Default factory

    def test_create_request_with_defaults(self):
        """Test creation request with default values."""
        data = {
            "name": "default-mind",
            "provider": "google",
            "datasources": [],  # Empty list
        }

        with pytest.raises(ValidationError, match="Provider 'google' is not supported"):
            MindCreateRequest(**data)

    def test_create_request_missing_required_field(self):
        """Test creation request missing required fields."""
        # Missing name (the only required field)
        with pytest.raises(ValidationError) as exc_info:
            MindCreateRequest()

        assert "name" in str(exc_info.value)

        # Provider is optional if omitted
        request = MindCreateRequest(name="test-mind")
        assert request.provider is None

    def test_create_request_name_constraints(self):
        """Test name field constraints."""
        # Valid names
        valid_names = ["test-mind", "mind_123"]
        for name in valid_names:
            request = MindCreateRequest(name=name, provider="openai")
            assert request.name == name

        # Test name length constraints (assuming max_length=256)
        long_name = "a" * 256
        request = MindCreateRequest(name=long_name, provider="openai")
        assert request.name == long_name

        # Too long name (assuming max_length=256)
        too_long_name = "a" * 257
        with pytest.raises(ValidationError):
            MindCreateRequest(name=too_long_name, provider="openai")

    def test_create_request_parameters_validation(self):
        """Test parameters field validation."""
        # Valid parameters (any dict)
        valid_params = [
            {"temperature": 0.7},
            {"temperature": 0.7, "max_tokens": 100, "top_p": 0.9},
            {},  # Empty dict
            {"custom_param": "custom_value"},
        ]

        for params in valid_params:
            request = MindCreateRequest(name="test", provider="openai", parameters=params)
            assert request.parameters == params

    def test_create_request_datasources_validation(self):
        """Test datasources field validation."""
        # Valid datasources
        valid_datasources = [
            [],
            [DatasourceConfig(name="single-datasource")],
            [
                DatasourceConfig(name="datasource1", tables=["table1"]),
                DatasourceConfig(name="datasource2", tables=None),
                DatasourceConfig(name="datasource3", tables=["table3a", "table3b"]),
            ],
        ]

        for datasources in valid_datasources:
            request = MindCreateRequest(name="test", provider="openai", datasources=datasources)
            assert request.datasources == datasources

    def test_create_request_lowercase_name_and_datasources(self):
        """Test that name and datasource names are lowercased."""
        data = {
            "name": "Test-Mind",
            "provider": "openai",
            "datasources": [
                DatasourceConfig(name="DataSource1", tables=["Table1"]),
                DatasourceConfig(name="DATA_SOURCE_2", tables=None),
            ],
        }

        request = MindCreateRequest(**data)

        assert request.name == "test-mind"
        assert request.datasources[0].name == "datasource1"
        assert request.datasources[1].name == "data_source_2"


class TestMindUpdateRequest:
    """Test suite for MindUpdateRequest schema."""

    def test_update_request_all_fields_optional(self):
        """Test that all fields are optional in update request."""
        request = MindUpdateRequest()

        assert request.name is None
        assert request.provider is None
        assert request.model_name is None
        assert request.parameters is None
        assert request.datasources is None

    def test_update_request_partial_update(self):
        """Test partial update with some fields."""
        request = MindUpdateRequest(name="updated-name", parameters={"temperature": 0.9})

        assert request.name == "updated-name"
        assert request.parameters == {"temperature": 0.9}
        assert request.provider is None  # Not updated
        assert request.datasources is None  # Not updated

    def test_update_request_full_update(self):
        """Test full update with all fields.

        Schema-level validation only checks provider normalization (no context
        available), so any valid LLMProvider is accepted here.  Model-selection
        rules are enforced at the endpoint/service layer where context exists.
        """
        data = {
            "name": "fully-updated-mind",
            "provider": "anthropic",
            "model_name": "claude-3",
            "parameters": {"temperature": 0.5},
            "datasources": [DatasourceConfig(name="new-datasource", tables=["table1"])],
        }

        request = MindUpdateRequest(**data)

        assert request.name == "fully-updated-mind"
        assert request.provider == "anthropic"
        assert request.model_name == "claude-3"
        assert request.parameters == {"temperature": 0.5}
        assert len(request.datasources) == 1

    def test_update_request_lowercase_name_and_datasources(self):
        """Test that name and datasource names are lowercased in update request."""
        data = {
            "name": "Updated-Mind",
            "datasources": [
                DatasourceConfig(name="DataSource1", tables=["Table1"]),
                DatasourceConfig(name="DATA_SOURCE_2", tables=None),
            ],
        }

        request = MindUpdateRequest(**data)

        assert request.name == "updated-mind"
        assert request.datasources[0].name == "datasource1"
        assert request.datasources[1].name == "data_source_2"


class TestMindResponse:
    """Test suite for MindResponse schema."""

    def test_valid_mind_response(self):
        """Test valid mind response creation."""
        data = {
            "name": "response-mind",
            "model_name": "gpt-4o",
            "provider": "openai",
            "parameters": {"temperature": 0.7},
            "datasources": [DatasourceConfig(name="datasource1", tables=["table1"])],
            "created_at": "2024-01-01T00:00:00Z",
            "modified_at": "2024-01-01T12:00:00Z",
        }

        response = MindResponse(**data)

        assert response.name == "response-mind"
        assert response.model_name == "gpt-4o"
        assert response.provider == "openai"
        assert response.parameters == {"temperature": 0.7}
        assert len(response.datasources) == 1
        assert response.datasources[0].name == "datasource1"
        assert response.datasources[0].tables == ["table1"]
        assert response.created_at == "2024-01-01T00:00:00Z"
        assert response.modified_at == "2024-01-01T12:00:00Z"

    def test_mind_response_with_optional_fields(self):
        """Test mind response with optional fields."""
        data = {
            "name": "minimal-response",
            "model_name": "gpt-4o",
            "provider": "openai",
            "parameters": {},
            "datasources": [],
        }

        response = MindResponse(**data)

        assert response.created_at is None
        assert response.modified_at is None

    def test_mind_response_serialization(self):
        """Test mind response serialization to dict."""
        data = {
            "name": "serializable-mind",
            "model_name": "gpt-4o",
            "provider": "openai",
            "parameters": {"temperature": 0.7},
            "datasources": [DatasourceConfig(name="datasource1", tables=["table1"])],
            "created_at": "2024-01-01T00:00:00Z",
        }

        response = MindResponse(**data)
        serialized = response.model_dump()

        assert isinstance(serialized, dict)
        assert serialized["name"] == "serializable-mind"
        assert serialized["parameters"] == {"temperature": 0.7}

    def test_mind_response_with_detailed_datasource_configs(self):
        """Test mind response with DetailedDatasourceConfig objects."""
        detailed_datasource = DetailedDatasourceConfig(
            name="detailed-datasource",
            tables=["users", "orders"],
            status=DetailedDataCatalogStatus(tasks=[], progress=1.0, overall_status=DataCatalogStatus.COMPLETED),
            engine="postgres",
            description="Production database",
            connection_data={"host": "prod.db.com", "port": 5432},
            created_at="2024-01-01T00:00:00Z",
            modified_at="2024-01-01T12:00:00Z",
        )

        data = {
            "name": "response-with-detailed",
            "model_name": "gpt-4o",
            "provider": "openai",
            "parameters": {"temperature": 0.7},
            "datasources": [detailed_datasource],
        }

        response = MindResponse(**data)

        assert response.name == "response-with-detailed"
        assert len(response.datasources) == 1
        assert response.datasources[0].name == "detailed-datasource"
        assert response.datasources[0].tables == ["users", "orders"]
        assert response.datasources[0].engine == "postgres"
        assert response.datasources[0].description == "Production database"
        assert response.datasources[0].connection_data == {"host": "prod.db.com", "port": 5432}

    def test_mind_response_serialization_with_detailed_datasources(self):
        """Test serialization with DetailedDatasourceConfig objects."""
        detailed_datasource = DetailedDatasourceConfig(
            name="serializable-detailed",
            tables=["table1"],
            status=DetailedDataCatalogStatus(tasks=[], progress=0.5, overall_status=DataCatalogStatus.LOADING),
            engine="postgres",
            description="Test datasource",
            connection_data={"host": "localhost", "port": 5432},
            created_at="2024-01-01T00:00:00Z",
        )

        data = {
            "name": "serializable-mind",
            "model_name": "gpt-4o",
            "provider": "openai",
            "parameters": {"temperature": 0.7},
            "datasources": [detailed_datasource],
        }

        response = MindResponse(**data)
        serialized = response.model_dump()

        assert isinstance(serialized, dict)
        assert "datasources" in serialized
        assert len(serialized["datasources"]) == 1

        ds_serialized = serialized["datasources"][0]
        assert ds_serialized["name"] == "serializable-detailed"
        assert ds_serialized["engine"] == "postgres"
        assert ds_serialized["description"] == "Test datasource"
        assert ds_serialized["connection_data"] == {"host": "localhost", "port": 5432}
        assert ds_serialized["status"]["overall_status"] == "LOADING"
        assert ds_serialized["status"]["progress"] == 0.5
        assert ds_serialized["status"]["tasks"] == []

    def test_mind_response_status_computation(self):
        """Test status computation based on datasource statuses."""
        # Test with all completed datasources
        completed_datasource = DatasourceConfig(
            name="completed-ds",
            status=DetailedDataCatalogStatus(tasks=[], progress=1.0, overall_status=DataCatalogStatus.COMPLETED),
        )
        data = {
            "name": "test-mind",
            "model_name": "gpt-4o",
            "provider": "openai",
            "parameters": {},
            "datasources": [completed_datasource],
        }
        response = MindResponse(**data)
        assert response.status == DataCatalogStatus.COMPLETED

        # Test with pending datasource
        pending_datasource = DatasourceConfig(
            name="pending-ds",
            status=DetailedDataCatalogStatus(tasks=[], progress=0.0, overall_status=DataCatalogStatus.PENDING),
        )
        data["datasources"] = [completed_datasource, pending_datasource]
        response = MindResponse(**data)
        assert response.status == DataCatalogStatus.PENDING

        # Test with loading datasource
        loading_datasource = DatasourceConfig(
            name="loading-ds",
            status=DetailedDataCatalogStatus(tasks=[], progress=0.5, overall_status=DataCatalogStatus.LOADING),
        )
        data["datasources"] = [completed_datasource, loading_datasource]
        response = MindResponse(**data)
        assert response.status == DataCatalogStatus.LOADING

        # Test with failed datasource (should take precedence)
        failed_datasource = DatasourceConfig(
            name="failed-ds",
            status=DetailedDataCatalogStatus(tasks=[], progress=0.0, overall_status=DataCatalogStatus.FAILED),
        )
        data["datasources"] = [completed_datasource, failed_datasource]
        response = MindResponse(**data)
        assert response.status == DataCatalogStatus.FAILED

        # Test with empty datasources list
        data["datasources"] = []
        response = MindResponse(**data)
        assert response.status == DataCatalogStatus.COMPLETED


class TestSchemaInteroperability:
    """Test schema interoperability and edge cases."""

    def test_create_to_response_conversion(self):
        """Test that create request data can be used in response."""
        create_data = {
            "name": "interop-mind",
            "provider": "openai",
            "model_name": "gpt-4o",
            "parameters": {"temperature": 0.7},
            "datasources": [DatasourceConfig(name="datasource1", tables=["table1"])],
        }

        create_request = MindCreateRequest(**create_data)

        # Simulate what happens in the service layer
        response_data = {**create_data, "created_at": "2024-01-01T00:00:00Z", "modified_at": "2024-01-01T00:00:00Z"}

        response = MindResponse(**response_data)

        assert response.name == create_request.name
        assert response.provider == create_request.provider
        assert response.parameters == create_request.parameters

    def test_update_to_response_conversion(self):
        """Test update request compatibility with response."""
        update_data = {"name": "updated-interop-mind", "parameters": {"temperature": 0.9}}

        update_request = MindUpdateRequest(**update_data)

        # Original response data
        original_response_data = {
            "name": "original-mind",
            "model_name": "gpt-4o",
            "provider": "openai",
            "parameters": {"temperature": 0.7},
            "datasources": [DatasourceConfig(name="datasource1", tables=["table1"])],
            "created_at": "2024-01-01T00:00:00Z",
            "modified_at": "2024-01-01T00:00:00Z",
        }

        # Simulate update merge
        updated_response_data = original_response_data.copy()
        if update_request.name is not None:
            updated_response_data["name"] = update_request.name
        if update_request.parameters is not None:
            updated_response_data["parameters"] = update_request.parameters

        response = MindResponse(**updated_response_data)

        assert response.name == "updated-interop-mind"
        assert response.parameters == {"temperature": 0.9}
        assert response.provider == "openai"  # Unchanged
