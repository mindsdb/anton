"""
Tests for mind schemas.

Tests Pydantic validation, defaults, and serialization for mind-related schemas.
"""

import pytest
from pydantic import ValidationError

from minds.model.mind_datasource import DataCatalogStatus
from minds.schemas.minds import (
    DatasourceConfig,
    DetailedDatasourceConfig,
    MindCreateRequest,
    MindResponse,
    MindUpdateRequest,
)


class TestDatasourceConfig:
    """Test DatasourceConfig schema."""

    def test_config_minimal(self):
        """Test config with only required name field."""
        data = {"name": "postgres-db"}
        config = DatasourceConfig(**data)

        assert config.name == "postgres-db"
        assert config.tables is None
        assert config.status == DataCatalogStatus.PENDING

    def test_config_with_tables(self):
        """Test config with specific tables."""
        data = {"name": "mysql-db", "tables": ["users", "orders", "products"]}
        config = DatasourceConfig(**data)

        assert config.name == "mysql-db"
        assert config.tables == ["users", "orders", "products"]
        assert config.status == DataCatalogStatus.PENDING

    def test_config_with_status(self):
        """Test config with custom status."""
        data = {"name": "test-db", "status": DataCatalogStatus.LOADING}
        config = DatasourceConfig(**data)

        assert config.name == "test-db"
        assert config.status == DataCatalogStatus.LOADING

    def test_config_all_fields(self):
        """Test config with all fields."""
        data = {
            "name": "prod-db",
            "tables": ["table1", "table2"],
            "status": DataCatalogStatus.COMPLETED,
        }
        config = DatasourceConfig(**data)

        assert config.name == "prod-db"
        assert config.tables == ["table1", "table2"]
        assert config.status == DataCatalogStatus.COMPLETED

    def test_config_missing_name(self):
        """Test validation error for missing name field."""
        with pytest.raises(ValidationError):
            DatasourceConfig()

    def test_config_empty_tables_list(self):
        """Test config with empty tables list."""
        data = {"name": "test-db", "tables": []}
        config = DatasourceConfig(**data)

        assert config.tables == []


class TestDetailedDatasourceConfig:
    """Test DetailedDatasourceConfig schema."""

    def test_detailed_config_minimal(self):
        """Test detailed config with only required name field."""
        data = {"name": "postgres-db"}
        config = DetailedDatasourceConfig(**data)

        assert config.name == "postgres-db"
        assert config.engine is None
        assert config.description is None
        assert config.connection_data is None
        assert config.created_at is None
        assert config.modified_at is None

    def test_detailed_config_full(self):
        """Test detailed config with all fields."""
        data = {
            "name": "postgres-db",
            "tables": ["users", "orders"],
            "status": DataCatalogStatus.COMPLETED,
            "engine": "postgres",
            "description": "Production PostgreSQL database",
            "connection_data": {"host": "db.example.com", "port": 5432},
            "created_at": "2023-01-01T00:00:00Z",
            "modified_at": "2023-12-01T00:00:00Z",
        }
        config = DetailedDatasourceConfig(**data)

        assert config.name == "postgres-db"
        assert config.engine == "postgres"
        assert config.description == "Production PostgreSQL database"
        assert config.connection_data == {"host": "db.example.com", "port": 5432}
        assert config.created_at == "2023-01-01T00:00:00Z"
        assert config.modified_at == "2023-12-01T00:00:00Z"

    def test_detailed_config_inheritance(self):
        """Test that detailed config inherits from DatasourceConfig."""
        data = {"name": "test-db", "tables": ["t1"], "status": DataCatalogStatus.LOADING}
        config = DetailedDatasourceConfig(**data)

        assert config.name == "test-db"
        assert config.tables == ["t1"]
        assert config.status == DataCatalogStatus.LOADING


class TestMindCreateRequest:
    """Test MindCreateRequest schema."""

    def test_create_request_minimal(self):
        """Test create request with only required fields."""
        data = {"name": "Test-Mind"}
        request = MindCreateRequest(**data)

        assert request.name == "test-mind"
        assert request.provider == "openai"
        assert request.model_name is None
        assert request.parameters == {}
        assert request.datasources == []

    def test_create_request_full(self):
        """Test create request with all fields."""
        data = {
            "name": "Advanced-Mind",
            "provider": "google",
            "model_name": "gemini-pro",
            "parameters": {"temperature": 0.7, "max_tokens": 1000},
            "datasources": [
                {"name": "DB1", "tables": ["users"]},
                {"name": "DB2"},
            ],
        }
        request = MindCreateRequest(**data)

        assert request.name == "advanced-mind"
        assert request.provider == "google"
        assert request.model_name == "gemini-pro"
        assert request.parameters == {"temperature": 0.7, "max_tokens": 1000}
        assert len(request.datasources) == 2
        assert request.datasources[0].name == "db1"
        assert request.datasources[1].name == "db2"

    def test_create_request_lowercase_name(self):
        """Test that name is lowercased."""
        data = {"name": "MyAwesome-Mind"}
        request = MindCreateRequest(**data)

        assert request.name == "myawesome-mind"

    def test_create_request_lowercase_datasource_names(self):
        """Test that datasource names are lowercased."""
        data = {
            "name": "test-mind",
            "datasources": [
                {"name": "PostgreSQL-DB"},
                {"name": "MySQL-DB"},
            ],
        }
        request = MindCreateRequest(**data)

        assert request.datasources[0].name == "postgresql-db"
        assert request.datasources[1].name == "mysql-db"

    def test_create_request_name_validation_empty(self):
        """Test validation error for empty name."""
        with pytest.raises(ValidationError):
            MindCreateRequest(name="")

    def test_create_request_name_validation_too_long(self):
        """Test validation error for name exceeding max length."""
        long_name = "a" * 257
        with pytest.raises(ValidationError):
            MindCreateRequest(name=long_name)

    def test_create_request_with_empty_datasources(self):
        """Test create request with empty datasources list."""
        data = {"name": "test-mind", "datasources": []}
        request = MindCreateRequest(**data)

        assert request.datasources == []

    def test_create_request_default_parameters(self):
        """Test create request with default parameters."""
        data = {"name": "test-mind"}
        request = MindCreateRequest(**data)

        assert isinstance(request.parameters, dict)
        assert len(request.parameters) == 0

    def test_create_request_custom_parameters(self):
        """Test create request with custom parameters."""
        data = {
            "name": "test-mind",
            "parameters": {
                "temperature": 0.5,
                "top_p": 0.9,
                "frequency_penalty": 0.2,
            },
        }
        request = MindCreateRequest(**data)

        assert request.parameters["temperature"] == 0.5
        assert request.parameters["top_p"] == 0.9
        assert request.parameters["frequency_penalty"] == 0.2


class TestMindUpdateRequest:
    """Test MindUpdateRequest schema."""

    def test_update_request_empty(self):
        """Test update request with no fields (all optional)."""
        request = MindUpdateRequest()

        assert request.name is None
        assert request.provider is None
        assert request.model_name is None
        assert request.parameters is None
        assert request.datasources is None

    def test_update_request_partial_name(self):
        """Test update request with only name."""
        data = {"name": "Updated-Mind"}
        request = MindUpdateRequest(**data)

        assert request.name == "updated-mind"
        assert request.provider is None
        assert request.model_name is None

    def test_update_request_partial_provider(self):
        """Test update request with only provider."""
        data = {"provider": "anthropic"}
        request = MindUpdateRequest(**data)

        assert request.name is None
        assert request.provider == "anthropic"
        assert request.model_name is None

    def test_update_request_partial_parameters(self):
        """Test update request with only parameters."""
        data = {"parameters": {"temperature": 0.9}}
        request = MindUpdateRequest(**data)

        assert request.name is None
        assert request.parameters == {"temperature": 0.9}

    def test_update_request_partial_datasources(self):
        """Test update request with only datasources."""
        data = {"datasources": [{"name": "NewDB"}]}
        request = MindUpdateRequest(**data)

        assert request.name is None
        assert len(request.datasources) == 1
        assert request.datasources[0].name == "newdb"

    def test_update_request_full(self):
        """Test update request with all fields."""
        data = {
            "name": "Updated-Mind",
            "provider": "openai",
            "model_name": "gpt-4",
            "parameters": {"temperature": 0.5},
            "datasources": [{"name": "DB1"}],
        }
        request = MindUpdateRequest(**data)

        assert request.name == "updated-mind"
        assert request.provider == "openai"
        assert request.model_name == "gpt-4"
        assert request.parameters == {"temperature": 0.5}
        assert request.datasources[0].name == "db1"

    def test_update_request_lowercase_name(self):
        """Test that name is lowercased in update."""
        data = {"name": "MyUpdated-Mind"}
        request = MindUpdateRequest(**data)

        assert request.name == "myupdated-mind"

    def test_update_request_lowercase_datasource_names(self):
        """Test that datasource names are lowercased in update."""
        data = {
            "datasources": [
                {"name": "PostgreSQL-Updated"},
                {"name": "MongoDB-New"},
            ]
        }
        request = MindUpdateRequest(**data)

        assert request.datasources[0].name == "postgresql-updated"
        assert request.datasources[1].name == "mongodb-new"

    def test_update_request_name_validation_empty(self):
        """Test validation error for empty name."""
        with pytest.raises(ValidationError):
            MindUpdateRequest(name="")

    def test_update_request_name_validation_too_long(self):
        """Test validation error for name exceeding max length."""
        long_name = "a" * 257
        with pytest.raises(ValidationError):
            MindUpdateRequest(name=long_name)


class TestMindResponse:
    """Test MindResponse schema."""

    def test_response_minimal(self):
        """Test response with only required fields."""
        data = {
            "name": "test-mind",
            "provider": "openai",
            "model_name": "gpt-3.5-turbo",
            "parameters": {"temperature": 0.7},
            "datasources": [],
        }
        response = MindResponse(**data)

        assert response.name == "test-mind"
        assert response.provider == "openai"
        assert response.model_name == "gpt-3.5-turbo"
        assert response.parameters == {"temperature": 0.7}
        assert response.datasources == []
        assert response.created_at is None
        assert response.modified_at is None

    def test_response_full(self):
        """Test response with all fields."""
        data = {
            "name": "advanced-mind",
            "provider": "anthropic",
            "model_name": "claude-3",
            "parameters": {
                "temperature": 0.5,
                "max_tokens": 2000,
                "system_prompt": "You are a helpful assistant",
            },
            "datasources": [
                {"name": "postgres-db", "tables": ["users", "orders"]},
                {
                    "name": "mysql-db",
                    "engine": "mysql",
                    "description": "Analytics database",
                    "connection_data": {"host": "analytics.example.com"},
                },
            ],
            "created_at": "2023-01-01T00:00:00Z",
            "modified_at": "2023-12-01T00:00:00Z",
        }
        response = MindResponse(**data)

        assert response.name == "advanced-mind"
        assert response.provider == "anthropic"
        assert response.model_name == "claude-3"
        assert response.parameters["system_prompt"] == "You are a helpful assistant"
        assert len(response.datasources) == 2
        assert response.created_at == "2023-01-01T00:00:00Z"
        assert response.modified_at == "2023-12-01T00:00:00Z"

    def test_response_with_basic_datasources(self):
        """Test response with basic datasource configs."""
        data = {
            "name": "test-mind",
            "provider": "openai",
            "model_name": "gpt-4",
            "parameters": {},
            "datasources": [
                {"name": "db1", "tables": ["t1"]},
                {"name": "db2"},
            ],
        }
        response = MindResponse(**data)

        assert len(response.datasources) == 2
        assert response.datasources[0].name == "db1"
        assert response.datasources[1].name == "db2"

    def test_response_with_detailed_datasources(self):
        """Test response with detailed datasource configs."""
        data = {
            "name": "test-mind",
            "provider": "openai",
            "model_name": "gpt-4",
            "parameters": {},
            "datasources": [
                {
                    "name": "postgres-db",
                    "engine": "postgres",
                    "description": "Main database",
                    "connection_data": {"host": "db.example.com"},
                }
            ],
        }
        response = MindResponse(**data)

        assert len(response.datasources) == 1
        ds = response.datasources[0]
        assert ds.name == "postgres-db"
        assert ds.engine == "postgres"
        assert ds.description == "Main database"

    def test_response_status_pending(self):
        """Test mind status when datasource is pending."""
        data = {
            "name": "test-mind",
            "provider": "openai",
            "model_name": "gpt-4",
            "parameters": {},
            "datasources": [
                {"name": "db1", "status": DataCatalogStatus.PENDING},
            ],
        }
        response = MindResponse(**data)

        assert response.status == DataCatalogStatus.PENDING

    def test_response_status_loading(self):
        """Test mind status when datasource is loading."""
        data = {
            "name": "test-mind",
            "provider": "openai",
            "model_name": "gpt-4",
            "parameters": {},
            "datasources": [
                {"name": "db1", "status": DataCatalogStatus.LOADING},
            ],
        }
        response = MindResponse(**data)

        assert response.status == DataCatalogStatus.LOADING

    def test_response_status_failed(self):
        """Test mind status when datasource failed."""
        data = {
            "name": "test-mind",
            "provider": "openai",
            "model_name": "gpt-4",
            "parameters": {},
            "datasources": [
                {"name": "db1", "status": DataCatalogStatus.FAILED},
            ],
        }
        response = MindResponse(**data)

        assert response.status == DataCatalogStatus.FAILED

    def test_response_status_completed(self):
        """Test mind status when all datasources are completed."""
        data = {
            "name": "test-mind",
            "provider": "openai",
            "model_name": "gpt-4",
            "parameters": {},
            "datasources": [
                {"name": "db1", "status": DataCatalogStatus.COMPLETED},
                {"name": "db2", "status": DataCatalogStatus.COMPLETED},
            ],
        }
        response = MindResponse(**data)

        assert response.status == DataCatalogStatus.COMPLETED

    def test_response_status_no_datasources(self):
        """Test mind status when there are no datasources."""
        data = {
            "name": "test-mind",
            "provider": "openai",
            "model_name": "gpt-4",
            "parameters": {},
            "datasources": [],
        }
        response = MindResponse(**data)

        assert response.status == DataCatalogStatus.COMPLETED

    def test_response_missing_required_field(self):
        """Test validation error for missing required field."""
        with pytest.raises(ValidationError):
            MindResponse(
                name="test-mind",
                provider="openai",
                # Missing model_name
                parameters={},
                datasources=[],
            )


class TestSchemaIntegration:
    """Test integration between mind schemas."""

    def test_create_to_response_conversion(self):
        """Test conversion from create request to response."""
        create_data = {
            "name": "Test-Mind",
            "provider": "openai",
            "model_name": "gpt-4",
            "parameters": {"temperature": 0.7},
            "datasources": [{"name": "DB1"}],
        }
        create_req = MindCreateRequest(**create_data)

        # Simulate conversion to response
        response_data = {
            "name": create_req.name,
            "provider": create_req.provider,
            "model_name": create_req.model_name,
            "parameters": create_req.parameters,
            "datasources": create_req.datasources,
        }
        response = MindResponse(**response_data)

        assert response.name == "test-mind"
        assert response.datasources[0].name == "db1"

    def test_update_preserves_existing_fields(self):
        """Test that partial update doesn't overwrite omitted fields."""
        update_data = {"name": "Updated-Mind"}
        update_req = MindUpdateRequest(**update_data)

        # Only name should be set
        assert update_req.name == "updated-mind"
        assert update_req.provider is None
        assert update_req.parameters is None

    def test_schemas_json_serialization(self):
        """Test that all schemas can be serialized to JSON."""
        create_req = MindCreateRequest(
            name="Test-Mind",
            provider="openai",
            model_name="gpt-4",
            parameters={"temperature": 0.7},
            datasources=[{"name": "DB1"}],
        )
        create_json = create_req.model_dump()
        assert "name" in create_json
        assert create_json["name"] == "test-mind"

        config = DatasourceConfig(name="TestDB", tables=["t1"])
        config_json = config.model_dump()
        assert "name" in config_json
        assert config_json["name"] == "TestDB"

        response = MindResponse(
            name="test-mind",
            provider="openai",
            model_name="gpt-4",
            parameters={},
            datasources=[],
        )
        response_json = response.model_dump()
        assert "name" in response_json
        assert "provider" in response_json

    def test_field_descriptions_present(self):
        """Test that important fields have descriptions."""
        create_fields = MindCreateRequest.model_fields
        assert "Name of the mind" in create_fields["name"].description
        assert "AI provider" in create_fields["provider"].description
        assert "Model name" in create_fields["model_name"].description
        assert "datasource" in create_fields["datasources"].description.lower()

        response_fields = MindResponse.model_fields
        assert "Mind name" in response_fields["name"].description
        assert "AI provider" in response_fields["provider"].description

        datasource_fields = DatasourceConfig.model_fields
        assert "Name of the datasource" in datasource_fields["name"].description
        assert "tables" in datasource_fields["tables"].description.lower()
