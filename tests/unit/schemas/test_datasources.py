"""
Tests for datasource schemas.

Tests Pydantic validation, defaults, and serialization for datasource-related schemas.
"""

import pytest
from pydantic import ValidationError

from minds.schemas.datasources import (
    DatasourceConnectionStatus,
    DatasourceCreateRequest,
    DatasourceDetailedResponse,
    DatasourceResponse,
    DatasourceUpdateRequest,
    DeleteDatasourceRequest,
)


class TestDatasourceCreateRequest:
    """Test DatasourceCreateRequest schema."""

    def test_create_request_valid(self):
        """Test valid datasource creation request."""
        data = {
            "name": "test-db",
            "engine": "postgres",
            "connection_data": {"host": "localhost", "port": 5432}
        }
        request = DatasourceCreateRequest(**data)
        
        assert request.name == "test-db"
        assert request.engine == "postgres"
        assert request.connection_data == {"host": "localhost", "port": 5432}

    def test_create_request_missing_required_fields(self):
        """Test validation error for missing required fields."""
        with pytest.raises(ValidationError):
            DatasourceCreateRequest(name="test")  # Missing engine and connection_data




class TestDatasourceUpdateRequest:
    """Test DatasourceUpdateRequest schema."""

    def test_update_request_empty(self):
        """Test update request with no fields (all optional)."""
        request = DatasourceUpdateRequest()
        
        assert request.connection_data is None

    def test_update_request_partial(self):
        """Test update request with some fields."""
        data = {
            "connection_data": {"host": "new-host"}
        }
        request = DatasourceUpdateRequest(**data)
        
        assert request.connection_data == {"host": "new-host"}


class TestDatasourceResponse:
    """Test DatasourceResponse schema."""

    def test_response_minimal(self):
        """Test response with minimal required data."""
        data = {"name": "test-db"}
        response = DatasourceResponse(**data)
        
        assert response.name == "test-db"
        assert response.engine is None
        assert response.connection_data is None
        assert response.created_at is None
        assert response.is_demo is None

    def test_response_full(self):
        """Test response with all fields."""
        data = {
            "name": "test-db",
            "engine": "postgres",
            "connection_data": {"host": "localhost"},
            "created_at": "2023-01-01T00:00:00Z",
            "is_demo": True
        }
        response = DatasourceResponse(**data)
        
        assert response.name == "test-db"
        assert response.engine == "postgres"
        assert response.is_demo is True


class TestDatasourceConnectionStatus:
    """Test DatasourceConnectionStatus schema."""

    def test_connection_status_success(self):
        """Test successful connection status."""
        data = {"success": True}
        status = DatasourceConnectionStatus(**data)
        
        assert status.success is True
        assert status.error_message is None

    def test_connection_status_failure(self):
        """Test failed connection status."""
        data = {"success": False, "error_message": "Connection timeout"}
        status = DatasourceConnectionStatus(**data)
        
        assert status.success is False
        assert status.error_message == "Connection timeout"

    def test_connection_status_missing_success(self):
        """Test validation error for missing success field."""
        with pytest.raises(ValidationError):
            DatasourceConnectionStatus(error_message="Some error")


class TestDatasourceDetailedResponse:
    """Test DatasourceDetailedResponse schema (extends DatasourceResponse)."""

    def test_detailed_response_inheritance(self):
        """Test that detailed response inherits from base response."""
        data = {
            "name": "test-db",
            "engine": "mysql",
            "connection_status": {"success": True}
        }
        response = DatasourceDetailedResponse(**data)
        
        # Base fields
        assert response.name == "test-db"
        assert response.engine == "mysql"
        
        # Extended field
        assert response.connection_status.success is True

    def test_detailed_response_without_connection_status(self):
        """Test detailed response without connection status."""
        data = {"name": "test-db"}
        response = DatasourceDetailedResponse(**data)
        
        assert response.name == "test-db"
        assert response.connection_status is None


class TestDeleteDatasourceRequest:
    """Test DeleteDatasourceRequest schema."""

    def test_delete_request_default(self):
        """Test delete request with default cascade value."""
        request = DeleteDatasourceRequest()
        
        assert request.cascade is False

    def test_delete_request_cascade_true(self):
        """Test delete request with cascade enabled."""
        data = {"cascade": True}
        request = DeleteDatasourceRequest(**data)
        
        assert request.cascade is True


class TestSchemaIntegration:
    """Test integration between schemas."""

    def test_schemas_json_serialization(self):
        """Test that all schemas can be serialized to JSON."""
        # Create request
        create_req = DatasourceCreateRequest(
            name="test",
            engine="postgres", 
            connection_data={"host": "localhost"}
        )
        create_json = create_req.model_dump()
        assert "name" in create_json
        
        # Response
        response = DatasourceResponse(name="test")
        response_json = response.model_dump()
        assert "name" in response_json
        
        # Connection status
        status = DatasourceConnectionStatus(success=True)
        status_json = status.model_dump()
        assert "success" in status_json

    def test_field_descriptions_present(self):
        """Test that important fields have descriptions."""
        # Check a few key field descriptions
        create_fields = DatasourceCreateRequest.model_fields
        assert "Datasource name" in create_fields["name"].description
        assert "Database engine" in create_fields["engine"].description
        
        response_fields = DatasourceResponse.model_fields  
        assert "Datasource name" in response_fields["name"].description
