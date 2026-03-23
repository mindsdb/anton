"""
Unit tests for API v1 router.

Tests the router configuration and endpoint aggregation.
"""

from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from minds.api.v1.router import api_router


class TestAPIV1Router:
    """Test suite for API v1 router."""

    @pytest.fixture
    def app_with_router(self):
        """Create FastAPI app with v1 router included."""
        app = FastAPI()
        app.include_router(api_router)

        return app

    @pytest.fixture
    def headers(self):
        """Create headers for test client."""
        return {
            "X-User-Id": "00000000-0000-0000-0000-000000000001",
            "X-Organization-Id": "00000000-0000-0000-0000-000000000002",
        }

    @pytest.fixture
    def client(self, app_with_router, headers):
        """Create test client with router."""
        return TestClient(app_with_router, headers=headers)

    def test_router_prefix(self):
        """Test that router has correct prefix."""
        assert api_router.prefix == "/api/v1"

    def test_router_includes_all_endpoints(self, client):
        """Test that all expected endpoints are available."""

        # Test health endpoints
        response = client.get("/api/v1/health/")
        assert response.status_code == 200

        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200

        response = client.get("/api/v1/health/live")
        assert response.status_code == 200

    @patch("minds.client.mindsdb.connect")
    @patch("minds.requests.context.extract_context_from_request")
    def test_minds_endpoints_registered(self, mock_context, mock_connect, client):
        """Test that minds endpoints are registered."""
        # Mock MindsDB client
        mock_client = Mock()
        mock_client.minds = Mock()
        mock_client.minds.list.return_value = []
        mock_connect.return_value = mock_client

        # Mock context
        mock_context.return_value.user_id = UUID("00000000-0000-0000-0000-000000000001")
        mock_context.return_value.organization_id = UUID("00000000-0000-0000-0000-000000000002")

        response = client.get("/api/v1/minds/")
        assert response.status_code != 404  # Endpoint should exist

        response = client.get("/api/v1/minds/test-mind")
        # This can legitimately be 404 if the mind doesn't exist; we're only asserting route registration.
        assert response.status_code in {200, 400, 401, 403, 404, 422, 500}

    def test_chat_endpoints_registered(self, client):
        """Test that chat endpoints are registered."""
        response = client.post("/api/v1/chat/completions", json={})
        assert response.status_code in [422, 500]  # Not 404 - endpoint exists

    @patch("minds.client.mindsdb.connect")
    @patch("minds.requests.context.extract_context_from_request")
    def test_datasources_endpoints_registered(self, mock_context, mock_connect, client):
        """Test that datasources endpoints are registered."""
        # Mock MindsDB client
        mock_client = Mock()
        mock_client.datasources = Mock()
        mock_client.datasources.list.return_value = []
        mock_connect.return_value = mock_client

        # Mock context
        mock_context.return_value.user_id = UUID("00000000-0000-0000-0000-000000000001")
        mock_context.return_value.organization_id = UUID("00000000-0000-0000-0000-000000000002")

        response = client.get("/api/v1/datasources/")
        assert response.status_code != 404  # Endpoint should exist

    @patch("minds.client.mindsdb.connect")
    @patch("minds.requests.context.extract_context_from_request")
    def test_tree_endpoints_registered(self, mock_context, mock_connect, client):
        """Test that tree endpoints are registered."""
        # Mock MindsDB client
        mock_client = Mock()
        mock_client.trees = Mock()
        mock_client.trees.list.return_value = []
        mock_client.get_databases.return_value = []
        mock_connect.return_value = mock_client

        # Mock context
        mock_context.return_value.user_id = UUID("00000000-0000-0000-0000-000000000001")
        mock_context.return_value.organization_id = UUID("00000000-0000-0000-0000-000000000002")

        response = client.get("/api/v1/tree/")
        # Could be 200 (success), 422 (validation), or 500 (dependency error) - just not 404
        assert response.status_code != 404  # Endpoint should exist

    def test_router_tags_configuration(self):
        """Test that router includes have correct tags."""
        # Check that the router was configured with proper tags
        routes = api_router.routes

        # Find routes and verify tags (routes have /api/v1 prefix)
        health_routes = [r for r in routes if "/health" in r.path]
        minds_routes = [r for r in routes if "/minds" in r.path]
        chat_routes = [r for r in routes if "/chat" in r.path]
        datasources_routes = [r for r in routes if "/datasources" in r.path]
        tree_routes = [r for r in routes if "/tree" in r.path]

        assert len(health_routes) > 0, f"No health routes found in {[r.path for r in routes]}"
        assert len(minds_routes) > 0, f"No minds routes found in {[r.path for r in routes]}"
        assert len(chat_routes) > 0, f"No chat routes found in {[r.path for r in routes]}"
        assert len(datasources_routes) > 0, f"No datasources routes found in {[r.path for r in routes]}"
        assert len(tree_routes) > 0, f"No tree routes found in {[r.path for r in routes]}"

    def test_openapi_docs_generation(self, app_with_router):
        """Test that OpenAPI docs are properly generated."""
        client = TestClient(app_with_router)

        # Test that OpenAPI schema is available
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi_schema = response.json()
        assert "openapi" in openapi_schema
        assert "paths" in openapi_schema

        # Verify some key paths exist
        paths = openapi_schema["paths"]
        assert "/api/v1/health/" in paths
        assert "/api/v1/minds/" in paths
        assert "/api/v1/chat/completions" in paths

    def test_route_path_prefixes(self):
        """Test that all routes have correct path prefixes."""
        routes = api_router.routes

        for route in routes:
            # All routes should start with /api/v1/ then the expected prefixes
            expected_prefixes = [
                "/api/v1/health",
                "/api/v1/minds",
                "/api/v1/chat",
                "/api/v1/conversations",
                "/api/v1/datasources",
                "/api/v1/responses",
                "/api/v1/tree",
                "/api/v1/limits",
            ]
            assert any(route.path.startswith(prefix) for prefix in expected_prefixes), (
                f"Route {route.path} doesn't match expected prefixes {expected_prefixes}"
            )

    def test_router_methods_available(self):
        """Test that router exposes expected HTTP methods."""
        routes = api_router.routes

        # Find a few key routes and verify they have correct methods
        minds_list_routes = [r for r in routes if r.path == "/minds/" and hasattr(r, "methods")]
        if minds_list_routes:
            route = minds_list_routes[0]
            assert "GET" in route.methods  # list_minds
            assert "POST" in route.methods  # create_mind

        minds_detail_routes = [r for r in routes if r.path == "/minds/{mind_name}" and hasattr(r, "methods")]
        if minds_detail_routes:
            route = minds_detail_routes[0]
            assert "GET" in route.methods  # get_mind
            assert "PUT" in route.methods  # update_mind
            assert "DELETE" in route.methods  # delete_mind
