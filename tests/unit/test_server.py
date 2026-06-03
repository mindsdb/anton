"""Tests for minds.server module."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from minds.server import app, create_app


class TestCreateApp:
    """Tests for create_app function."""

    def test_create_app_returns_fastapi_instance(self):
        """Test that create_app returns a FastAPI instance."""
        test_app = create_app()
        assert test_app is not None
        assert hasattr(test_app, "routes")
        assert hasattr(test_app, "middleware")

    def test_app_metadata(self):
        """Test that app has correct metadata."""
        test_app = create_app()
        assert test_app.title == "Minds API"
        assert "OpenAI-compatible" in test_app.description
        assert test_app.version == "1.9.1"

    def test_cors_middleware_configured(self):
        """Test that CORS middleware is configured."""
        test_app = create_app()
        # Check middleware list contains CORS configuration
        # FastAPI's user_middleware contains the middleware
        assert len(test_app.user_middleware) > 0
        # Verify at least one middleware is added (CORS)
        middleware_classes = [str(m.cls) for m in test_app.user_middleware]
        assert any("CORS" in str(m) for m in middleware_classes)

    def test_v1_routes_registered(self):
        """Test that /v1 routes are registered."""
        test_app = create_app()
        route_paths = [route.path for route in test_app.routes]
        # Check for core endpoints
        assert any("/v1/health" in path for path in route_paths)
        assert any("/v1/models" in path for path in route_paths)
        assert any("/v1/chat" in path for path in route_paths)

    def test_legacy_api_v1_routes_registered(self):
        """Test that /api/v1 legacy routes are registered."""
        test_app = create_app()
        route_paths = [route.path for route in test_app.routes]
        # Check for legacy endpoint prefixes
        assert any("/api/v1" in path for path in route_paths)

    @pytest.mark.asyncio
    async def test_lifespan_startup_shutdown(self):
        """Test that lifespan startup and shutdown work correctly."""
        with patch("minds.server.init_statsig") as mock_init_statsig, patch(
            "minds.server.shutdown_statsig"
        ) as mock_shutdown_statsig:
            mock_statsig_instance = MagicMock()
            mock_init_statsig.return_value = mock_statsig_instance

            test_app = create_app()

            # Simulate the lifespan context manager
            async with test_app.router.lifespan_context(test_app):
                # Verify init_statsig was called during startup
                mock_init_statsig.assert_called_once()
                # Verify statsig instance is stored in app state
                assert test_app.state.statsig == mock_statsig_instance

            # Verify shutdown_statsig was called during shutdown
            mock_shutdown_statsig.assert_called_once()

    def test_health_endpoint_accessible(self):
        """Test that health endpoint is accessible."""
        test_app = create_app()
        client = TestClient(test_app)
        response = client.get("/v1/health/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_legacy_health_endpoint_accessible(self):
        """Test that legacy /api/v1/health endpoint is accessible."""
        test_app = create_app()
        client = TestClient(test_app)
        response = client.get("/api/v1/health/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_models_endpoint_requires_headers(self):
        """Test that models endpoint requires tenant headers."""
        test_app = create_app()
        client = TestClient(test_app)
        # Without headers should get 401
        response = client.get("/v1/models/")
        assert response.status_code == 401

    def test_models_endpoint_with_headers(self):
        """Test that models endpoint works with proper headers."""
        test_app = create_app()
        client = TestClient(test_app)
        headers = {
            "X-User-Id": "00000000-0000-0000-0000-000000000001",
            "X-Organization-Id": "00000000-0000-0000-0000-000000000002",
        }
        response = client.get("/v1/models/", headers=headers)
        assert response.status_code == 200
        assert "data" in response.json()


class TestModuleLevel:
    """Tests for module-level app instantiation."""

    def test_app_is_instantiated(self):
        """Test that module-level app is instantiated."""
        assert app is not None
        assert hasattr(app, "routes")

    def test_app_has_routes(self):
        """Test that module-level app has routes."""
        route_paths = [route.path for route in app.routes]
        assert len(route_paths) > 0
        assert any("/v1/health" in path for path in route_paths)
