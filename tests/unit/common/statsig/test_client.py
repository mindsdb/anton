"""
Unit tests for Statsig client lifecycle.

Tests init, get, shutdown, and thread-safety of the singleton client.
"""

from unittest.mock import MagicMock, patch

import pytest

import minds.common.statsig.client as client_module
from minds.common.settings.app_settings import AppSettings


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure every test starts with a clean singleton."""
    client_module._client = None
    yield
    client_module._client = None


def _make_settings(**overrides) -> AppSettings:
    defaults = {
        "statsig": {
            "sdk_key": "secret-test-key",
            "environment": "test",
            "disable_network": True,
            "disable_all_logging": True,
        }
    }
    defaults.update(overrides)
    return AppSettings(**defaults)


class TestBuildStatsigOptions:
    """Tests for build_statsig_options."""

    def test_options_reflect_settings(self):
        settings = _make_settings()
        opts = client_module.build_statsig_options(settings)
        assert opts.environment == "test", f"environment should be 'test', got '{opts.environment}'"
        assert opts.disable_network is True, f"disable_network should be True, got {opts.disable_network}"
        assert opts.disable_all_logging is True, f"disable_all_logging should be True, got {opts.disable_all_logging}"


class TestInitStatsig:
    """Tests for init_statsig."""

    @patch("minds.common.statsig.client.apply_offline_overrides")
    @patch("minds.common.statsig.client.Statsig")
    def test_initializes_and_returns_client(self, mock_statsig_cls, mock_overrides):
        mock_instance = MagicMock()
        mock_instance.initialize.return_value = MagicMock()
        mock_instance.initialize.return_value.wait.return_value = None
        mock_statsig_cls.return_value = mock_instance

        settings = _make_settings()
        result = client_module.init_statsig(settings=settings)

        mock_statsig_cls.assert_called_once()
        mock_instance.initialize.assert_called_once()
        mock_overrides.assert_called_once_with(statsig=mock_instance, settings=settings)
        assert result is mock_instance, "init_statsig should return the Statsig instance"

    @patch("minds.common.statsig.client.apply_offline_overrides")
    @patch("minds.common.statsig.client.Statsig")
    def test_returns_existing_client_on_second_call(self, mock_statsig_cls, mock_overrides):
        mock_instance = MagicMock()
        mock_instance.initialize.return_value.wait.return_value = None
        mock_statsig_cls.return_value = mock_instance

        settings = _make_settings()
        first = client_module.init_statsig(settings=settings)
        second = client_module.init_statsig(settings=settings)

        assert first is second, "Second call should return the same cached client"
        assert mock_statsig_cls.call_count == 1, (
            f"Statsig constructor should be called once, was called {mock_statsig_cls.call_count} times"
        )


class TestGetStatsig:
    """Tests for get_statsig."""

    @patch("minds.common.statsig.client.init_statsig")
    def test_initializes_if_client_is_none(self, mock_init):
        mock_client = MagicMock()
        mock_init.side_effect = lambda settings: setattr(client_module, "_client", mock_client)

        result = client_module.get_statsig(settings=_make_settings())
        mock_init.assert_called_once()
        assert result is mock_client, "get_statsig should return the newly initialized client"

    def test_returns_existing_client(self):
        mock_client = MagicMock()
        client_module._client = mock_client

        result = client_module.get_statsig(settings=_make_settings())
        assert result is mock_client, "get_statsig should return the existing client"


class TestShutdownStatsig:
    """Tests for shutdown_statsig."""

    def test_shutdown_clears_singleton(self):
        mock_client = MagicMock()
        mock_client.shutdown.return_value.wait.return_value = None
        client_module._client = mock_client

        client_module.shutdown_statsig()

        mock_client.shutdown.assert_called_once()
        assert client_module._client is None, "Singleton should be None after shutdown"

    def test_shutdown_noop_when_no_client(self):
        """Should not raise if there's nothing to shut down."""
        client_module._client = None
        client_module.shutdown_statsig()  # no error
        assert client_module._client is None, "Singleton should remain None"

    def test_shutdown_clears_even_on_exception(self):
        mock_client = MagicMock()
        mock_client.shutdown.return_value.wait.side_effect = RuntimeError("boom")
        client_module._client = mock_client

        with pytest.raises(RuntimeError, match="boom"):
            client_module.shutdown_statsig()

        # Client should still be cleared thanks to finally block
        assert client_module._client is None, "Singleton should be cleared even when shutdown raises"
