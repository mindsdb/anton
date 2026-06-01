"""Tests for the ProviderAdapter interface."""

import pytest

from minds.inference.adapter import ProviderAdapter


def test_provider_adapter_is_abstract():
    """ProviderAdapter cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ProviderAdapter()


def test_provider_adapter_requires_complete():
    """Subclasses must implement complete()."""

    class IncompleteAdapter(ProviderAdapter):
        pass

    with pytest.raises(TypeError):
        IncompleteAdapter()


def test_provider_adapter_requires_get_last_usage():
    """Subclasses must implement get_last_usage()."""

    class IncompleteAdapter(ProviderAdapter):
        async def complete(self, *args, **kwargs):
            pass

    with pytest.raises(TypeError):
        IncompleteAdapter()


def test_provider_adapter_requires_get_last_output():
    """Subclasses must implement get_last_output()."""

    class IncompleteAdapter(ProviderAdapter):
        async def complete(self, *args, **kwargs):
            pass

        async def get_last_usage(self):
            pass

    with pytest.raises(TypeError):
        IncompleteAdapter()


def test_provider_adapter_requires_get_last_artifacts():
    """Subclasses must implement get_last_artifacts()."""

    class IncompleteAdapter(ProviderAdapter):
        async def complete(self, *args, **kwargs):
            pass

        async def get_last_usage(self):
            pass

        def get_last_output(self):
            pass

    with pytest.raises(TypeError):
        IncompleteAdapter()


def test_provider_adapter_full_interface():
    """A complete adapter implementation can be instantiated."""

    class ConcreteAdapter(ProviderAdapter):
        async def complete(self, *args, **kwargs):
            return None

        async def get_last_usage(self):
            return None

        def get_last_output(self):
            return None

        def get_last_artifacts(self):
            return []

    adapter = ConcreteAdapter()
    assert adapter is not None
