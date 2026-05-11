from __future__ import annotations

import asyncio
import importlib
import pkgutil

import pytest


pytest.importorskip("fastapi")


def test_cowork_server_modules_import() -> None:
    import anton.cowork.server as server_pkg

    failures: list[tuple[str, str]] = []
    for module_info in pkgutil.walk_packages(
        server_pkg.__path__,
        prefix=f"{server_pkg.__name__}.",
    ):
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:  # pragma: no cover - assertion prints details
            failures.append((module_info.name, repr(exc)))

    assert failures == []


def test_cowork_server_health_versions_and_protocol() -> None:
    from anton.cowork.server import (
        COWORK_SERVER_PROTOCOL_VERSION,
        COWORK_SERVER_VERSION,
    )
    from anton.cowork.server.main import health

    payload = asyncio.run(health())

    assert payload["status"] == "ok"
    assert payload["anton_version"] == COWORK_SERVER_VERSION
    assert payload["cowork_server_version"] == COWORK_SERVER_VERSION
    assert payload["cowork_server_protocol_version"] == COWORK_SERVER_PROTOCOL_VERSION


def test_cowork_connector_registry_loads_package_data() -> None:
    from anton.cowork.server.anton_api import connectors_registry

    connectors_registry.reload_connectors()
    connectors = connectors_registry.all_connectors()

    assert "github" in connectors
    assert len(connectors) > 100
