"""Driver selection and schema loading.

Schema comes from the manifest file (single source, the same one the publish
pipeline reads); the driver is chosen by backend.STATE (dict → DynamoDB,
otherwise the local SQLite driver).
"""
from __future__ import annotations

import os

from .base import Store
from .schema import StateSchema
from .sqlite_driver import SQLiteDriver

_DEFAULT_LOCAL = "./.anton_state.db"
_ENV_PATH = "ANTON_ARTIFACT_STATE_PATH"
_DEFAULT_MANIFEST = "./state_manifest.json"
_ENV_MANIFEST = "ANTON_STATE_MANIFEST"


def _resolve_schema(schema: StateSchema | None, manifest_path: str | None) -> StateSchema:
    if schema is not None:
        return schema
    path = manifest_path or os.environ.get(_ENV_MANIFEST) or _DEFAULT_MANIFEST
    return StateSchema.from_manifest(path)


def open_store(
    schema: StateSchema | None = None,
    *,
    state: dict | None = None,
    local_path: str | None = None,
    manifest_path: str | None = None,
) -> Store:
    schema = _resolve_schema(schema, manifest_path)
    if state:
        from .dynamo_driver import DynamoDBDriver  # lazy: boto3 only needed in the cloud

        driver = DynamoDBDriver(
            table=state["table"],
            region=state["region"],
            credentials=state["credentials"],
            schema=schema,
        )
        return Store(driver, schema)
    path = local_path or os.environ.get(_ENV_PATH) or _DEFAULT_LOCAL
    return Store(SQLiteDriver(path, schema), schema)


def from_backend_state(
    state: dict | None, schema: StateSchema | None = None, *, manifest_path: str | None = None
) -> Store:
    """Entry point for the backend: pass backend.STATE here (schema from the manifest)."""
    return open_store(schema, state=state, manifest_path=manifest_path)
