"""Tests for /v1/health/* endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from minds.api.v1.router import api_router
from minds.db.pg_session import get_session


@pytest.fixture
def fake_session():
    session = MagicMock()
    session.execute.return_value = MagicMock()
    return session


@pytest.fixture
def client(fake_session):
    app = FastAPI()
    app.include_router(api_router)
    app.dependency_overrides[get_session] = lambda: fake_session
    return TestClient(app)


def test_healthz_returns_ok(client):
    response = client.get("/v1/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "v1"}


def test_liveness_is_independent_of_db(client, fake_session):
    fake_session.execute.side_effect = OperationalError("boom", {}, Exception("db down"))
    response = client.get("/v1/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive", "version": "v1"}


def test_readiness_succeeds_when_db_reachable(client, fake_session):
    response = client.get("/v1/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "version": "v1"}
    fake_session.execute.assert_called_once()


def test_readiness_returns_503_when_db_unreachable(client, fake_session):
    fake_session.execute.side_effect = OperationalError("boom", {}, Exception("db down"))
    response = client.get("/v1/health/ready")
    assert response.status_code == 503
    assert "database unreachable" in response.json()["detail"]
