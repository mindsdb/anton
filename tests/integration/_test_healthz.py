import pytest

from minds.api.v1.endpoints.health import healthz


@pytest.mark.asyncio
async def test_healthz():
    response = await healthz()
    assert response == {"status": "ok", "version": "v1"}
