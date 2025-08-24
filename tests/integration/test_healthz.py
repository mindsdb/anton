import pytest

from minds.server import healthz


@pytest.mark.asyncio
async def test_healthz():
    response = await healthz()
    assert response == {"status": "ok"}
