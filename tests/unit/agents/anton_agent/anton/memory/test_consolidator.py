from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_consolidator_should_replay_and_replay_and_extract():
    from minds.agents.anton_agent.anton.backends.base import Cell
    from minds.agents.anton_agent.anton.memory.consolidator import Consolidator

    cons = Consolidator()
    assert (
        cons.should_replay(
            [
                Cell(code="x", stdout="", stderr="", error=None),
                Cell(code="y", stdout="", stderr="", error=None),
            ]
        )
        is False
    )
    assert (
        cons.should_replay(
            [
                Cell(code="x", stdout="", stderr="", error="boom"),
                Cell(code="y", stdout="", stderr="", error=None),
            ]
        )
        is True
    )

    class FakeLLM:
        async def code(self, *, system, messages, max_tokens=2048):
            content = (
                "```json\n"
                + json.dumps(
                    [{"text": "Always call progress()", "kind": "always", "scope": "project", "confidence": "high"}]
                )
                + "\n```"
            )
            return SimpleNamespace(content=content)

    cells = [
        Cell(code="print(1)", stdout="1", stderr="", error=None, description="d"),
        Cell(code="x", stdout="", stderr="", error="Traceback...\nValueError", description="e"),
    ]
    engrams = await cons.replay_and_extract(cells, FakeLLM())
    assert engrams and engrams[0].kind == "always"


def test_consolidator_additional_branches():
    from minds.agents.anton_agent.anton.backends.base import Cell
    from minds.agents.anton_agent.anton.memory.consolidator import Consolidator

    cons = Consolidator()
    assert cons.should_replay([Cell(code="x", stdout="", stderr="", error=None) for _ in range(5)]) is True
    assert (
        cons.should_replay(
            [
                Cell(code="x", stdout="", stderr="Cancelled", error=None),
                Cell(code="y", stdout="", stderr="", error=None),
            ]
        )
        is True
    )


@pytest.mark.asyncio
async def test_consolidator_replay_and_extract_invalid_json_returns_empty():
    from minds.agents.anton_agent.anton.backends.base import Cell
    from minds.agents.anton_agent.anton.memory.consolidator import Consolidator

    class BadLLM:
        async def code(self, *, system, messages, max_tokens=2048):
            return SimpleNamespace(content="not json")

    cons = Consolidator()
    cells = [
        Cell(code="x", stdout="1", stderr="", error=None),
        Cell(code="y", stdout="", stderr="", error=None),
    ]
    assert await cons.replay_and_extract(cells, BadLLM()) == []
