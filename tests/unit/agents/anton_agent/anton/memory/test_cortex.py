from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_cortex_encode_and_build_memory_context(tmp_path):
    from minds.agents.anton_agent.anton.memory.cortex import Cortex
    from minds.agents.anton_agent.anton.memory.hippocampus import Engram

    c = Cortex(global_dir=tmp_path / "g", project_dir=tmp_path / "p", mode="autopilot", llm_client=None)
    actions = await c.encode(
        [
            Engram(text="Prefers dark mode", kind="profile", scope="global", confidence="high", source="user"),
            Engram(text="Never use time.sleep()", kind="never", scope="project", confidence="high", source="user"),
            Engram(
                text="Use progress() for long jobs", kind="lesson", scope="project", confidence="high", source="user"
            ),
        ]
    )
    assert any("Updated identity" in a for a in actions)
    assert any("Encoded never rule" in a for a in actions)
    assert any("Encoded lesson" in a for a in actions)

    ctx = c.build_memory_context()
    assert "Your Memory" in ctx


@pytest.mark.asyncio
async def test_cortex_compaction_and_identity_update(tmp_path):
    from minds.agents.anton_agent.anton.memory.cortex import Cortex
    from minds.agents.anton_agent.anton.memory.hippocampus import Hippocampus

    class FakeLLM:
        async def code(self, *, system, messages, max_tokens=4096):
            return SimpleNamespace(content=json.dumps({"kept": ["Always use httpx", "Never use time.sleep()"]}))

    c = Cortex(global_dir=tmp_path / "g", project_dir=tmp_path / "p", mode="manual", llm_client=FakeLLM())
    c._COMPACTION_THRESHOLD = 1

    hc: Hippocampus = c.global_hc
    hc._dir.mkdir(parents=True, exist_ok=True)
    hc._lessons_path.write_text("# Lessons\n" + "\n".join(f"- lesson {i}" for i in range(30)) + "\n", encoding="utf-8")
    await c.compact_all()
    assert "Always use httpx" in hc._lessons_path.read_text(encoding="utf-8")

    hc.rewrite_identity(["Name: Alice"])

    class FakeLLM2:
        async def code(self, *, system, messages, max_tokens=512):
            return SimpleNamespace(content=json.dumps(["Name: Bob", "Timezone: PST"]))

    c2 = Cortex(global_dir=tmp_path / "g2", project_dir=tmp_path / "p2", mode="autopilot", llm_client=FakeLLM2())
    c2.global_hc.rewrite_identity(["Name: Alice"])
    await c2.maybe_update_identity("hi")
    ident = c2.global_hc.recall_identity()
    assert "Name: Bob" in ident and "Timezone: PST" in ident


@pytest.mark.asyncio
async def test_cortex_encoding_gate_and_compact_file_early_returns(tmp_path):
    from minds.agents.anton_agent.anton.memory.cortex import Cortex
    from minds.agents.anton_agent.anton.memory.hippocampus import Engram

    c = Cortex(global_dir=tmp_path / "g", project_dir=tmp_path / "p", mode="manual", llm_client=None)
    assert c.encoding_gate(Engram(text="x", kind="lesson", scope="project", confidence="medium")) is True
    c.mode = "off"
    assert c.encoding_gate(Engram(text="x", kind="lesson", scope="project", confidence="medium")) is False

    await c.compact_all()


@pytest.mark.asyncio
async def test_cortex_build_memory_context_empty_and_needs_compaction(tmp_path):
    from minds.agents.anton_agent.anton.memory.cortex import Cortex

    c = Cortex(global_dir=tmp_path / "g", project_dir=tmp_path / "p", mode="autopilot", llm_client=None)
    assert c.build_memory_context() == ""
    assert c.needs_compaction() is False
