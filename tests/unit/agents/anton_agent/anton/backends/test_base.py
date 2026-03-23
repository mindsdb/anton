from __future__ import annotations

import asyncio
import types
from uuid import UUID

import pytest


# ScratchPad
def test_scratchpad_runtime_base_helpers_and_compaction():
    from minds.agents.anton_agent.anton.backends.base import Cell, ScratchpadRuntime

    class Dummy(ScratchpadRuntime):
        async def start(self):
            return None

        async def reset(self):
            return None

        async def close(self, cleanup: bool = True):
            return None

        async def cancel(self):
            return None

        async def install_packages(self, packages):
            return ""

        async def execute_streaming(
            self, code: str, *, description: str = "", estimated_time: str = "", estimated_seconds: int = 0
        ):
            yield Cell(
                code=code, stdout="ok", stderr="", error=None, description=description, estimated_time=estimated_time
            )

        async def report_exists(self, organization_id, user_id, conversation_id, message_id):
            return True

        async def get_report(self, organization_id, user_id, conversation_id, message_id):
            return ""

    rt = Dummy(name="x")
    assert ScratchpadRuntime._truncate_output("a\n" * 100, max_lines=2).startswith("a\n")

    cell = asyncio.run(rt.execute("print(1)", description="d"))
    assert cell.stdout == "ok"

    rt.cells = [Cell(code=f"print({i})", stdout=str(i), stderr="", error=None) for i in range(10)]
    assert "Scratchpad:" in rt.render_notebook()
    assert rt._compact_cells() is True


def test_scratchpad_runtime_factory_discover_and_create(monkeypatch, tmp_path):
    from minds.agents.anton_agent.anton.backends.base import ScratchpadRuntime, ScratchpadRuntimeFactory

    class Impl(ScratchpadRuntime):
        async def start(self): ...

        async def reset(self): ...

        async def close(self, cleanup: bool = True): ...

        async def cancel(self): ...

        async def install_packages(self, packages): ...

        async def execute_streaming(
            self, code: str, *, description: str = "", estimated_time: str = "", estimated_seconds: int = 0
        ): ...

        async def report_exists(self, organization_id, user_id, conversation_id, message_id): ...

        async def get_report(self, organization_id, user_id, conversation_id, message_id): ...

    fake_mod = types.ModuleType("minds.agents.anton_agent.anton.backends.fake")
    Impl.__module__ = fake_mod.__name__
    fake_mod.Impl = Impl

    class _ModInfo:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.backends.base.pkgutil.iter_modules",
        lambda _paths: [_ModInfo("fake")],
    )
    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.backends.base.importlib.import_module",
        lambda _name: fake_mod,
    )

    fac = ScratchpadRuntimeFactory()
    reg = fac.discover()
    assert "fake" in reg
    rt = fac.create(
        name="n",
        backend="fake",
        coding_provider="anthropic",
        coding_model="m",
        coding_api_key="k",
        workspace_path=tmp_path,
        extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"},
    )
    assert isinstance(rt, Impl)


def test_scratchpad_runtime_view_and_notebook_and_compaction_edges():
    from minds.agents.anton_agent.anton.backends.base import Cell, ScratchpadRuntime

    class Dummy(ScratchpadRuntime):
        async def start(self): ...

        async def reset(self): ...

        async def close(self, cleanup: bool = True): ...

        async def cancel(self): ...

        async def install_packages(self, packages): ...

        async def execute_streaming(
            self, code: str, *, description: str = "", estimated_time: str = "", estimated_seconds: int = 0
        ): ...

        async def report_exists(self, organization_id, user_id, conversation_id, message_id): ...

        async def get_report(self, organization_id, user_id, conversation_id, message_id): ...

    rt = Dummy(name="pad")
    assert rt.view() == "Scratchpad 'pad' is empty."
    assert rt.render_notebook() == "Scratchpad 'pad' has no cells."
    assert rt._compact_cells() is False

    rt.cells = [
        Cell(code="  \n", stdout="", stderr="", error=None),
        Cell(code="print(1)", stdout="", stderr="", error=None, logs="", description="d"),
        Cell(code="x", stdout="hello", stderr="", error=None, logs="log line\n" * 50),
        Cell(code="y", stdout="out", stderr="err", error="Traceback...\nValueError: boom"),
    ]
    view = rt.view()
    assert "--- Cell 1" in view
    assert "(no output)" in view
    nb = rt.render_notebook()
    assert "## Scratchpad: pad" in nb
    assert "**Logs:**" in nb
    assert "**Error:**" in nb


def test_base_truncate_output_char_limit_and_timeouts():
    from minds.agents.anton_agent.anton.backends.base import ScratchpadRuntime

    s = "line1\n" + ("x" * 3000)
    out = ScratchpadRuntime._truncate_output(s, max_lines=999, max_chars=50)
    assert out.endswith("\n... (truncated)")

    total, inactivity = ScratchpadRuntime._compute_timeouts(estimated_seconds=10)
    assert total >= 20
    assert inactivity >= 30


@pytest.mark.asyncio
async def test_runtime_factory_create_unknown_backend_raises():
    from minds.agents.anton_agent.anton.backends.base import ScratchpadRuntimeFactory

    fac = ScratchpadRuntimeFactory()
    fac.discover = lambda: {}  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="Unknown backend"):
        fac.create(name="x", backend="nope")


@pytest.mark.asyncio
async def test_runtime_factory_report_exists_and_get_report_delegates():
    from minds.agents.anton_agent.anton.backends.base import ScratchpadRuntime, ScratchpadRuntimeFactory

    class Impl(ScratchpadRuntime):
        async def start(self): ...

        async def reset(self): ...

        async def close(self, cleanup: bool = True): ...

        async def cancel(self): ...

        async def install_packages(self, packages): ...

        async def execute_streaming(
            self, code: str, *, description: str = "", estimated_time: str = "", estimated_seconds: int = 0
        ): ...

        async def report_exists(self, organization_id, user_id, conversation_id, message_id):
            return True

        async def get_report(self, organization_id, user_id, conversation_id, message_id):
            return "HTML"

    fac = ScratchpadRuntimeFactory()

    def _discover():
        fac._registry = {"fake": Impl}
        return fac._registry

    fac.discover = _discover  # type: ignore[method-assign]

    org = UUID("00000000-0000-0000-0000-000000000002")
    user = UUID("00000000-0000-0000-0000-000000000001")
    conv = UUID("00000000-0000-0000-0000-0000000000aa")
    msg = UUID("00000000-0000-0000-0000-0000000000bb")

    assert await fac.report_exists("fake", org, user, conv, msg) is True
    assert await fac.get_report("fake", org, user, conv, msg) == "HTML"
