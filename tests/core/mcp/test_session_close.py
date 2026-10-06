"""ChatSession.close()'s guarantee that MCP session cleanup runs even when
an earlier step in close() raises — see ENG-1816 code review."""
from __future__ import annotations

import anton.core.mcp.wiring as wiring_mod


async def test_mcp_sessions_close_even_if_scratchpad_close_raises(make_session, monkeypatch):
    """ScratchpadManager.close_all() has no internal try/except around each
    pad's own close() — without close()'s own guarantee, one bad scratchpad
    close would leak every open MCP session's transport for the rest of the
    process, the same way the LLM client's cleanup is already guaranteed."""
    closed: list = []

    async def fake_close_mcp_sessions(sessions):
        closed.extend(sessions)

    monkeypatch.setattr(wiring_mod, "close_mcp_sessions", fake_close_mcp_sessions)

    fake_mcp_sessions = [object(), object()]
    session = make_session(mcp_sessions=fake_mcp_sessions)

    async def raise_on_close_all():
        raise RuntimeError("simulated bad scratchpad close")

    monkeypatch.setattr(session._scratchpads, "close_all", raise_on_close_all)

    try:
        await session.close()
    except RuntimeError:
        pass  # the original failure still propagates — only cleanup is guaranteed, not silence

    assert closed == fake_mcp_sessions


async def test_close_is_a_no_op_for_cleanup_when_there_are_no_mcp_sessions(make_session, monkeypatch):
    """The cleanup branch must not blow up when there was nothing to clean
    up — the common case, since Stage 1 never has a real method=="mcp"
    connection in production yet."""
    called = {"n": 0}

    async def fake_close_mcp_sessions(sessions):
        called["n"] += 1

    monkeypatch.setattr(wiring_mod, "close_mcp_sessions", fake_close_mcp_sessions)

    session = make_session()
    await session.close()

    assert called["n"] == 0
