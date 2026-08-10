"""Deferred tools (`unlock_skill`) must coexist with a `tool_allowlist`.

Regression for ENG-764: a deferred tool listed in the allowlist used to make
`_build_tools` raise, because the up-front "unknown name" check ran before the
tool was ever registered.
"""

from __future__ import annotations

import pytest

from anton.core.tools.tool_defs import ToolDef


async def _noop_handler(session, tc_input) -> str:  # pragma: no cover - never called
    return ""


def _deferred_tool(name: str = "deferred_tool") -> ToolDef:
    return ToolDef(
        name=name,
        description="deferred",
        input_schema={"type": "object", "properties": {}},
        handler=_noop_handler,
        unlock_skill="test-skill",
    )


def _names(session) -> set[str]:
    return {t["name"] for t in session._build_tools()}


def test_allowlisted_deferred_tool_does_not_raise_and_is_hidden(make_session):
    tool = _deferred_tool()
    session = make_session(tools=[tool], tool_allowlist=frozenset({tool.name}))
    # Before the fix this raised ValueError: the name was in the allowlist but
    # not yet registered, so it counted as "unknown".
    assert tool.name not in _names(session)


def test_deferred_tool_appears_after_its_bundle_unlocks(make_session):
    tool = _deferred_tool()
    session = make_session(tools=[tool], tool_allowlist=frozenset({tool.name}))
    _names(session)  # first build populates _deferred_bundles
    session._register_tool_bundle("test-skill")
    # It survives the allowlist re-enforcement on the next build.
    assert tool.name in _names(session)


def test_deferred_tool_left_out_of_allowlist_stays_filtered(make_session):
    """Deferral only changes *timing*; the allowlist still governs whether an
    unlocked tool is allowed at all."""
    tool = _deferred_tool()
    session = make_session(tools=[tool], tool_allowlist=frozenset({"scratchpad"}))
    _names(session)
    session._register_tool_bundle("test-skill")
    assert tool.name not in _names(session)


if __name__ == "__main__":  # allow a bare `python tests/test_deferred_tools.py`
    raise SystemExit(pytest.main([__file__, "-q"]))
