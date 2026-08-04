"""ENG-1166 regression: resumed / rebuilt sessions must register the same
extra tools as fresh sessions.

The bug: `chat.py` (fresh session) passed `tools=[CONNECT_DATASOURCE_TOOL,
PUBLISH_TOOL]`, but `chat_session.py::rebuild_session` (used by `/resume` and
after a settings/model change) built its config with NO `tools=` argument — so
every resumed session silently lost `publish_or_preview` + `connect_new_datasource`
and could no longer publish. Both builders now derive their extra-tool list from
the single `DEFAULT_SESSION_TOOLS` source; these tests guard against re-drift.
"""
from __future__ import annotations

from pathlib import Path

import anton
from anton.tools import DEFAULT_SESSION_TOOLS

_ANTON_DIR = Path(anton.__file__).parent


def test_default_session_tools_includes_publish_and_datasource():
    names = {t.name for t in DEFAULT_SESSION_TOOLS}
    assert "publish_or_preview" in names
    assert "connect_new_datasource" in names


def test_both_session_builders_use_the_shared_tool_list():
    # Drift guard: if either builder stops referencing DEFAULT_SESSION_TOOLS,
    # they can diverge again (which is exactly how resumed sessions lost the
    # publish tool). Read source rather than import chat.py (heavy).
    fresh = (_ANTON_DIR / "chat.py").read_text()
    rebuild = (_ANTON_DIR / "chat_session.py").read_text()
    assert "DEFAULT_SESSION_TOOLS" in fresh, "fresh session builder (chat.py) must use the shared tool list"
    assert "DEFAULT_SESSION_TOOLS" in rebuild, "rebuild_session (chat_session.py) must use the shared tool list"
