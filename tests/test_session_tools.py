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


def test_both_session_builders_identify_the_host_as_cli():
    """Same drift class as the tool list above, one field over.

    `ChatSessionConfig.harness` reserves `None` for "a host that did not
    identify itself". The CLI leaving it unset is what made `""` mean both
    "this is the CLI" and "nobody said", so the two could never be told apart —
    and that value reaches PostHog and Langfuse tags, not just a log line.

    Source inspection rather than a behavioural test, for the same reason the
    test above gives: exercising `rebuild_session` means standing up an
    LLMClient, a Cortex and a knowledge refresh. The property being guarded is
    structural — both builders must pass the argument — so asserting on the
    call site is the honest check rather than a weaker substitute.

    Verified to be needed: deleting `harness="cli"` from both files leaves the
    entire suite green.
    """
    fresh = (_ANTON_DIR / "chat.py").read_text()
    rebuild = (_ANTON_DIR / "chat_session.py").read_text()
    assert 'harness="cli"' in fresh, (
        "fresh session builder (chat.py) must identify the host, or telemetry "
        "and Langfuse tags cannot tell the CLI from an unidentified host"
    )
    assert 'harness="cli"' in rebuild, (
        "rebuild_session (chat_session.py) must identify the host — a resumed "
        "session that drops it reintroduces the ambiguity mid-session"
    )
