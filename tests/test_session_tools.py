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

    ENG-1495's property, unchanged: a host that does not identify itself is
    indistinguishable from the CLI, because `None` is reserved for exactly that
    case. The CLI leaving it unset is what made `""` mean both "this is the CLI"
    and "nobody said", and that value reaches PostHog and Langfuse tags, not
    just a log line.

    **ENG-1694 moved which field carries it.** `harness` now names the AGENT
    (`anton` / `hermes`) and `surface` names the HOST (`cli` / `desktop` /
    `web`) — one field could not answer both, which is why a `cli` trace could
    not say whether anton or hermes ran. So the CLI identifies itself via
    `surface=SURFACE_CLI` now, and additionally reports `harness=HARNESS_ANTON`
    because the CLI does run the anton agent. Both must be present: dropping
    the surface reintroduces exactly the ENG-1495 ambiguity, and dropping the
    harness makes the agent unknowable.

    Source inspection rather than a behavioural test, for the same reason the
    test above gives: exercising `rebuild_session` means standing up an
    LLMClient, a Cortex and a knowledge refresh. The property being guarded is
    structural — both builders must pass the arguments — so asserting on the
    call site is the honest check rather than a weaker substitute.

    Verified to be needed: deleting either kwarg from both files leaves the
    rest of the suite green.
    """
    fresh = (_ANTON_DIR / "chat.py").read_text()
    rebuild = (_ANTON_DIR / "chat_session.py").read_text()
    for name, src in (("chat.py", fresh), ("chat_session.py", rebuild)):
        assert "surface=SURFACE_CLI" in src, (
            f"{name} must identify the host, or telemetry and Langfuse tags "
            "cannot tell the CLI from an unidentified host (ENG-1495)"
        )
        assert "harness=HARNESS_ANTON" in src, (
            f"{name} must name the agent it runs, or a CLI trace cannot say "
            "whether anton or hermes produced it (ENG-1694)"
        )
        assert 'harness="cli"' not in src, (
            f"{name} still names a PLACE as its harness — that is the "
            "two-vocabularies bug ENG-1694 removed"
        )
