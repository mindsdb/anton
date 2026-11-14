"""The forget tool: removing saved memory from chat.

In-process (desktop, self-hosted) the tool deletes the entries it matches and
drops the conversation's memory pin. On a cloud turn it deletes nothing: the
cortex records each matched entry for cowork to remove after the turn. A
requested line (normalize_prompt_line) names an entry when it equals the
entry's text in the form cowork-server compares too (normalize_entry_text).
"""

from __future__ import annotations

import json
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.core.llm.identity import DeploymentIdentity
from anton.core.memory.cortex import Cortex, ForgetRequest
from anton.core.memory.hippocampus import Hippocampus, normalize_entry_text, normalize_prompt_line
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.tool_handlers import handle_forget, handle_memorize
from anton.workspace import Workspace

from tests.test_cloud_turn_session import _MEMORY, _mock_llm, _real_cloud_session, memory_dir  # noqa: F401

CONV = "0f4a7c9e-0000-4000-8000-00000000f001"
OTHER_CONV = "0f4a7c9e-0000-4000-8000-00000000f002"

TEAL = "Always use teal (#008080) for chart bars"


@pytest.fixture()
def memory_dirs(tmp_path):
    global_dir, project_dir = tmp_path / "global", tmp_path / "project"
    global_dir.mkdir()
    project_dir.mkdir()
    return global_dir, project_dir


def _session(tmp_path, memory_dirs, session_id=CONV, mode="autopilot", **config) -> ChatSession:
    """One in-process host turn: a real Cortex over real directories."""
    global_dir, project_dir = memory_dirs
    ws = Workspace(tmp_path / "ws")
    ws.initialize()
    return ChatSession(ChatSessionConfig(
        llm_client=_mock_llm(),
        workspace=ws,
        session_id=session_id,
        cortex=Cortex(global_hc=Hippocampus(global_dir), project_hc=Hippocampus(project_dir),
                      mode=mode, llm_client=_mock_llm()),
        **config,
    ))


def _save(memory_dirs):
    """A rule, a lesson and a profile fact in the global tier, and a rule in the project's."""
    global_dir, project_dir = memory_dirs
    (global_dir / "rules.md").write_text(f"# Rules\n\n## Always\n- {TEAL}\n- Answer in English\n")
    (global_dir / "lessons.md").write_text(
        "# Lessons\n- CoinGecko rate-limits at 50/min <!-- topic:api-coingecko ts:2026-11-01 -->\n"
        "- Orders are partitioned by month <!-- ts:2026-11-01 -->\n"
    )
    (global_dir / "profile.md").write_text("# Profile\n- Name: Ada\n- Timezone: PST\n")
    (project_dir / "rules.md").write_text("# Rules\n\n## Never\n- Never show salaries\n")


def _files(memory_dirs) -> dict[str, str]:
    return {
        f"{tier.name}/{path.name}": path.read_text()
        for tier in memory_dirs for path in sorted(tier.glob("*.md"))
    }


def _prompt_line(prompt: str, phrase: str) -> str:
    """The line of the system prompt that shows `phrase`, as the model reads it."""
    return next(line for line in prompt.splitlines() if phrase in line)


# ── the matching rule ────────────────────────────────────────────────────────

@pytest.mark.parametrize("line", [
    "Always use teal (#008080) for chart bars",
    "- Always use teal (#008080) for chart bars",
    "  -   always USE teal (#008080)\tfor  chart bars  ",
    "- Always use teal (#008080) for chart bars <!-- confidence:high source:user ts:2026-11-12 -->",
])
def test_a_line_as_the_prompt_shows_it_names_the_entry(line):
    assert normalize_prompt_line(line) == normalize_entry_text(TEAL)
    assert normalize_entry_text(TEAL) == "always use teal (#008080) for chart bars"


@pytest.mark.parametrize("line", [
    "Always use teal",                                      # part of the line
    "Always use teal (#008080) for chart bars.",            # punctuation added
    "Use teal (#008080) for chart bars",                    # reworded
    "Always use teal (#008080) for all chart bars",
])
def test_anything_else_names_nothing(line):
    assert normalize_prompt_line(line) != normalize_entry_text(TEAL)


def test_the_rule_folds_case_and_is_stable_on_its_own_output():
    assert normalize_entry_text("STRASSE") == normalize_entry_text("Straße")
    for text in (TEAL, "- Name: Ada", "İstanbul ΣΑΣ <!-- ts:2026-01-01 -->"):
        once = normalize_entry_text(text)
        assert normalize_entry_text(once) == once


# cowork-server's rule for a saved entry (normalize_memory_line in
# cowork/services/memory.py), restated here to pin anton's to it.
_COWORK_TAIL = re.compile(r"\s*<!--[^>]*-->\s*$")


def _cowork_form(text: str) -> str:
    return " ".join(_COWORK_TAIL.sub("", text).split()).casefold()


@pytest.mark.parametrize("text", [
    TEAL,
    "- Always use teal for bars",                           # an entry saved with its own bullet
    "  CoinGecko   rate-limits at 50/min <!-- topic:api ts:2026-11-01 -->",
    "Name: Ada",
])
def test_an_entry_normalizes_as_cowork_compares_it(text):
    assert normalize_entry_text(text) == _cowork_form(text)


def test_only_the_prompts_own_list_marker_leaves_a_requested_line():
    # The prompt shows an entry saved as "- text" as "- - text".
    assert normalize_prompt_line("- - Always use teal for bars") == "- always use teal for bars"
    assert normalize_entry_text("- Always use teal for bars") == "- always use teal for bars"


# ── in-process: the entry is deleted ─────────────────────────────────────────

async def test_a_rule_a_lesson_and_a_profile_fact_are_removed_by_their_prompt_lines(tmp_path, memory_dirs):
    global_dir, _ = memory_dirs
    _save(memory_dirs)
    session = _session(tmp_path, memory_dirs)
    prompt = await session._build_system_prompt("hello")

    result = await handle_forget(session, {"entries": [
        {"text": _prompt_line(prompt, "teal"), "scope": "global"},
        {"text": _prompt_line(prompt, "CoinGecko rate-limits"), "scope": "any"},     # metadata tail and all
        {"text": _prompt_line(prompt, "Name: Ada"), "scope": "global"},
    ]})

    assert result.ok is True
    assert result.content.startswith("Removed from memory:")
    assert f'"{TEAL}" (global rule)' in result.content
    assert '"CoinGecko rate-limits at 50/min" (global lesson)' in result.content
    assert '"Name: Ada" (global profile fact)' in result.content
    assert "Not found" not in result.content

    rules = (global_dir / "rules.md").read_text()
    assert TEAL not in rules and "Answer in English" in rules
    lessons = (global_dir / "lessons.md").read_text()
    assert "CoinGecko" not in lessons and "Orders are partitioned by month" in lessons
    profile = (global_dir / "profile.md").read_text()
    assert "Name: Ada" not in profile and "Timezone: PST" in profile


async def test_a_line_that_matches_nothing_is_reported_and_nothing_is_removed(tmp_path, memory_dirs):
    _save(memory_dirs)
    before = _files(memory_dirs)
    session = _session(tmp_path, memory_dirs)
    await session._build_system_prompt("hello")
    pinned = session._memory_pin()

    result = await handle_forget(session, {"entries": [
        {"text": "my preferred chart color", "scope": "any"},
        {"text": "Always use teal", "scope": "global"},       # part of a saved line
    ]})

    assert result.ok is False
    assert result.reason == "invalid_fields"
    assert "Removed" not in result.content
    assert 'Not found, so nothing was removed' in result.content
    assert '"my preferred chart color" (scope any)' in result.content
    assert '"Always use teal" (scope global)' in result.content
    assert _files(memory_dirs) == before
    assert session._memory_pin() is pinned        # nothing removed, so the pin stays


async def test_a_mixed_call_reports_each_part(tmp_path, memory_dirs):
    _save(memory_dirs)
    session = _session(tmp_path, memory_dirs)

    result = await handle_forget(session, {"entries": [
        {"text": TEAL, "scope": "global"},
        {"text": "no such line", "scope": "any"},
        {"text": "   ", "scope": "any"},
        {"text": "Answer in English", "scope": "everywhere"},
    ]})

    assert result.ok is True
    assert f'Removed from memory:\n- "{TEAL}" (global rule)' in result.content
    assert '"no such line" (scope any)' in result.content
    assert "Skipped 2 entries: each needs a text and a scope of global, project or any." in result.content
    assert "Answer in English" in (memory_dirs[0] / "rules.md").read_text()


@pytest.mark.parametrize("scope, global_left, project_left", [
    ("project", True, False),
    ("global", False, True),
    ("any", False, False),
])
async def test_the_scope_names_the_tiers_searched(tmp_path, memory_dirs, scope, global_left, project_left):
    global_dir, project_dir = memory_dirs
    for tier in memory_dirs:
        (tier / "rules.md").write_text("# Rules\n\n## Always\n- Show amounts in EUR\n")
    session = _session(tmp_path, memory_dirs)

    result = await handle_forget(session, {"entries": [{"text": "Show amounts in EUR", "scope": scope}]})

    assert result.ok is True
    assert ("Show amounts in EUR" in (global_dir / "rules.md").read_text()) is global_left
    assert ("Show amounts in EUR" in (project_dir / "rules.md").read_text()) is project_left
    removed_tiers = {t for t in ("global", "project") if f"({t} rule)" in result.content}
    assert removed_tiers == {t for t, left in (("global", global_left), ("project", project_left)) if not left}


async def test_a_line_in_the_other_tier_is_not_found(tmp_path, memory_dirs):
    _save(memory_dirs)
    before = _files(memory_dirs)
    session = _session(tmp_path, memory_dirs)

    result = await handle_forget(session, {"entries": [{"text": "Never show salaries", "scope": "global"}]})

    assert result.ok is False
    assert '"Never show salaries" (scope global)' in result.content
    assert _files(memory_dirs) == before


async def test_two_requests_for_one_entry_both_report_it(tmp_path, memory_dirs):
    _save(memory_dirs)
    session = _session(tmp_path, memory_dirs)

    result = await handle_forget(session, {"entries": [
        {"text": TEAL, "scope": "global"},
        {"text": f"- {TEAL.upper()}", "scope": "any"},
    ]})

    assert result.ok is True
    assert result.content.count(TEAL) == 1
    assert "Not found" not in result.content


async def test_the_operators_layer_is_never_removed(tmp_path, memory_dirs):
    global_dir, project_dir = memory_dirs
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "rules.md").write_text("# Rules\n\n## Always\n- Cite the data source\n")
    ws = Workspace(tmp_path / "ws")
    ws.initialize()
    session = ChatSession(ChatSessionConfig(
        llm_client=_mock_llm(), workspace=ws, session_id=CONV,
        cortex=Cortex(global_hc=Hippocampus(global_dir), project_hc=Hippocampus(project_dir),
                      mode="autopilot", managed_hc=Hippocampus(managed)),
    ))

    result = await handle_forget(session, {"entries": [{"text": "Cite the data source", "scope": "any"}]})

    assert result.ok is False
    assert "Cite the data source" in (managed / "rules.md").read_text()


async def test_memory_mode_off_removes_nothing(tmp_path, memory_dirs):
    # "off" leaves saved memory as it is, as memorize does. A host's model
    # comparison side reads the person's own memory in this mode and must
    # never change it, and a cloud host drops a removal in this mode too.
    _save(memory_dirs)
    before = _files(memory_dirs)
    session = _session(tmp_path, memory_dirs, mode="off")
    await session._build_system_prompt("hello")
    pinned = session._memory_pin()
    recalled = session.recalled_preferences()

    result = await handle_forget(session, {"entries": [{"text": TEAL, "scope": "global"}]})

    # Tier 3, as memorize's "encoding is disabled": a configured state.
    assert isinstance(result, str)
    assert result == (
        "Saved memory is read-only in this conversation, so nothing was removed. "
        "The user can remove saved entries where the host's memory settings allow it."
    )
    assert _files(memory_dirs) == before
    assert session._memory_pin() is pinned
    assert session.recalled_preferences() == recalled


_READ_ONLY_SAVE = "Saved memory is read-only in this conversation, so nothing was saved."


@pytest.mark.parametrize(("surface", "reply"), [
    ("cli", "Memory encoding is disabled. Change memory mode via /setup to enable."),
    ("desktop", _READ_ONLY_SAVE),
    ("web", _READ_ONLY_SAVE),
    (None, _READ_ONLY_SAVE),    # the host reported no surface
])
async def test_memorize_in_memory_mode_off_points_at_setup_only_in_the_cli(
    tmp_path, memory_dirs, surface, reply,
):
    # /setup is a CLI command. A desktop or web user has no such command, and
    # on a comparison side no setting would turn saving on.
    _save(memory_dirs)
    before = _files(memory_dirs)
    session = _session(tmp_path, memory_dirs, mode="off", surface=surface)

    result = await handle_memorize(session, {"entries": [
        {"text": "Answer in French", "kind": "always", "scope": "global"},
    ]})

    # Tier 3, a configured state, on every surface.
    assert result == reply
    assert _files(memory_dirs) == before


@pytest.mark.parametrize("identity", [
    DeploymentIdentity(kind="self-hosted", product_name="Cowork"),
    None,                       # a host can share memory without naming a deployment
])
async def test_shared_memory_is_not_removed_from_chat(tmp_path, memory_dirs, identity):
    # One store serves everyone on such a deployment; one person's request
    # would remove the entry for all of them.
    _save(memory_dirs)
    before = _files(memory_dirs)
    session = _session(
        tmp_path, memory_dirs, deployment_identity=identity, learned_memory_scope="shared",
    )

    result = await handle_forget(session, {"entries": [{"text": TEAL, "scope": "any"}]})

    assert result.ok is False
    assert result.reason == "access_denied"
    assert "Nothing was removed" in result.content
    assert _files(memory_dirs) == before


async def test_personal_memory_on_a_deployment_is_removed(tmp_path, memory_dirs):
    _save(memory_dirs)
    session = _session(
        tmp_path, memory_dirs,
        deployment_identity=DeploymentIdentity(kind="self-hosted", product_name="Cowork"),
        learned_memory_scope="personal",
    )

    result = await handle_forget(session, {"entries": [{"text": TEAL, "scope": "any"}]})

    assert result.ok is True
    assert TEAL not in (memory_dirs[0] / "rules.md").read_text()


# ── in-process: the conversation stops recalling it ──────────────────────────

async def test_after_a_removal_no_later_prompt_recalls_the_rule(tmp_path, memory_dirs):
    _save(memory_dirs)
    session = _session(tmp_path, memory_dirs)
    await session._build_system_prompt("hello")
    assert TEAL in session.recalled_preferences()

    result = await handle_forget(session, {"entries": [{"text": TEAL, "scope": "global"}]})
    assert result.ok is True

    # This turn's page build no longer gets it, though the turn's prompt is built.
    assert TEAL not in session.recalled_preferences()
    assert "Answer in English" in session.recalled_preferences()

    # The conversation's next turn builds its prompt again instead of reusing the pin.
    next_turn = _session(tmp_path, memory_dirs)
    prompt = await next_turn._build_system_prompt("chart my sales")
    assert TEAL not in prompt
    assert TEAL not in next_turn.recalled_preferences()
    assert "Answer in English" in next_turn.recalled_preferences()

    # And a new conversation starts without it.
    other = _session(tmp_path, memory_dirs, session_id=OTHER_CONV)
    await other._build_system_prompt("chart my sales")
    assert TEAL not in other.recalled_preferences()
    assert "Answer in English" in other.recalled_preferences()


async def test_a_removal_leaves_the_same_line_in_the_operators_and_the_other_tiers_sections(
    tmp_path, memory_dirs,
):
    """The page build keeps the operator's rule and the project's copy, which
    are still saved, though the user's global copy goes."""
    global_dir, project_dir = memory_dirs
    managed = tmp_path / "managed"
    managed.mkdir()
    line = "Never load scripts from external CDNs"
    for tier in (managed, global_dir, project_dir):
        (tier / "rules.md").write_text(f"# Rules\n\n## Never\n- {line}\n")
    ws = Workspace(tmp_path / "ws")
    ws.initialize()
    session = ChatSession(ChatSessionConfig(
        llm_client=_mock_llm(), workspace=ws, session_id=CONV,
        cortex=Cortex(global_hc=Hippocampus(global_dir), project_hc=Hippocampus(project_dir),
                      mode="autopilot", managed_hc=Hippocampus(managed), llm_client=_mock_llm()),
    ))
    await session._build_system_prompt("hello")
    assert session.recalled_preferences().count(line) == 3

    result = await handle_forget(session, {"entries": [{"text": line, "scope": "global"}]})

    assert result.ok is True
    recalled = session.recalled_preferences()
    operator, rest = recalled.split("## Your Memory — Global Rules")
    global_section, project_section = rest.split("## Your Memory — Project Rules")
    assert line in operator
    assert line not in global_section
    assert line in project_section
    assert line not in (global_dir / "rules.md").read_text()
    assert line in (managed / "rules.md").read_text()
    assert line in (project_dir / "rules.md").read_text()


async def test_an_entry_saved_with_a_bullet_is_removed_by_its_prompt_line(tmp_path, memory_dirs):
    global_dir, _ = memory_dirs
    session = _session(tmp_path, memory_dirs)
    await handle_memorize(session, {"entries": [
        {"text": "- Always use teal for bars", "kind": "always", "scope": "global"},
    ]})
    await session.settle_memory_writes()
    prompt = await session._build_system_prompt("hello")
    line = _prompt_line(prompt, "teal")
    assert line == "- - Always use teal for bars"

    result = await handle_forget(session, {"entries": [{"text": line, "scope": "global"}]})

    assert result.ok is True
    assert '"- Always use teal for bars" (global rule)' in result.content
    assert "teal" not in (global_dir / "rules.md").read_text()
    assert "teal" not in session.recalled_preferences()


async def test_one_call_reads_each_tier_once_and_at_most_100_entries(tmp_path, memory_dirs, monkeypatch):
    _save(memory_dirs)
    before = _files(memory_dirs)
    session = _session(tmp_path, memory_dirs)
    reads: list[Hippocampus] = []
    real_get_rules = Hippocampus.get_rules

    def counting_get_rules(self, *args, **kwargs):
        reads.append(self)
        return real_get_rules(self, *args, **kwargs)

    monkeypatch.setattr(Hippocampus, "get_rules", counting_get_rules)
    entries = [{"text": f"no such line {i}", "scope": "any"} for i in range(100)]
    entries.append({"text": TEAL, "scope": "global"})            # past the cap: never read

    result = await handle_forget(session, {"entries": entries})

    assert len(reads) == 2                                      # one read per tier
    assert result.ok is False
    assert result.content.count("(scope any)") == 100
    assert result.content.endswith(
        "Skipped 1 entry after the first 100: one call reads at most 100. "
        "Call forget again for the rest."
    )
    assert _files(memory_dirs) == before


# ── registration ─────────────────────────────────────────────────────────────

def _tool_names(session) -> set[str]:
    session._scratchpads = MagicMock(available_packages=[], pads={})
    session._build_tools()
    return {t.name for t in session.tool_registry.get_tool_defs()}


def test_forget_is_offered_with_a_cortex(tmp_path, memory_dirs):
    assert {"memorize", "forget"} <= _tool_names(_session(tmp_path, memory_dirs))


def test_forget_is_not_offered_without_a_cortex(tmp_path):
    session = ChatSession(ChatSessionConfig(llm_client=_mock_llm(), cortex=None))
    assert "forget" not in _tool_names(session)


def test_forget_is_not_offered_on_the_legacy_self_awareness_path(tmp_path):
    session = ChatSession(ChatSessionConfig(llm_client=_mock_llm(), cortex=None, self_awareness=MagicMock()))
    names = _tool_names(session)
    assert "memorize" in names and "forget" not in names


def test_the_description_names_the_cap_and_calls_a_cloud_removal_a_request():
    """A host may refuse, or never receive, a removal the tool asked for, so
    the description never tells the model that one happens."""
    from anton.core.tools.tool_defs import FORGET_TOOL
    from anton.core.tools.tool_handlers import FORGET_MAX_ENTRIES

    description = FORGET_TOOL.description
    assert f"at most {FORGET_MAX_ENTRIES} per call" in description
    # The opening sentence says what the tool does on either host.
    assert description.startswith(
        "Remove saved rules, lessons and profile facts from long-term memory, or "
        "ask the host that keeps memory to remove them."
    )
    assert "an asked-for removal is a request, not a removal" in description
    for promise in ("they happen", "will be removed", "when this turn ends"):
        assert promise not in description
    # A person on a deployment whose memory is shared or off can't remove an
    # entry on the Memories page either, so the description names no place.
    assert description.endswith(
        "The user can also remove saved memory where the host's memory settings allow it."
    )


# ── cloud: recorded for cowork, nothing deleted in the pod ───────────────────

async def test_a_cloud_turn_records_the_request_and_deletes_nothing(tmp_path, monkeypatch, memory_dir):
    from anton.cloud_turn.session import drain_pending_forget

    session = _real_cloud_session(tmp_path, monkeypatch, memory=_MEMORY)
    before = {p.name: p.read_text() for p in sorted((memory_dir / "global").iterdir())}

    result = await handle_forget(session, {"entries": [
        {"text": "-   reply in SPANISH", "scope": "global"},
        {"text": "Name: Zoran", "scope": "any"},
        {"text": "my favorite color", "scope": "any"},
    ]})

    assert result.ok is True
    assert "Removed" not in result.content
    # cowork removes these after the turn and may refuse one, so the reply
    # says the removal is asked for, not done.
    assert result.content.startswith(
        "Asked the host to remove these; it decides after this turn, and none is "
        'removed now:\n- "Reply in Spanish" (global rule)\n'
        '- "Name: Zoran" (global profile fact)\n'
        "The host keeps an entry this user may not change, such as a project entry "
        "another member saved; only that member or an organization admin can "
        "remove it, on the Memories page. Tell the user you asked for the removal, "
        "not that it is done."
    )
    for promise in ("will be removed", "when this turn ends", "not removed yet", "an entry goes"):
        assert promise not in result.content.lower()
    assert '"my favorite color" (scope any)' in result.content
    # Nothing persisted pod-side: the staged copies are untouched.
    assert {p.name: p.read_text() for p in sorted((memory_dir / "global").iterdir())} == before

    # Each matched entry in cowork's form, scoped to the tier it was found in;
    # a request that named nothing is not sent.
    assert drain_pending_forget(session) == [
        {"text": "reply in spanish", "scope": "global"},
        {"text": "name: zoran", "scope": "global"},
    ]
    assert drain_pending_forget(session) == []     # cleared: no double-apply


@pytest.mark.parametrize("compacted", [False, True])
async def test_a_cloud_prompt_built_again_in_the_turn_still_leaves_the_line_out(
    tmp_path, monkeypatch, memory_dir, compacted,
):
    """A retried model call builds the prompt again: from the pin, or after a
    compaction from the copies the pod read. Both still hold the line, since
    the host removes it only after the turn, so a page built later in the turn
    must still not get it."""
    memory = {"global": {"rules": f"## Always\n- {TEAL}\n- Answer in English"}, "project": {}}
    session = _real_cloud_session(tmp_path, monkeypatch, memory=memory)
    await session._build_system_prompt("chart my sales")
    assert TEAL in session.recalled_preferences()

    result = await handle_forget(session, {"entries": [{"text": TEAL, "scope": "global"}]})
    assert result.ok is True
    assert TEAL not in session.recalled_preferences()

    if compacted:
        session._drop_memory_pin()      # what a compaction does to the pin
    await session._build_system_prompt("chart my sales")

    assert TEAL not in session.recalled_preferences()
    assert "Answer in English" in session.recalled_preferences()


async def test_a_cloud_request_named_twice_is_sent_once(tmp_path, monkeypatch, memory_dir):
    from anton.cloud_turn.session import drain_pending_forget

    session = _real_cloud_session(tmp_path, monkeypatch, memory=_MEMORY)
    for text in ("Reply in Spanish", "- reply in spanish"):
        await handle_forget(session, {"entries": [{"text": text, "scope": "global"}]})

    assert drain_pending_forget(session) == [{"text": "reply in spanish", "scope": "global"}]


async def test_a_cloud_turn_that_matched_nothing_sends_nothing(tmp_path, monkeypatch, memory_dir):
    from anton.cloud_turn.session import drain_pending_forget

    session = _real_cloud_session(tmp_path, monkeypatch, memory=_MEMORY)
    result = await handle_forget(session, {"entries": [{"text": "Reply in Spanish", "scope": "project"}]})

    assert result.ok is False
    assert drain_pending_forget(session) == []


async def test_a_cloud_move_to_another_tier_removes_only_the_tier_it_was_found_in(
    tmp_path, monkeypatch, memory_dir,
):
    """forget with scope any plus memorize of the same line to global: cowork
    must remove the project rule and keep the global one the turn saves."""
    from anton.cloud_turn.session import drain_pending_forget, drain_pending_memory

    memory = {"global": {}, "project": {"rules": "## Always\n- Show amounts in EUR"}}
    session = _real_cloud_session(tmp_path, monkeypatch, memory=memory)

    forgot = await handle_forget(session, {"entries": [{"text": "- Show amounts in EUR", "scope": "any"}]})
    saved = await handle_memorize(session, {"entries": [
        {"text": "Show amounts in EUR", "kind": "always", "scope": "global"},
    ]})
    await session.settle_memory_writes()

    assert forgot.ok is True and saved.ok is True
    assert '"Show amounts in EUR" (project rule)' in forgot.content
    assert drain_pending_forget(session) == [{"text": "show amounts in eur", "scope": "project"}]
    assert [(e["text"], e["scope"]) for e in drain_pending_memory(session)] == [
        ("Show amounts in EUR", "global"),
    ]


async def test_a_cloud_entry_saved_with_a_bullet_is_sent_in_coworks_form(tmp_path, monkeypatch, memory_dir):
    """An entry whose own text starts with "- " keeps it in the form cowork
    compares, so cowork removes the entry anton reported."""
    from anton.cloud_turn.session import drain_pending_forget

    memory = {"global": {"rules": "## Always\n- - Always use teal for bars"}, "project": {}}
    session = _real_cloud_session(tmp_path, monkeypatch, memory=memory)
    entry = Hippocampus(memory_dir / "global").get_rules()[0]
    assert entry.text == "- Always use teal for bars"

    # One bullet short of the prompt's line names nothing, so nothing is sent.
    missed = await handle_forget(session, {"entries": [{"text": "- Always use teal for bars", "scope": "global"}]})
    assert missed.ok is False
    assert drain_pending_forget(session) == []

    result = await handle_forget(session, {"entries": [{"text": "- - Always use teal for bars", "scope": "global"}]})

    assert result.ok is True
    sent = drain_pending_forget(session)
    assert sent == [{"text": "- always use teal for bars", "scope": "global"}]
    assert sent[0]["text"] == _cowork_form(entry.text)


# ── cloud: a turn whose writes cowork keeps none of (memory_writable false) ──

async def _run_memory_tools_turn(tmp_path, monkeypatch, **fields):
    """One cloud turn off a wire request that carries _MEMORY and ``fields``,
    run through the entrypoint on the real session builder. The turn calls
    forget on a saved rule and memorize on a new one, as the model would.
    Returns the two tool replies and the events the pod wrote."""
    import anton.core.llm.client as llm_client_mod
    from anton.cloud_turn.__main__ import stream_turn
    from anton.cloud_turn.session import _WORKSPACE_PATH_ENV, build_cloud_chat_session
    from anton.core.llm.provider import StreamTextDelta

    monkeypatch.setenv(_WORKSPACE_PATH_ENV, str(tmp_path))
    monkeypatch.setattr(
        llm_client_mod.LLMClient, "from_settings", classmethod(lambda cls, settings: _mock_llm()),
    )
    replies: dict = {}

    def build(request, **kwargs):
        session = build_cloud_chat_session(request, **kwargs)
        # The entrypoint closes the session when the turn ends.
        session._scratchpads = MagicMock(available_packages=[], close_all=AsyncMock())

        async def turn_stream(user_input, **_):
            replies["forget"] = await handle_forget(session, {"entries": [
                {"text": "- Reply in Spanish", "scope": "global"},
            ]})
            replies["memorize"] = await handle_memorize(session, {"entries": [
                {"text": "Answer in French", "kind": "always", "scope": "global"},
            ]})
            yield StreamTextDelta(text="ok")

        session.turn_stream = turn_stream
        return session

    line = json.dumps({
        "protocol_version": 1, "conversation_id": "conv_1", "input": "hi", "memory": _MEMORY, **fields,
    })
    events: list[dict] = []
    await stream_turn(line, events.append, session_builder=build)
    return replies, events


async def test_a_read_only_cloud_turn_changes_nothing_and_sends_no_memory_line(
    tmp_path, monkeypatch, memory_dir,
):
    """cowork drops every save and removal of a turn whose memory is off or
    that is a comparison side, and sends memory_writable false for it. The
    tools then answer as an in-process turn in memory mode "off" does, and the
    pod sends no line for cowork to drop."""
    replies, events = await _run_memory_tools_turn(tmp_path, monkeypatch, memory_writable=False)

    assert replies["forget"] == (
        "Saved memory is read-only in this conversation, so nothing was removed. "
        "The user can remove saved entries where the host's memory settings allow it."
    )
    # Tier 3, a configured state, as in-process with memory off. A cloud turn
    # is never the CLI, so the reply names no /setup.
    assert replies["memorize"] == "Saved memory is read-only in this conversation, so nothing was saved."
    assert [e["kind"] for e in events] == ["delta", "turn_completed"]


@pytest.mark.parametrize("fields", [{}, {"memory_writable": True}])
async def test_a_writable_cloud_turn_still_sends_its_changes(tmp_path, monkeypatch, memory_dir, fields):
    """The same turn off a request without the field, or with true, asks
    cowork for the removal and the save as it did before the field."""
    replies, events = await _run_memory_tools_turn(tmp_path, monkeypatch, **fields)

    assert replies["forget"].ok is True and replies["memorize"].ok is True
    assert {"kind": "memory_forget", "entries": [{"text": "reply in spanish", "scope": "global"}]} in events
    saved = [entry["text"] for event in events if event["kind"] == "memory" for entry in event["entries"]]
    assert saved == ["Answer in French"]


async def test_a_read_only_cloud_turn_recalls_memory_as_a_writable_one(tmp_path, monkeypatch, memory_dir):
    """Read-only stops the writes only. The prompt recalls the request's memory
    exactly as on a writable turn, as an in-process turn in memory mode "off"
    reads the person's memory."""
    import anton.core.session as session_module

    built = {}
    for writable in (True, False):
        # Each session builds its own memory section rather than reading the
        # other's pin, which is shared by conversation id.
        session_module._MEMORY_PINS.clear()
        session = _real_cloud_session(tmp_path, monkeypatch, memory=_MEMORY, memory_writable=writable)
        assert session._cortex.mode == ("autopilot" if writable else "off")
        built[writable] = (await session._build_system_prompt("hola"), session.recalled_preferences())

    prompt, preferences = built[False]
    for line in ("Name: Zoran", "Reply in Spanish", "The staging DB is read-only"):
        assert line in prompt
    assert "Reply in Spanish" in preferences
    assert built[False] == built[True]


def test_requests_carry_their_scope_or_any():
    assert ForgetRequest(text="x").scope == "any"
    for bad in ({"text": ""}, {"text": "- "}, {"text": "x", "scope": "everywhere"}, {"scope": "any"}):
        with pytest.raises(ValueError):
            ForgetRequest.model_validate(bad)


# ── memorize never saves an entry under a guessed kind ───────────────────────

async def test_memorize_rejects_an_unknown_kind_instead_of_saving_a_lesson(tmp_path, memory_dirs):
    global_dir, project_dir = memory_dirs
    _save(memory_dirs)
    before = _files(memory_dirs)
    session = _session(tmp_path, memory_dirs)

    result = await handle_memorize(session, {"entries": [
        {"text": TEAL, "kind": "forget", "scope": "global"},
        {"text": "Prefer bar charts", "scope": "project"},          # no kind at all
    ]})
    await session.settle_memory_writes()

    assert result.ok is False
    assert result.reason == "invalid_type"
    assert f"Not saved: {TEAL!r} (kind 'forget'); 'Prefer bar charts' (kind None)." in result.content
    assert "The kind must be one of always, never, when, lesson, profile." in result.content
    assert "call forget" in result.content
    assert _files(memory_dirs) == before


async def test_memorize_saves_the_valid_entries_and_names_the_rest(tmp_path, memory_dirs):
    global_dir, _ = memory_dirs
    session = _session(tmp_path, memory_dirs)

    result = await handle_memorize(session, {"entries": [
        {"text": "Answer in Spanish", "kind": "always", "scope": "global"},
        {"text": "Prefer bar charts", "kind": "preference", "scope": "global"},
    ]})
    await session.settle_memory_writes()

    assert result.ok is True
    assert result.content.startswith("Memory updated: Encoded always: Answer in Spanish\n")
    assert "Not saved: 'Prefer bar charts' (kind 'preference')." in result.content
    assert "Answer in Spanish" in (global_dir / "rules.md").read_text()
    assert not (global_dir / "lessons.md").exists()
