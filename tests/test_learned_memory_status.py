"""Empty saved memory differs from filtered, shared, disabled or unreadable memory."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from anton.core.llm.identity import DeploymentIdentity
from anton.core.memory.cortex import Cortex
from anton.core.memory.hippocampus import Hippocampus
from anton.core.session import ChatSession, ChatSessionConfig
from tests.conftest import make_mock_llm


@pytest.fixture
def stores(tmp_path):
    return [Hippocampus(tmp_path / name) for name in ("global", "project", "managed")]


@pytest.fixture
def cortex(stores):
    global_hc, project_hc, managed_hc = stores
    return Cortex(global_hc, project_hc, managed_hc=managed_hc)


async def test_default_empty_context_is_unchanged(cortex):
    assert await cortex.build_memory_context() == ""
    assert cortex.get_scratchpad_context() == ""


async def test_empty_personal_memory_with_operator_rules(cortex, stores):
    stores[2].encode_rule("Use calendar quarters", kind="always")
    context = await cortex.build_memory_context(learned_memory_scope="personal")
    assert "No personal memories are saved" in context
    assert "Operator Memory" in context
    assert "Use calendar quarters" in context
    assert "not remembered user preferences" in context
    assert "## Your Memory" not in context


@pytest.mark.parametrize("slot", ["profile.md", "rules.md", "lessons.md"])
async def test_heading_only_slots_are_empty(cortex, stores, slot):
    store = stores[0]
    store._dir.mkdir(parents=True)
    (store._dir / slot).write_text("# Memory\n\n## Always\n", encoding="utf-8")
    assert store.memory_presence() == "empty"
    assert "No personal memories are saved" in await cortex.build_memory_context(learned_memory_scope="personal")


@pytest.mark.parametrize("tier", [0, 1])
@pytest.mark.parametrize("kind", ["rule", "scratchpad", "topic"])
async def test_unfiltered_saved_content_prevents_empty_claim(cortex, stores, tier, kind):
    store = stores[tier]
    if kind == "rule":
        store.encode_rule("If discussing invoices, use fiscal quarters", kind="when")
        cortex._retrieve_relevant_rules = AsyncMock(return_value=[])
    elif kind == "scratchpad":
        store.encode_lesson("Use a separate scratchpad for each report", topic="scratchpad")
    else:
        store.encode_lesson("Revenue uses fiscal quarters", topic="reporting")
    context = await cortex.build_memory_context("hello", learned_memory_scope="personal")
    assert "contains saved memories" in context
    assert "No personal memories are saved" not in context
    if kind == "scratchpad":
        assert "Use a separate scratchpad" not in context
        assert "Use a separate scratchpad" in cortex.get_scratchpad_context(learned_memory_scope="personal")


async def test_saved_profile_is_personal_content(cortex, stores):
    stores[0].rewrite_identity(["Name: Ada"])
    context = await cortex.build_memory_context(learned_memory_scope="personal")
    assert "contains saved memories" in context
    assert "Name: Ada" in context


async def test_legacy_topic_content_prevents_claiming_nothing_is_saved(cortex, stores):
    topics = stores[0]._dir / "topics"
    topics.mkdir(parents=True)
    (topics / "planning.md").write_text("# Planning\nUse calendar quarters\n", encoding="utf-8")
    context = await cortex.build_memory_context(learned_memory_scope="personal")
    assert "contains saved memories" in context
    assert "No personal memories are saved" not in context


@pytest.mark.parametrize("scope", ["off", "unavailable"])
async def test_disabled_and_no_person_turns_do_not_inspect_learned_stores(cortex, stores, monkeypatch, scope):
    stores[0].rewrite_identity(["Name: Ada"])
    stores[2].encode_rule("If using scratchpad, page large tables", kind="when")
    for store in stores[:2]:
        for name in ("memory_presence", "recall_identities", "get_rules", "recall_lessons", "recall_topic", "recall_scratchpad_wisdom"):
            monkeypatch.setattr(store, name, Mock(side_effect=AssertionError("must not inspect learned memory")))
    context = await cortex.build_memory_context(learned_memory_scope=scope)
    scratchpad = cortex.get_scratchpad_context(learned_memory_scope=scope)
    assert "No personal memories are saved" not in context
    assert "Name: Ada" not in context
    assert "page large tables" in scratchpad
    for store in stores[:2]:
        store.memory_presence.assert_not_called()


@pytest.mark.parametrize("has_content", [False, True])
async def test_shared_memory_is_never_labeled_personal(cortex, stores, has_content):
    if has_content:
        stores[0].rewrite_identity(["Name: Another person"])
        stores[0].encode_lesson("Use scratchpad for tables", topic="scratchpad")
        stores[2].encode_rule("Use ISO dates", kind="always")
    context = await cortex.build_memory_context(learned_memory_scope="shared")
    assert "shared across people" in context
    assert "not a private profile of this user" in context
    assert "## Your Memory" not in context
    assert "No personal memories are saved" not in context
    if has_content:
        assert "## Shared Memory" in context
        assert "Shared learned memory" in cortex.get_scratchpad_context(learned_memory_scope="shared")
    else:
        assert "shared store has no saved memories" in context


# An omitted scope on a deployment session means "unavailable".
@pytest.mark.parametrize("scope", ["shared", "off", None])
async def test_session_scratchpad_tool_labels_learned_memory_by_scope(cortex, stores, scope):
    stores[2].encode_rule("If using scratchpad, page large tables", kind="when")
    stores[0].encode_lesson("Use a separate scratchpad for each report", topic="scratchpad")
    session = ChatSession(ChatSessionConfig(
        llm_client=make_mock_llm(), cortex=cortex, learned_memory_scope=scope,
        deployment_identity=DeploymentIdentity(kind="self-hosted", product_name="Acme Cowork"),
    ))
    try:
        tools = session._build_tools()
        description = next(t["description"] for t in tools if t["name"] == "scratchpad")
    finally:
        await session.close()
    assert "Operator (applies to everyone, wins on conflict):" in description
    assert "page large tables" in description
    assert "Your own:" not in description
    if scope == "shared":
        assert "Shared learned memory:\n- Use a separate scratchpad for each report" in description
    else:
        assert "Use a separate scratchpad for each report" not in description


@pytest.mark.parametrize("error", [PermissionError("unreadable"), UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")])
async def test_read_failure_does_not_become_empty(cortex, stores, monkeypatch, error):
    stores[0].encode_rule("Use ISO dates", kind="always")
    original = Path.read_text
    def fail(path, *args, **kwargs):
        if path == stores[0]._rules_path:
            raise error
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", fail)
    assert stores[0].memory_presence() == "unavailable"
    context = await cortex.build_memory_context(learned_memory_scope="personal")
    assert "could not be verified" in context
    assert "No personal memories are saved" not in context


async def test_failed_presence_api_does_not_become_empty(cortex, stores, monkeypatch):
    monkeypatch.setattr(stores[0], "memory_presence", Mock(side_effect=RuntimeError("backend down")))
    context = await cortex.build_memory_context(learned_memory_scope="personal")
    assert "could not be verified" in context
    assert "No personal memories are saved" not in context


async def test_old_store_without_presence_api_is_unknown(cortex, stores, monkeypatch):
    monkeypatch.delattr(Hippocampus, "memory_presence")
    context = await cortex.build_memory_context(learned_memory_scope="personal")
    assert "could not be verified" in context
    assert "No personal memories are saved" not in context
