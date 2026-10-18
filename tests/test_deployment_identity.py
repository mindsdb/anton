"""Host-supplied branding changes self-hosted prompts without changing defaults."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from anton.core.llm.identity import DeploymentIdentity, product_lines
from anton.core.llm.provider import LLMResponse, ToolCall
from anton.core.memory.cortex import Cortex
from anton.core.memory.hippocampus import Hippocampus
from anton.core.memory import skills as skills_module
from anton.core.memory.skills import SkillStore
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.recall_skill import handle_recall_skill
from tests.conftest import make_mock_llm, run_turn


def deployment(name="Acme Cowork"):
    return DeploymentIdentity(kind="self-hosted", product_name=name)


@pytest.mark.parametrize("name", ["Cowork", "Acme Cowork", "Café 🚀", "Acme\u00a0Cowork", "Acme\u2007Cowork", "Ada's $HOME `whoami` $(id)", 'A "quoted" \\ name', "A" * 64])
def test_names_round_trip_without_rewriting(name):
    identity = deployment(name)
    assert identity.product_name == name
    assert DeploymentIdentity.model_validate_json(identity.model_dump_json()) == identity


@pytest.mark.parametrize("name", ["", "A" * 65, " Acme", "Acme ", "\tAcme", "A\nB", "A\rB", "A\x00B", "A\x7fB", "A\u200bB", "A\u202eB", "A\u2028B", 123, None])
def test_invalid_names_are_rejected_without_truncation(name):
    with pytest.raises(ValidationError):
        deployment(name)


def test_deployment_kind_and_fields_are_closed():
    with pytest.raises(ValidationError):
        DeploymentIdentity(kind="web", product_name="Acme")
    with pytest.raises(ValidationError):
        DeploymentIdentity(kind="self-hosted", product_name="Acme", maker="Other")
    with pytest.raises(ValidationError):
        deployment().product_name = "Changed"


_DEFAULT_LINES = [
    "- You are Cowork, an AI agent made by MindsHub. Never identify as, or as "
    "built by, any other AI product or company — you are not ChatGPT, Claude, "
    "Gemini or Copilot, and MindsHub is not OpenAI, Anthropic or Google.",
    "- Cowork runs as a desktop app on macOS, Windows and Linux, and in the browser. "
    "Some features exist on only one of those, so never assume a feature you know "
    "about is reachable from where this user is.",
]
_DEFAULT_RULE = (
    "- If asked what Cowork is, who makes it, how to install, update or remove it, "
    "which platforms or editions exist, where a setting or feature lives, which models "
    "it can use (including local models such as Ollama or LM Studio), or what it "
    "costs, call `recall_skill(\"cowork-product\")` and answer from it. Do not answer "
    "those from general knowledge, and do not guess or web-search for them. If it "
    "does not cover the question, say so plainly and point the user at "
    "https://docs.mindshub.ai — never fill the gap with a guess."
)


@pytest.mark.parametrize(("surface", "line"), [
    ("desktop", "- This conversation is running in the Cowork desktop app."),
    ("web", "- This conversation is running in Cowork in the browser. Desktop-only features are not reachable from here."),
    ("cli", "- This conversation is running in the anton command-line interface."),
    (None, None), ("nonsense", None),
])
def test_default_product_lines_are_byte_identical(surface, line):
    expected = _DEFAULT_LINES + ([line] if line else []) + [_DEFAULT_RULE]
    assert product_lines(surface) == expected
    assert product_lines(surface, None) == expected


@pytest.mark.parametrize("name", ["Acme Cowork", "Cowork"])
async def test_host_identity_and_memory_status_reach_the_model(tmp_path, name):
    llm = make_mock_llm()
    cortex = Cortex(Hippocampus(tmp_path / "global"), Hippocampus(tmp_path / "project"))
    session = ChatSession(ChatSessionConfig(
        llm_client=llm, surface="web", deployment_identity=deployment(name),
        learned_memory_scope="personal", cortex=cortex, background_memory=False,
    ))
    try:
        await run_turn(session, "Who are you and what do you remember about me?")
        prompt = llm.plan.call_args.kwargs["system"]
        assert f'Deployment product name: "{name}"' in prompt
        assert "made by MindsHub" in prompt
        assert "self-hosted deployment accessed in the browser" in prompt
        assert "This conversation is running in the Cowork desktop app" not in prompt
        assert "The operator installs, updates and removes this deployment" in prompt
        assert "No personal memories are saved" in prompt
        assert "not remembered user preferences" in prompt
    finally:
        await session.close()


async def test_scope_without_deployment_identity_does_not_change_prompt(tmp_path):
    cortex = Cortex(Hippocampus(tmp_path / "global"), Hippocampus(tmp_path / "project"))
    prompts = []
    for scope in (None, "personal"):
        session = ChatSession(ChatSessionConfig(
            llm_client=make_mock_llm(), surface="desktop", cortex=cortex,
            learned_memory_scope=scope, started_at=datetime(2026, 10, 18),
        ))
        try:
            prompts.append(await session._build_system_prompt("hello"))
        finally:
            await session.close()
    assert prompts[0] == prompts[1]
    assert "Learned memory status" not in prompts[0]


@pytest.mark.parametrize("scope", [None, "personal", "shared", "unavailable", "off"])
async def test_missing_cortex_never_claims_empty_personal_memory(scope):
    session = ChatSession(ChatSessionConfig(
        llm_client=make_mock_llm(), deployment_identity=deployment(), learned_memory_scope=scope,
    ))
    try:
        prompt = await session._build_system_prompt("what do you remember?")
        assert "No personal memories are saved" not in prompt
        assert ("Learned memory is off" if scope == "off" else "memory is unavailable") in prompt
    finally:
        await session.close()


async def test_recalled_product_skill_defers_to_trusted_deployment(tmp_path):
    store = SkillStore(
        root=tmp_path / "skills",
        builtin_root=Path(skills_module.__file__).parent / "builtin_skills",
    )
    result = await handle_recall_skill(SimpleNamespace(_skill_store=store, history=[]), {"label": "cowork-product"})
    assert result.ok
    text = result.content
    assert text.index("TRUSTED DEPLOYMENT FACTS COME FIRST") < text.index("GETTING THE DESKTOP APP")
    assert "take precedence" in text
    assert "Keep using that name after recalling this skill. MindsHub remains the maker." in text
    assert "The hosted browser's provider restrictions" in text
    assert "below do not describe self-hosted deployments" in text
    assert "When no self-hosted deployment is declared, use the default facts below" in text
    # Default instructions remain available when the host supplies no override.
    assert "https://mindshub.ai/download" in text
    assert "Desktop updates itself" in text
    assert "https://cowork.mindshub.ai" in text


async def test_identity_remains_authoritative_in_model_call_after_recall(tmp_path):
    llm = make_mock_llm()
    llm.plan.side_effect = [
        LLMResponse(
            content="", stop_reason="tool_use",
            tool_calls=[ToolCall(id="product", name="recall_skill", input={"label": "cowork-product"})],
        ),
        LLMResponse(content="Ask your operator about updates.", stop_reason="end_turn"),
    ]
    session = ChatSession(ChatSessionConfig(llm_client=llm, deployment_identity=deployment()))
    session._skill_store = SkillStore(
        root=tmp_path / "skills",
        builtin_root=Path(skills_module.__file__).parent / "builtin_skills",
    )
    try:
        await run_turn(session, "How do I update this deployment?")
        assert llm.plan.call_count == 2
        after_recall = llm.plan.call_args.kwargs
        assert 'Deployment product name: "Acme Cowork"' in after_recall["system"]
        assert "Keep the deployment facts above authoritative after recall" in after_recall["system"]
        results = [
            block["content"]
            for message in after_recall["messages"] if isinstance(message.get("content"), list)
            for block in message["content"] if block.get("type") == "tool_result"
        ]
        assert any("TRUSTED DEPLOYMENT FACTS COME FIRST" in content for content in results)
    finally:
        await session.close()
