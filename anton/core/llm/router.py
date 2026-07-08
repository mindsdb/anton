"""Cheap front-model routing — respond directly or delegate to planning.

ENG-648: every user turn used to go straight to the planning model with
the full system prompt and tool schemas, even for "thanks!" or a
follow-up question about text already in the conversation. The router
runs first on a cost-effective model with a minimal prompt and NO tool
schemas, and either:

  • RESPOND — answers directly when the reply is fully derivable from
    the conversation plus stable general knowledge (no scratchpad, no
    tools, nothing created or fetched), or
  • DELEGATE — hands off to the planning model, optionally naming
    procedural skills to preload so the expensive model doesn't spend a
    round calling ``recall_skill`` itself.

The bias is deliberately conservative: any need for tools, fresh data,
files, or computation must delegate. A wrong direct answer costs far
more than the delegation overhead, so every ambiguity resolves to
DELEGATE — including router errors, empty responses, and answers that
blow past the router's output budget (a long answer is evidence the
turn wasn't trivial).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from anton.core.llm.provider import LLMResponse

if TYPE_CHECKING:
    from anton.core.llm.client import LLMClient


ROUTE_RESPOND = "respond"
ROUTE_DELEGATE = "delegate"

# The router sees a condensed, text-only view of history: full transcripts
# (and the tool_use/tool_result blocks inside them) are what the planning
# model needs, not what a yes/no routing call needs.
_MAX_ROUTER_MESSAGES = 16
_MAX_CHARS_PER_MESSAGE = 1_500


DELEGATE_TOOL: dict = {
    "name": "delegate",
    "description": (
        "Hand the current request to the full agent, which can execute "
        "code, query data sources, search the web, read files, and build "
        "artifacts. Call this immediately — with no preamble text — "
        "whenever the request needs anything beyond a direct "
        "conversational answer."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "One short sentence: why the full agent is needed.",
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Labels of known procedures (from the list in your "
                    "system prompt) that clearly apply to this request, so "
                    "they are preloaded for the full agent. Omit if none "
                    "apply."
                ),
            },
        },
        "required": ["reason"],
    },
}


_ROUTER_SYSTEM_PROMPT = """\
You are the fast front-line responder for Anton, an agent that analyzes data, \
executes code, and builds things for the user. Decide, for the latest user \
message, between exactly two actions:

1. ANSWER DIRECTLY — reply yourself, only when ALL of these hold:
   - No tools are needed: no code execution, no file or attachment access, no \
data queries, no web search or fetching, no artifacts or dashboards.
   - Nothing must be created, modified, computed, fetched, scheduled, or verified.
   - You are not guessing: the answer is already in the conversation, or is \
stable general knowledge.
   Typical direct cases: greetings and small talk, answering questions about \
results already shown in the conversation, explaining or rephrasing something \
already said, asking the user a clarifying question when their request is too \
vague to act on.

2. DELEGATE — call the `delegate` tool immediately, with NO text before it, \
when the request involves data analysis, files or attachments, running or \
writing code, building or updating anything, web lookups or information that \
could be stale, connecting to services, or memory/skill management — or \
whenever you are unsure. When in doubt, delegate: a wrong or stale direct \
answer is far worse than the small cost of delegating.
{skills_section}
Direct answers must be short, helpful, and in the user's language. Never \
mention this routing step, the `delegate` tool, or "the full agent" — from \
the user's point of view there is only one assistant."""


_SKILLS_SECTION_HEADER = """
When you delegate, include in `skills` any of these known procedures that \
clearly apply (their labels exactly as written), so they are preloaded:
"""


def build_router_system_prompt(
    skill_summaries: list[dict] | None = None,
) -> str:
    """Render the router's minimal system prompt.

    ``skill_summaries`` is the ``SkillStore.list_summaries()`` shape:
    dicts with ``label`` and ``description`` keys. Listed one per line so
    the router can name preloads at delegation time.
    """
    skills_section = ""
    lines: list[str] = []
    for s in skill_summaries or []:
        label = (s.get("label") or "").strip()
        if not label:
            continue
        when = (s.get("description") or "").strip()
        lines.append(f"- `{label}` — {when}" if when else f"- `{label}`")
    if lines:
        skills_section = _SKILLS_SECTION_HEADER + "\n".join(lines) + "\n"
    return _ROUTER_SYSTEM_PROMPT.format(skills_section=skills_section)


def condense_history(
    history: list[dict],
    *,
    max_messages: int = _MAX_ROUTER_MESSAGES,
    max_chars: int = _MAX_CHARS_PER_MESSAGE,
) -> list[dict]:
    """Build the router's text-only view of the conversation.

    Tool blocks collapse to one-line markers (the router only needs to
    know work happened, not its payload), long messages are middle-
    truncated, and consecutive same-role messages merge so the result
    keeps the role alternation providers require. Only the most recent
    ``max_messages`` survive, and a leading assistant message left over
    from the cut is dropped so the list starts with a user message.
    """
    condensed: list[dict] = []
    for msg in history:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    parts.append(str(block.get("text", "")))
                elif btype == "tool_use":
                    parts.append(f"[ran tool: {block.get('name', '?')}]")
                elif btype == "tool_result":
                    parts.append("[tool output omitted]")
                elif btype == "image":
                    parts.append("[image]")
            text = "\n".join(p for p in parts if p)
        else:
            continue
        text = text.strip()
        if not text:
            continue
        if len(text) > max_chars:
            half = max_chars // 2
            text = f"{text[:half]}\n[… truncated …]\n{text[-half:]}"
        if condensed and condensed[-1]["role"] == role:
            condensed[-1]["content"] += "\n" + text
        else:
            condensed.append({"role": role, "content": text})

    condensed = condensed[-max_messages:]
    while condensed and condensed[0]["role"] != "user":
        condensed.pop(0)
    return condensed


@dataclass
class RouterDecision:
    """Outcome of the routing call.

    ``action`` is ROUTE_RESPOND (``text`` carries the direct answer) or
    ROUTE_DELEGATE (``skills`` carries validated-later preload labels).
    ``response`` is the raw router LLMResponse, kept so callers can
    surface its usage in stream events.
    """

    action: str
    text: str = ""
    skills: list[str] = field(default_factory=list)
    reason: str = ""
    response: LLMResponse | None = None


async def route_turn(
    llm: "LLMClient",
    *,
    history: list[dict],
    skill_summaries: list[dict] | None = None,
    max_tokens: int = 1024,
) -> RouterDecision:
    """Run the routing call and interpret its outcome.

    Every non-answer shape resolves to DELEGATE: a `delegate` tool call
    (the explicit path), an empty response, or a direct answer cut off
    by ``max_tokens`` — an answer that long is evidence the turn wasn't
    trivial, and a truncated reply must never reach the user.
    """
    messages = condense_history(history)
    if not messages:
        return RouterDecision(action=ROUTE_DELEGATE, reason="no routable history")

    response = await llm.route(
        system=build_router_system_prompt(skill_summaries),
        messages=messages,
        tools=[DELEGATE_TOOL],
        max_tokens=max_tokens,
    )

    for tc in response.tool_calls:
        if tc.name != "delegate":
            continue
        tc_input = tc.input if isinstance(tc.input, dict) else {}
        skills = [
            s.strip()
            for s in (tc_input.get("skills") or [])
            if isinstance(s, str) and s.strip()
        ]
        return RouterDecision(
            action=ROUTE_DELEGATE,
            skills=skills,
            reason=str(tc_input.get("reason", "")),
            response=response,
        )

    if response.stop_reason in ("max_tokens", "length"):
        return RouterDecision(
            action=ROUTE_DELEGATE,
            reason="direct answer exceeded the router output budget",
            response=response,
        )

    text = (response.content or "").strip()
    if not text:
        return RouterDecision(
            action=ROUTE_DELEGATE,
            reason="router produced no answer",
            response=response,
        )

    return RouterDecision(action=ROUTE_RESPOND, text=text, response=response)


__all__ = [
    "DELEGATE_TOOL",
    "ROUTE_DELEGATE",
    "ROUTE_RESPOND",
    "RouterDecision",
    "build_router_system_prompt",
    "condense_history",
    "route_turn",
]
