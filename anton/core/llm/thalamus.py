"""Thalamus — the gate between an incoming turn and the cortex.

Brain analogue
==============

The thalamus is the brain's central relay. Almost every sensory and
motor signal passes through it on the way to the cortex — it is the
first thing a signal hits, and it decides what happens next. Four of
its properties are worth stealing:

1. **It gates; it does not compute.** The thalamus relays signals and
   modulates their gain. The heavy interpretation happens in the cortex.
   A thalamus that started "thinking" would be both slow and a liability.

2. **Its default posture is inhibitory.** The thalamic reticular
   nucleus (TRN) is a shell of inhibitory neurons that suppresses relay
   by default and selectively *dis*inhibits the one channel that earns
   passage — Crick's "searchlight." Nothing gets through on a maybe.

3. **It has two firing modes.** *Tonic* mode faithfully relays a simple,
   attended signal. *Burst* mode is an alerting spike that says "cortex,
   wake up, this one needs you." Every signal resolves to one or the
   other.

4. **The cortex talks back — a lot.** Corticothalamic fibers outnumber
   the feed-forward ones roughly ten to one: the cortex constantly
   biases what the thalamus lets through, based on context and
   expectation. The gate is not context-free.

Anton's analogue
================

Anton's thalamus is the cheap front-model that every text turn hits
first (ENG-648). Mapping the four properties:

1. *Gate, don't compute* — the gating call runs on a cost-effective
   model with a minimal system prompt and **no tool schemas**. It may
   answer, or it may hand off, but it never does real work (no
   scratchpad, no data, no files). Keeping it thin is the whole point.

2. *Default-inhibit* — the direct-answer path (the TRN "searchlight")
   opens only for signals that are unambiguously trivial: derivable
   from the conversation plus stable general knowledge, needing nothing
   created, fetched, or computed. Every ambiguity — an empty response,
   an answer that overran the output budget, an outright error — falls
   through to the cortex. A wrong direct answer is far costlier than the
   gating overhead, so "when in doubt, relay up" is the resting state.

3. *Two modes* — the two decisions ARE tonic vs burst. ``ACTION_RESPOND``
   is tonic relay: the thalamus itself passes a faithful, complete
   answer for a simple signal. ``ACTION_DELEGATE`` is the burst: a
   forced ``delegate`` tool call that alerts the planning model (the
   cortex) and hands the turn up, optionally naming procedural skills to
   preload so the cortex doesn't spend a round fetching them.

4. *Corticothalamic feedback* — partial today, and the clearest place to
   grow. The gate already receives one top-down signal: the list of
   learned skills (a cortical/hippocampal product) shapes what it can
   preload. The principled next step is to let recent cortical outcomes
   bias the gate — e.g. if this project's turns almost always need
   tools, lower the bar to delegate. That hook is called out at
   ``gate_turn`` rather than built here.

Because the thalamus is the first relay, a misfire starves everything
downstream — so, like a lesioned thalamus that must not silently drop
signals, ours fails *open*: any doubt or error relays the turn up to the
cortex untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from anton.core.llm.provider import LLMResponse

if TYPE_CHECKING:
    from anton.core.llm.client import LLMClient


# The two gate outcomes — tonic relay (answer here) vs burst (alert the
# cortex / planning model). See the module docstring.
ACTION_RESPOND = "respond"
ACTION_DELEGATE = "delegate"

# The thalamus sees a condensed, text-only view of history: full
# transcripts (and the tool_use/tool_result blocks inside them) are what
# the cortex needs, not what a gating call needs.
_MAX_GATE_MESSAGES = 16
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


_THALAMUS_SYSTEM_PROMPT = """\
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


def build_thalamus_system_prompt(
    skill_summaries: list[dict] | None = None,
) -> str:
    """Render the thalamus's minimal system prompt.

    ``skill_summaries`` is the ``SkillStore.list_summaries()`` shape:
    dicts with ``label`` and ``description`` keys. Listed one per line so
    the thalamus can name preloads at delegation time. This list is the
    one corticothalamic signal the gate currently receives — top-down
    context (learned procedures) biasing what it can relay upward.
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
    return _THALAMUS_SYSTEM_PROMPT.format(skills_section=skills_section)


def condense_history(
    history: list[dict],
    *,
    max_messages: int = _MAX_GATE_MESSAGES,
    max_chars: int = _MAX_CHARS_PER_MESSAGE,
) -> list[dict]:
    """Build the thalamus's text-only view of the conversation.

    Tool blocks collapse to one-line markers (the gate only needs to
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
class ThalamicDecision:
    """Outcome of the gating call.

    ``action`` is ACTION_RESPOND (tonic relay — ``text`` carries the
    direct answer) or ACTION_DELEGATE (burst — ``skills`` carries
    preload labels validated later). ``response`` is the raw gating
    LLMResponse, kept so callers can surface its usage in stream events.
    """

    action: str
    text: str = ""
    skills: list[str] = field(default_factory=list)
    reason: str = ""
    response: LLMResponse | None = None


async def gate_turn(
    llm: "LLMClient",
    *,
    history: list[dict],
    skill_summaries: list[dict] | None = None,
    max_tokens: int = 1024,
) -> ThalamicDecision:
    """Run the gating call and interpret its outcome.

    Every non-answer shape resolves to DELEGATE (default-inhibit): a
    `delegate` tool call (the explicit burst), an empty response, or a
    direct answer cut off by ``max_tokens`` — an answer that long is
    evidence the turn wasn't trivial, and a truncated reply must never
    reach the user.

    Corticothalamic-feedback hook: today the only top-down signal is
    ``skill_summaries``. A future version can accept recent cortical
    outcomes (delegation rate per project, last-turn tool usage) here and
    bias the gate — lowering the bar to delegate where the cortex keeps
    getting woken anyway. Deliberately not built yet; see the module
    docstring.
    """
    messages = condense_history(history)
    if not messages:
        return ThalamicDecision(action=ACTION_DELEGATE, reason="no routable history")

    response = await llm.gate(
        system=build_thalamus_system_prompt(skill_summaries),
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
        return ThalamicDecision(
            action=ACTION_DELEGATE,
            skills=skills,
            reason=str(tc_input.get("reason", "")),
            response=response,
        )

    if response.stop_reason in ("max_tokens", "length"):
        return ThalamicDecision(
            action=ACTION_DELEGATE,
            reason="direct answer exceeded the gating output budget",
            response=response,
        )

    text = (response.content or "").strip()
    if not text:
        return ThalamicDecision(
            action=ACTION_DELEGATE,
            reason="thalamus produced no answer",
            response=response,
        )

    return ThalamicDecision(action=ACTION_RESPOND, text=text, response=response)


__all__ = [
    "ACTION_DELEGATE",
    "ACTION_RESPOND",
    "DELEGATE_TOOL",
    "ThalamicDecision",
    "build_thalamus_system_prompt",
    "condense_history",
    "gate_turn",
]
