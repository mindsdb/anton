"""The prompt section for skills an organization applies on every turn.

Cowork shares skills org-wide and with teams, and those skills hold for every
task rather than waiting to be recalled. The cloud pod appends this section to
its prompt suffix; cowork-server builds the same text for desktop, so the
format here is a contract and must change in step with it.
"""

from __future__ import annotations

ALWAYS_ON_SKILL_MAX_CHARS = 6000
ALWAYS_ON_TOTAL_MAX_CHARS = 24000

_HEADER = (
    "\n\n## Organization and team skills (always apply)\n"
    "These are how this organization works. Follow them on every task where "
    "they apply; they override general defaults but never the user's explicit "
    "instructions in this conversation."
)
_TRUNCATED = "\n[…truncated; the full skill is available on demand]"


def _truncate(body: str) -> str:
    """*body* cut to the per-skill cap at a line boundary, unmarked."""
    if len(body) <= ALWAYS_ON_SKILL_MAX_CHARS:
        return body
    head = body[:ALWAYS_ON_SKILL_MAX_CHARS]
    # A single line longer than the cap has no boundary to cut at.
    cut = head.rfind("\n")
    return (head[:cut] if cut > 0 else head).rstrip()


def render_always_on_skills(skills: list[dict]) -> str:
    """The always-apply section for *skills*, or "" when there are none.

    *skills* is ``[{"name": str, "body": str}]`` in the order cowork sent them.
    Bodies count against the total budget before the truncation marker; the
    first skill that no longer fits, and every one after it, is left out and
    named on a closing line instead, since it is still staged for recall.
    """
    blocks: list[str] = []
    omitted: list[str] = []
    used = 0
    for skill in skills:
        name, body = skill["name"], skill["body"]
        text = _truncate(body)
        if omitted or used + len(text) > ALWAYS_ON_TOTAL_MAX_CHARS:
            omitted.append(name)
            continue
        used += len(text)
        if text != body:
            text += _TRUNCATED
        blocks.append(f"### {name}\n{text}")
    if not blocks and not omitted:
        return ""
    section = _HEADER + "".join(f"\n\n{block}" for block in blocks)
    if omitted:
        section += f"\n\n(Also available on demand: {', '.join(omitted)}.)"
    return section
