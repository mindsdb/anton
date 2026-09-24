"""Deterministic renderers for what the discovery phases found.

Both outputs are built by code, not summarised by a model, and both are
capped. They are the only channels from phases A-C into the generation
nodes: the shared message list is dropped at the spec boundary, so anything
a generator needs verbatim has to be here or in `spec.md`.
"""

from __future__ import annotations

import re

# Caps for the exec-code record: per-cell code, per-cell output snippet, and
# the whole section. Oldest cells are dropped first — the most recent ones are
# the ones that worked.
EXEC_CODE_MAX = 2000
EXEC_OUTPUT_MAX = 300
EXEC_NOTES_MAX = 8000

# Caps for the web record. Deliberately much smaller than a fetched page: a
# measured run pulled a 9.5KB article dump, and phase E re-sends its context
# on every write round. What travels is the pointer plus enough text to
# recognise the source; the body itself was already read by `make_tech_spec`,
# which is the last node that sees it.
WEB_EXCERPT_MAX = 1500
WEB_NOTES_MAX = 6000


def render_exec_notes(
    execs: list[dict], *, header: str = "### Code executed while fetching"
) -> str:
    """Deterministic record of the Python the discovery phase ran.

    Appended to data_notes so later steps (tech spec, backend generation) see
    the exact working data-access code instead of relying on the model's
    `finish` summary to mention it.
    """
    blocks: list[str] = []
    for e in execs:
        code = (e.get("code") or "").strip()
        if not code:
            continue
        if len(code) > EXEC_CODE_MAX:
            code = code[:EXEC_CODE_MAX] + "\n# … truncated …"
        out = " ".join((e.get("output") or "").split())
        if len(out) > EXEC_OUTPUT_MAX:
            out = out[:EXEC_OUTPUT_MAX] + " …"
        block = f"Scratchpad `{e.get('name')}`:\n```python\n{code}\n```"
        if out:
            block += f"\nOutput: {out}"
        blocks.append(block)
    dropped = 0
    while blocks and sum(len(b) for b in blocks) > EXEC_NOTES_MAX:
        blocks.pop(0)
        dropped += 1
    if not blocks:
        return ""
    if dropped:
        header += f" (first {dropped} cell(s) omitted for size)"
    return header + "\n" + "\n\n".join(blocks)


def render_web_notes(calls: list[dict]) -> str:
    """Record of what the discovery phase pulled off the web.

    `calls` items carry `kind` ("web_fetch" | "web_search"), `url`, `title`,
    `query` and `excerpt`. A call with neither a url nor a query is dropped:
    there is nothing for a generator to point at.

    This exists because a generator that must link back to a source, or embed
    its images, cannot depend on the tech spec having carried those URLs
    forward — a model summarising a page is exactly where a URL goes missing.
    """
    header = "### Sources read from the web"
    blocks: list[str] = []
    for c in calls:
        url = (c.get("url") or "").strip()
        query = (c.get("query") or "").strip()
        if not url and not query:
            continue
        title = " ".join((c.get("title") or "").split())
        excerpt = " ".join((c.get("excerpt") or "").split())
        if len(excerpt) > WEB_EXCERPT_MAX:
            excerpt = excerpt[:WEB_EXCERPT_MAX] + " …"
        head = f"- {c.get('kind') or 'web'}: {url or query}"
        if title:
            head += f" — {title}"
        blocks.append(head + (f"\n  {excerpt}" if excerpt else ""))
    dropped = 0
    while blocks and sum(len(b) for b in blocks) > WEB_NOTES_MAX:
        blocks.pop(0)
        dropped += 1
    if not blocks:
        return ""
    if dropped:
        header += f" (first {dropped} source(s) omitted for size)"
    return header + "\n" + "\n".join(blocks)


# Residue of the model's own tool-call syntax leaking into a JSON value.
# Seen live: `open_points` arrived as the string
# '\n<parameter name="open_points">What does ...' (the question itself was in
# the user's language) — the model wrote the field's opening tag inside the
# field. It is markup, never content.
_TOOL_CALL_MARKUP_RE = re.compile(r"</?parameter(?:\s[^>]*)?>")

# Leading list markers a model puts in front of each line when it sends a
# list as one multi-line string: "- ", "* ", "• ", "1. ", "2) ".
_LIST_MARKER_RE = re.compile(r"^(?:[-*•]|\d{1,2}[.)])\s+")


def string_list(value) -> list[str]:
    """A schema `array` of strings as the model actually sent it. The
    model's JSON is never trusted to match the schema:

    - a list: items stringified, `null` and blanks dropped;
    - a string: one item per non-blank line, list markers removed. Seen live:
      `constraints`, `assumptions` and `open_points`
      all came as plain strings, the old "not a list → empty" dropped every
      one of them, and `discovery.json` recorded no assumption and no open
      point — a cold start would have redrawn the brief without them;
    - anything else (a number, an object): empty.

    Every item is also cleaned of tool-call markup (`<parameter ...>`)."""
    if isinstance(value, str):
        raw = [_LIST_MARKER_RE.sub("", line.strip()) for line in value.splitlines()]
    elif isinstance(value, list):
        raw = [str(v) for v in value if v is not None]
    else:
        return []
    cleaned = (_TOOL_CALL_MARKUP_RE.sub("", item).strip() for item in raw)
    return [item for item in cleaned if item]


def render_gathering_notes(inp: dict) -> str:
    """Markdown record of a `finish_gathering` call, built by code.

    The structured fields (`data_findings`, `constraints`, `assumptions`,
    `open_points`) replaced the free-form `notes` so the
    gathering step records facts and open decisions instead of drafting the
    brief. A call that still carries `notes` — an older prompt, a model that
    ignored the schema — falls back to it, then to `summary`, so nothing the
    model wrote is dropped on the floor.
    """
    parts: list[str] = []
    summary = str(inp.get("summary") or "").strip()
    if summary:
        parts.append(summary)
    findings = inp.get("data_findings")
    if isinstance(findings, list):
        rows: list[str] = []
        for f in findings:
            if not isinstance(f, dict):
                continue
            source = str(f.get("source") or "").strip()
            if not source:
                continue
            row = f"- {source}"
            for key, label in (
                ("verified_by", "verified by"),
                ("shape", "shape"),
                ("sample", "sample"),
            ):
                value = " ".join(str(f.get(key) or "").split())
                if value:
                    row += f"\n  {label}: {value}"
            rows.append(row)
        if rows:
            parts.append("### Data findings\n" + "\n".join(rows))
    for key, header in (
        ("constraints", "### Constraints"),
        ("assumptions", "### Assumptions"),
        ("open_points", "### Open points"),
    ):
        lines = string_list(inp.get(key))
        if lines:
            parts.append(header + "\n" + "\n".join(f"- {line}" for line in lines))
    # The structured fields decide, not the count: `summary` is `parts[0]`
    # only when present, so "more than one part" silently returned an empty
    # string for a call that carried findings and constraints but no summary
    # (I-46). Sections present → the rendered record; none → the legacy
    # `notes`, then whatever summary there was.
    has_sections = len(parts) > (1 if summary else 0)
    if has_sections:
        return "\n\n".join(parts)
    legacy = str(inp.get("notes") or "").strip()
    return legacy or summary
