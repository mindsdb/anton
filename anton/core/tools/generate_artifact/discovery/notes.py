"""Deterministic renderers for what the discovery phases found.

Both outputs are built by code, not summarised by a model, and both are
capped. They are the only channels from phases A-C into the generation
nodes: the shared message list is dropped at the spec boundary, so anything
a generator needs verbatim has to be here or in `spec.md`.
"""

from __future__ import annotations

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
