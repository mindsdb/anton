"""The page builder: one focused agent loop that turns a request into a page with the kit.

`generate_artifact` runs it for every `html-app`. The builder works in the session's scratchpad:
it fetches the data, computes every figure with code, composes the page with `kit` and saves it,
reads the outline, warnings and browser check the cell returns, fixes what they flag, and calls
`finish`. One loop, no brief/PRD/spec stages: the kit carries the design and the rendering, so the
model's job is the data and the composition.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anton.core.session import ChatSession

logger = logging.getLogger(__name__)

MAX_ROUNDS = 10
CALL_TOKENS = 16000

KIT_REFERENCE = '''\
```
from anton.core.artifacts import kit
page = kit.Page(title, subtitle=None, lang="en", theme="light"|"dark", badges=[...], meta=[...],
                width="normal"|"wide"|"narrow"|"mobile", currency="USD", sources=[...], notes=[...],
                labels={field: label}, formats={field: format})
# formats: int, number:1, percent (98.4 -> 98.4%), fraction (0.984 -> 98.4%), currency, currency:2,
#   currency_compact ($1.57M), compact, days, date (Nov 20, 2026), date_short, date_iso, month, text
page.kpis([kit.kpi(label, value, format, sub=None, delta=None, better="up"|"down", tone="good"|"bad"|"warn"|"accent")])
page.text(markdown, style="lead") · page.callout(markdown, title=None, tone="info"|"good"|"warn"|"bad")
page.compare(["Base", "Option B"], [kit.metric(label, [v1, v2], format, better="higher"|"lower")], title=)
page.chart(kind, df, x, y, series=None, title=, format=, stack=False, horizontal=False, sort="desc",
           threshold={"value": 100, "label": "Capacity"}, ref_lines=[{"value": v, "label": l}],
           y2=None, colors={name: color_or_tone}, show_values=None)
           # kind: bar hbar line area combo(y bars + y2 line) scatter pie donut heatmap waterfall
page.table(df, [field, kit.col(field, label, format, pill=True, tones={value: tone}, bar=True)], title=,
           search=None, filters=[fields], sort={"field": f, "desc": True}, totals={field: "sum"},
           key=None, link=None, detail="drawer"|"inline"|False, row_tone={"field": f, "map": {value: tone}})
page.map(kit.points(df, lat, lon, label=, sublabel=, color=field, colors={value: tone}, shape="circle"|
           "square"|"diamond", tooltip=[fields], key=, link=, pulse=False, name=legend_name),
         kit.routes(df, from_lat, from_lon, to_lat, to_lon, via_lat=None, via_lon=None, color=, colors=,
           animate=False, key=, link=, tooltip=[...]),
         kit.regions(df, country, value, format=), title=, fit="data"|"world")
page.globe(<the same layers>, title=, rotate="spin"|"sway"|False)    # drag to turn; routes arc above it
page.network(links_df, source=, target=, value=None, nodes=nodes_df, id_field="id", label=, group=,
             tiers=[group order, left to right], style="graph"|"sankey", link_color=field,
             link_colors={value: tone}, dashed=field, value_format=, title=)
page.timeline(bars=[kit.bars(df, row, start, end, label=None, name=legend, tone=)],
              milestones=[kit.milestones(df, row, date, label=, tooltip=[fields])],
              markers=[kit.marker(date, label, tone)], title=)    # one lane per bars layer: plan vs actual
with page.section(title, subtitle): ...      with page.grid(2, widths=[2, 1]): ...
t = page.tabs(["Base Plan", "Option B"], style="segmented"|"tabs");  with t["Base Plan"]: ...
page.echarts(option_dict) for any other chart · page.html(html, js=...) for anything else · page.image(fig)
deck = kit.Deck(title, subtitle, lang=...); with deck.slide(title, subtitle, layout="title"|"content"|
       "big_number"|"split"|"section", value=, caption=): deck.chart(...)   # slides take the same blocks
page.save(folder)                        # prints the outline (with its figures) and WARNINGS
page = kit.load(folder); print(page.outline())   # edit a saved page:
with page.after("b3"): page.table(...)   # page.update("b2", title=...), page.set(theme="dark"), page.remove(id)
kit.geocode("City, Country") -> (lat, lon) or None   # offline gazetteer of towns
```
Colours: tones good/bad/warn/info/accent/neutral or any CSS colour; categories get a consistent palette.
Values must be numbers (not formatted strings); dates as "YYYY-MM-DD" strings or datetimes.'''

BUILDER_PROMPT = f'''\
You build one web page for the user, end to end, then call `finish`. The page is an artifact \
(type html-app) already registered; its folder is given below. You build it with the artifact kit \
in the scratchpad, from data you fetch and compute there. The kit writes one self-contained HTML \
file: a designed, responsive page (desktop and phone, light or dark) whose charts, maps, globes, \
networks, timelines, tables and KPIs are pre-built and tested, with the data embedded straight from \
your DataFrames. Never write the page's HTML, CSS or JavaScript yourself and never re-type numbers \
into it.

Tools: `scratchpad` runs Python (query connected data sources with HTTP or SQL using their \
credential environment variables, compute with pandas, build and save the page); `recall_skill` \
loads a skill's instructions (for example how to query a data source); `finish` ends the build. \
You work alone: nobody answers questions, so decide sensibly and say what you assumed in `finish`.

How to work (fast: every round costs the user time):
1. If a skill below describes the data source you need, recall it first (nothing else in that reply).
2. One scratchpad cell that fetches every table the page needs, computes every figure, prints \
those figures, and builds and saves the page with `page.save(<folder>)`. Name the scratchpad `page` \
unless one listed below already holds the data.
3. Read the cell's output: the page outline with its figures, any WARNINGS, and the browser check \
of the page. Fix anything wrong (an error, a warning, a wrong or missing figure, a failed check) in \
one more cell.
4. Call `finish` with a short summary for the user: the key figures, what the page shows, anything \
you could not show and why, and what you assumed.

Getting it right:
- Use exactly the scope asked for: the named scenario, period, site, status or population, and \
nothing else. Take each figure from that scope's own rows; never mix values from different \
scenarios or versions of the data, and never add up values that should not be summed. With several \
scenarios, fetch and compute each one separately and label it.
- Count and total from rows with code, then sanity-check (row counts, ranges, that parts add up \
to their total) before you build. Keep the data's own wording for statuses and measures (e.g. "at \
risk" is not "lost"); dates follow the data's calendar, not today's.
- Never invent records, ids, places or values. When the data has no answer, say so on the page. \
Places: use coordinates from the data; for place names use `kit.geocode("City, Country")` \
(None when unsure: leave that point out and say so).
- A page that already exists (its outline is below) is an edit: `page = kit.load(folder)`, change \
only what the request asks, keep every other block, figure and setting, and save. Re-running the \
code that built it with the change is fine when that code is in the scratchpad. When asked to check \
a figure, recompute it from the source data, print old and new, fix the page if needed, and add a \
short note on the page saying what was checked.

Designing it:
- Lead with the answer: a title naming the subject, a subtitle with the main finding in numbers, \
badges for the scope (scenario, period), a meta line such as "Data as of 2026-05-04". Then the \
elements the request names, in its order, and a short sources note.
- Show what was asked, all of it, and nothing nobody asked for: a KPI row only when the request \
asks for figures, totals, a dashboard or a brief; no extra charts, panels or metrics "for context". \
Every figure the user asked for appears as text (a KPI, a table cell, a label or a compare row), \
not only inside a chart.
- Pick the view for the request: places on a map (`page.map`); "3D" or "globe" on `page.globe`; \
flows, lanes or supply networks on `page.network` (style "sankey" for volumes); schedules, \
milestones or Gantt on `page.timeline`; options or scenarios side by side on `page.compare`; trends \
and breakdowns on `page.chart`; records on `page.table`.
- Interactions: "switch/toggle between A and B" is `page.tabs([...], style="segmented")` with each \
option's blocks inside; "filter" is `filters=[...]`/`search=True` on the table; "click a row to see \
its detail" is `detail="drawer"`; "clicking a row highlights it on the map/globe/chart" is the same \
`key` and `link` on the table and on the map layers; hovering a map point, bar or milestone shows \
its tooltip.
- Write every word on the page in the user's language and set `lang` ("en", "es", "fr", ...): \
titles, labels (`labels={{field: label}}` and column labels), notes. "Phone/mobile" pages use \
`width="mobile"`. "Premium", "board" or "leadership" pages use `theme="dark"` or a calm light page \
with a strong headline, few large elements and subtle motion (built in). Animate map routes with \
`animate=True` only when motion is asked for.

The kit (use it as documented here; no need to read its source):
{KIT_REFERENCE}
'''

FINISH_TOOL = {
    "name": "finish",
    "description": (
        "End the build once the page is saved and its output checked. `summary` is for the user: "
        "the key figures, what the page shows, anything not shown and why, assumptions made."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"summary": {"type": "string", "description": "Short summary for the user."}},
        "required": ["summary"],
    },
}


def _context_sections(session: "ChatSession") -> str:
    """Connected data sources (names of their credential variables, never values), the skills the
    builder may recall, and the project's rules."""
    parts: list[str] = []
    try:
        from anton.utils.datasources import build_datasource_context

        ds = build_datasource_context(
            session._data_vault,
            active_only=getattr(session, "_active_datasource", None),
            usage_notes=getattr(session, "_connector_usage_notes", None),
        )
        if ds and ds.strip():
            parts.append(ds.strip())
    except Exception:  # a build without a vault still works on public or attached data
        logger.debug("page builder: no datasource context", exc_info=True)
    store = getattr(session, "_skill_store", None)
    if store is not None:
        try:
            skills = [s for s in store.list_summaries() if s.get("label") and not s["label"].startswith("build-")]
        except Exception:
            skills = []
        if skills:
            lines = ["## Skills you can recall"]
            lines += [f"- `{s['label']}` — {s.get('description', '').strip()}" for s in skills]
            parts.append("\n".join(lines))
    read = getattr(getattr(session, "_workspace", None), "read_anton_md", None)
    if read is not None:
        try:
            rules = read()
            if isinstance(rules, str) and rules.strip():
                parts.append("## Project rules (follow them)\n" + rules.strip())
        except Exception:
            pass
    return "\n\n".join(parts)


def _existing_page(folder: Path) -> str:
    page = folder / "index.html"
    if not page.exists():
        return "No page yet: build it."
    try:
        from anton.core.artifacts import kit

        loaded = kit.load(page)
        return ("The page already exists (an edit). Its outline:\n" + loaded.outline())
    except Exception:
        return (f"A page already exists at {page} but was not made with the kit. Rebuild it with the kit, "
                "keeping its content and look as close as the request allows.")


def _pads(session: "ChatSession") -> str:
    try:
        pads = session._scratchpads.pads if hasattr(session._scratchpads, "pads") else {}
        names = sorted(pads) if isinstance(pads, dict) else []
    except Exception:
        names = []
    if not names:
        return ""
    return ("## Scratchpads already in this session\n" + ", ".join(f"`{n}`" for n in names)
            + "\nA pad keeps the variables of earlier cells (data already fetched); reuse it to save time.")


def kickoff(*, folder: Path, user_request: str, user_language: str, agent_understanding: str,
            known_data: str, session: "ChatSession") -> str:
    sections = [
        f"## The request\n{user_request.strip()}",
        f"## Language of the page\n{user_language.strip() or 'the language of the request'}",
    ]
    if agent_understanding.strip():
        sections.append(f"## Notes from the conversation\n{agent_understanding.strip()}")
    if known_data.strip():
        sections.append(f"## Data already known\n{known_data.strip()}")
    sections.append(f"## Artifact folder\n{folder}")
    sections.append(f"## The page\n{_existing_page(folder)}")
    pads = _pads(session)
    if pads:
        sections.append(pads)
    return "\n\n".join(sections)


async def build_page(session: "ChatSession", *, slug: str, folder: Path, user_request: str,
                     user_language: str, agent_understanding: str = "", known_data: str = "",
                     progress=None) -> dict:
    """Run the builder loop. Returns {status, slug, page, summary, rounds}. `progress` is an
    asyncio.Queue for the progress lines the UI shows (closed with None by the caller)."""
    from anton.core.tools.generate_artifact.sub_tools import (assistant_blocks, malformed_input_result,
                                                             tool_result, tool_schema, unwrap_outcome)
    from anton.core.tools.recall_skill import RECALL_SKILL_TOOL, handle_recall_skill
    from anton.core.tools.tool_defs import SCRATCHPAD_TOOL
    from anton.core.tools.tool_handlers import handle_scratchpad

    def say(line: str) -> None:
        if progress is not None:
            progress.put_nowait(line)

    folder = Path(folder)
    page_path = folder / "index.html"
    before = page_path.stat().st_mtime if page_path.exists() else None
    system = BUILDER_PROMPT
    context = _context_sections(session)
    if context:
        system += "\n\n" + context
    tools = [tool_schema(SCRATCHPAD_TOOL), tool_schema(RECALL_SKILL_TOOL), FINISH_TOOL]
    messages: list[dict] = [{"role": "user", "content": kickoff(
        folder=folder, user_request=user_request, user_language=user_language,
        agent_understanding=agent_understanding, known_data=known_data, session=session)}]
    summary, nudged, last_text = None, 0, ""
    say("Reading the request")
    rounds = 0
    for rounds in range(1, MAX_ROUNDS + 1):
        response = await session._llm.plan(system=system, messages=messages, tools=tools, max_tokens=CALL_TOKENS)
        messages.append({"role": "assistant", "content": assistant_blocks(response)})
        last_text = (response.content or "").strip() or last_text
        if not response.tool_calls:
            if page_path.exists() and (before is None or page_path.stat().st_mtime != before):
                summary = last_text
                break
            nudged += 1
            if nudged > 2:
                break
            messages.append({"role": "user", "content": (
                "No page is saved yet. Run the scratchpad now: fetch the data, compute the figures, build "
                f"the page with the kit and save it to {folder}. The scratchpad can reach the connected "
                "data sources and the internet.")})
            continue
        results = []
        done = None
        for tc in response.tool_calls:
            if getattr(tc, "parse_error", None):
                results.append(malformed_input_result(tc))
                continue
            if tc.name == "scratchpad":
                say((tc.input or {}).get("one_line_description") or "Working in the scratchpad")
                out = unwrap_outcome(await handle_scratchpad(session, tc.input or {}))
                results.append(tool_result(tc.id, out))
            elif tc.name == "recall_skill":
                say("Reading " + str((tc.input or {}).get("label") or "a skill"))
                results.append(tool_result(tc.id, unwrap_outcome(await handle_recall_skill(session, tc.input or {}))))
            elif tc.name == "finish":
                saved = page_path.exists() and (before is None or page_path.stat().st_mtime != before)
                if saved:
                    done = str((tc.input or {}).get("summary") or "")
                    results.append(tool_result(tc.id, "ok"))
                else:
                    results.append(tool_result(tc.id, (
                        f"Not finished: no page was saved to {folder} in this build. Build and save it "
                        "with the kit first.")))
            else:
                results.append(tool_result(tc.id, f"Unknown tool `{tc.name}`: use scratchpad, recall_skill or finish."))
        messages.append({"role": "user", "content": results})
        if done is not None:
            summary = done
            break
        if rounds == MAX_ROUNDS - 1:
            messages.append({"role": "user", "content": "Last round: if the page is saved, call `finish` now."})
    saved = page_path.exists() and (before is None or page_path.stat().st_mtime != before)
    say("Page ready" if saved else "The page could not be finished")
    return {
        "status": "generated" if saved else "failed",
        "slug": slug,
        "page": str(page_path) if saved else None,
        "summary": summary or last_text,
        "rounds": rounds,
    }


def result_text(result: dict) -> str:
    """What `generate_artifact` returns to the agent for a page."""
    if result["status"] == "generated":
        instruction = ("Reply to the user in their language: what the page shows and its key figures (from the "
                       "summary), anything not shown and why, and the artifact's name. Do not rebuild or edit the "
                       "page in this reply.")
    else:
        instruction = ("The page was not finished. Tell the user plainly what went wrong, from the summary; do not "
                       "claim a page was built.")
    return json.dumps({**result, "instruction": instruction}, ensure_ascii=False)
