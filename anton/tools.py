"""Extra tools for the open source terminal agent."""

from __future__ import annotations
from typing import TYPE_CHECKING

from anton.core.tools.tool_defs import ToolDef

if TYPE_CHECKING:
    from anton.core.session import ChatSession


async def handle_connect_datasource(session: ChatSession, tc_input: dict) -> str:
    """Handle connect_new_datasource tool call — interactive connection flow."""
    engine = tc_input.get("engine", "")
    if not engine:
        return "Engine name is required."

    raw_known = tc_input.get("known_variables") or {}
    known_variables: dict[str, str] = (
        {str(k): str(v) for k, v in raw_known.items() if v is not None and v != ""}
        if isinstance(raw_known, dict) else {}
    )

    console = session._console
    if console is None:
        return "Cannot connect datasource — no console available."

    console.print()
    console.print(
        f"[anton.prompt]anton>[/] I can help with that \u2014 let's connect [bold]{engine}[/] to Anton."
    )

    from anton.commands.datasource import handle_connect_datasource
    from anton.core.datasources.data_vault import DataVault

    # Check which connections exist before
    vault = DataVault()
    before = {f"{c['engine']}-{c['name']}" for c in vault.list_connections()}

    # Clear any stale status from a previous run
    setattr(session, "_pending_connect_redirect", None)
    setattr(session, "_pending_connect_status", None)

    await handle_connect_datasource(
        console,
        session._scratchpads,
        session,
        prefill=engine,
        known_variables=known_variables or None,
        from_tool_call=True,
    )

    # Check if a new connection was actually added
    after = {f"{c['engine']}-{c['name']}" for c in vault.list_connections()}
    new_connections = after - before

    if new_connections:
        slug = next(iter(new_connections))
        return (
            f"Successfully connected '{slug}'. The datasource is now available. "
            f"Continue helping the user with their original request using this data source."
        )

    # Did the flow record a mid-flow redirect? Read it from the session
    # attribute stashed by _build_redirect_message. We CANNOT append to
    # session._history from within the handler — we're between the
    # tool_use and tool_result blocks and doing so breaks the Anthropic
    # API invariant that every tool_use must be immediately followed by
    # its tool_result.
    redirect_text = getattr(session, "_pending_connect_redirect", None)
    if redirect_text:
        setattr(session, "_pending_connect_redirect", None)
        return redirect_text

    # No new connection was saved. Distinguish *why* — the LLM should
    # not be told "user pressed Escape" when really the test failed.
    status = getattr(session, "_pending_connect_status", None)
    setattr(session, "_pending_connect_status", None)

    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text
    import asyncio

    console.print()
    console.print("[anton.muted]  No worries, let's continue where we left off.[/]")
    with Live(
        Spinner("dots", text=Text("", style="anton.muted"), style="anton.cyan"),
        console=console,
        refresh_per_second=10,
        transient=True,
    ):
        await asyncio.sleep(1.5)
    console.print()

    if status == "test_failed":
        return (
            f"CONNECTION TEST FAILED: The connection test for '{engine}' did not "
            f"succeed and the user declined to re-enter credentials. Nothing was "
            f"saved.\n\n"
            f"You have exactly TWO mutually exclusive options — pick ONE, do NOT "
            f"mix them:\n\n"
            f"OPTION A — Retry silently (only if you suspect a transient issue "
            f"like a network glitch or first-connection cold start):\n"
            f"  Emit ZERO text in your response. Output ONLY a tool_use block "
            f"calling connect_new_datasource again with the same known_variables. "
            f"The user will only see the final result — clean and uncluttered.\n\n"
            f"OPTION B — Give up and troubleshoot (if you believe the failure is "
            f"real — bad credentials, wrong host, firewall, etc.):\n"
            f"  Respond with TEXT ONLY, NO tool calls. Briefly explain what "
            f"likely went wrong and ask the user what to do.\n\n"
            f"CRITICAL: Mixing text + a retry tool call in the same response "
            f"produces a confusing two-message stack for the user (failure text "
            f"followed by success text). Pick A or B, never both."
        )

    # Default: user cancelled (pressed Escape) at some point
    return (
        f"CANCELLED: The user cancelled the '{engine}' connection setup before "
        f"it completed. Ask the user what they'd like to do instead. "
        f"Do NOT immediately call connect_new_datasource again unless they "
        f"explicitly ask for it. Respond with TEXT ONLY — no tool calls."
    )


CONNECT_DATASOURCE_TOOL = ToolDef(
    name = "connect_new_datasource",
    description = (
        "Connect a new data source to Anton's Local Vault. Call this when the user "
        "asks a question that requires data from a source that isn't connected yet "
        "(e.g. email, database, CRM, API). This starts an interactive connection flow "
        "where the user enters their credentials.\n\n"
        "Pass the datasource type/name (e.g. 'gmail', 'postgres', 'salesforce', 'hubspot'). "
        "Anton will match it to the right connector and guide the user through setup.\n\n"
        "If the user has ALREADY mentioned credential values in the conversation "
        "(e.g. 'connect to dynamodb, my access key is AKIA... and region is us-east-1'), "
        "pass them as `known_variables` so the user is not asked again.\n\n"
        "Do NOT print any message before calling this tool — it handles the user-facing output."
    ),
    input_schema = {
        "type": "object",
        "properties": {
            "engine": {
                "type": "string",
                "description": "The datasource type or name (e.g. 'gmail', 'postgres', 'snowflake', 'hubspot')",
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation of why this datasource is needed",
            },
            "known_variables": {
                "type": "object",
                "description": (
                    "Pre-extracted credential field values from the conversation. "
                    "Use snake_case field names (e.g. {\"host\": \"db.example.com\", "
                    "\"port\": \"5432\", \"user\": \"admin\"}). Only pass fields the "
                    "user actually mentioned — never invent values."
                ),
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["engine"],
    },
    handler = handle_connect_datasource,
)


async def handle_publish_or_preview(session: ChatSession, tc_input: dict) -> str:
    """Interactive preview/publish flow after dashboard creation."""
    import os
    import webbrowser
    from pathlib import Path

    console = session._console

    raw_path = tc_input.get("file_path", "")
    title = tc_input.get("title", "Dashboard")
    action = tc_input.get("action", "ask")
    file_path = Path(raw_path)
    if not file_path.is_absolute() and session._workspace:
        file_path = Path(session._workspace.base) / raw_path

    if not file_path.exists():
        return f"File not found: {file_path}"

    # Direct preview — just open and return, no prompts
    if action in ("preview", "ask"):
        abs_path = os.path.abspath(str(file_path))
        webbrowser.open(f"file://{abs_path}")
        return f"Opened {title} in browser. The user can ask for changes or say /publish to publish it to the web."

    # Publish flow
    from anton.config.settings import AntonSettings
    from anton.publisher import publish

    settings = AntonSettings()

    if not settings.minds_api_key:
        console.print()
        console.print("  [anton.muted]To publish you need a free Minds account.[/]")
        console.print("  [anton.muted]Run [bold]/publish[/bold] to set up your API key and publish.[/]")
        console.print()
        return (
            "STOP: No Minds API key configured. Do NOT call this tool again. "
            "Tell the user to run the /publish command to set up their mdb.ai API key "
            "and publish their dashboard. The /publish command handles the interactive "
            "API key setup flow."
        )

    from rich.live import Live
    from rich.spinner import Spinner

    with Live(Spinner("dots", text="  Publishing...", style="anton.cyan"), console=console, transient=True):
        try:
            result = publish(
                file_path,
                api_key=settings.minds_api_key,
                publish_url=settings.publish_url,
                ssl_verify=settings.minds_ssl_verify,
            )
        except Exception as e:
            console.print(f"  [anton.error]Publish failed: {e}[/]")
            console.print()
            return f"PUBLISH FAILED: {e}"

    view_url = result.get("view_url", "")
    console.print(f"  [anton.success]Published![/]")
    console.print(f"  [link={view_url}]{view_url}[/link]")
    console.print()

    if view_url:
        webbrowser.open(view_url)

    return f"Published successfully!\nView URL: {view_url}"


PUBLISH_TOOL = ToolDef(
    name = "publish_or_preview",
    description = (
        "Call this after generating an HTML dashboard or report in .anton/output/. "
        "Actions: 'ask' (default) prompts the user to preview/publish/skip interactively. "
        "'preview' opens the file in the browser immediately. "
        "'publish' publishes to the web immediately. "
        "Use 'preview' or 'publish' when the user has already stated their intent. "
        "Use 'ask' after generating a new dashboard to let the user choose."
    ),
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the HTML file (e.g. .anton/output/dashboard.html)",
            },
            "title": {
                "type": "string",
                "description": "Short title describing the dashboard (e.g. 'BTC & Macro Dashboard')",
            },
            "action": {
                "type": "string",
                "enum": ["ask", "preview", "publish"],
                "description": "What to do: 'ask' prompts user, 'preview' opens locally, 'publish' publishes to web",
            },
        },
        "required": ["file_path"],
    },
    handler = handle_publish_or_preview,
    prompt = (
        "CONTENT SHARING POLICY:\n"
        "- Publishing dashboards or reports to the web is done ONLY via the `publish_or_preview` tool. \n"
        "- Do NOT upload, post, or share generated files (HTML, data, images) to external hosting \n"
        "- services (paste sites, gists, CDNs, file hosts) via scratchpad code — unless the user \n"
        "- explicitly names the service and confirms. Reading from public APIs and writing to the \n"
        "- user's connected datasources (databases, CRMs, etc.) is fine — this rule only applies to \n"
        "- sharing generated output with the public internet."
    ),
)


async def handle_generate_dashboard(session: "ChatSession", tc_input: dict) -> str:
    """Run a focused inner LLM loop to generate a dashboard function, then execute it."""
    import ast
    from anton.core.llm.prompts import DASHBOARD_BUILDER_SYSTEM_PROMPT
    from anton.core.utils.scratchpad import format_cell_result

    variables: dict = tc_input.get("variables") or {}
    output_path: str = tc_input.get("output_path", ".anton/output/dashboard.html")
    spec: str = tc_input.get("spec", "")
    scratchpad_name: str = tc_input.get("scratchpad_name", "main")
    title: str = tc_input.get("title", "Dashboard")

    if not variables:
        return "Error: 'variables' is required — provide a dict of variable_name → description."

    param_names = list(variables.keys())
    params_str = ", ".join(param_names)
    call_args = ", ".join([f'"{output_path}"'] + param_names)
    def _var_line(k: str, v) -> str:
        if isinstance(v, dict):
            py_type = v.get("type", "")
            desc = v.get("description", "")
            return f"- {k} ({py_type}): {desc}" if py_type else f"- {k}: {desc}"
        return f"- {k}: {v}"

    var_block = "\n".join(_var_line(k, v) for k, v in variables.items())

    # Gather scratchpad cell outputs so the inner LLM sees actual data
    # (column names, value ranges, sample rows) not just variable descriptions.
    pad = await session._scratchpads.get_or_create(scratchpad_name)
    cell_context_parts: list[str] = []
    for i, cell in enumerate(pad.cells):
        out = format_cell_result(cell)
        if out and out != "Code executed successfully (no output).":
            label = cell.description or f"cell {i + 1}"
            cell_context_parts.append(f"[{label}]\n{out}")
    scratchpad_context = (
        "\n\n".join(cell_context_parts)
        if cell_context_parts
        else "(no output from scratchpad cells)"
    )

    import textwrap
    user_message = textwrap.dedent(f"""
Your task is to produce ONE single self-contained HTML file with a dashboard that fully meets the requirements described below.

CRITICAL: The final dashboard MUST be a single .html file with ALL data, CSS, and JS inlined. \
Do NOT reference external local files (like data.js) — browsers block local file:// cross-references \
for security reasons and the dashboard will silently fail to load data.

SECURITY (critical): Dashboards may be published to the web. NEVER embed API keys, tokens, \
passwords, connection strings, or any credentials in the HTML, JS, or inline data. Fetch data \
in scratchpad cells using credentials from environment variables, then serialize only the \
resulting data into the dashboard. If the user explicitly asks to embed a credential \
(e.g. for a live-updating dashboard), warn them that publishing will expose it and get \
confirmation before proceeding.

WRITE A DASHBOARD BRIEF: Before coding the HTML, plan the dashboard out loud:
- What story does each chart tell? (not "a bar chart of X" but "this shows how Y \
is driving Z, annotated at the inflection point")
- What is the visual hierarchy? Hero KPIs at top, main narrative chart first, \
supporting charts below.
- What should be annotated? Key dates, threshold crossings, outliers.
- What color scheme ties it together? Consistent meaning (green=positive, red=negative) \
across all charts.

SELF-CONTAINED OUTPUT (critical):
Prefer inlining everything — CSS in `<style>`, JS in `<script>`, data as JS variables. \
A single .html file is the most portable and publishable format. \
If the dataset is very large (>100KB of JSON), you may write it to a separate .js file \
in the SAME directory and reference it with a \
relative `<script src="dashboard_data.js">` tag. The publisher will auto-bundle sibling \
files referenced in the HTML. Never reference files outside the output directory.

WHY: (1) Browsers block local file:// cross-references across directories. \
(2) Splitting the build across cells catches JS/CSS errors early — if a cell has a syntax issue \
in a string, you'll see it before the final assembly. (3) Large datasets in single cells timeout. \
(4) Self-contained files can be published to the web via /publish without missing assets.

PYTHON → JS STRING SAFETY (critical):
When building JS code inside Python strings, escape sequences get resolved by Python BEFORE \
writing to the file. This means '\\n' in Python becomes a literal newline in the output, which \
breaks JavaScript string literals. Rules:
- Use '\\\\n' in Python if you need a literal \\n in the JS output
- Use raw strings (r"...") for JS code blocks when possible
- NEVER use '\\n', '\\t', or '\\\"' inside JS strings within Python — double-escape them
- After writing the file, sanity-check that no string literals span multiple lines

Output format:
- Unless the user explicitly asks for a different format, always output visualizations \
as polished, single-file HTML pages — never raw PNGs or bare image files.

Visual design:
- Make it look good by default. Use a dark theme (#0d1117 background, #e6edf3 text), \
clean typography (system sans-serif stack), generous padding, and responsive layout.
- ALWAYS use Apache ECharts for interactive charts. Load it via CDN: \
`<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>`. \
No Python dependencies needed — just write the HTML with inline JS. Use ECharts' built-in \
dark theme: `echarts.init(dom, 'dark')`, then customize colors to match #0d1117 background.
- NEVER use Plotly, matplotlib, or other charting libraries unless the user explicitly asks.

Line smoothing (critical — smooth: true misrepresents volatile data):
- DEFAULT: `smooth: false` on ALL line series. Straight segments between data points are \
the honest representation — they show actual volatility, drawdowns, and inflection points.
- EXCEPTION: Use `smooth: true` ONLY for cumulative/monotonic series (cumulative returns, \
running totals, growth curves) where the trend matters more than point-to-point moves.
- Decision heuristic: Does the line ever reverse direction meaningfully? If yes → smooth: false. \
Is it a running sum, cumulative metric, or long-horizon trend? → smooth: true is acceptable.
- Line widths: 2.5 for hero/primary lines, 1.5 for multi-line comparisons, 1 for secondary/reference lines.

Chart readability (critical — labels must NEVER overlap):
- Use `axisLabel: {{ rotate: -45 }}` or `{{ rotate: 45 }}` on crowded axes. \
Set `grid: {{ containLabel: true }}` so labels never clip. Use `legend: {{ type: 'scroll', \
bottom: 0 }}` to place scrollable legends below the chart. For pie/donut charts use \
`label: {{ show: true, position: 'outside' }}` with `labelLayout: {{ hideOverlap: true }}`. \
For bar charts with many categories, use horizontal bars (`yAxis` as category) or \
abbreviate labels with `axisLabel: {{ formatter }}`. Always configure rich `tooltip` with \
`formatter` functions for precise value display on hover. Use `dataZoom` for time series \
so users can zoom into ranges.

Layout and composition:
- For non-chart visualizations (tables, reports, dashboards), write clean HTML/CSS directly. \
Use CSS grid or flexbox. Add subtle styling: rounded corners, soft shadows, hover effects.
- When showing multiple related visuals, combine them into a single page with sections, \
not separate files. Ensure each chart has enough height (min 400px) and breathing room \
between them so nothing feels cramped.
- Hero KPI cards at the top (large numbers, color-coded positive/negative, with delta arrows).
- Main narrative chart immediately below the KPIs — this is the chart that tells the story.
- Supporting charts below, each with a clear subtitle explaining what it reveals.
- Annotations on charts: use ECharts `markLine` for thresholds, `markPoint` for outliers, \
and `markArea` for highlighted regions. A chart without annotations is a missed opportunity.
- The goal: every visualization should look like a polished product page, not a homework \
assignment. Think dark-mode dashboard, not Jupyter default.\


Variables available in the scratchpad:
{var_block}

Scratchpad output (actual printed output from data-fetching cells — use this to understand \
real column names, value ranges, and data shape):
{scratchpad_context}

Dashboard spec:
{spec}

Output path: {output_path}
Title: {title}

OUTPUT FORMAT — STRICT (critical):
Your entire response MUST be valid Python code and NOTHING ELSE.
No markdown fences, no prose, no comments outside the function.

You MUST follow this exact structure:
```python
def _build_dashboard(output_path: str, {params_str}):
    # All HTML generation logic lives INSIDE this function.
    # Only the parameters above are available — do NOT reference
    # any variables from outside this function's scope.
    ...

_build_dashboard({call_args})
```

WHY this structure matters:
- The function runs inside an existing scratchpad namespace that contains
  many other variables. Wrapping all logic in a function creates an isolated
  local scope — any variables you define stay inside the function and do not
  pollute or collide with the outer namespace.
- All data you need is already passed as parameters. Do not read globals,
  do not import scratchpad state — use ONLY what is in the function signature.
    """).strip()

    messages: list[dict] = [{"role": "user", "content": user_message}]
    last_error = ""
    code = ""

    for attempt in range(2):
        if attempt > 0:
            messages.append({
                "role": "user",
                "content": f"Syntax error: {last_error}. Fix and return the complete corrected code.",
            })

        response = await session._llm.plan(
            system=DASHBOARD_BUILDER_SYSTEM_PROMPT,
            messages=messages,
            max_tokens=8192,
        )

        # Strip markdown fences the LLM may add despite instructions
        raw = response.content.strip()
        for fence in ("```python\n", "```py\n", "```\n"):
            if raw.startswith(fence):
                raw = raw[len(fence):]
                break
        if raw.endswith("```"):
            raw = raw[:-3].rstrip()
        code = raw

        try:
            ast.parse(code)
            break
        except SyntaxError as e:
            last_error = str(e)
            if attempt == 0:
                messages.append({"role": "assistant", "content": response.content})
    else:
        return (
            f"Failed to generate valid Python code after 2 attempts. "
            f"Last syntax error: {last_error}"
        )

    cell = await pad.execute(
        code,
        description=f"Generate {title} HTML dashboard",
        estimated_time="15s",
        estimated_seconds=15,
    )

    if cell.error:
        return f"Dashboard generation failed:\n{format_cell_result(cell)}"

    return f"Dashboard written to {output_path}."

# # TODO move to main viz prompt
# _GENERATE_DASHBOARD_WORKFLOW = """\
# VISUALIZATIONS — dashboard workflow:
# 1. FETCH DATA: Use a scratchpad cell to pull data and compute key metrics. Print variable \
# names, structure, and key values — you'll use these as descriptions in step 3.
# 2. STREAM INSIGHTS before building anything: narrate findings to the user immediately.
#   - DATA HIGHLIGHTS: Compact markdown table of key numbers at a glance.
#   - HEADLINE: One sentence — the single most important finding.
#   - CONTEXT: Compare against a benchmark or expectation. Raw numbers without comparison \
# are meaningless.
#   - THE NON-OBVIOUS: What would an expert notice? Patterns the table doesn't show.
#   - ASSUMPTIONS: Data source, time range, real-time vs delayed.
#   - ACTIONABLE EDGE: Risks, thresholds, scenarios worth considering.
# 3. CALL generate_dashboard with:
#    - scratchpad_name: the scratchpad where variables live
#    - variables: {name: {type, description}} for each variable (based on step 1 printed output)
#    - output_path: ".anton/output/<descriptive_name>.html"
#    - spec: full description of charts, layout, KPI cards, user preferences, title
# 4. CALL publish_or_preview with the output path.

# Do NOT build HTML dashboards manually in scratchpad cells. \
# The generate_dashboard tool handles all HTML/CSS/JS/ECharts rendering.\

# PROACTIVE FOLLOW-UP SUGGESTIONS:
# After completing analysis, if the user's own data could complement the findings, \
# offer it in ONE sentence at the end. Examples:
# - After stock/market analysis → "I can also analyze your portfolio against these benchmarks."
# - After economic analysis → "I can pull in your company's data to compare."
# Keep it brief, not pushy. Don't repeat the offer if ignored.\
# """

GENERATE_DASHBOARD_TOOL = ToolDef(
    name="generate_dashboard",
    description=(
        "Generate a self-contained HTML dashboard from data already computed in a scratchpad. "
        "Use this instead of writing dashboard HTML manually in scratchpad cells."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "scratchpad_name": {
                "type": "string",
                "description": "Name of the scratchpad where the data variables live.",
            },
            "variables": {
                "type": "object",
                "description": "Map of variable_name → {type, description} for each variable to pass into the dashboard.",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": "Python type of the variable (e.g. 'pandas.DataFrame', 'list', 'float', 'dict')",
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Full description of the variable: what data it holds, its structure "
                                "(e.g. list of tuples, DataFrame with columns X/Y/Z, dict with keys A/B), "
                                "value ranges and units, and any relationships to other variables "
                                "(e.g. 'shares the same timestamps as prices'). "
                                "The more detail here, the better the generated dashboard."
                            ),
                        },
                    },
                    "required": ["type", "description"],
                },
            },
            "output_path": {
                "type": "string",
                "description": "Where to write the HTML file (e.g. .anton/output/my_dashboard.html).",
            },
            "spec": {
                "type": "string",
                "description": (
                    "Full description of the dashboard: which charts (line, bar, pie, etc.), "
                    "layout, KPI cards to show, title, and time range label. "
                    "Include color preferences, chart library, or style details ONLY if the user explicitly stated them."
                ),
            },
            "title": {
                "type": "string",
                "description": "Dashboard title (e.g. 'TON/USD — Last 1 Hour').",
            },
        },
        "required": ["scratchpad_name", "variables", "output_path", "spec"],
    },
    handler=handle_generate_dashboard,
    # prompt=_GENERATE_DASHBOARD_WORKFLOW,
)
