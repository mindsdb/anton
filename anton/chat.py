from __future__ import annotations

import asyncio
import os
import urllib.error
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import anthropic

from anton.clipboard import (
    cleanup_old_uploads,
    grab_clipboard,
    is_clipboard_supported,
    parse_dropped_paths as _parse_dropped_paths,
    save_clipboard_image,
)
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.llm.prompt_builder import SystemPromptContext
from anton.core.llm.provider import (
    TokenLimitExceeded,
    StreamComplete,
    StreamContextCompacted,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolResult,
    StreamToolUseDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
)
from anton.checks import TokenLimitInfo, TokenLimitStatus, check_minds_token_limits
from anton.commands.setup import (
    handle_memory,
    handle_setup,
    handle_setup_models,
)
from anton.commands.ui import handle_explain, handle_theme, print_slash_help
from anton.utils.clipboard import (
    ensure_clipboard,
    format_clipboard_image_message,
    format_file_message,
    human_size,
)
from anton.chat_session import build_runtime_context, rebuild_session
from anton.commands.session import handle_resume
from anton.commands.datasource import (
    handle_list_data_sources,
    handle_remove_data_source,
    handle_connect_datasource,
    handle_test_datasource,
)
from anton.commands.skills import (
    handle_skill_remove,
    handle_skill_save,
    handle_skill_show,
    handle_skills_list,
)
from anton.tools import CONNECT_DATASOURCE_TOOL, PUBLISH_TOOL
from anton.utils.prompt import (
    prompt_or_cancel,
    prompt_minds_api_key,
)

from anton.minds_client import (
    normalize_minds_url,
    describe_minds_connection_error,
    list_minds,
    list_datasources,
    test_llm,
)
from anton.core.datasources.data_vault import LocalDataVault
from anton.utils.datasources import (
    register_secret_vars,
)
from anton.core.datasources.datasource_registry import (
    DatasourceRegistry,
)

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style as PTStyle
from rich.prompt import Prompt

if TYPE_CHECKING:
    from rich.console import Console

    from anton.config.settings import AntonSettings
    from anton.core.memory.episodes import EpisodicMemory
    from anton.workspace import Workspace


async def _handle_connect(
    console: Console,
    settings: AntonSettings,
    workspace: Workspace,
    state: dict,
    self_awareness,
    cortex,
    session: ChatSession,
    episodic: EpisodicMemory | None = None,
) -> ChatSession:
    """Connect to a Minds server: select a Mind, then optionally a datasource."""
    from anton.workspace import Workspace as _Workspace

    global_ws = _Workspace(Path.home())

    console.print()

    # --- Prompt for URL and API key (use saved values as defaults) ---
    saved_url = normalize_minds_url(settings.minds_url)
    minds_url = await prompt_or_cancel("(anton) Minds server URL", default=saved_url)
    if minds_url is None:
        return session
    minds_url = normalize_minds_url(minds_url)

    saved_key = settings.minds_api_key or ""
    api_key = await prompt_minds_api_key(
        console,
        current_key=saved_key,
        allow_empty_keep=True,
    )
    if not api_key:
        console.print("[anton.error]API key is required.[/]")
        console.print()
        return session

    ssl_verify = settings.minds_ssl_verify

    # --- Try to connect ---
    minds = None
    while minds is None:
        console.print()
        console.print(f"[anton.muted]Connecting to {minds_url}...[/]")
        try:
            minds = list_minds(minds_url, api_key, verify=ssl_verify)
            break
        except (urllib.error.URLError, urllib.error.HTTPError) as err:
            headline, advice = describe_minds_connection_error(err)
            console.print(f"[anton.error]{headline}[/]")
            console.print(f"[anton.muted]{advice}[/]")
        except Exception as err:
            headline, advice = describe_minds_connection_error(err)
            console.print(f"[anton.error]{headline}[/]")
            console.print(f"[anton.muted]{advice}[/]")

        console.print()
        console.print("  Recovery options:")
        console.print("    [bold]1[/]  Reconfigure API key")
        console.print("    [bold]2[/]  Retry without SSL verification")
        console.print("    [bold]q[/]  Back")
        console.print()

        action = await prompt_or_cancel("(anton) Select", choices=["1", "2", "q"], default="q")
        if action is None or action == "q":
            console.print("[anton.muted]Aborted.[/]")
            console.print()
            return session
        if action == "1":
            new_key = await prompt_minds_api_key(
                console,
                current_key=api_key,
                allow_empty_keep=False,
            )
            if new_key is None:
                console.print("[anton.muted]API key unchanged.[/]")
                continue
            api_key = new_key
            ssl_verify = settings.minds_ssl_verify
            continue

        ssl_verify = False

    if not minds:
        console.print("[anton.warning]No minds found on this server.[/]")
        console.print()
        return session

    # --- Select a Mind ---
    console.print()
    console.print("[anton.cyan]Available minds:[/]")
    for i, mind in enumerate(minds, 1):
        name = mind.get("name", "?")
        ds_list = mind.get("datasources", [])
        ds_count = len(ds_list)
        ds_label = (
            f"{ds_count} datasource{'s' if ds_count != 1 else ''}"
            if ds_count
            else "no datasources"
        )
        console.print(f"    [bold]{i}[/]  {name} [dim]({ds_label})[/]")
    console.print()

    choices = [str(i) for i in range(1, len(minds) + 1)]
    pick = await prompt_or_cancel("(anton) Select mind", choices=choices)
    if pick is None:
        return session
    selected_mind = minds[int(pick) - 1]
    mind_name = selected_mind.get("name", "")

    # --- Datasource selection within the mind ---
    mind_datasources = selected_mind.get("datasources", [])
    ds_name = ""
    ds_engine = ""

    if len(mind_datasources) > 1:
        console.print()
        console.print(f"[anton.cyan]Datasources in mind '{mind_name}':[/]")
        for i, ds_ref in enumerate(mind_datasources, 1):
            # datasource refs may be strings or dicts
            ref_name = ds_ref if isinstance(ds_ref, str) else ds_ref.get("name", "?")
            console.print(f"    [bold]{i}[/]  {ref_name}")
        console.print()
        ds_choices = [str(i) for i in range(1, len(mind_datasources) + 1)]
        ds_pick = await prompt_or_cancel("(anton) Select datasource", choices=ds_choices)
        if ds_pick is None:
            return session
        picked_ds = mind_datasources[int(ds_pick) - 1]
        ds_name = picked_ds if isinstance(picked_ds, str) else picked_ds.get("name", "")
    elif len(mind_datasources) == 1:
        picked_ds = mind_datasources[0]
        ds_name = picked_ds if isinstance(picked_ds, str) else picked_ds.get("name", "")
        console.print(f"[anton.muted]Auto-selected datasource: {ds_name}[/]")

    if ds_name:
        try:
            all_datasources = list_datasources(
                minds_url, api_key, verify=ssl_verify
            )
            for ds in all_datasources:
                if ds.get("name") == ds_name:
                    ds_engine = ds.get("engine", "unknown")
                    break
        except Exception:
            ds_engine = "unknown"

    # --- Persist to global ~/.anton/.env ---
    global_ws.set_secret("ANTON_MINDS_API_KEY", api_key)
    global_ws.set_secret("ANTON_MINDS_URL", minds_url)
    global_ws.set_secret("ANTON_MINDS_MIND_NAME", mind_name)
    global_ws.set_secret("ANTON_MINDS_DATASOURCE", ds_name)
    global_ws.set_secret("ANTON_MINDS_DATASOURCE_ENGINE", ds_engine)
    global_ws.set_secret("ANTON_MINDS_SSL_VERIFY", "true" if ssl_verify else "false")

    settings.minds_api_key = api_key
    settings.minds_url = minds_url
    settings.minds_mind_name = mind_name
    settings.minds_datasource = ds_name
    settings.minds_datasource_engine = ds_engine
    settings.minds_ssl_verify = ssl_verify

    console.print()
    status = f"[anton.success]Selected mind: {mind_name}[/]"
    if ds_name:
        status += f" [anton.success]| datasource: {ds_name} ({ds_engine})[/]"
    console.print(status)

    # --- Test if the Minds server also supports LLM endpoints ---
    # (silenced: was printing "Testing LLM endpoints..." and "not available" messages)
    llm_ok = test_llm(minds_url, api_key, verify=ssl_verify)

    if llm_ok:
        console.print(
            "[anton.success]LLM endpoints available — using Minds server as LLM provider.[/]"
        )
        settings.planning_provider = "openai-compatible"
        settings.coding_provider = "openai-compatible"
        settings.planning_model = "_reason_"
        settings.coding_model = "_code_"
        # openai_api_key and openai_base_url are derived at runtime from
        # minds_api_key and minds_url via model_post_init — no need to persist them.
        settings.model_post_init(None)
        global_ws.set_secret("ANTON_PLANNING_PROVIDER", "openai-compatible")
        global_ws.set_secret("ANTON_CODING_PROVIDER", "openai-compatible")
        global_ws.set_secret("ANTON_PLANNING_MODEL", "_reason_")
        global_ws.set_secret("ANTON_CODING_MODEL", "_code_")
    else:
        # Check if Anthropic key is already configured
        has_anthropic = settings.anthropic_api_key or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        if not has_anthropic:
            anthropic_key = Prompt.ask("Anthropic API key (for LLM)", console=console)
            if anthropic_key.strip():
                anthropic_key = anthropic_key.strip()
                settings.anthropic_api_key = anthropic_key
                settings.planning_provider = "anthropic"
                settings.coding_provider = "anthropic"
                settings.planning_model = "claude-sonnet-4-6"
                settings.coding_model = "claude-haiku-4-5-20251001"
                global_ws.set_secret("ANTON_ANTHROPIC_API_KEY", anthropic_key)
                global_ws.set_secret("ANTON_PLANNING_PROVIDER", "anthropic")
                global_ws.set_secret("ANTON_CODING_PROVIDER", "anthropic")
                global_ws.set_secret("ANTON_PLANNING_MODEL", "claude-sonnet-4-6")
                global_ws.set_secret("ANTON_CODING_MODEL", "claude-haiku-4-5-20251001")
                console.print("[anton.success]Anthropic API key saved.[/]")
            else:
                console.print(
                    "[anton.warning]No API key provided — LLM calls will not work.[/]"
                )

    global_ws.apply_env_to_process()
    console.print()

    return rebuild_session(
        settings=settings,
        state=state,
        self_awareness=self_awareness,
        cortex=cortex,
        workspace=workspace,
        console=console,
        episodic=episodic,
    )




def _extract_html_title(path, re_module) -> str:
    """Extract <title> content from an HTML file. Returns '' if not found."""
    try:
        # Read only the first 4KB — title is always near the top
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(4096)
        m = re_module.search(r"<title[^>]*>(.*?)</title>", head, re_module.IGNORECASE | re_module.DOTALL)
        return m.group(1).strip() if m else ""
    except Exception:
        return ""


async def _handle_publish(
    console: Console,
    settings,
    workspace,
    file_arg: str = "",
) -> None:
    """Handle /publish command — publish an HTML report to the web."""
    import json
    import webbrowser
    from pathlib import Path

    from anton.publisher import publish

    console.print()

    # 1. Ensure Minds API key is available
    if not settings.minds_api_key:
        console.print("  [anton.muted]To publish dashboards you need a free Minds account.[/]")
        console.print()
        has_key = await prompt_or_cancel(
            "  Do you have an mdb.ai API key?",
            choices=["y", "n"],
            choices_display="y/n",
            default="y",
        )
        if has_key is None:
            console.print()
            return
        if has_key.lower() == "n":
            webbrowser.open(
                "https://mdb.ai/auth/realms/mindsdb/protocol/openid-connect/registrations"
                "?client_id=public-client&response_type=code&scope=openid"
                "&redirect_uri=https%3A%2F%2Fmdb.ai"
            )
            console.print()

        api_key = await prompt_or_cancel("  API key", password=True)
        if api_key is None or not api_key.strip():
            console.print()
            return
        api_key = api_key.strip()
        settings.minds_api_key = api_key
        if workspace:
            workspace.set_secret("ANTON_MINDS_API_KEY", api_key)
        console.print()

    # 2. Find the HTML file to publish
    import re

    output_dir = Path(settings.workspace_path) / ".anton" / "output"

    if file_arg:
        target = Path(file_arg)
        if not target.is_absolute():
            target = Path(settings.workspace_path) / file_arg
    else:
        # List HTML files sorted by modification time (most recent first)
        html_files = sorted(
            output_dir.glob("*.html"), key=lambda f: f.stat().st_mtime, reverse=True
        ) if output_dir.is_dir() else []
        if not html_files:
            console.print("  [anton.warning]No HTML files found in .anton/output/[/]")
            console.print()
            return

        PAGE_SIZE = 10
        offset = 0

        while True:
            page = html_files[offset:offset + PAGE_SIZE]
            has_more = offset + PAGE_SIZE < len(html_files)

            console.print("  [anton.cyan]Available reports:[/]")
            console.print()
            for i, f in enumerate(page, offset + 1):
                title = _extract_html_title(f, re)
                label = title or f.name
                console.print(f"  [bold]{i}[/]  {label}  [anton.muted]{f.name}[/]")

            if has_more:
                console.print(f"\n  [anton.muted]m  Show more ({len(html_files) - offset - PAGE_SIZE} remaining)[/]")

            console.print()
            choice = await prompt_or_cancel("  Select", default="1")
            if choice is None:
                console.print()
                return

            if choice.strip().lower() == "m" and has_more:
                offset += PAGE_SIZE
                console.print()
                continue

            try:
                idx = int(choice) - 1
                if idx < 0 or idx >= len(html_files):
                    raise ValueError
                target = html_files[idx]
                break
            except (ValueError, IndexError):
                console.print("  [anton.warning]Invalid choice.[/]")
                console.print()
                return

    if not target.exists():
        console.print(f"  [anton.warning]File not found: {target}[/]")
        console.print()
        return

    # 3. Check if this file was previously published
    published_json = output_dir / ".published.json"
    published_map = {}
    try:
        if published_json.is_file():
            published_map = json.loads(published_json.read_text())
    except Exception:
        pass

    report_id = None
    file_key = target.name
    prev = published_map.get(file_key)

    if prev and prev.get("report_id"):
        console.print(f"  [anton.muted]Previously published: {prev.get('url', '')}[/]")
        update_choice = await prompt_or_cancel(
            "  Update existing report, or publish as new?",
            choices=["update", "new", "u", "n"],
            choices_display="update/new",
            default="update",
        )
        if update_choice is None:
            console.print()
            return
        if update_choice in ("update", "u"):
            report_id = prev["report_id"]

    # 4. Publish
    from rich.live import Live
    from rich.spinner import Spinner

    action_text = "  Updating..." if report_id else "  Publishing..."
    with Live(Spinner("dots", text=action_text, style="anton.cyan"), console=console, transient=True):
        try:
            result = publish(
                target,
                api_key=settings.minds_api_key,
                report_id=report_id,
                publish_url=settings.publish_url,
                ssl_verify=settings.minds_ssl_verify,
            )
        except Exception as e:
            console.print(f"  [anton.error]Publish failed: {e}[/]")
            console.print()
            return

    view_url = result.get("view_url", "")
    returned_report_id = result.get("report_id", "")
    version = result.get("version", 1)
    unchanged = result.get("unchanged", False)

    if unchanged:
        console.print(f"  [anton.muted]Already up to date (v{version})[/]")
    elif report_id:
        console.print(f"  [anton.success]Updated! (v{version})[/]")
    else:
        console.print(f"  [anton.success]Published![/]")
    console.print(f"  [link={view_url}]{view_url}[/link]")
    console.print()

    # 5. Save mapping
    if returned_report_id:
        published_map[file_key] = {
            "report_id": returned_report_id,
            "url": view_url,
            "last_md5": result.get("md5", ""),
        }
        try:
            published_json.write_text(json.dumps(published_map, indent=2))
        except Exception:
            pass

    if view_url:
        webbrowser.open(view_url)




async def _handle_unpublish(
    console: Console,
    settings,
    workspace,
) -> None:
    """Handle /unpublish command — list published reports and delete one."""
    from anton.publisher import list_published, unpublish

    console.print()

    # 1. Ensure Minds API key is available
    if not settings.minds_api_key:
        console.print("  [anton.warning]No Minds API key configured. Run /publish first.[/]")
        console.print()
        return

    # 2. Fetch published reports
    from rich.live import Live
    from rich.spinner import Spinner

    reports = []
    with Live(Spinner("dots", text="  Loading published reports...", style="anton.cyan"), console=console, transient=True):
        try:
            reports = list_published(
                api_key=settings.minds_api_key,
                publish_url=settings.publish_url,
                ssl_verify=settings.minds_ssl_verify,
            )
        except Exception as e:
            console.print(f"  [anton.error]Failed to list reports: {e}[/]")
            console.print()
            return

    if not reports:
        console.print("  [anton.muted]No published reports found.[/]")
        console.print()
        return

    # 3. Display paginated list
    PAGE_SIZE = 10
    offset = 0

    while True:
        page = reports[offset:offset + PAGE_SIZE]
        has_more = offset + PAGE_SIZE < len(reports)

        console.print("  [anton.cyan]Published reports:[/]")
        console.print()
        for i, r in enumerate(page, offset + 1):
            title = r.get("title", "Untitled")
            url = r.get("view_url", "")
            console.print(f"  [bold]{i}[/]  {title}  [anton.muted]{url}[/]")

        if has_more:
            console.print(f"\n  [anton.muted]m  Show more ({len(reports) - offset - PAGE_SIZE} remaining)[/]")

        console.print()
        choice = await prompt_or_cancel("  Select report to unpublish")
        if choice is None:
            console.print()
            return

        if choice.strip().lower() == "m" and has_more:
            offset += PAGE_SIZE
            console.print()
            continue

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(reports):
                raise ValueError
            selected = reports[idx]
            break
        except (ValueError, IndexError):
            console.print("  [anton.warning]Invalid choice.[/]")
            console.print()
            return

    # 4. Confirm
    title = selected.get("title", "Untitled")
    console.print(f"  [anton.warning]This will remove:[/] {title}")
    confirm = await prompt_or_cancel(
        "  Are you sure?",
        choices=["y", "n"],
        choices_display="y/n",
        default="n",
    )
    if confirm is None or confirm != "y":
        console.print()
        return

    # 5. Delete
    with Live(Spinner("dots", text="  Removing...", style="anton.cyan"), console=console, transient=True):
        try:
            unpublish(
                selected.get("report_id") or selected["md5"],
                api_key=settings.minds_api_key,
                publish_url=settings.publish_url,
                ssl_verify=settings.minds_ssl_verify,
            )
        except Exception as e:
            console.print(f"  [anton.error]Failed to remove: {e}[/]")
            console.print()
            return

    console.print(f"  [anton.success]Removed:[/] {title}")
    console.print()


async def _agent_zero(console: Console, session: "ChatSession", settings) -> str | None:
    """First-run staged demo. Runs the backup script in a real scratchpad cell.

    Returns "_AGENT_ZERO_DONE" if demo ran, None if skipped/failed.
    """
    import os as _os
    import time as _time

    script_path = Path(__file__).resolve().parent / "demo_data" / "nvda_btc_scratchpad_backup.py"
    if not script_path.is_file():
        return None

    # Clear screen
    _os.system("cls" if sys.platform == "win32" else "clear")

    console.print()
    _line1 = "All set! To test things out, I\u2019ll pull NVIDIA vs Bitcoin data from"
    _line2 = "the web and build you a 5-year investment comparison dashboard."
    console.print("[anton.prompt]anton>[/] ", end="")
    for ch in _line1:
        console.file.write(ch)
        console.file.flush()
        _time.sleep(0.02)
    console.print()
    console.print("       ", end="")
    for ch in _line2:
        console.file.write(ch)
        console.file.flush()
        _time.sleep(0.02)
    console.print()
    console.print()
    console.print()

    answer = await prompt_or_cancel(
        "(anton) Run analysis, or skip straight to chatting?",
        choices_display="run/skip",
        default="run",
        allow_cancel=True,
    )
    if answer is None:
        return None

    answer_text = (answer or "").strip().lower()

    # Classify: does the user want to run it?
    _skip_words = {"no", "n", "skip", "nah", "pass", "nope", "later", "chat", "straight"}
    _go_words = {"yes", "y", "ok", "sure", "go", "yeah", "yep", "run", "do it", "let's go", "lets go", "go for it"}

    wants_demo = None
    for w in _go_words:
        if w in answer_text:
            wants_demo = True
            break
    if wants_demo is None:
        for w in _skip_words:
            if w in answer_text:
                wants_demo = False
                break
    if wants_demo is None:
        # Default to yes if ambiguous
        wants_demo = True if not answer_text else True

    if not wants_demo:
        console.print()
        console.print("  [anton.muted]All good! Ask me anything \u2014 data questions, dashboards, analysis, you name it.[/]")
        console.print()
        return None

    # Typed message with ellipsis animation
    console.print()
    from anton.channel.theme import get_palette as _gp3
    _c = _gp3().cyan
    _r, _g, _b = int(_c[1:3], 16), int(_c[3:5], 16), int(_c[5:7], 16)
    _ac = f"\033[1;38;2;{_r};{_g};{_b}m"
    _ar = "\033[0m"

    _prefix = f"{_ac}anton>{_ar} "
    _typed_msg = "Perfect! Fetching live data, crunching numbers, and building the dashboard"
    console.file.write(_prefix)
    console.file.flush()
    for ch in _typed_msg:
        console.file.write(ch)
        console.file.flush()
        _time.sleep(0.02)

    # Ellipsis + spinner for ~10 seconds
    console.file.write("...\n")
    console.file.flush()

    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text

    with Live(
        Spinner("dots", text=Text("", style="anton.muted"), style="anton.cyan"),
        console=console,
        refresh_per_second=10,
        transient=True,
    ):
        await asyncio.sleep(10)
    console.print()

    # Read the script and patch for scratchpad execution.
    # 1. __file__ doesn't exist inside exec() — set it so os.path.dirname works
    # 2. Override OUTPUT_PATH to write to .anton/output/ instead of demo_data/
    code = script_path.read_text()
    output_dir = str(Path(settings.workspace_path) / ".anton" / "output")
    output_html = str(Path(output_dir) / "nvda_btc_dashboard.html")
    code = (
        f"import os as _os; _os.makedirs({output_dir!r}, exist_ok=True)\n"
        f"__file__ = {str(script_path)!r}\n"
        + code
    )
    # Replace the OUTPUT_PATH line so the dashboard goes to .anton/output/
    code = code.replace(
        'OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvda_btc_dashboard.html")',
        f'OUTPUT_PATH = {output_html!r}',
    )

    from anton.core.backends.base import Cell
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text

    pad = await session._scratchpads.get_or_create("main")

    # Pre-install dependencies so the main script doesn't fail mid-run
    install_spinner = Text("  Installing dependencies (yfinance, pandas, numpy)...", style="anton.muted")
    with Live(
        Spinner("dots", text=install_spinner, style="anton.cyan"),
        console=console,
        refresh_per_second=10,
        transient=True,
    ):
        await pad.install_packages(["yfinance", "pandas", "numpy"])
    console.print(f"  [anton.success]\u2714[/] [anton.muted]Dependencies ready[/]")

    spinner_text = Text("  Scratchpad(Building NVDA vs BTC dashboard...)", style="anton.muted")
    cell = None
    with Live(
        Spinner("dots", text=spinner_text, style="anton.cyan"),
        console=console,
        refresh_per_second=10,
        transient=True,
    ):
        async for item in pad.execute_streaming(
            code,
            description="Build NVDA vs BTC investment dashboard",
            estimated_time="~2 min",
            estimated_seconds=120,
        ):
            if isinstance(item, str):
                # Progress message from the script — update spinner
                spinner_text = Text(f"  Scratchpad({item})", style="anton.muted")
            elif isinstance(item, Cell):
                cell = item

    if cell is None or cell.error:
        err = cell.error if cell else "No result"
        console.print()
        err_line = err.strip().split("\n")[-1] if err else err
        console.print(f"[anton.error]  Demo encountered an issue: {err_line}[/]")
        console.print("[anton.muted]  You can still use Anton normally.[/]")
        console.print()
        return None

    console.print(f"  [anton.success]\u2714[/] [anton.muted]Dashboard built successfully[/]")

    # Inject context into session history so the LLM knows data is live
    _demo_stdout = (cell.stdout or "")[:3000]
    session._history.append({
        "role": "assistant",
        "content": (
            "I built an interactive NVIDIA vs Bitcoin 5-year investment dashboard. "
            "The dashboard HTML is at: " + output_html + "\n\n"
            "The scratchpad 'main' is still running with all data loaded in memory:\n"
            "- prices DataFrame (monthly OHLCV, returns, cumulative, drawdowns)\n"
            "- risk DataFrame (annual stats, Sharpe, Sortino, Calmar, win rate)\n"
            "- annual DataFrame (year-by-year breakdown)\n"
            "- mc DataFrame (1,000-path Monte Carlo, 60 months)\n"
            "- scorecard DataFrame (12-metric head-to-head comparison)\n\n"
            "All variables are live in the 'main' scratchpad — the user can ask "
            "follow-up questions and I can use the existing data without re-fetching.\n\n"
            f"Script output:\n{_demo_stdout}"
        ),
    })

    # Show findings — typed out like the intro message
    console.print()
    _lines = [
        "Everything worked! I pulled 5 years of data from Yahoo Finance,",
        "ran the numbers on NVIDIA vs Bitcoin, and built you a full",
        "interactive dashboard \u2014 it\u2019s open in your browser.",
        "",
        "6 tabs to explore: Performance \u00b7 Risk \u00b7 Monte Carlo \u00b7 Annual \u00b7",
        "Scorecard \u00b7 Decision.",
        "",
        "My take? If I had money to put down, NVIDIA wins this one.",
        "",
    ]
    from anton.channel.theme import get_palette as _gp2
    _cyan = _gp2().cyan
    # Convert hex color to ANSI 24-bit escape
    _r, _g, _b = int(_cyan[1:3], 16), int(_cyan[3:5], 16), int(_cyan[5:7], 16)
    _ansi_cyan = f"\033[1;38;2;{_r};{_g};{_b}m"
    _ansi_reset = "\033[0m"

    for li, line in enumerate(_lines):
        console.file.write("  ")
        for ch in line:
            console.file.write(ch)
            console.file.flush()
            _time.sleep(0.015)
        console.file.write("\n")
        console.file.flush()
    console.print()
    console.print("[anton.muted] Ask me follow-ups, a completely different question, or connect your own data (using the /connect command).[/]")
    console.print("[anton.muted] What\u2019s next, boss?[/]")
    console.print()

    return "_AGENT_ZERO_DONE"


def _persist_first_run_done(settings) -> None:
    """Write ANTON_FIRST_RUN_DONE=true to ~/.anton/.env."""
    from pathlib import Path

    env_path = Path.home() / ".anton" / ".env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    existing = env_path.read_text() if env_path.is_file() else ""
    if "ANTON_FIRST_RUN_DONE" not in existing:
        with env_path.open("a") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write("ANTON_FIRST_RUN_DONE=true\n")
    settings.first_run_done = True


_GREETING_EXAMPLES = [
    (
        "Go through my inbox, find every subscription I never read,\n"
        "       and build me a dashboard with unsubscribe links right there."
    ),
    (
        "Classify my last 200 emails \u2014 what actually needs my\n"
        "       attention vs what\u2019s noise? Show me a breakdown."
    ),
    (
        "Show me all my meetings next month \u2014 who\u2019s taking most\n"
        "       of my time? Build me a dashboard."
    ),
    (
        "Find all recurring meetings I haven\u2019t attended in 3+ months \u2014\n"
        "       should I drop them? Give me a report."
    ),
    (
        "Compare AAPL, NVDA, and TSLA over the last year \u2014\n"
        "       full interactive investment dashboard."
    ),
    (
        "What\u2019s the latest tech news today? Pull the headlines\n"
        "       and summarize what actually matters."
    ),
    (
        "I have a spreadsheet with sales data \u2014 analyze it and\n"
        "       build me an interactive dashboard with the key insights."
    ),
    (
        "Help me plan a trip to Tokyo \u2014 flights, hotels, budget,\n"
        "       all in one dashboard."
    ),
]


def _desktop_greeting(console: Console, settings) -> None:
    """First-time greeting for desktop app users. Types out a welcome + example."""
    import random
    import time as _time

    from anton.channel.theme import get_palette as _gp

    _c = _gp().cyan
    _r, _g, _b = int(_c[1:3], 16), int(_c[3:5], 16), int(_c[5:7], 16)
    _ac = f"\033[1;38;2;{_r};{_g};{_b}m"
    _ar = "\033[0m"

    example = random.choice(_GREETING_EXAMPLES)  # noqa: S311

    console.print()

    # Line 1: "Hi Boss! I'm Anton — here to help with anything."
    _line1 = "Hi Boss! I\u2019m Anton \u2014 here to help with anything."
    console.file.write(f"{_ac}anton>{_ar} ")
    for ch in _line1:
        console.file.write(ch)
        console.file.flush()
        _time.sleep(0.02)
    console.file.write("\n")
    console.file.flush()

    _time.sleep(0.3)

    # Line 2: blank
    console.file.write("\n")

    # Line 3: "For example, try something like:"
    _line2 = "For example, try something like:"
    console.file.write("       ")
    for ch in _line2:
        console.file.write(ch)
        console.file.flush()
        _time.sleep(0.02)
    console.file.write("\n")
    console.file.flush()

    _time.sleep(0.2)

    # Line 4: blank
    console.file.write("\n")

    # Line 5+: the example (quoted, italic feel)
    console.file.write("       \u201c")
    for ch in example:
        console.file.write(ch)
        console.file.flush()
        _time.sleep(0.015)
    console.file.write("\u201d\n")
    console.file.flush()

    console.print()

    _persist_first_run_done(settings)


def run_chat(
    console: Console, settings: AntonSettings, *, resume: bool = False, first_run: bool = False, desktop_first_run: bool = False
) -> None:
    """Launch the interactive chat REPL."""
    asyncio.run(_chat_loop(console, settings, resume=resume, first_run=first_run, desktop_first_run=desktop_first_run))


async def _chat_loop(
    console: Console, settings: AntonSettings, *, resume: bool = False, first_run: bool = False, desktop_first_run: bool = False
) -> None:
    from anton.context.self_awareness import SelfAwarenessContext
    from anton.core.llm.client import LLMClient
    from anton.core.memory.cortex import Cortex
    from anton.core.memory.hippocampus import Hippocampus
    from anton.workspace import Workspace

    # Use a mutable container so closures always see the current client
    state: dict = {"llm_client": LLMClient.from_settings(settings)}

    # Self-awareness context (legacy, kept for backward compatibility)
    self_awareness = SelfAwarenessContext(Path(settings.context_dir))

    # Workspace for anton.md and secret vault
    workspace = Workspace(settings.workspace_path)
    workspace.apply_env_to_process()

    # Inject all Local Vault connections as namespaced DS_* env vars so every
    # scratchpad subprocess inherits them. Must happen before any ChatSession is created.
    dv = LocalDataVault()
    dreg = DatasourceRegistry()
    for conn in dv.list_connections():
        dv.inject_env(conn["engine"], conn["name"])  # flat=False by default
        edef = dreg.get(conn["engine"])
        if edef is not None:
            register_secret_vars(edef, engine=conn["engine"], name=conn["name"])
    del dv, dreg

    global_memory_dir = Path.home() / ".anton" / "memory"
    project_memory_dir = settings.workspace_path / ".anton" / "memory"

    cortex = Cortex(
        global_hc=Hippocampus(global_memory_dir),
        project_hc=Hippocampus(project_memory_dir),
        mode=settings.memory_mode,
        llm_client=state["llm_client"],
    )

    # Reconsolidation: migrate legacy memory formats on first run
    from anton.memory.reconsolidator import needs_reconsolidation, reconsolidate

    project_anton_dir = settings.workspace_path / ".anton"
    if needs_reconsolidation(project_anton_dir):
        actions = reconsolidate(project_anton_dir)
        if actions:
            console.print(f"[anton.muted]  Memory migration: {actions[0]}[/]")

    # Background compaction if needed
    if cortex.needs_compaction():
        asyncio.create_task(cortex.compact_all())

    from anton.core.memory.episodes import EpisodicMemory

    episodes_dir = settings.workspace_path / ".anton" / "episodes"
    episodic = EpisodicMemory(episodes_dir, enabled=settings.episodic_memory)
    if episodic.enabled:
        episodic.start_session()

    from anton.memory.history_store import HistoryStore

    history_store = HistoryStore(episodes_dir)
    current_session_id = episodic._session_id if episodic.enabled else None

    # Clean up old clipboard uploads
    uploads_dir = Path(settings.workspace_path) / ".anton" / "uploads"
    cleanup_old_uploads(uploads_dir)

    # Build runtime context so the LLM knows what it's running on
    runtime_context = build_runtime_context(settings)

    output_path = f"{settings.output_dir.rstrip('/')}/"
    session = ChatSession(ChatSessionConfig(
        llm_client=state["llm_client"],
        self_awareness=self_awareness,
        cortex=cortex,
        episodic=episodic,
        system_prompt_context=SystemPromptContext(
            runtime_context=runtime_context,
            output_context=f"Save output to `{output_path}` (create it if needed).",
        ),
        workspace=workspace,
        console=console,
        history_store=history_store,
        session_id=current_session_id,
        proactive_dashboards=settings.proactive_dashboards,
        tools=[CONNECT_DATASOURCE_TOOL, PUBLISH_TOOL],
    ))

    # Handle --resume flag at startup
    if resume:
        session, resumed_id = await handle_resume(
            console,
            settings,
            state,
            self_awareness,
            cortex,
            workspace,
            session,
            episodic=episodic,
            history_store=history_store,
        )
        if resumed_id:
            current_session_id = resumed_id

    if desktop_first_run and not settings.first_run_done:
        try:
            _desktop_greeting(console, settings)
        except Exception:
            pass

    _agent_zero_query: str | None = None
    if first_run and not settings.first_run_done:
        try:
            _agent_zero_result = await _agent_zero(console, session, settings)
            if _agent_zero_result == "_AGENT_ZERO_DONE":
                _agent_zero_query = None
            else:
                _agent_zero_query = _agent_zero_result
        except Exception:
            pass
        _persist_first_run_done(settings)

    if not first_run and not desktop_first_run:
        console.print(f"[anton.cyan_dim] {'━' * 40}[/]")
    console.print("[anton.muted] type '/help' for commands or 'exit' to quit.[/]")
    console.print()

    from anton.analytics import send_event
    _query_count = 0
    _total_questions = 0  # tracks first 10 questions for time estimates

    from anton.chat_ui import StreamDisplay, EscapeWatcher, ClosingSpinner

    toolbar = {"stats": "", "status": ""}
    display = StreamDisplay(console, toolbar=toolbar)
    last_token_status: TokenLimitInfo | None = None
    last_token_status_checked_at: float | None = None

    def _bottom_toolbar():
        stats = toolbar["stats"]
        status = toolbar["status"]
        if not stats and not status:
            return ""
        try:
            width = os.get_terminal_size().columns
        except OSError:
            width = 80
        gap = width - len(status) - len(stats)
        if gap < 1:
            gap = 1
        line = status + " " * gap + stats
        return HTML(f"\n<style fg='#555570'>{line}</style>")

    pt_style = PTStyle.from_dict(
        {
            "bottom-toolbar": "noreverse nounderline bg:default",
        }
    )

    prompt_session: PromptSession[str] = PromptSession(
        mouse_support=False,
        bottom_toolbar=_bottom_toolbar,
        style=pt_style,
    )

    try:
        while True:
            # Memory confirmation UX — show pending lessons before prompt
            if session._pending_memory_confirmations:
                pending = session._pending_memory_confirmations
                console.print("[anton.muted]Lessons learned from this session:[/]")
                for i, engram in enumerate(pending, 1):
                    console.print(f"  [bold]{i}.[/] [{engram.kind}] {engram.text}")
                console.print()
                confirm = (
                    console.input("[bold]Save to memory? (y/n/pick numbers):[/] ")
                    .strip()
                    .lower()
                )
                if confirm in ("y", "yes"):
                    if cortex is not None:
                        await cortex.encode(pending)
                    console.print("[anton.muted]Saved.[/]")
                elif confirm in ("n", "no"):
                    console.print("[anton.muted]Discarded.[/]")
                else:
                    # Parse number selections like "1 3" or "1,3"
                    try:
                        nums = [
                            int(x.strip())
                            for x in confirm.replace(",", " ").split()
                            if x.strip().isdigit()
                        ]
                        selected = [
                            pending[n - 1] for n in nums if 1 <= n <= len(pending)
                        ]
                        if selected and cortex is not None:
                            await cortex.encode(selected)
                            console.print(
                                f"[anton.muted]Saved {len(selected)} entries.[/]"
                            )
                        else:
                            console.print("[anton.muted]Discarded.[/]")
                    except (ValueError, IndexError):
                        console.print("[anton.muted]Discarded.[/]")
                session._pending_memory_confirmations = []
                console.print()

            try:
                from anton.channel.theme import get_palette as _gp
                _you_color = _gp().user_prompt
                user_input = await prompt_session.prompt_async(
                    [(f"bold fg:{_you_color}", "you>"), ("", '\u2009')]
                )
            except EOFError:
                break

            stripped = user_input.strip()
            # message_content holds what we send to the LLM — may be str or
            # list[dict] (multimodal content blocks for images).
            message_content: str | list[dict] | None = None

            # Empty input → check clipboard for an image
            if not stripped:
                if is_clipboard_supported():
                    clip = grab_clipboard()
                    if clip.image:
                        uploaded = save_clipboard_image(clip.image.image, uploads_dir)
                        console.print(
                            f"  [anton.muted]attached: clipboard image "
                            f"({uploaded.width}x{uploaded.height}, "
                            f"{human_size(uploaded.size_bytes)})[/]"
                        )
                        message_content = format_clipboard_image_message(uploaded)
                    elif clip.file_paths:
                        stripped = format_file_message("", clip.file_paths, console)
                if not stripped and message_content is None:
                    continue

            if message_content is None and stripped.lower() in ("exit", "quit", "bye"):
                break

            # Detect dragged file paths early — a dragged absolute path like
            # "/Users/foo/bar.txt" starts with "/" and would otherwise be
            # mistaken for a slash command.
            if message_content is None and stripped.startswith("/"):
                dropped_early = _parse_dropped_paths(stripped)
                if dropped_early:
                    stripped = format_file_message(stripped, dropped_early, console)
                    message_content = stripped

            # Slash command dispatch
            if message_content is None and stripped.startswith("/"):
                parts = stripped.split(maxsplit=1)
                cmd = parts[0].lower()
                if cmd == "/llm":
                    session = await handle_setup_models(
                        console,
                        settings,
                        workspace,
                        state,
                        self_awareness,
                        cortex,
                        session,
                        episodic=episodic,
                        history_store=history_store,
                        session_id=current_session_id,
                    )
                    continue
                elif cmd == "/minds":
                    session = await _handle_connect(
                        console,
                        settings,
                        workspace,
                        state,
                        self_awareness,
                        cortex,
                        session,
                        episodic=episodic,
                    )
                    continue
                elif cmd == "/setup":
                    session = await handle_setup(
                        console,
                        settings,
                        workspace,
                        state,
                        self_awareness,
                        cortex,
                        session,
                        episodic=episodic,
                        history_store=history_store,
                        session_id=current_session_id,
                    )
                    continue
                elif cmd == "/memory":
                    handle_memory(console, settings, cortex, episodic=episodic)
                    continue
                elif cmd == "/connect":
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    session = await handle_connect_datasource(
                        console,
                        session._scratchpads,
                        session,
                        prefill=arg or None,
                        vault=session._data_vault,
                    )
                    continue
                elif cmd == "/list":
                    handle_list_data_sources(console, vault=session._data_vault)
                    continue
                elif cmd == "/remove":
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    await handle_remove_data_source(console, arg, vault=session._data_vault)
                    continue
                elif cmd == "/edit":
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    if not arg:
                        console.print(
                            "[anton.warning]Usage: /edit <engine-name>[/]"
                        )
                        console.print()
                    else:
                        session = await handle_connect_datasource(
                            console,
                            session._scratchpads,
                            session,
                            datasource_name=arg,
                            vault=session._data_vault,
                        )
                    continue
                elif cmd == "/test":
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    await handle_test_datasource(
                        console, session._scratchpads, arg, vault=session._data_vault
                    )
                    continue
                elif cmd == "/skill":
                    # /skill save [name hint] | /skill show <label> | /skill remove <label>
                    sub_parts = parts[1].strip().split(maxsplit=1) if len(parts) > 1 else []
                    sub = sub_parts[0] if sub_parts else ""
                    rest = sub_parts[1] if len(sub_parts) > 1 else ""
                    if sub == "save":
                        await handle_skill_save(
                            console, session, name_hint=rest
                        )
                    elif sub == "show":
                        handle_skill_show(console, rest)
                    elif sub == "remove":
                        handle_skill_remove(console, rest)
                    elif sub == "list" or sub == "":
                        handle_skills_list(console)
                    else:
                        console.print()
                        console.print(
                            "[anton.warning]Usage: /skill save [name] | "
                            "/skill list | /skill show <label> | /skill remove <label>[/]"
                        )
                        console.print()
                    continue
                elif cmd == "/skills":
                    handle_skills_list(console)
                    continue
                elif cmd == "/resume":
                    session, resumed_id = await handle_resume(
                        console,
                        settings,
                        state,
                        self_awareness,
                        cortex,
                        workspace,
                        session,
                        episodic=episodic,
                        history_store=history_store,
                    )
                    if resumed_id:
                        current_session_id = resumed_id
                    continue
                elif cmd == "/theme":
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    handle_theme(console, arg)
                    continue
                elif cmd == "/publish":
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    await _handle_publish(console, settings, workspace, arg)
                    continue
                elif cmd == "/unpublish":
                    await _handle_unpublish(console, settings, workspace)
                    continue
                elif cmd == "/explain":
                    handle_explain(console, settings.workspace_path)
                    continue
                elif cmd == "/help":
                    print_slash_help(console)
                    continue
                elif cmd == "/paste":
                    if not await ensure_clipboard(console):
                        continue
                    clip = grab_clipboard()
                    if clip.image:
                        uploaded = save_clipboard_image(clip.image.image, uploads_dir)
                        console.print(
                            f"  [anton.muted]attached: clipboard image "
                            f"({uploaded.width}x{uploaded.height}, "
                            f"{human_size(uploaded.size_bytes)})[/]"
                        )
                        user_text = parts[1] if len(parts) > 1 else ""
                        message_content = format_clipboard_image_message(
                            uploaded, user_text
                        )
                        # Fall through to turn_stream (don't continue)
                    else:
                        console.print("[anton.warning]No image found on clipboard.[/]")
                        continue
                else:
                    console.print(f"[anton.warning]Unknown command: {cmd}[/]")
                    continue

            # Detect dragged file paths and reformat the message
            if message_content is None:
                dropped = _parse_dropped_paths(stripped)
                if dropped:
                    stripped = format_file_message(stripped, dropped, console)

            # Use multimodal content if set, otherwise the text string
            if message_content is None:
                message_content = stripped

            _query_count += 1
            _total_questions += 1

            # Determine the LLM provider label for telemetry
            _provider = settings.planning_provider or ""
            if _provider == "openai-compatible" and settings.minds_api_key:
                _llm_provider = "mdb_ai"
            elif _provider == "anthropic":
                _llm_provider = "anthropic"
            elif "gemini" in (settings.planning_model or "").lower():
                _llm_provider = "gemini"
            elif _provider in ("openai", "openai-compatible"):
                _llm_provider = "openai"
            else:
                _llm_provider = "other"

            if _query_count == 1:
                send_event(settings, "anton_first_query", llm_provider=_llm_provider)
            else:
                send_event(settings, "anton_query", llm_provider=_llm_provider)

            display.start()
            t0 = time.monotonic()
            ttft: float | None = None
            total_input = 0
            total_output = 0
            session._cancel_event.clear()

            try:
                async with EscapeWatcher(on_cancel=display.show_cancelling) as esc:
                    session._escape_watcher = esc
                    async for event in session.turn_stream(message_content):
                        if esc.cancelled.is_set():
                            session._cancel_event.set()
                            raise KeyboardInterrupt
                        if isinstance(event, StreamTextDelta):
                            if ttft is None:
                                ttft = time.monotonic() - t0
                            display.append_text(event.text)
                        elif isinstance(event, StreamToolResult):
                            display.show_tool_result(event.content)
                        elif isinstance(event, StreamToolUseStart):
                            display.on_tool_use_start(event.id, event.name)
                        elif isinstance(event, StreamToolUseDelta):
                            display.on_tool_use_delta(event.id, event.json_delta)
                        elif isinstance(event, StreamToolUseEnd):
                            display.on_tool_use_end(event.id)
                        elif isinstance(event, StreamTaskProgress):
                            display.update_progress(
                                event.phase, event.message, event.eta_seconds
                            )
                        elif isinstance(event, StreamContextCompacted):
                            display.show_context_compacted(event.message)
                        elif isinstance(event, StreamComplete):
                            total_input += event.response.usage.input_tokens
                            total_output += event.response.usage.output_tokens

                elapsed = time.monotonic() - t0
                parts = []

                if settings.minds_api_key and settings.minds_url:
                    #TODO: Lets check if this is best solution
                    now = time.monotonic()
                    if last_token_status_checked_at is None or (now - last_token_status_checked_at) >= settings.token_status_cache_ttl:
                        last_token_status = check_minds_token_limits(
                            settings.minds_url.rstrip("/"),
                            settings.minds_api_key,
                            verify=settings.minds_ssl_verify,
                        )
                        last_token_status_checked_at = now
                    if last_token_status.billing_cycle_limit > 0:
                        _pct = last_token_status.billing_cycle_used * 100 // last_token_status.billing_cycle_limit
                        parts.append(f"{last_token_status.billing_cycle_used:,} / {last_token_status.billing_cycle_limit:,} ({_pct}%)")

                parts.append(f"{elapsed:.1f}s")
                if not settings.minds_api_key and not settings.minds_url:
                    parts.append(f"{total_input} in / {total_output} out")
                if ttft is not None:
                    parts.append(f"TTFT {int(ttft * 1000)}ms")
                toolbar["stats"] = "  ".join(parts)
                toolbar["status"] = ""
                display.finish()
                if settings.minds_api_key and settings.minds_url and last_token_status is not None and last_token_status.status is TokenLimitStatus.WARNING:
                    pct = int(last_token_status.used / last_token_status.limit * 100) if last_token_status.limit else 80
                    console.print(
                        f"[anton.warning]Approaching token limit: {last_token_status.used:,} / "
                        f"{last_token_status.limit:,} tokens used ({pct}%). "
                        "Visit mdb.ai to upgrade your plan or top up your tokens.[/]"
                    )
                    console.print()
                if _query_count == 1:
                    send_event(settings, "anton_first_answer")
            except anthropic.AuthenticationError:
                display.abort()
                console.print()
                console.print(
                    "[anton.error]Invalid API key. Let's set up a new one.[/]"
                )
                settings.anthropic_api_key = None
                from anton.cli import _ensure_api_key

                _ensure_api_key(settings)
                session = rebuild_session(
                    settings=settings,
                    state=state,
                    self_awareness=self_awareness,
                    cortex=cortex,
                    workspace=workspace,
                    console=console,
                    episodic=episodic,
                    history_store=history_store,
                    session_id=current_session_id,
                )
            except KeyboardInterrupt:
                display.abort()
                session.repair_history()
                # Kill any running scratchpad processes (they may have
                # spawned subprocesses that would otherwise be orphaned).
                if session._scratchpads.list_pads():
                    console.print()
                    _closing = ClosingSpinner(console)
                    _closing.start()
                    try:
                        await session._scratchpads.close_all()
                    finally:
                        _closing.stop()
                else:
                    console.print()
                console.print("[anton.muted]Cancelled.[/]")
                console.print()
                # Cancel the turn but stay in the chat loop
                continue
            except (TokenLimitExceeded, ConnectionError) as exc:
                display.abort()
                console.print()
                console.print(f"[anton.warning]{exc}[/]")
                console.print()
                choice = await prompt_or_cancel(
                    "  (anton) Switch LLM provider, update API key, or retry?",
                    choices=["setup", "retry", "s", "r"],
                    choices_display="setup/retry",
                    default="retry" if isinstance(exc, ConnectionError) else "setup",
                )
                if choice in ("setup", "s"):
                    session = await handle_setup_models(
                        console,
                        settings,
                        workspace,
                        state,
                        self_awareness,
                        cortex,
                        session,
                        episodic=episodic,
                        history_store=history_store,
                        session_id=current_session_id,
                    )
                # retry or after setup — loop continues and re-sends the last message
                continue
            except Exception as exc:
                display.abort()
                console.print()
                console.print(f"[anton.error]{exc}[/]")
                console.print()
                choice = await prompt_or_cancel(
                    "  (anton) Switch LLM provider, or retry?",
                    choices=["setup", "retry", "s", "r"],
                    choices_display="setup/retry",
                    default="retry",
                )
                if choice in ("setup", "s"):
                    session = await handle_setup_models(
                        console,
                        settings,
                        workspace,
                        state,
                        self_awareness,
                        cortex,
                        session,
                        episodic=episodic,
                        history_store=history_store,
                        session_id=current_session_id,
                    )
                continue
    except KeyboardInterrupt:
        pass

    console.print()
    console.print("[anton.muted]See you.[/]")
    await session.close()
