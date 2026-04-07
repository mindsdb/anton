"""Slash-command handlers for /minds and /publish."""

from __future__ import annotations

import os
import re
import urllib.error
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.prompt import Prompt
from rich.spinner import Spinner

from anton.chat_session import rebuild_session
from anton.minds_client import (
    describe_minds_connection_error,
    list_datasources,
    list_minds,
    normalize_minds_url,
    test_llm,
)
from anton.publisher import publish
from anton.utils.prompt import prompt_minds_api_key, prompt_or_cancel
from anton.workspace import Workspace

if TYPE_CHECKING:
    from anton.chat_session import ChatSession
    from anton.config.settings import AntonSettings
    from anton.memory.cortex import Cortex
    from anton.memory.episodes import EpisodicMemory


async def handle_connect_minds(
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
    global_ws = Workspace(Path.home())

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


async def handle_publish(
    console: Console,
    settings,
    workspace,
    file_arg: str = "",
) -> None:
    """Handle /publish command — publish an HTML report to the web."""
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

    # 3. Publish
    with Live(Spinner("dots", text="  Publishing...", style="anton.cyan"), console=console, transient=True):
        try:
            result = publish(
                target,
                api_key=settings.minds_api_key,
                publish_url=settings.publish_url,
                ssl_verify=settings.minds_ssl_verify,
            )
        except Exception as e:
            console.print(f"  [anton.error]Publish failed: {e}[/]")
            console.print()
            return

    view_url = result.get("view_url", "")
    console.print(f"  [anton.success]Published![/]")
    console.print(f"  [link={view_url}]{view_url}[/link]")
    console.print()

    if view_url:
        webbrowser.open(view_url)


def handle_unpublish(
    console: Console,
    settings,
    workspace,
    report_arg: str = "",
) -> None:
    """Handle /unpublish command — unpublish a previously published report."""
    console.print()
    console.print("  [anton.warning]Unpublish is not implemented yet.[/]")
    console.print()