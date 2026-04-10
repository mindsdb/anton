"""Bug report command handler."""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from anton.diagnostics import collect_diagnostics, save_diagnostics_file
from anton.utils.prompt import prompt_or_cancel

from anton.publisher import publish_bug_report

if TYPE_CHECKING:
    from anton.chat_session import ChatSession
    from anton.config.settings import AntonSettings
    from anton.memory.cortex import Cortex
    from anton.workspace import Workspace


async def handle_report_bug(
    console: Console,
    settings: AntonSettings,
    workspace: Workspace | None,
    session: ChatSession | None,
    cortex: Cortex | None,
) -> None:
    """Handle /report-bug command - collect diagnostics and send to bug report endpoint."""
    console.print()
    console.print("[anton.cyan]Bug Report[/]")
    console.print()

    # Privacy consent prompt
    console.print("[anton.warning]⚠️  Important Privacy Notice[/]")
    console.print()
    console.print("  This bug report will include:")
    console.print("  • Your conversation history from this session")
    console.print("  • System information and Anton configuration")
    console.print("  • Connected datasource names (no credentials)")
    console.print("  • Recent logs and memory state")
    console.print()
    console.print(
        "  [bold]Our dev team will be able to see all of this information.[/]"
    )
    console.print()

    consent = await prompt_or_cancel(
        "  Do you agree to share this information?",
        choices=["y", "n"],
        choices_display="y/n",
        default="n",
    )

    if consent is None or consent.lower() != "y":
        console.print()
        console.print("  [anton.muted]Bug report cancelled.[/]")
        console.print()
        return

    console.print()

    # Optional bug description
    add_description = await prompt_or_cancel(
        "  Would you like to add a description of the bug?",
        choices=["y", "n"],
        choices_display="y/n",
        default="y",
    )

    bug_description = None
    if add_description and add_description.lower() == "y":
        console.print()
        console.print(
            "  [anton.muted]Please describe the bug (press Enter when done):[/]"
        )
        bug_description = await prompt_or_cancel("  ")
        if bug_description is None:
            bug_description = ""

    console.print()

    # Collect diagnostics
    from rich.live import Live
    from rich.spinner import Spinner

    with Live(
        Spinner(
            "dots", text="  Collecting diagnostic information...", style="anton.cyan"
        ),
        console=console,
        transient=True,
    ):
        try:
            diagnostics = collect_diagnostics(settings, session, workspace, cortex)

            # Add bug description if provided
            if bug_description:
                diagnostics["user_description"] = bug_description

            # Save to file
            output_dir = Path(settings.workspace_path) / ".anton" / "output"
            diagnostics_file = save_diagnostics_file(diagnostics, output_dir)

        except Exception as e:
            console.print(f"  [anton.error]Failed to collect diagnostics: {e}[/]")
            console.print()
            return

    # Ensure Minds API key is available
    if not settings.minds_api_key:
        console.print(
            "  [anton.muted]To submit bug reports you need a free Minds account.[/]"
        )
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
            webbrowser.open("https://mdb.ai/")
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

    # Submit bug report
    with Live(
        Spinner("dots", text="  Submitting bug report...", style="anton.cyan"),
        console=console,
        transient=True,
    ):
        try:
            publish_bug_report(
                diagnostics_file,
                api_key=settings.minds_api_key,
                bug_report_url=settings.bug_report_url or settings.publish_url,
                ssl_verify=settings.minds_ssl_verify,
            )

        except Exception as e:
            console.print(f"  [anton.error]Failed to submit bug report: {e}[/]")
            console.print()
            return

    console.print("  [anton.success]Bug report submitted successfully![/]")
    console.print("  [anton.muted]Thank you for helping us improve Anton.[/]")
    console.print()
