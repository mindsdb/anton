"""Connection testing commands."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Awaitable, Callable

from anton.commands.datasource.helpers import prompt_field_value
from anton.core.backends.base import Cell
from anton.core.datasources.data_vault import DataVault, LocalDataVault
from anton.core.datasources.datasource_registry import DatasourceEngine, DatasourceField, DatasourceRegistry
from anton.utils.datasources import (
    parse_connection_slug,
    register_secret_vars,
    restore_namespaced_env,
    set_ds_env_values,
)
from anton.utils.prompt import prompt_or_cancel

if TYPE_CHECKING:
    from rich.console import Console
    from anton.core.backends.manager import ScratchpadManager

# Bounds only the connect attempt itself (pad.execute of the test_snippet).
# A ModuleNotFoundError retry's pad.install_packages() call is not covered —
# it has its own, separate cell_install_timeout (default 120s) — so the
# worst-case wall clock for a full test is higher than this number suggests.
CONNECTION_TEST_TIMEOUT_SECONDS = 30


async def run_connection_test(
    console: "Console",
    scratchpads: "ScratchpadManager",
    vault: "DataVault",
    engine_def: "DatasourceEngine",
    credentials: dict[str, str],
    retry_fields: "list[DatasourceField]",
    retry_edit_callback: "Callable[[], Awaitable[bool]] | None" = None,
    *,
    interactive: bool = True,
) -> bool:
    """Run engine_def.test_snippet with flat DS_* vars in the pad's own env.

    Returns True on success, False if the user declines retry after failure.
    Mutates credentials in-place when the user re-enters secrets on retry.
    `interactive=False` (mode (a) callers with no terminal to prompt in)
    skips the "retry?" prompt entirely and fails closed on the first error —
    prompt_or_cancel drives a real terminal regardless of the `console`
    passed in, so it must never be reached without one.
    """
    while True:
        console.print()
        console.print("[anton.cyan](anton)[/] Got it. Testing connection…")

        flat_ds_env: dict[str, str] = {
            f"DS_{key.upper()}": value
            for key, value in credentials.items()
            if not key.startswith("_")
        }
        register_secret_vars(engine_def)  # flat mode, for scrubbing during test
        set_ds_env_values(flat_ds_env)

        try:
            pad = await scratchpads.get_or_create(
                "__datasource_test__", ds_env_override=flat_ds_env
            )
            await pad.reset()
            if engine_def.pip:
                if isinstance(engine_def.pip, list):
                    pip_pkgs = engine_def.pip
                else:
                    pip_pkgs = engine_def.pip.split()
                install_result = await pad.install_packages(pip_pkgs)
                if "failed" in (install_result or "").lower():
                    console.print()
                    console.print(f"[anton.warning](anton)[/] Package install issue: {install_result[:200]}")

            cell = None
            for attempt in range(3):
                task = asyncio.ensure_future(pad.execute(engine_def.test_snippet))
                done, _pending = await asyncio.wait(
                    {task}, timeout=CONNECTION_TEST_TIMEOUT_SECONDS
                )
                if task not in done:
                    # The local backend catches its own internal
                    # CancelledError and returns a Cell instead of
                    # re-raising, so `await task` here would silently hand
                    # back that Cell (with a generic kill-tree message)
                    # rather than surfacing our timeout — cancel and
                    # discard its result/exception either way, and always
                    # report our own message.
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                    cell = Cell(
                        code=engine_def.test_snippet,
                        stdout="",
                        stderr="",
                        error=(
                            f"Connection test timed out after "
                            f"{CONNECTION_TEST_TIMEOUT_SECONDS}s — the server "
                            f"did not respond."
                        ),
                    )
                    break
                cell = task.result()
                if cell.error and "ModuleNotFoundError" in cell.error:
                    match = re.search(r"No module named '([^']+)'", cell.error)
                    if match:
                        missing = match.group(1).split(".")[0]
                        await pad.install_packages([missing])
                        continue
                break
        finally:
            restore_namespaced_env(vault)

        if cell.error or (cell.stdout.strip() != "ok" and cell.stderr.strip()):
            error_text = cell.error or cell.stderr.strip() or cell.stdout.strip()
            last_line = next(
                (ln for ln in reversed(error_text.splitlines()) if ln.strip()), error_text
            )
            console.print()
            console.print("[anton.warning](anton)[/] ✗ Connection failed.")
            console.print()
            console.print(f"        Error: {last_line}")
            console.print()
            if not interactive:
                return False
            retry = await prompt_or_cancel(
                "(anton) Would you like to re-enter your credentials?",
                choices=["y", "n"], default="n",
            )
            if retry is None or retry.strip().lower() != "y":
                return False
            console.print()
            if retry_edit_callback is not None:
                if not await retry_edit_callback():
                    return False
                continue
            for f in retry_fields:
                if not await prompt_field_value(f, credentials):
                    return False
            continue

        console.print("[anton.success]        ✓ Connected successfully![/]")
        return True


async def handle_test_datasource(
    console: "Console",
    scratchpads: "ScratchpadManager",
    slug: str,
    vault: DataVault | None = None,
) -> None:
    """Test an existing Local Vault connection by running its test_snippet."""
    if not slug:
        console.print(
            "[anton.warning]Usage: /test <engine-name>[/]"
        )
        console.print()
        return

    vault = vault or LocalDataVault()
    registry = DatasourceRegistry()
    parsed = parse_connection_slug(slug, [e.engine for e in registry.all_engines()], vault=vault)
    if parsed is None:
        console.print(
            f"[anton.warning]Invalid name '{slug}'. Use engine-name format.[/]"
        )
        console.print()
        return
    engine, name = parsed
    fields = vault.load(engine, name)
    if fields is None:
        console.print(
            f"[anton.warning]No connection '{slug}' found in Local Vault.[/]"
        )
        console.print()
        return

    engine_def = registry.get(engine)
    if engine_def is None:
        console.print(
            f"[anton.warning]Unknown engine '{engine}'. Cannot test.[/]"
        )
        console.print()
        return

    if not engine_def.test_snippet:
        console.print(
            f"[anton.warning]No test snippet defined for '{engine}'. Cannot test.[/]"
        )
        console.print()
        return

    console.print()
    console.print(
        f"[anton.cyan](anton)[/] Testing connection [bold]{slug}[/bold]…"
    )

    register_secret_vars(engine_def)  # flat names for scrubbing during test
    flat_ds_env = vault.env_for(engine, name, flat=True) or {}
    set_ds_env_values(flat_ds_env)

    cell = None
    try:
        pad = await scratchpads.get_or_create(
            "__datasource_test__", ds_env_override=flat_ds_env
        )
        await pad.reset()
        if engine_def.pip:
            await pad.install_packages([engine_def.pip])
        cell = await pad.execute(engine_def.test_snippet)
    finally:
        restore_namespaced_env(vault)

    if cell is None or cell.error or (
        cell.stdout.strip() != "ok" and cell.stderr.strip()
    ):
        error_text = ""
        if cell is not None:
            error_text = cell.error or cell.stderr.strip() or cell.stdout.strip()
        first_line = (
            next((ln for ln in error_text.splitlines() if ln.strip()), error_text)
            if error_text
            else "unknown error"
        )
        console.print()
        console.print(
            f"[anton.warning](anton)[/] ✗ Connection test failed for"
            f" [bold]{slug}[/bold]."
        )
        console.print()
        console.print(f"        Error: {first_line}")
    else:
        console.print(
            f"[anton.success]        ✓ Connection test passed for"
            f" [bold]{slug}[/bold]![/]"
        )
    console.print()
