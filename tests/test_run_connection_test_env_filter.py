from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.commands.datasource.verify import run_connection_test
from anton.core.datasources.data_vault import LocalDataVault
from anton.core.datasources.datasource_registry import DatasourceEngine, DatasourceField


@pytest.mark.asyncio
async def test_underscore_prefixed_credentials_not_injected_as_env(tmp_path):
    vault = LocalDataVault(vault_dir=tmp_path / "vault")
    engine_def = DatasourceEngine(
        engine="postgresql",
        display_name="PostgreSQL",
        fields=[DatasourceField(name="host", required=True, description="host")],
        test_snippet="print('ok')",
    )
    console = MagicMock()
    scratchpads = MagicMock()
    pad = AsyncMock()
    pad.reset = AsyncMock()
    pad.install_packages = AsyncMock(return_value="")

    # `run_connection_test()` calls `restore_namespaced_env(vault)` in a
    # `finally` block right after the snippet runs — that clears os.environ
    # and reinjects it from the vault, so asserting on os.environ *after*
    # the function returns would see post-restore state, not what the test
    # snippet actually saw. Capture DS_* env at execution time instead, via
    # the mock the code actually calls (`pad.execute`, not `pad.run`).
    captured: dict[str, str] = {}

    async def _execute(_snippet):
        captured.update({k: v for k, v in os.environ.items() if k.startswith("DS_")})
        return MagicMock(stdout="ok", stderr="", error=None)

    pad.execute = AsyncMock(side_effect=_execute)
    scratchpads.get_or_create = AsyncMock(return_value=pad)

    credentials = {"host": "db.example.com", "_user_label": "postgres 2"}
    await run_connection_test(
        console, scratchpads, vault, engine_def, credentials, engine_def.fields
    )
    assert "DS__USER_LABEL" not in captured
    assert captured.get("DS_HOST") == "db.example.com"


@pytest.mark.asyncio
async def test_hung_snippet_times_out_instead_of_blocking_forever(tmp_path):
    """A test_snippet stuck on e.g. a dead TCP connect() must not hang the
    caller forever — it's bounded by CONNECTION_TEST_TIMEOUT_SECONDS and
    reported as a normal connection failure."""
    vault = LocalDataVault(vault_dir=tmp_path / "vault")
    engine_def = DatasourceEngine(
        engine="postgresql",
        display_name="PostgreSQL",
        fields=[DatasourceField(name="host", required=True, description="host")],
        test_snippet="connect_forever()",
    )
    console = MagicMock()
    scratchpads = MagicMock()
    pad = AsyncMock()
    pad.reset = AsyncMock()
    pad.install_packages = AsyncMock(return_value="")

    async def _execute_hangs(_snippet):
        await asyncio.sleep(3600)  # simulates a wedged connect() to a dead host
        raise AssertionError("should have been cancelled by the timeout")

    pad.execute = AsyncMock(side_effect=_execute_hangs)
    scratchpads.get_or_create = AsyncMock(return_value=pad)

    with (
        patch("anton.commands.datasource.verify.CONNECTION_TEST_TIMEOUT_SECONDS", 0.05),
        patch(
            "anton.commands.datasource.verify.prompt_or_cancel",
            new=AsyncMock(return_value="n"),  # decline the "retry?" prompt
        ),
    ):
        ok = await asyncio.wait_for(
            run_connection_test(
                console, scratchpads, vault, engine_def,
                {"host": "unreachable.example.com"}, engine_def.fields,
            ),
            timeout=5,
        )
    assert ok is False
    printed = " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
    assert "timed out" in printed


@pytest.mark.asyncio
async def test_timeout_message_survives_a_backend_that_swallows_cancellation(tmp_path):
    """LocalScratchpadRuntime.execute_streaming catches its own
    CancelledError and returns a Cell instead of re-raising it, so a bare
    `asyncio.wait_for(pad.execute(...))` would silently hand back that
    Cell's generic kill-tree message rather than raising TimeoutError. The
    friendly "timed out after Ns" message must still win."""
    vault = LocalDataVault(vault_dir=tmp_path / "vault")
    engine_def = DatasourceEngine(
        engine="postgresql",
        display_name="PostgreSQL",
        fields=[DatasourceField(name="host", required=True, description="host")],
        test_snippet="connect_forever()",
    )
    console = MagicMock()
    scratchpads = MagicMock()
    pad = AsyncMock()
    pad.reset = AsyncMock()
    pad.install_packages = AsyncMock(return_value="")

    async def _execute_swallows_cancellation(_snippet):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            # What the real local backend does on cancellation: catch it and
            # return a Cell instead of letting it propagate.
            return MagicMock(stdout="", stderr="", error="Killed: cancelled by caller.")
        raise AssertionError("should have been cancelled by the timeout")

    pad.execute = AsyncMock(side_effect=_execute_swallows_cancellation)
    scratchpads.get_or_create = AsyncMock(return_value=pad)

    with (
        patch("anton.commands.datasource.verify.CONNECTION_TEST_TIMEOUT_SECONDS", 0.05),
        patch(
            "anton.commands.datasource.verify.prompt_or_cancel",
            new=AsyncMock(return_value="n"),
        ),
    ):
        ok = await asyncio.wait_for(
            run_connection_test(
                console, scratchpads, vault, engine_def,
                {"host": "unreachable.example.com"}, engine_def.fields,
            ),
            timeout=5,
        )
    assert ok is False
    printed = " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
    assert "timed out after" in printed
    assert "Killed: cancelled by caller" not in printed


@pytest.mark.asyncio
async def test_non_interactive_skips_retry_prompt_on_failure(tmp_path):
    """Mode (a) callers pass interactive=False — there's no human present to
    answer "retry?", so a failed test must fail closed without prompting.
    prompt_or_cancel drives a real terminal regardless of `console`, so
    reaching it here would hang/crash in a console-less host."""
    vault = LocalDataVault(vault_dir=tmp_path / "vault")
    engine_def = DatasourceEngine(
        engine="postgresql",
        display_name="PostgreSQL",
        fields=[DatasourceField(name="host", required=True, description="host")],
        test_snippet="print('fail')",
    )
    console = MagicMock()
    scratchpads = MagicMock()
    pad = AsyncMock()
    pad.reset = AsyncMock()
    pad.install_packages = AsyncMock(return_value="")
    pad.execute = AsyncMock(return_value=MagicMock(stdout="", stderr="boom", error=None))
    scratchpads.get_or_create = AsyncMock(return_value=pad)

    with patch(
        "anton.commands.datasource.verify.prompt_or_cancel",
        new=AsyncMock(side_effect=AssertionError("must not prompt when non-interactive")),
    ):
        ok = await run_connection_test(
            console, scratchpads, vault, engine_def,
            {"host": "db.example.com"}, engine_def.fields,
            interactive=False,
        )
    assert ok is False


@pytest.mark.asyncio
async def test_run_connection_test_scrubs_the_credential_under_test(tmp_path):
    """The credential under test survives redaction even if a concurrent
    turn wipes os.environ mid-test — proves the explicit override, not just the os.environ fallback."""
    from anton.utils.datasources import scrub_credentials

    vault = LocalDataVault(vault_dir=tmp_path / "vault")
    engine_def = DatasourceEngine(
        engine="postgresql",
        display_name="PostgreSQL",
        fields=[
            DatasourceField(name="host", required=True, description="host"),
            DatasourceField(name="password", required=True, description="password", secret=True),
        ],
        test_snippet="print('ok')",
    )
    console = MagicMock()
    scratchpads = MagicMock()
    pad = AsyncMock()
    pad.reset = AsyncMock()
    pad.install_packages = AsyncMock(return_value="")

    captured = {}

    async def _execute(_snippet):
        # Simulate a concurrent turn's clear_ds_env() wiping shared
        # os.environ mid-test — the exact race this fix closes.
        for key in [k for k in os.environ if k.startswith("DS_")]:
            del os.environ[key]
        captured["scrubbed"] = scrub_credentials("connecting with password s3cr3t_under_test")
        return MagicMock(stdout="ok", stderr="", error=None)

    pad.execute = AsyncMock(side_effect=_execute)
    scratchpads.get_or_create = AsyncMock(return_value=pad)

    credentials = {"host": "db.example.com", "password": "s3cr3t_under_test"}
    await run_connection_test(
        console, scratchpads, vault, engine_def, credentials, engine_def.fields
    )

    assert "s3cr3t_under_test" not in captured["scrubbed"]
    assert "[DS_PASSWORD]" in captured["scrubbed"]
