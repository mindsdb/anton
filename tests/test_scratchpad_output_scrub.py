"""A token a cell prints never leaves the scratchpad readable.

A skill that fetches an OAuth token and prints the token endpoint's response
puts the token in the cell's output. Every copy of that output is scrubbed:
the streamed tool result a host saves, the progress strings, the tool result
the model reads next round, the episodic log, and the notebook dump. A cell
killed after printing the token keeps its salvaged output, scrubbed too.

The cells build the token at runtime so it never appears in the cell's code,
which is the model's own text and is not scrubbed.
"""
from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.chat_session import get_runtime_factory
from anton.core.backends.base import Cell
from anton.core.backends.local import _SALVAGE_RAW_MAX, LocalScratchpadRuntime, _add_salvage_chunk, _salvage_tail
from anton.core.datasources.data_vault import LocalDataVault
from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamTaskProgress,
    StreamToolResult,
    ToolCall,
    Usage,
)
from anton.core.memory.episodes import EpisodicMemory
from anton.core.session import ChatSession, ChatSessionConfig, _VerifierVerdict
from anton.core.tools.tool_defs import ToolDef
from anton.utils.datasources import begin_ds_turn_scope, restore_namespaced_env, set_ds_env_values
from tests.conftest import make_mock_llm

_DEFAULTS = dict(
    coding_provider="anthropic",
    coding_model="",
    coding_api_key="",
    coding_base_url="",
)

TOKEN = "opq" + "Z9" * 15
_MAKE_TOKEN = "tok = 'opq' + 'Z9' * 15\n"
_PRINT_TOKEN_RESPONSE = (
    "import json\n"
    + _MAKE_TOKEN
    + "progress('Bearer ' + tok)\n"
    + "print(json.dumps({'access_token': tok, 'expires_in': 3600}))\n"
)


def _exec_response(code: str) -> LLMResponse:
    """A model round that asks the scratchpad to run ``code``."""
    return LLMResponse(
        content="running",
        tool_calls=[ToolCall(id="tc_cell", name="scratchpad",
                             input={"action": "exec", "name": "main", "code": code})],
        usage=Usage(input_tokens=1, output_tokens=1),
        stop_reason="tool_use",
    )


async def _run_one_cell_turn(tmp_path, code: str, data_vault=None) -> dict:
    """Run a turn whose model runs ``code`` in a real local pad, then finishes.

    Returns every copy of the output: streamed results, progress lines, the
    model's next input, the episodic log and the pad's notebook.
    """
    workspace = MagicMock(base=tmp_path)
    workspace.artifacts_dir = tmp_path / "artifacts"
    llm = make_mock_llm()
    llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="COMPLETE", reason="done")
    )
    model_inputs: list[str] = []

    def plan_stream(**kwargs):
        """Record what the model is sent; run the cell first, then finish."""
        model_inputs.append(repr(kwargs.get("messages")))
        response = _exec_response(code) if len(model_inputs) == 1 else LLMResponse(
            content="done", tool_calls=[], usage=Usage(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
        )

        async def gen():
            yield StreamComplete(response=response)

        return gen()

    llm.plan_stream = plan_stream
    episodic = EpisodicMemory(tmp_path / "episodes")
    episodic.start_session()
    session = ChatSession(
        ChatSessionConfig(llm_client=llm, workspace=workspace, episodic=episodic, data_vault=data_vault)
    )
    pad = get_runtime_factory(SimpleNamespace(backend="local"))(
        name="main", cells=None, workspace_path=tmp_path, **_DEFAULTS
    )
    await pad.start()
    session._scratchpads.get_or_create = AsyncMock(return_value=pad)
    events = []
    try:
        with patch("anton.analytics.send_event"):
            async for event in session.turn_stream("connect to the API"):
                events.append(event)
        notebook = pad.render_notebook()
        recorded = list(pad.cells)
    finally:
        await pad.close()
        await session.close()

    assert len(model_inputs) == 2, "the model never got the tool result"
    return {
        "results": [e.content for e in events if isinstance(e, StreamToolResult)],
        "progress": [e.message or "" for e in events if isinstance(e, StreamTaskProgress)],
        "next_model_input": model_inputs[1],
        "episodes": "".join(p.read_text() for p in (tmp_path / "episodes").glob("*.jsonl")),
        "notebook": notebook,
        "recorded": recorded,
    }


async def test_a_printed_token_is_scrubbed_from_every_copy_of_the_output(tmp_path):
    """Stream, progress, next model input, episodic log and dump never hold the token."""
    out = await _run_one_cell_turn(tmp_path, _PRINT_TOKEN_RESPONSE)

    results, progress = out["results"], out["progress"]
    assert results, "no tool result was streamed"
    for copy in [*results, *progress, out["next_model_input"], out["episodes"], out["notebook"]]:
        assert TOKEN not in copy
    assert "access_token" in results[0]
    assert "[REDACTED_TOKEN]" in results[0]
    assert "[REDACTED_TOKEN]" in out["next_model_input"]
    assert any("Bearer [REDACTED_TOKEN]" in m for m in progress)


_SAVED_PASSWORD = "Pw0" + "Q8" * 15


async def test_a_printed_saved_secret_shows_its_placeholder_in_the_stream(tmp_path):
    """A secret-flagged saved value shows as its `[DS_...]` placeholder in the
    streamed result and the recorded cell, not only in the model's input."""
    vault = LocalDataVault(tmp_path / "vault")
    vault.save("postgres", "prod", {"host": "db.example.com", "password": _SAVED_PASSWORD})

    # Read from a file so the value is in the output only, not in the cell's code.
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text(_SAVED_PASSWORD)
    out = await _run_one_cell_turn(
        tmp_path, f"print('connecting with ' + open({str(secret_file)!r}).read())\n", data_vault=vault
    )

    results, recorded = out["results"], out["recorded"]
    assert results, "no tool result was streamed"
    for copy in [*results, out["next_model_input"], out["episodes"], out["notebook"], recorded[-1].stdout]:
        assert _SAVED_PASSWORD not in copy
    assert "connecting with [DS_POSTGRES_PROD__PASSWORD]" in results[0]
    assert "connecting with [DS_POSTGRES_PROD__PASSWORD]" in recorded[-1].stdout


@pytest.mark.parametrize("secret_expression", [
    "'abc' + chr(39) + 'def' + 'Gh' + 'Ij' + 'Kl' + 'Mn' + 'OpQr'",
    "'abc' + chr(39) + ',def' + 'Gh' + 'Ij' + 'Kl' + 'Mn' + 'OpQrStu'",
    "'+ AbCdEf' + 'Gh' + 'Ij' + 'Kl' + 'Mn' + 'OpQrSt'",
], ids=["apostrophe", "quote-and-comma", "leading-plus"])
async def test_a_printed_secret_holding_an_apostrophe_is_redacted_whole(tmp_path, secret_expression):
    """A client secret can hold a quote or an operator; the printed response still loses all of it."""
    out = await _run_one_cell_turn(
        tmp_path,
        "import json\n"
        f"secret = {secret_expression}\n"
        "print(json.dumps({'client_secret': secret, 'expires_in': 3600}))\n",
    )

    results, recorded = out["results"], out["recorded"]
    assert results, "no tool result was streamed"
    for copy in [*results, out["next_model_input"], out["episodes"], out["notebook"], recorded[-1].stdout]:
        assert "GhIjKlMn" not in copy
    assert json.loads(results[0])["stdout"] == '{"client_secret": "[REDACTED_TOKEN]", "expires_in": 3600}\n'
    assert recorded[-1].stdout == '{"client_secret": "[REDACTED_TOKEN]", "expires_in": 3600}\n'


async def _print_saved_secret(tmp_path, secret: str, printed: str):
    """Run one cell that prints the ``printed`` expression of ``secret``, saved
    as a postgres password and read from a file so it is not in the code."""
    vault = LocalDataVault(tmp_path / "vault")
    vault.save("postgres", "prod", {"host": "db.example.com", "password": secret})
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text(secret)
    begin_ds_turn_scope()
    restore_namespaced_env(vault)
    pad = LocalScratchpadRuntime(name="cut-secret-cell", **_DEFAULTS)
    await pad.start()
    try:
        return await pad.execute(f"secret = open({str(secret_file)!r}).read()\nprint({printed})\n")
    finally:
        await pad.close()


def _secret_of(length: int) -> str:
    return ("Pw0" + "Q8" * length)[:length]


@pytest.mark.parametrize("length, printed", [
    (33, "'x' * 9967 + ' ' + secret"),
    (101, "(' ' + secret + chr(10)) * 130 + 'x' * 655 + ' ' + secret + 'y' * 100"),
    (6000, "'x' * 9095 + ' ' + secret + 'y' * 100"),
], ids=["at-the-output-limit", "at-the-worker-cut-after-text-the-scrub-shrinks", "longer-than-the-margin"])
async def test_a_saved_secret_cut_by_the_output_limit_is_still_redacted(tmp_path, length, printed):
    """Long output is cut to its limit; a saved secret across a cut must not
    lose its tail and with it the exact match that redacts it."""
    secret = _secret_of(length)
    cell = await _print_saved_secret(tmp_path, secret, printed)

    assert "chars total)" in cell.stdout
    assert len(cell.stdout) < 10_100
    for start in range(len(secret) - 7):
        assert secret[start : start + 8] not in cell.stdout


async def test_output_that_scrubs_away_entirely_says_it_was_withheld(tmp_path):
    """When nothing safe is left to show, the cell says output was withheld."""
    secret = _secret_of(300)
    cell = await _print_saved_secret(tmp_path, secret, "(' ' + secret + chr(10)) * 100")

    assert "withheld" in cell.stdout
    for start in range(len(secret) - 7):
        assert secret[start : start + 8] not in cell.stdout


async def test_a_long_saved_value_that_is_not_secret_does_not_withhold_output(tmp_path):
    """Only values the scrub matches set how much is held back; a long
    non-secret field must not hide normal output."""
    vault = LocalDataVault(tmp_path / "vault")
    vault.save("postgres", "prod", {"host": "h" * 20_000, "password": _SAVED_PASSWORD})
    begin_ds_turn_scope()
    restore_namespaced_env(vault)
    pad = LocalScratchpadRuntime(name="long-host-cell", **_DEFAULTS)
    await pad.start()
    try:
        cell = await pad.execute("print('normal line\\n' * 2000)\n")
    finally:
        await pad.close()

    assert "withheld" not in cell.stdout
    assert cell.stdout.startswith("normal line\n")
    assert cell.stdout.count("normal line") > 800


async def test_short_and_cut_output_both_read_as_printed():
    """Output under the limit comes back as printed; output over it is cut
    with a note giving the full length."""
    pad = LocalScratchpadRuntime(name="uncut-field-cell", **_DEFAULTS)
    await pad.start()
    try:
        short = await pad.execute("print('short')\n")
        long = await pad.execute("print('x' * 12000)\n")
    finally:
        await pad.close()

    assert short.stdout == "short\n"
    assert long.stdout.startswith("x" * 10_000)
    assert "(truncated, 12001 chars total)" in long.stdout


def test_cells_replayed_from_stored_history_are_scrubbed():
    """Cells a host rebuilds from stored events come back scrubbed in dump and view."""
    stored = Cell(code="fetch()", stdout=f'{{"access_token": "{TOKEN}"}}', stderr="", error=None)
    pad = LocalScratchpadRuntime(name="replayed", cells=[stored], **_DEFAULTS)

    assert TOKEN not in pad.cells[0].stdout
    assert TOKEN not in pad.render_notebook()
    assert TOKEN not in pad.view()
    assert '"access_token": "[REDACTED_TOKEN]"' in pad.cells[0].stdout


async def test_a_cell_killed_after_printing_a_token_keeps_scrubbed_output(monkeypatch):
    """A cell killed by its timeout keeps its salvaged output, with the token redacted."""
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "1")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "1")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_AFTER_PROGRESS", "1")
    monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "2")
    monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.2")
    pad = LocalScratchpadRuntime(name="killed-token-cell", **_DEFAULTS)
    await pad.start()
    code = _MAKE_TOKEN + "print('access_token=' + tok, flush=True)\nimport time; time.sleep(30)\n"
    try:
        items = [item async for item in pad.execute_streaming(code)]
    finally:
        await pad.close()

    cells = [item for item in items if isinstance(item, Cell)]
    assert len(cells) == 1
    assert "timed out" in cells[0].error.lower()
    assert TOKEN not in cells[0].stdout
    assert "access_token=[REDACTED_TOKEN]" in cells[0].stdout
    assert TOKEN not in pad.cells[-1].stdout


@pytest.mark.parametrize("key, filler", [
    ("' access_token: ', end=''", "'line\\n' * 1600"),
    ("' access_token: ', end=''", "'x' * 8000"),
    ("' Authorization: Bearer'", "'line\\n' * 1600"),
], ids=["lines", "one-line", "bearer-on-its-own-line"])
async def test_a_killed_cell_over_the_salvage_budget_does_not_strip_a_token_key(monkeypatch, key, filler):
    """Salvage arrives in heartbeat chunks; trimming the oldest chunk must not
    drop a token's key and leave the token itself in the kept output. The first
    chunk ends with the key and the second starts with the token, and the
    salvage is cut to its budget after the scrub."""
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "2")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "2")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_AFTER_PROGRESS", "2")
    monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "3")
    monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.2")
    pad = LocalScratchpadRuntime(name="salvage-split-cell", **_DEFAULTS)
    await pad.start()
    code = (
        _MAKE_TOKEN
        + "import time\n"
        + f"print('a' * 2000 + {key}, flush=True)\n"
        + "time.sleep(0.6)\n"
        + f"print(tok + ' ' + {filler}, flush=True)\n"
        + "time.sleep(30)\n"
    )
    try:
        items = [item async for item in pad.execute_streaming(code)]
    finally:
        await pad.close()

    cells = [item for item in items if isinstance(item, Cell)]
    assert len(cells) == 1
    assert "timed out" in cells[0].error.lower()
    assert cells[0].stdout
    assert TOKEN not in cells[0].stdout


def _salvage_from(stream: str) -> str:
    """Feed `stream` through the salvage path in worker-sized chunks."""
    kept: list[str] = []
    dropped = False
    for start in range(0, len(stream), 8192):
        dropped = _add_salvage_chunk(kept, stream[start : start + 8192]) or dropped
    return _salvage_tail(kept, dropped)


_LONG_TOKEN = "opq" + "Z9" * 500


@pytest.mark.parametrize("key, before_token", [
    ("access_token: ", ""),
    ("Authorization: Bearer\n", ""),
    ("Authorization: Bearer", "\n"),
    ('"access_token":', '\n\n  "'),
    ("", "partial\nAuthorization: Bearer\n"),
    ('"access_token"', '\n:\n"'),
    ("access_token", '\n\n=\n\n  "'),
    ('"access_to', 'ken"\n:\n"'),
], ids=[
    "same-line", "bearer-newline-in-dropped", "bearer-newline-in-kept", "blank-lines",
    "key-on-a-kept-line", "separator-on-its-own-line", "separator-between-blank-lines",
    "cut-inside-the-key",
])
def test_dropped_salvage_never_leaves_a_token_without_its_key(key, before_token):
    """Token-dense output scrubs to far less than the budget, so the start of
    the kept chunks reaches the output; a token whose key was in a dropped
    chunk must not be there. Sized so only the first chunk is dropped."""
    first = "x" * (8192 - len(key) - 1) + "\n" + key
    rest = before_token + TOKEN + "\n"
    rest += "".join(f"access_token: {_LONG_TOKEN}\n" for _ in range(35))
    salvaged = _salvage_from(first + rest)

    assert TOKEN not in salvaged
    assert "access_token: [REDACTED_TOKEN]" in salvaged
    assert salvaged.startswith("(truncated to most recent output)\n")


def test_dropped_salvage_on_one_long_line_reports_nothing_rather_than_a_bare_token():
    """With no line start to cut at, the kept text cannot be told safe."""
    line = "x" * 8178 + "access_token: " + TOKEN + " "
    line += "".join(f"access_token: {_LONG_TOKEN} " for _ in range(32))
    assert _salvage_from(line) == ""


def test_salvage_under_the_budget_is_kept_whole():
    """Nothing is dropped or cut, so the salvage comes back as printed."""
    assert _salvage_from("sent 1/3\nsent 2/3\n") == "sent 1/3\nsent 2/3\n"


def _save_password(tmp_path, secret: str) -> None:
    """Register `secret` as this turn's saved postgres password."""
    vault = LocalDataVault(tmp_path / "vault")
    vault.save("postgres", "prod", {"host": "db.example.com", "password": secret})
    begin_ds_turn_scope()
    restore_namespaced_env(vault)


def _unknown_ds_value(tmp_path, secret: str) -> None:
    """Make `secret` this turn's value of a DS_* var no engine registered."""
    begin_ds_turn_scope()
    set_ds_env_values({"DS_CUSTOM_PROD__TOKEN": secret})


@pytest.mark.parametrize("register", [_save_password, _unknown_ds_value], ids=["saved-secret", "unknown-ds-var"])
def test_salvage_ending_partway_through_a_secret_withholds_the_part(tmp_path, register):
    """A crash right after a heartbeat leaves the last chunk ending partway
    through a secret, which an exact-value scrub cannot match."""
    secret = _secret_of(33)
    register(tmp_path, secret)
    salvaged = _salvage_tail(["x" * 8159 + " " + secret[:32]], False)

    for start in range(25):
        assert secret[start : start + 8] not in salvaged
    assert salvaged.startswith("(truncated to most recent output)\n")
    assert salvaged.endswith("x ")


def test_salvage_ending_partway_through_a_base64_secret_withholds_the_part(tmp_path):
    """A secret printed in base64 and cut is withheld like one printed as typed."""
    secret = _secret_of(33)
    _save_password(tmp_path, secret)
    printed = base64.b64encode(secret.encode()).decode()
    salvaged = _salvage_tail(["x" * 8159 + " " + printed[:40]], False)

    for start in range(33):
        assert printed[start : start + 8] not in salvaged
    assert salvaged.startswith("(truncated to most recent output)\n")
    assert salvaged.endswith("x ")


def test_salvage_ending_with_a_whole_secret_shows_its_placeholder(tmp_path):
    """A secret whose end repeats its start is whole, so it is redacted, not cut."""
    secret = "Ab3" + "Q8" * 15 + "Ab3"
    _save_password(tmp_path, secret)
    salvaged = _salvage_tail(["sent 1/3\nconnecting with " + secret], False)

    assert salvaged == "sent 1/3\nconnecting with [DS_POSTGRES_PROD__PASSWORD]"


def test_salvage_that_ends_on_no_secret_is_unchanged(tmp_path):
    """Only the start of a secret is withheld; other text keeps its last characters."""
    _save_password(tmp_path, _SAVED_PASSWORD)

    assert _salvage_tail(["sent 1/3\nsent 2/3 Px"], False) == "sent 1/3\nsent 2/3 Px"
    assert _salvage_tail(["sent 1/3\nsent 2/3"], False) == "sent 1/3\nsent 2/3"


_PEM_LINES = [f"MIIEv{i:02d}" + "Q8" * 30 for i in range(20)]
_PEM = "-----BEGIN PRIVATE KEY-----\n" + "\n".join(_PEM_LINES) + "\n-----END PRIVATE KEY-----"


@pytest.mark.parametrize("register", [_save_password, _unknown_ds_value], ids=["saved-secret", "unknown-ds-var"])
def test_dropped_salvage_starting_partway_through_a_multi_line_secret_withholds_the_part(tmp_path, register):
    """A dropped chunk can cut a multi-line secret, such as a private key, so the
    kept text starts with lines the exact-value scrub cannot match."""
    register(tmp_path, _PEM)
    kept: list[str] = []
    _add_salvage_chunk(kept, "a" * (_SALVAGE_RAW_MAX - 100) + _PEM[:200])
    dropped = _add_salvage_chunk(kept, _PEM[200:] + "\nsent 3/3\nsent 4/4\ndone\n")
    salvaged = _salvage_tail(kept, dropped)

    assert dropped
    for line in _PEM_LINES:
        assert line[5:] not in salvaged
    assert salvaged == "(truncated to most recent output)\ndone\n"


def test_dropped_salvage_ending_partway_through_a_secret_withholds_the_part(tmp_path):
    """Both edges apply at once: a drop at the start and a cut secret at the end."""
    secret = _secret_of(33)
    _save_password(tmp_path, secret)
    salvaged = _salvage_from("line\n" * 8000 + secret[:32])

    for start in range(25):
        assert secret[start : start + 8] not in salvaged
    assert salvaged.startswith("(truncated to most recent output)\n")
    assert salvaged.endswith("line\n")


def test_salvage_ending_partway_through_a_provider_key_withholds_the_part(monkeypatch):
    """Provider keys are exact-match secrets too, whatever their shape."""
    key = "Zt7" + "Q8" * 15
    monkeypatch.setenv("ANTON_MINDS_API_KEY", key)
    begin_ds_turn_scope()
    salvaged = _salvage_tail(["calling the gateway with " + key[:24]], False)

    for start in range(17):
        assert key[start : start + 8] not in salvaged
    assert salvaged.endswith("calling the gateway with ")


async def test_a_crashed_cell_ending_partway_through_a_saved_secret_withholds_the_part(tmp_path, monkeypatch):
    """A crash right after the worker ships part of a saved secret keeps the
    output before it and none of the secret."""
    monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.1")
    secret = _secret_of(33)
    _save_password(tmp_path, secret)
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text(secret)
    pad = LocalScratchpadRuntime(name="crashed-mid-secret-cell", **_DEFAULTS)
    await pad.start()
    code = (
        "import os, time\n"
        f"secret = open({str(secret_file)!r}).read()\n"
        "print('sent 1/3\\nconnecting with ' + secret[:32], end='', flush=True)\n"
        "time.sleep(0.5)\n"
        "os._exit(1)\n"
    )
    try:
        items = [item async for item in pad.execute_streaming(code)]
    finally:
        await pad.close()

    cells = [item for item in items if isinstance(item, Cell)]
    assert len(cells) == 1
    assert "exited unexpectedly" in cells[0].error.lower()
    for start in range(25):
        assert secret[start : start + 8] not in cells[0].stdout
    assert "sent 1/3\nconnecting with " in cells[0].stdout


async def test_a_killed_cell_with_a_saved_secret_keeps_its_last_line(tmp_path, monkeypatch):
    """A saved secret in the turn must not cost a killed cell its latest
    output, which says what side effects already happened."""
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "1")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "1")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_AFTER_PROGRESS", "1")
    monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "2")
    monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.2")
    _save_password(tmp_path, _SAVED_PASSWORD)
    pad = LocalScratchpadRuntime(name="killed-saved-secret-cell", **_DEFAULTS)
    await pad.start()
    try:
        items = [item async for item in pad.execute_streaming(
            "print('sent 1/3', flush=True)\nimport time; time.sleep(30)\n"
        )]
    finally:
        await pad.close()

    cells = [item for item in items if isinstance(item, Cell)]
    assert len(cells) == 1
    assert "timed out" in cells[0].error.lower()
    assert cells[0].stdout == "sent 1/3\n"


async def test_a_crashed_cell_with_dropped_salvage_says_it_is_partial(monkeypatch):
    """A crash reports salvaged stdout like a kill does: the most recent
    output, scrubbed, and marked as cut instead of passing for the whole output."""
    monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.1")
    pad = LocalScratchpadRuntime(name="crashed-cell", **_DEFAULTS)
    await pad.start()
    code = (
        _MAKE_TOKEN
        + "import os, time\n"
        "for i in range(8):\n"
        "    print(f'line {i}\\n' + 'x' * 8000, flush=True)\n"
        "    time.sleep(0.4)\n"
        "print('access_token: ' + tok, flush=True)\n"
        "time.sleep(0.4)\n"
        "os._exit(1)\n"
    )
    try:
        items = [item async for item in pad.execute_streaming(code)]
    finally:
        await pad.close()

    cells = [item for item in items if isinstance(item, Cell)]
    assert len(cells) == 1
    assert "exited unexpectedly" in cells[0].error.lower()
    assert cells[0].stdout.startswith("(truncated to most recent output)\n")
    assert "line 7" in cells[0].stdout
    assert "access_token: [REDACTED_TOKEN]" in cells[0].stdout
    assert TOKEN not in cells[0].stdout


async def test_a_killed_cell_whose_salvage_is_withheld_says_output_existed(monkeypatch):
    """One long line with no line start to cut at cannot be shown safely after
    a drop; the error still tells the agent the cell printed something."""
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "2")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_MAX", "2")
    monkeypatch.setenv("ANTON_CELL_INACTIVITY_AFTER_PROGRESS", "2")
    monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "5")
    monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.1")
    pad = LocalScratchpadRuntime(name="withheld-salvage-cell", **_DEFAULTS)
    await pad.start()
    code = (
        "import time\n"
        "for i in range(7):\n"
        "    print('x' * 8000, end='', flush=True)\n"
        "    time.sleep(0.3)\n"
        "time.sleep(30)\n"
    )
    try:
        items = [item async for item in pad.execute_streaming(code)]
    finally:
        await pad.close()

    cells = [item for item in items if isinstance(item, Cell)]
    assert len(cells) == 1
    assert cells[0].stdout == ""
    assert "printed output" in cells[0].error and "withheld" in cells[0].error


async def test_another_tool_result_keeps_quoted_placeholders(tmp_path):
    """Only scratchpad output gets the token filter; source another tool returns reaches the model as written."""
    source = 'export const config = { accessToken: "some-placeholder-value" };'

    async def read_source(_session, _input):
        """Return artifact source the way open_artifact does."""
        return source

    workspace = MagicMock(base=tmp_path)
    workspace.artifacts_dir = tmp_path / "artifacts"
    llm = make_mock_llm()
    llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="COMPLETE", reason="done")
    )
    model_inputs: list[str] = []

    def plan_stream(**kwargs):
        """Ask for the tool once, then finish."""
        model_inputs.append(repr(kwargs.get("messages")))
        response = LLMResponse(
            content="reading",
            tool_calls=[ToolCall(id="tc_read", name="read_source", input={})],
            usage=Usage(input_tokens=1, output_tokens=1),
            stop_reason="tool_use",
        ) if len(model_inputs) == 1 else LLMResponse(
            content="done", tool_calls=[], usage=Usage(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
        )

        async def gen():
            yield StreamComplete(response=response)

        return gen()

    llm.plan_stream = plan_stream
    session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace))
    session.tool_registry.register_tool(ToolDef(
        name="read_source", description="read source", input_schema={"type": "object", "properties": {}},
        handler=read_source,
    ))
    try:
        with patch("anton.analytics.send_event"):
            async for _event in session.turn_stream("read the source"):
                pass
    finally:
        await session.close()

    assert len(model_inputs) == 2
    assert 'some-placeholder-value' in model_inputs[1]


def test_a_shared_replay_list_stays_correct_across_pads():
    """Pads built from one replayed list share it; scrubbing it again for each pad changes nothing more."""
    stored = Cell(code="fetch()", stdout=f'{{"access_token": "{TOKEN}"}}', stderr="", error=None)
    shared = [stored]

    first = LocalScratchpadRuntime(name="first", cells=shared, **_DEFAULTS)
    after_first = list(shared)
    second = LocalScratchpadRuntime(name="second", cells=shared, **_DEFAULTS)

    assert first.cells is shared and second.cells is shared
    assert shared == after_first
    assert shared[0].stdout == '{"access_token": "[REDACTED_TOKEN]"}'
    assert shared[0].code == "fetch()"
