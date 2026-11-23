"""Unit tests for native coding & cowork tools, secret/sandbox/write-gate
enforcement, and read_text_file policy boundaries.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from anton.core.backends.sandbox import CellSandbox
from anton.core.tools import file_access
from anton.core.tools.coding_tools import (
    handle_bash,
    handle_edit_file,
    handle_find_files,
    handle_grep_search,
    handle_write_file,
    is_protected_secret_path,
)
from anton.core.tools.tool_handlers import handle_read_text_file


def _make_session(root: Path, *, sandbox=None, write_gate=None):
    pads = SimpleNamespace(_cell_sandbox=sandbox, cell_sandbox=sandbox) if sandbox is not None else None
    return SimpleNamespace(
        _workspace=SimpleNamespace(base=root, artifacts_dir=root / ".anton" / "artifacts"),
        _settings=None,
        _scratchpads=pads,
        _write_gate=write_gate,
        _skill_drafts_root=None,
    )


def test_file_access_under_policy_unchanged_for_attachments():
    # Ensure file_access.under_policy still fences policy=None callers (e.g. attachments)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        outside = Path("/etc/hosts").resolve()
        reason = file_access.read_refusal(outside, workspace=root)
        assert reason == file_access.REFUSED_OUTSIDE
        assert file_access.under_policy(reason, outside, None, outside=file_access.REFUSED_OUTSIDE) == file_access.REFUSED_OUTSIDE


def test_secret_path_blocklist():
    assert is_protected_secret_path(Path("/work/.anton/.env"))
    assert is_protected_secret_path(Path("/work/.anton/.env.superseded"))
    assert is_protected_secret_path(Path("/work/.anton/data_vault/pg.json"))
    assert is_protected_secret_path(Path("/work/.anton/.published.json"))
    assert is_protected_secret_path(Path("/work/.anton/artifacts/app/.published.json"))
    assert is_protected_secret_path(Path("/work/.anton/secrets.json"))
    assert is_protected_secret_path(Path("/home/u/.cowork/.env"))
    assert is_protected_secret_path(Path("/home/u/.cowork-dev/.env"))
    assert is_protected_secret_path(Path("/home/u/.ssh/id_rsa"))
    assert is_protected_secret_path(Path("/home/u/.aws/credentials"))
    assert is_protected_secret_path(Path("/work/credentials.json"))
    assert is_protected_secret_path(Path("/work/.env"))
    assert is_protected_secret_path(Path("/work/.env.local"))
    assert not is_protected_secret_path(Path("/work/.env.example"))
    assert not is_protected_secret_path(Path("/work/.gitignore"))
    assert not is_protected_secret_path(Path("/work/.github/workflows/ci.yml"))


def test_coding_tools_crud_search_bash_and_secret_scrub():
    async def _run():
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            sess = _make_session(root)

            # write_file + edit_file + grep_search + find_files + read_text_file dotfile
            r = await handle_write_file(sess, {"path": "pkg/app.py", "content": "def add(a, b):\n    return a + b\n"})
            assert r.ok, r
            r = await handle_edit_file(sess, {"path": "pkg/app.py", "old_string": "return a + b", "new_string": "return a + b + 1"})
            assert r.ok and "+    return a + b + 1" in str(r.content), r
            r = await handle_grep_search(sess, {"pattern": "return a \\+ b \\+ 1", "path": "."})
            assert r.ok and "pkg/app.py:2:" in str(r.content), r
            r = await handle_find_files(sess, {"pattern": "*.py", "path": "."})
            assert r.ok and "pkg/app.py" in str(r.content), r

            # Dotfile allowed for read_text_file when unlocked
            (root / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
            r = await handle_read_text_file(sess, {"path": ".gitignore"})
            assert r.ok and "*.pyc" in str(r.content), r

            # Secrets strictly blocked across all tools (including rg --glob bypass attempts)
            (root / ".anton").mkdir()
            (root / ".anton" / ".env").write_text("OPENAI_API_KEY=sk-secret\n", encoding="utf-8")
            (root / ".env").write_text("DB_PASS=secret123\n", encoding="utf-8")
            (root / "credentials.json").write_text('{"secret":"topsecret_cred"}\n', encoding="utf-8")
            for p in (".anton/.env", ".env", "credentials.json"):
                assert not (await handle_read_text_file(sess, {"path": p})).ok
                assert not (await handle_write_file(sess, {"path": p, "content": "x"})).ok
                assert not (await handle_edit_file(sess, {"path": p, "old_string": "=", "new_string": ":"})).ok
                assert not (await handle_grep_search(sess, {"pattern": "secret", "path": p})).ok
            # Regression: glob override must not leak secrets via ripgrep
            for bad_glob in (".anton/.env", ".env", "credentials.json"):
                rg_res = await handle_grep_search(sess, {"pattern": "secret", "path": ".", "glob": bad_glob})
                assert "sk-secret" not in str(rg_res.content)
                assert "secret123" not in str(rg_res.content)
                assert "topsecret_cred" not in str(rg_res.content)

            # Regression: editing an artifact file surfaces artifact lint via cell_content_with_lint
            from unittest.mock import AsyncMock, patch
            from anton.core.tools.tool_handlers import ArtifactLint

            art_dir = root / ".anton" / "artifacts" / "demo"
            art_dir.mkdir(parents=True)
            (art_dir / "index.html").write_text("<html><body>ok</body></html>\n", encoding="utf-8")
            fake_store = SimpleNamespace(root=root / ".anton" / "artifacts")
            fake_lint = ArtifactLint(messages=("[artifact lint] demo/index.html: test warning",))
            with (
                patch("anton.core.tools.tool_handlers.resolve_artifact_store", return_value=fake_store),
                patch("anton.core.tools.tool_handlers.capture_turn_pages", new=AsyncMock()),
                patch("anton.core.tools.tool_handlers.snapshot_cell_artifacts", new=AsyncMock(return_value={})),
                patch("anton.core.tools.tool_handlers.check_cell_artifacts", new=AsyncMock(return_value=fake_lint)),
            ):
                er = await handle_edit_file(
                    sess,
                    {"path": ".anton/artifacts/demo/index.html", "old_string": "ok", "new_string": "broken"},
                )
                assert er.ok and "[artifact lint] demo/index.html: test warning" in str(er.content), er

            # bash strips provider secrets from env
            os.environ["ANTON_TEST_INTERNAL_SECRET"] = "must_not_leak"
            try:
                r = await handle_bash(sess, {"command": "echo VAL=${ANTON_TEST_INTERNAL_SECRET:-GONE}"})
                assert r.ok and "VAL=GONE" in str(r.content), r
            finally:
                os.environ.pop("ANTON_TEST_INTERNAL_SECRET", None)

            # bash timeout cleans up cleanly
            r = await handle_bash(sess, {"command": "sleep 5", "timeout_seconds": 1})
            assert not r.ok and r.reason == "timeout", r

    asyncio.run(_run())


def test_sandbox_and_write_gate_fail_closed():
    async def _run():
        with tempfile.TemporaryDirectory() as td:
            base = Path(td).resolve()
            root = base / "proj"
            other = base / "other"
            home = base / "home"
            tmp = base / "tmp"
            for d in (root, other, home, tmp):
                d.mkdir()
            (root / "ok.txt").write_text("hello\n", encoding="utf-8")
            (other / "secret.txt").write_text("other_tenant\n", encoding="utf-8")

            # WriteGate blocks write_file, edit_file, bash
            wg_sess = _make_session(root, write_gate=object())
            assert not (await handle_bash(wg_sess, {"command": "echo hi"})).ok
            assert not (await handle_write_file(wg_sess, {"path": "x.txt", "content": "x"})).ok
            assert not (await handle_edit_file(wg_sess, {"path": "ok.txt", "old_string": "hello", "new_string": "bye"})).ok

            # Real CellSandbox blocks write tools and blocks reads outside project_dir
            sandbox = CellSandbox(project_dir=root, home_dir=home, tmp_dir=tmp, user_name="testuser")
            sb_sess = _make_session(root, sandbox=sandbox)
            assert not (await handle_bash(sb_sess, {"command": "echo hi"})).ok
            assert not (await handle_write_file(sb_sess, {"path": "x.txt", "content": "x"})).ok
            assert not (await handle_edit_file(sb_sess, {"path": "ok.txt", "old_string": "hello", "new_string": "bye"})).ok
            assert not (await handle_grep_search(sb_sess, {"pattern": "other_tenant", "path": str(other)})).ok
            assert not (await handle_find_files(sb_sess, {"pattern": "*", "path": str(other)})).ok
            # Inside granted project_dir, read-only grep/find succeed
            assert (await handle_grep_search(sb_sess, {"pattern": "hello", "path": "."})).ok
            assert (await handle_find_files(sb_sess, {"pattern": "*.txt", "path": "."})).ok

    asyncio.run(_run())


def test_bash_daemon_does_not_hang_and_timeout_kills_hung_foreground():
    import re
    import signal
    import time

    async def _run():
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            sess = _make_session(root)

            # Background daemon WITHOUT stdout/stderr redirection genuinely inherits
            # the pipe write FD — handle_bash must still return promptly on shell exit
            # and leave the background process alive.
            t0 = time.monotonic()
            res = await handle_bash(
                sess,
                {
                    "command": f'cd "{root}" && nohup sleep 30 & echo "PID=$!"; sleep 0.2; echo "DONE"',
                    "timeout_seconds": 10,
                },
            )
            elapsed = time.monotonic() - t0
            assert res.ok, res.content
            assert elapsed < 3.0, f"handle_bash hung on inherited pipe FD ({elapsed:.2f}s)"
            assert "DONE" in res.content
            m = re.search(r"PID=(\d+)", res.content)
            assert m, res.content
            bg_pid = int(m.group(1))
            try:
                # Confirm background process is still alive (kill signal 0 succeeds)
                os.kill(bg_pid, 0)
            finally:
                try:
                    os.kill(bg_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            # Genuine hung foreground command must time out and clean up cleanly.
            t1 = time.monotonic()
            to_res = await handle_bash(sess, {"command": "sleep 30", "timeout_seconds": 1})
            to_elapsed = time.monotonic() - t1
            assert not to_res.ok and to_res.reason == "timeout", to_res
            assert to_elapsed < 7.0, f"timeout took too long ({to_elapsed:.2f}s)"

            # CancelledError during a long-running bash command must promptly kill the
            # child process group and not leak it.
            pid_file = root / "cancel.pid"
            task = asyncio.create_task(
                handle_bash(
                    sess,
                    {
                        "command": f'echo $$ > "{pid_file}"; exec sleep 30',
                        "timeout_seconds": 30,
                    },
                )
            )
            for _ in range(40):
                if pid_file.exists() and pid_file.read_text().strip():
                    break
                await asyncio.sleep(0.05)
            assert pid_file.exists(), "child never wrote pid file"
            cancel_pid = int(pid_file.read_text().strip())
            task.cancel()
            try:
                await task
                assert False, "expected CancelledError"
            except asyncio.CancelledError:
                pass
            # Poll briefly for kernel to reap killed process
            alive = True
            for _ in range(20):
                try:
                    os.kill(cancel_pid, 0)
                    await asyncio.sleep(0.05)
                except ProcessLookupError:
                    alive = False
                    break
            assert not alive, f"cancelled bash child {cancel_pid} survived cancellation"

    asyncio.run(_run())


def test_edit_file_atomic_multi_hunk_and_rollback():
    async def _run():
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            sess = _make_session(root)
            f = root / "app.py"
            original = "alpha = 1\nbeta = 2\ngamma = 3\n"
            f.write_text(original, encoding="utf-8")

            # Mutually exclusive args rejected (including top-level replace_all with edits)
            bad = await handle_edit_file(
                sess,
                {
                    "path": "app.py",
                    "old_string": "alpha = 1",
                    "new_string": "alpha = 10",
                    "edits": [{"old_string": "beta = 2", "new_string": "beta = 20"}],
                },
            )
            assert not bad.ok and bad.reason == "invalid_args"

            bad_ra = await handle_edit_file(
                sess,
                {
                    "path": "app.py",
                    "replace_all": True,
                    "edits": [{"old_string": "beta = 2", "new_string": "beta = 20"}],
                },
            )
            assert not bad_ra.ok and bad_ra.reason == "invalid_args"

            # 1-element edits array reports edits[0].old_string on mismatch
            single_err = await handle_edit_file(
                sess,
                {
                    "path": "app.py",
                    "edits": [{"old_string": "missing", "new_string": "x"}],
                },
            )
            assert not single_err.ok and "edits[0].old_string" in single_err.content

            # Atomic rollback: edit 0 matches, edit 1 fails -> file unchanged on disk
            rb = await handle_edit_file(
                sess,
                {
                    "path": "app.py",
                    "edits": [
                        {"old_string": "alpha = 1", "new_string": "alpha = 10"},
                        {"old_string": "nonexistent = 99", "new_string": "x = 0"},
                    ],
                },
            )
            assert not rb.ok and rb.reason == "old_string_not_found"
            assert "edits[1]" in rb.content
            assert "No changes were written" in rb.content
            assert f.read_text(encoding="utf-8") == original

            # Atomic success across multiple hunks in one call
            ok = await handle_edit_file(
                sess,
                {
                    "path": "app.py",
                    "edits": [
                        {"old_string": "alpha = 1", "new_string": "alpha = 10"},
                        {"old_string": "gamma = 3", "new_string": "gamma = 30"},
                    ],
                },
            )
            assert ok.ok, ok.content
            assert "2 replacements" in ok.content
            assert f.read_text(encoding="utf-8") == "alpha = 10\nbeta = 2\ngamma = 30\n"

            # Mismatch error returns actionable numbered context snippet
            (root / "ctx.py").write_text(
                "def foo():\n    x = 1\n    target_val = 42\n    y = 2\n    return target_val\n",
                encoding="utf-8",
            )
            miss = await handle_edit_file(
                sess,
                {
                    "path": "ctx.py",
                    "old_string": "  target_val = 42\n  y = 999",
                    "new_string": "  target_val = 43\n  y = 999",
                },
            )
            assert not miss.ok and miss.reason == "old_string_not_found"
            assert "Context around line 3:" in miss.content
            assert "3:     target_val = 42" in miss.content
            assert "4:     y = 2" in miss.content

    asyncio.run(_run())


def test_grep_search_context_lines_rg_and_python_walker():
    async def _run():
        with tempfile.TemporaryDirectory() as td:
            base = Path(td).resolve()
            root = base / "proj"
            home = base / "home"
            tmp = base / "tmp"
            for d in (root, home, tmp):
                d.mkdir()
            (root / "sample-1-file.py").write_text(
                "\n".join(f"line_{i:02d}" for i in range(1, 21)) + "\n",
                encoding="utf-8",
            )
            (root / ".env").write_text("SECRET_LINE=line_05\n", encoding="utf-8")

            # 1. Unlocked session (uses rg if installed)
            sess = _make_session(root)
            res = await handle_grep_search(
                sess,
                {"pattern": "line_(05|15)", "path": ".", "context_lines": 1},
            )
            assert res.ok, res.content
            assert "sample-1-file.py-4-line_04" in res.content
            assert "sample-1-file.py:5:line_05" in res.content
            assert "sample-1-file.py-6-line_06" in res.content
            assert "--" in res.content
            assert "sample-1-file.py:15:line_15" in res.content
            assert "SECRET_LINE" not in res.content

            # max_results=1 with context_lines=1 must not bleed into match 2's trailing context
            res_cap = await handle_grep_search(
                sess,
                {"pattern": "line_(05|07)", "path": ".", "context_lines": 1, "max_results": 1},
            )
            assert res_cap.ok, res_cap.content
            assert "line_05" in res_cap.content
            assert "line_06" in res_cap.content
            assert "line_07" not in res_cap.content
            assert "line_08" not in res_cap.content
            assert "1 more matches" in res_cap.content

            # 2. CellSandbox session (exercises pure-Python walker with context_lines)
            sandbox = CellSandbox(project_dir=root, home_dir=home, tmp_dir=tmp, user_name="testuser")
            sb_sess = _make_session(root, sandbox=sandbox)
            res_py = await handle_grep_search(
                sb_sess,
                {"pattern": "line_(05|15)", "path": ".", "context_lines": 1},
            )
            assert res_py.ok, res_py.content
            assert "sample-1-file.py-4-line_04" in res_py.content
            assert "sample-1-file.py:5:line_05" in res_py.content
            assert "sample-1-file.py-6-line_06" in res_py.content
            assert "--" in res_py.content
            assert "sample-1-file.py:15:line_15" in res_py.content
            assert "SECRET_LINE" not in res_py.content

    asyncio.run(_run())


def test_read_text_file_containment_and_mismatched_sandbox_dotfile():
    async def _run():
        with tempfile.TemporaryDirectory() as td:
            base = Path(td).resolve()
            root = base / "proj"
            other = base / "other"
            home = base / "home"
            tmp = base / "tmp"
            for d in (root, other, home, tmp):
                d.mkdir()
            (root / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
            (other / "outside.txt").write_text("outside\n", encoding="utf-8")
            (other / ".gitignore").write_text("other\n", encoding="utf-8")

            # Workspace containment holds for unlocked sessions
            sess = _make_session(root)
            assert (await handle_read_text_file(sess, {"path": ".gitignore"})).ok
            outside_res = await handle_read_text_file(sess, {"path": str(other / "outside.txt")})
            assert not outside_res.ok and outside_res.reason == "access_denied"
            assert file_access.REFUSED_OUTSIDE in str(outside_res.content)

            # CellSandbox with a narrower project_dir must still be the final gate on dotfiles
            sandbox = CellSandbox(project_dir=other, home_dir=home, tmp_dir=tmp, user_name="testuser")
            mismatched = _make_session(root, sandbox=sandbox)
            dot_res = await handle_read_text_file(mismatched, {"path": ".gitignore"})
            assert not dot_res.ok and dot_res.reason == "access_denied"
            assert file_access.REFUSED_OUTSIDE in str(dot_res.content)

    asyncio.run(_run())


def test_parallel_read_only_batching_order_and_concurrency(tmp_path):
    """ChatSession.turn_stream prefetches independent read-only tool calls concurrently
    while preserving exact tool_call -> tool_start -> tool_done event order."""
    import time
    from unittest.mock import AsyncMock, patch
    from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage
    from anton.core.session import ChatSession, ChatSessionConfig, _VerifierVerdict
    from anton.workspace import Workspace
    from tests.conftest import make_mock_llm

    async def _run():
        ws = Workspace(tmp_path / "ws")
        ws.initialize()
        (ws.base / "a.txt").write_text("alpha\n", encoding="utf-8")
        (ws.base / "b.txt").write_text("beta\n", encoding="utf-8")

        llm = make_mock_llm()
        turns = [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc_1", name="read_text_file", input={"path": "a.txt"}),
                    ToolCall(id="tc_2", name="read_text_file", input={"path": "b.txt"}),
                    ToolCall(id="tc_3", name="grep_search", input={"pattern": "alpha", "path": "."}),
                ],
                usage=Usage(input_tokens=100, output_tokens=50),
                stop_reason="tool_use",
            ),
            LLMResponse(
                content="done",
                tool_calls=[],
                usage=Usage(input_tokens=150, output_tokens=20),
                stop_reason="end_turn",
            ),
        ]

        sent_messages: list[list[dict]] = []

        def _plan_stream(**kwargs):
            import copy
            sent_messages.append(copy.deepcopy(kwargs.get("messages") or []))
            async def _gen():
                yield StreamComplete(response=turns.pop(0))
            return _gen()

        llm.usage_listener = None
        llm.plan_stream = _plan_stream
        session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=ws))
        session._jev_enabled = False
        session._llm.generate_object_code = AsyncMock(
            return_value=_VerifierVerdict(status="COMPLETE", reason="ok", close_to_done=False)
        )

        active = 0
        max_active = 0
        delays = {"tc_1": 0.22, "tc_2": 0.14, "tc_3": 0.04}
        orig_dispatch = session.tool_registry.dispatch_tool

        async def _slow_dispatch(sess_arg, name, tool_input, **kw):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(delays.get(kw.get("tool_call_id", ""), 0.1))
                return await orig_dispatch(sess_arg, name, tool_input, **kw)
            finally:
                active -= 1

        session.tool_registry.dispatch_tool = _slow_dispatch
        t0 = time.monotonic()
        try:
            with patch("anton.analytics.send_event"):
                events = [ev async for ev in session.turn_stream("inspect files")]
        finally:
            await session.close()
        elapsed = time.monotonic() - t0

        assert max_active >= 2, f"expected concurrent read-only dispatch, got max_active={max_active}"
        assert elapsed < 0.38, f"batch appeared serialized ({elapsed:.2f}s)"
        # Even though tc_3 finished first and tc_1 last, exact tool_result order is preserved
        assert len(sent_messages) == 2
        result_blocks = [
            b
            for m in sent_messages[1]
            if m.get("role") == "user" and isinstance(m.get("content"), list)
            for b in m["content"]
            if b.get("type") == "tool_result"
        ]
        assert [b["tool_use_id"] for b in result_blocks] == ["tc_1", "tc_2", "tc_3"]
        assert "alpha" in str(result_blocks[0]["content"])
        assert "beta" in str(result_blocks[1]["content"])
        assert "a.txt:1:alpha" in str(result_blocks[2]["content"])

    asyncio.run(_run())


def test_artifact_prompts_and_tool_never_hijack_user_specified_paths():
    from anton.core.llm.prompts import ARTIFACTS_PROMPT
    from anton.core.llm.static_artifact_prompt import ARTIFACTS, STATIC_ARTIFACTS
    from anton.core.tools.tool_defs import CREATE_ARTIFACT_TOOL, STATIC_CREATE_ARTIFACT_TOOL

    for text in (ARTIFACTS, STATIC_ARTIFACTS, ARTIFACTS_PROMPT):
        head = text[:550].lower()
        assert "specific workspace or filesystem path" in head
        assert "never call" in head
    for tool in (CREATE_ARTIFACT_TOOL, STATIC_CREATE_ARTIFACT_TOOL):
        head = tool.description[:400].lower()
        assert "no user-specified file path" in head


def test_workspace_dotconfig_allowed_while_secret_parity_holds():
    async def _run():
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            sess = _make_session(root)

            # Legitimate workspace .config/nextest.toml allowed across read/write/edit/grep
            (root / ".config").mkdir()
            (root / ".config" / "nextest.toml").write_text("[profile.default]\nretries = 1\n", encoding="utf-8")
            assert (await handle_read_text_file(sess, {"path": ".config/nextest.toml"})).ok
            assert (await handle_edit_file(sess, {"path": ".config/nextest.toml", "old_string": "retries = 1", "new_string": "retries = 2"})).ok
            g_ok = await handle_grep_search(sess, {"pattern": "retries = 2", "path": ".config"})
            assert g_ok.ok and "nextest.toml" in str(g_ok.content), g_ok

            # Credential .config subdirs & top-level secret files strictly blocked everywhere
            (root / ".config" / "gh").mkdir()
            (root / ".config" / "gh" / "hosts.yml").write_text("oauth_token: gh_secret_xyz\n", encoding="utf-8")
            (root / ".kube").mkdir()
            (root / ".kube" / "config").write_text("token: kube_secret_xyz\n", encoding="utf-8")
            (root / "secrets.json").write_text('{"tok":"top_secret_json_xyz"}\n', encoding="utf-8")
            (root / "id_rsa").write_text("PRIVATE KEY secret_rsa_xyz\n", encoding="utf-8")
            (root / ".git-credentials").write_text("https://u:git_cred_xyz@host\n", encoding="utf-8")
            for p in (".config/gh/hosts.yml", ".kube/config", "secrets.json", "id_rsa", ".git-credentials"):
                assert not (await handle_read_text_file(sess, {"path": p})).ok, p
                assert not (await handle_write_file(sess, {"path": p, "content": "x"})).ok, p
                assert not (await handle_edit_file(sess, {"path": p, "old_string": "x", "new_string": "y"})).ok, p
                assert not (await handle_grep_search(sess, {"pattern": "xyz", "path": p})).ok, p
            # Whole-workspace grep must not leak any of them
            g_all = await handle_grep_search(sess, {"pattern": "xyz", "path": "."})
            for leaked in ("gh_secret_xyz", "kube_secret_xyz", "top_secret_json_xyz", "secret_rsa_xyz", "git_cred_xyz"):
                assert leaked not in str(g_all.content)

    asyncio.run(_run())


def test_head_tail_stdout_truncation_and_orphan_scrub_safety(monkeypatch):
    from anton.core.backends import local as local_mod
    from anton.core.backends.base import Cell
    from anton.core.backends.local import _clip_stdout
    from anton.core.backends.wire import (
        HEAD_OUTPUT,
        MAX_OUTPUT,
        STDOUT_SCRUB_MARGIN,
        TAIL_OUTPUT,
        TRUNCATION_PREFIX,
    )
    from anton.core.utils.scratchpad import format_cell_result

    # format_cell_result keeps both head and tail on long output
    body = "HEAD_MARKER\n" + ("x" * 15_000) + "\nTAIL_SUMMARY: 3 failed, 12 passed"
    formatted = format_cell_result(Cell(code="x", stdout=body, stderr="", error=None))
    assert formatted.count(TRUNCATION_PREFIX) == 1
    assert "HEAD_MARKER" in formatted and "TAIL_SUMMARY: 3 failed, 12 passed" in formatted

    # Giant >4,096-char orphaned access_token fails closed on single-line and preserves tail on multi-line
    big_tok = "S" * 5_500
    full_single = ("A" * 20_000) + f'"access_token": "{big_tok}"' + ("B" * 3_532)
    clipped_single = _clip_stdout(
        full_single[: HEAD_OUTPUT + STDOUT_SCRUB_MARGIN],
        total=len(full_single),
        cwd_note="Note: cwd changed",
        raw_tail=full_single[-(TAIL_OUTPUT + STDOUT_SCRUB_MARGIN) :],
    )
    assert "S" * 50 not in clipped_single and "Note: cwd changed" in clipped_single

    full_big = (
        ("A" * 20_000)
        + f'"access_token": "{big_tok}"\nline1\nline2\nline3\n'
        + ("B" * 3_500)
        + "\nEND_OK"
    )
    clipped_big = _clip_stdout(
        full_big[: HEAD_OUTPUT + STDOUT_SCRUB_MARGIN],
        total=len(full_big),
        cwd_note="Note: cwd changed",
        raw_tail=full_big[-(TAIL_OUTPUT + STDOUT_SCRUB_MARGIN) :],
    )
    assert "S" * 50 not in clipped_big and "END_OK" in clipped_big

    # total <= len(raw): full stdout scrubbed with no cut edges -> no leak or duplication
    full_mid = ("A" * 1_000) + f'"access_token": "{big_tok}"' + ("B" * 3_532)
    assert MAX_OUTPUT < len(full_mid) <= HEAD_OUTPUT + STDOUT_SCRUB_MARGIN
    clipped_mid = _clip_stdout(
        full_mid, total=len(full_mid), cwd_note="", raw_tail=full_mid[-(TAIL_OUTPUT + STDOUT_SCRUB_MARGIN) :]
    )
    assert "S" * 50 not in clipped_mid and TRUNCATION_PREFIX not in clipped_mid

    # Multi-line orphaned key + short-secret expansion inside raw_tail left margin
    orig_scrub = local_mod.scrub_output
    monkeypatch.setattr(
        local_mod,
        "scrub_output",
        lambda s: orig_scrub(s.replace("TINY", "[REDACTED_SHORT_SECRET_VALUE_EXPANSION_0000]")),
    )
    exp_full = (
        ("H" * 20_000)
        + '"access_token": "'
        + ("Q" * 100)
        + ("TINY" * 50)
        + ("Z" * 4_300)
        + '"\nline_a\nline_b\nline_c\n'
        + ("C" * 3_500)
        + "\nEXP_END"
    )
    clipped_exp = _clip_stdout(
        exp_full[: HEAD_OUTPUT + STDOUT_SCRUB_MARGIN],
        total=len(exp_full),
        cwd_note="",
        raw_tail=exp_full[-(TAIL_OUTPUT + STDOUT_SCRUB_MARGIN) :],
    )
    assert "Z" * 100 not in clipped_exp and "EXP_END" in clipped_exp


def test_ghost_tool_prompt_gating():
    from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext
    from anton.core.llm.prompts import (
        ARTIFACTS_PROMPT,
        BACKEND_GENERATION_PROMPT,
        NO_DIRECT_REPORTS_ARTIFACTS_PROMPT,
        NO_DIRECT_REPORTS_STATIC_ARTIFACTS_PROMPT,
        STATIC_ARTIFACTS_PROMPT,
    )
    from anton.core.tools.edit_artifact import route_section_to_editor
    from anton.core.tools.tool_defs import CREATE_ARTIFACT_TOOL

    # Positive check that ENG-4207 edit_artifact SECTION_ROUTES fire on all ARTIFACTS_PROMPT variants
    expected_route = "when the user names it, call `edit_artifact` with that name as `slug` directly"
    for variant in (
        ARTIFACTS_PROMPT,
        STATIC_ARTIFACTS_PROMPT,
        NO_DIRECT_REPORTS_ARTIFACTS_PROMPT,
        NO_DIRECT_REPORTS_STATIC_ARTIFACTS_PROMPT,
    ):
        routed = route_section_to_editor(variant)
        assert expected_route in routed
        assert "Change it with `edit_artifact(slug, task)`" in routed

    store = SimpleNamespace(
        list_summaries=lambda: [
            {"label": "browser", "description": "Use browser tool"},
            {"label": "build-fullstack-backend", "description": "Fullstack skill"},
            {"label": "public-data-sources", "description": "Public data"},
        ]
    )
    builder = ChatSystemPromptBuilder()
    common = dict(
        conversation_started="2026-11-23",
        system_prompt_context=SystemPromptContext(),
        proactive_dashboards=False,
        output_dir="/tmp",
        skill_store=store,
        fullstack_offered=True,
    )
    # tool_defs=None preserves backward-compatible inclusion of all three
    p_none = builder.build(**common, tool_defs=None)
    assert BACKEND_GENERATION_PROMPT in p_none
    assert "- `browser`" in p_none
    assert "- `build-fullstack-backend`" in p_none
    assert "fullstack-stateless-app" in p_none

    # Bare coding toolset without artifact/backend/browser tools omits all three consistently
    minimal_tools = [SimpleNamespace(name="scratchpad", input_schema={"properties": {}}, prompt=None)]
    p_min = builder.build(**common, tool_defs=minimal_tools)
    assert BACKEND_GENERATION_PROMPT not in p_min
    assert "- `browser`" not in p_min
    assert "- `build-fullstack-backend`" not in p_min
    assert "fullstack-stateless-app" not in p_min
    assert "- `public-data-sources`" in p_min

    # cloud_turn-like toolset: fullstack-capable create_artifact without browser/generate_artifact/launch_backend
    # omits browser skill while keeping fullstack ARTIFACTS_PROMPT, BACKEND_GENERATION_PROMPT, and build-fullstack-backend
    cloud_tools = [
        SimpleNamespace(name="scratchpad", input_schema={"properties": {}}, prompt=None),
        CREATE_ARTIFACT_TOOL,
    ]
    p_cloud = builder.build(**common, tool_defs=cloud_tools)
    assert "- `browser`" not in p_cloud
    assert BACKEND_GENERATION_PROMPT in p_cloud
    assert "- `build-fullstack-backend`" in p_cloud
    assert "fullstack-stateless-app" in p_cloud

    # Static-only create_artifact (no fullstack enum values) omits fullstack guidance consistently
    static_create = SimpleNamespace(
        name="create_artifact",
        input_schema={"properties": {"type": {"enum": ["html-app", "document"]}}},
        prompt=None,
    )
    p_static = builder.build(**common, tool_defs=[static_create])
    assert BACKEND_GENERATION_PROMPT not in p_static
    assert "- `build-fullstack-backend`" not in p_static
    assert "fullstack-stateless-app" not in p_static

    # With browser + launch_backend registered: all are included
    full_tools = [
        SimpleNamespace(name="browser", input_schema={"properties": {}}, prompt=None),
        SimpleNamespace(name="launch_backend", input_schema={"properties": {}}, prompt=None),
    ]
    p_full = builder.build(**common, tool_defs=full_tools)
    assert BACKEND_GENERATION_PROMPT in p_full
    assert "- `browser`" in p_full
    assert "- `build-fullstack-backend`" in p_full


def test_read_image_exposed_in_cloud_allowlist_and_readonly_parallel():
    from anton.cloud_turn.session import CLOUD_TOOL_ALLOWLIST
    from anton.core.tools.coding_tools import READ_ONLY_TOOLS

    assert "read_image" in CLOUD_TOOL_ALLOWLIST
    assert "read_image" in READ_ONLY_TOOLS


def test_read_image_pre_stat_size_cap_and_non_image_precedence():
    from PIL import Image as _PILImage
    from anton.core.tools.tool_handlers import handle_read_image

    async def _run():
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            sess = _make_session(root)

            # Valid PNG loads cleanly with dimensions via single-open worker
            ok_png = root / "ok.png"
            _PILImage.new("RGB", (12, 8), color=(10, 20, 30)).save(ok_png, format="PNG")
            out_ok = await handle_read_image(sess, {"file_path": str(ok_png)})
            assert out_ok.ok and isinstance(out_ok.content, list)
            assert out_ok.content[0]["source"]["media_type"] == "image/png"
            assert "12x8" in out_ok.content[1]["text"]

            # Valid BMP transcodes cleanly to PNG off the event loop
            ok_bmp = root / "ok.bmp"
            _PILImage.new("RGB", (6, 4), color=(40, 50, 60)).save(ok_bmp, format="BMP")
            out_bmp = await handle_read_image(sess, {"file_path": str(ok_bmp)})
            assert out_bmp.ok and isinstance(out_bmp.content, list)
            assert out_bmp.content[0]["source"]["media_type"] == "image/png"
            assert "6x4" in out_bmp.content[1]["text"]

            # Oversized valid-header PNG rejected via fstat before reading full file
            big_png = root / "huge.png"
            with big_png.open("wb") as fh:
                fh.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
                fh.seek(5 * 1024 * 1024)
                fh.write(b"\x00")
            out_big = await handle_read_image(sess, {"file_path": str(big_png)})
            assert not out_big.ok and out_big.reason == "image_too_large"

            # Oversized BMP (>50 MB) rejected via fstat before reading full file
            big_bmp = root / "huge.bmp"
            with big_bmp.open("wb") as fh:
                fh.write(b"BM" + b"\x00" * 14)
                fh.seek(51 * 1024 * 1024)
                fh.write(b"\x00")
            out_big_bmp = await handle_read_image(sess, {"file_path": str(big_bmp)})
            assert not out_big_bmp.ok and out_big_bmp.reason == "image_too_large"

            # Huge non-image named .png still returns not_an_image (header sniff precedes size cap)
            fake_png = root / "fake.png"
            with fake_png.open("wb") as fh:
                fh.write(b"NOT_A_PNG_HEADER")
                fh.seek(5 * 1024 * 1024)
                fh.write(b"\x00")
            out_fake = await handle_read_image(sess, {"file_path": str(fake_png)})
            assert not out_fake.ok and out_fake.reason == "not_an_image"

    asyncio.run(_run())


def test_edit_file_compact_diff_cap_and_scrub_before_truncate():
    from anton.utils.datasources import (
        _DS_KNOWN_VARS,
        _DS_SECRET_VARS,
        _reset_registered_ds_vars,
        set_ds_env_values,
    )

    async def _run():
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            sess = _make_session(root)

            # 1. Large edit caps preview and reports exact line deltas
            f = root / "big.py"
            f.write_text("header = 1\nbody = 'old'\nfooter = 2\n", encoding="utf-8")
            huge_block = "\n".join(f"line_{i} = {'x' * 50}" for i in range(60))
            res = await handle_edit_file(
                sess,
                {"path": "big.py", "old_string": "body = 'old'", "new_string": huge_block},
            )
            assert res.ok, res.content
            text = str(res.content)
            assert "+60/-1 lines; total 62 lines" in text
            assert "diff preview capped" in text
            assert len(text) < 1_600

            # 2. Single-line secret straddling the 1,200-char boundary is scrubbed
            # BEFORE truncation so zero secret prefix leaks across the cut.
            secret_tok = "sk-proj-BOUNDARYSECRET1234567890abcdefghijKLMNOPQRST"
            pad = "a" * 1080
            f2 = root / "boundary.txt"
            f2.write_text("start\nPLACEHOLDER\nend\n", encoding="utf-8")
            res2 = await handle_edit_file(
                sess,
                {
                    "path": "boundary.txt",
                    "old_string": "PLACEHOLDER",
                    "new_string": f"{pad} {secret_tok}",
                },
            )
            assert res2.ok, res2.content
            out2 = str(res2.content)
            assert "sk-proj-BOUNDARY" not in out2
            assert "BOUNDARYSECRET" not in out2

            # 3. Multi-line secrets (registered DS_* PEM key + two-line Bearer)
            # are scrubbed from raw contents BEFORE unified_diff adds `+`/`-` prefixes.
            pem_secret = (
                "-----BEGIN PRIVATE KEY-----\n"
                "MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwMULTILINESECRET\n"
                "-----END PRIVATE KEY-----"
            )
            bearer_tok = "eyJhbGciOiJSUzI1NiJ9.MULTILINEBEARERPAYLOAD9876543210"
            prev_ds = {k: v for k, v in os.environ.items() if k.startswith("DS_")}
            try:
                _DS_SECRET_VARS.add("DS_SF_PROD__PRIVATE_KEY")
                _DS_KNOWN_VARS.add("DS_SF_PROD__PRIVATE_KEY")
                set_ds_env_values({"DS_SF_PROD__PRIVATE_KEY": pem_secret})

                f3 = root / "creds_note.txt"
                f3.write_text("config:\nSLOT\n", encoding="utf-8")
                res3 = await handle_edit_file(
                    sess,
                    {
                        "path": "creds_note.txt",
                        "old_string": "SLOT",
                        "new_string": f"{pem_secret}\nAuthorization: Bearer\n{bearer_tok}",
                    },
                )
                assert res3.ok, res3.content
                out3 = str(res3.content)
                assert "MULTILINESECRET" not in out3
                assert "MULTILINEBEARERPAYLOAD" not in out3
            finally:
                _reset_registered_ds_vars()
                for k in list(os.environ):
                    if k.startswith("DS_"):
                        del os.environ[k]
                os.environ.update(prev_ds)

            # 4. Lines starting with `--` or `++` (SQL comments, YAML `---`, C++ `++i`)
            # are counted accurately (header filtering uses index slicing, not prefix).
            f4 = root / "query.sql"
            f4.write_text("-- old comment\n--- old yaml sep\nSELECT 1;\n", encoding="utf-8")
            res4 = await handle_edit_file(
                sess,
                {
                    "path": "query.sql",
                    "old_string": "-- old comment\n--- old yaml sep",
                    "new_string": "++count;\n--- new yaml sep\n-- new comment",
                },
            )
            assert res4.ok, res4.content
            assert "+3/-2 lines; total 4 lines" in str(res4.content), res4.content

    asyncio.run(_run())


def test_edit_file_smart_mismatch_hint_and_scrubbing():
    from anton.utils.datasources import (
        _DS_KNOWN_VARS,
        _DS_SECRET_VARS,
        _reset_registered_ds_vars,
        set_ds_env_values,
    )

    async def _run():
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            sess = _make_session(root)

            # 1. Distinctive anchor skips trivial leading `{` / blank lines that
            # appear earlier in the file and centers context on the real target.
            src = (
                "func helper() {\n"
                "    return 1\n"
                "}\n"
                + "".join(f"// filler line {i}\n" for i in range(15))
                + "func computeScore() {\n"
                "    total := accumulateWeightedMetrics(items, threshold)\n"
                "    return total + 1\n"
                "}\n"
            )
            f1 = root / "score.go"
            f1.write_text(src, encoding="utf-8")
            miss1 = await handle_edit_file(
                sess,
                {
                    "path": "score.go",
                    "old_string": "\n{\n    total := accumulateWeightedMetrics(items, threshold)\n    return total + 999\n}",
                    "new_string": "replacement",
                },
            )
            assert not miss1.ok and miss1.reason == "old_string_not_found"
            out1 = str(miss1.content)
            assert "accumulateWeightedMetrics" in out1
            assert "return total + 1" in out1

            # 2. Whitespace/indentation mismatch is explicitly diagnosed.
            f2 = root / "indent.py"
            f2.write_text(
                "def outer():\n"
                "    if ready:\n"
                "        result = compute_final_value(alpha, beta)\n"
                "        return result\n",
                encoding="utf-8",
            )
            miss2 = await handle_edit_file(
                sess,
                {
                    "path": "indent.py",
                    "old_string": "  if ready:\n    result = compute_final_value(alpha, beta)\n    return result",
                    "new_string": "    return 0",
                },
            )
            assert not miss2.ok and miss2.reason == "old_string_not_found"
            out2 = str(miss2.content)
            assert "different whitespace/indentation" in out2
            assert "compute_final_value" in out2

            # 3. Single-line (straddling 200-char line cap) and multi-line secrets
            # (registered DS_* PEM + two-line Bearer) are scrubbed BEFORE slicing.
            tok = "sk-proj-MISMATCHBOUNDARYSECRET1234567890abcdefGHIJKLM"
            pem_secret = (
                "-----BEGIN PRIVATE KEY-----\n"
                "MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwMISMATCHPEMSECRET\n"
                "-----END PRIVATE KEY-----"
            )
            bearer_tok = "eyJhbGciOiJSUzI1NiJ9.MISMATCHBEARERPAYLOAD9876543210"
            prev_ds = {k: v for k, v in os.environ.items() if k.startswith("DS_")}
            try:
                _DS_SECRET_VARS.add("DS_SF_PROD__PRIVATE_KEY")
                _DS_KNOWN_VARS.add("DS_SF_PROD__PRIVATE_KEY")
                set_ds_env_values({"DS_SF_PROD__PRIVATE_KEY": pem_secret})

                f3 = root / "secret_ctx.txt"
                f3.write_text(
                    f"{pem_secret}\n"
                    f"Authorization: Bearer\n{bearer_tok}\n"
                    f"anchor_token_here = {'z' * 170} {tok}\n"
                    "tail_line_value = 42\n",
                    encoding="utf-8",
                )
                miss3 = await handle_edit_file(
                    sess,
                    {
                        "path": "secret_ctx.txt",
                        "old_string": "anchor_token_here\ntail_line_value = 999\n",
                        "new_string": "ok",
                    },
                )
                assert not miss3.ok and miss3.reason == "old_string_not_found"
                out3 = str(miss3.content)
                assert "MISMATCHBOUNDARY" not in out3
                assert "sk-proj-MISMATCH" not in out3
                assert "MISMATCHPEMSECRET" not in out3
                assert "MISMATCHBEARERPAYLOAD" not in out3
                assert "tail_line_value = 42" in out3
            finally:
                _reset_registered_ds_vars()
                for k in list(os.environ):
                    if k.startswith("DS_"):
                        del os.environ[k]
                os.environ.update(prev_ds)

    asyncio.run(_run())
