from __future__ import annotations

import asyncio
import os

import pytest

from anton.core.backends.base import Cell
from anton.core.backends.local import LocalScratchpadRuntime
from anton.core.backends.utils import compute_timeouts as _compute_timeouts
from anton.core.backends.manager import ScratchpadManager
from anton.core.backends.local import local_scratchpad_runtime_factory

# Alias for brevity in tests
Scratchpad = LocalScratchpadRuntime

_SCRATCHPAD_DEFAULTS = dict(
    coding_provider="anthropic",
    coding_model="",
    coding_api_key="",
    coding_base_url="",
)

_MANAGER_DEFAULTS = dict(
    runtime_factory=local_scratchpad_runtime_factory,
    **_SCRATCHPAD_DEFAULTS,
)


def make_scratchpad(name: str, **kwargs) -> LocalScratchpadRuntime:
    return Scratchpad(name=name, **{**_SCRATCHPAD_DEFAULTS, **kwargs})


def make_manager(**kwargs) -> ScratchpadManager:
    return ScratchpadManager(**{**_MANAGER_DEFAULTS, **kwargs})


class TestScratchpadBasicExecution:
    async def test_basic_execution(self):
        """print(42) should return '42' in stdout."""
        pad = make_scratchpad(name="test")
        await pad.start()
        try:
            cell = await pad.execute("print(42)")
            assert cell.stdout.strip() == "42"
            assert cell.error is None
        finally:
            await pad.close()

    async def test_state_persists(self):
        """Variable from cell 1 should be available in cell 2."""
        pad = make_scratchpad(name="test")
        await pad.start()
        try:
            await pad.execute("x = 123")
            cell = await pad.execute("print(x)")
            assert cell.stdout.strip() == "123"
            assert cell.error is None
        finally:
            await pad.close()

    async def test_error_captured_process_survives(self):
        """Exception doesn't kill process; next cell works."""
        pad = make_scratchpad(name="test")
        await pad.start()
        try:
            cell1 = await pad.execute("raise ValueError('boom')")
            assert cell1.error is not None
            assert "ValueError" in cell1.error
            assert "boom" in cell1.error

            # Process should still work
            cell2 = await pad.execute("print('alive')")
            assert cell2.stdout.strip() == "alive"
            assert cell2.error is None
        finally:
            await pad.close()

    async def test_imports_persist(self):
        """import json in cell 1, json.dumps(...) in cell 2."""
        pad = make_scratchpad(name="test")
        await pad.start()
        try:
            await pad.execute("import json")
            cell = await pad.execute('print(json.dumps({"a": 1}))')
            assert cell.stdout.strip() == '{"a": 1}'
            assert cell.error is None
        finally:
            await pad.close()


class TestScratchpadView:
    async def test_view_history(self):
        """view() should show all cells with outputs."""
        pad = make_scratchpad(name="test")
        await pad.start()
        try:
            await pad.execute("x = 10")
            await pad.execute("print(x + 5)")
            output = pad.view()
            assert "Cell 1" in output
            assert "Cell 2" in output
            assert "x = 10" in output
            assert "15" in output
        finally:
            await pad.close()

    async def test_view_empty(self):
        """view() on empty pad returns a message."""
        pad = make_scratchpad(name="empty")
        await pad.start()
        try:
            output = pad.view()
            assert "empty" in output.lower()
        finally:
            await pad.close()


class TestScratchpadReset:
    async def test_reset_clears_state(self):
        """Variables should be gone after reset."""
        pad = make_scratchpad(name="test")
        await pad.start()
        try:
            await pad.execute("x = 42")
            await pad.reset()
            cell = await pad.execute("print(x)")
            assert cell.error is not None
            assert "NameError" in cell.error
            # Cells list should only have the post-reset cell
            assert len(pad.cells) == 1
        finally:
            await pad.close()


class TestAutoResume:
    """ENG-1273: `resume()` restarts without discarding the snapshot, and
    `_auto_resume()`'s cap falls back to a real `reset()` instead of
    retrying `resume()` forever."""

    async def test_resume_preserves_namespace_reset_does_not(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        pad = make_scratchpad(
            name="resume-vs-reset", _venvs_base=tmp_path / "venvs", session_id="conv"
        )
        await pad.start()
        try:
            await pad.execute("kept = 'alive'")

            await pad.resume()
            cell = await pad.execute("print(kept)")
            assert cell.error is None, cell.error
            assert cell.stdout.strip() == "alive"

            await pad.reset()
            cell = await pad.execute("print('kept' in dir())")
            assert cell.stdout.strip() == "False"
        finally:
            await pad.cleanup()

    async def test_falls_back_to_reset_after_max_consecutive_deaths(self, monkeypatch):
        pad = make_scratchpad(name="cap-test")
        calls: list[str] = []

        async def _fake_resume():
            calls.append("resume")

        async def _fake_reset():
            calls.append("reset")

        monkeypatch.setattr(pad, "resume", _fake_resume)
        monkeypatch.setattr(pad, "reset", _fake_reset)

        cap = pad._MAX_CONSECUTIVE_AUTO_RESUMES
        for _ in range(cap):
            await pad._auto_resume()
        assert calls == ["resume"] * cap

        await pad._auto_resume()
        assert calls[-1] == "reset"

    async def test_a_successful_cell_resets_the_death_counter(self):
        pad = make_scratchpad(name="counter-reset")
        await pad.start()
        try:
            pad._consecutive_deaths = 1  # simulate one prior death this streak
            cell = await pad.execute("print('alive')")
            assert cell.error is None
            assert pad._consecutive_deaths == 0
        finally:
            await pad.close()

    async def test_manual_reset_also_clears_the_death_counter(self):
        pad = make_scratchpad(name="manual-reset-clears")
        await pad.start()
        try:
            pad._consecutive_deaths = 2
            await pad.reset()
            assert pad._consecutive_deaths == 0
        finally:
            await pad.cleanup()

    async def test_auto_resume_reports_the_underlying_error_when_unrecoverable(
        self, monkeypatch
    ):
        pad = make_scratchpad(name="unrecoverable")

        async def _broken_resume():
            raise RuntimeError("venv is gone")

        monkeypatch.setattr(pad, "resume", _broken_resume)
        ok = await pad._auto_resume()
        assert ok is False
        assert "venv is gone" in pad._last_resume_error


class TestScratchpadEdgeCases:
    async def test_timeout_kills_process(self, monkeypatch):
        """Long-running code triggers timeout."""
        monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "1")
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "1")
        pad = make_scratchpad(name="test")
        await pad.start()
        try:
            cell = await pad.execute("import time; time.sleep(60)")
            assert cell.error is not None
            assert "timed out" in cell.error.lower() or "inactivity" in cell.error.lower()
        finally:
            await pad.close()

    async def test_output_truncation(self):
        """stdout exceeding _MAX_OUTPUT is capped in the boot script."""
        pad = make_scratchpad(name="test")
        await pad.start()
        try:
            cell = await pad.execute("print('x' * 20000)")
            assert "truncated" in cell.stdout
            assert len(cell.stdout) < 20000
            assert cell.error is None
        finally:
            await pad.close()

    async def test_dead_process_detected(self):
        """If process is dead, execute reports it."""
        pad = make_scratchpad(name="test")
        await pad.start()
        # Kill the process manually
        pad._proc.kill()
        await pad._proc.wait()
        cell = await pad.execute("print(1)")
        assert cell.error is not None
        assert "not running" in cell.error.lower()
        await pad.close()

    async def test_stderr_captured(self):
        """stderr output is captured separately."""
        pad = make_scratchpad(name="test")
        await pad.start()
        try:
            cell = await pad.execute("import sys; sys.stderr.write('warn\\n')")
            assert "warn" in cell.stderr
        finally:
            await pad.close()


class TestScratchpadManager:
    async def test_get_or_create(self):
        """Auto-creates a scratchpad on first access."""
        mgr = make_manager()
        try:
            pad = await mgr.get_or_create("alpha")
            assert pad.name == "alpha"
            assert "alpha" in mgr.list_pads()

            # Second call returns the same pad
            pad2 = await mgr.get_or_create("alpha")
            assert pad2 is pad
        finally:
            await mgr.close_all()

    async def test_remove(self):
        """remove() kills and deletes the scratchpad."""
        mgr = make_manager()
        try:
            await mgr.get_or_create("beta")
            result = await mgr.remove("beta")
            assert "beta" in result
            assert "beta" not in mgr.list_pads()
        finally:
            await mgr.close_all()

    async def test_remove_nonexistent(self):
        """remove() on unknown name returns a message."""
        mgr = make_manager()
        result = await mgr.remove("nope")
        assert "nope" in result

    async def test_close_all(self):
        """close_all() cleans up everything."""
        mgr = make_manager()
        await mgr.get_or_create("a")
        await mgr.get_or_create("b")
        assert len(mgr.list_pads()) == 2
        await mgr.close_all()
        assert len(mgr.list_pads()) == 0

    async def test_close_all_does_not_restart_processes(self):
        """close_all() kills worker processes without restarting them.

        cancel_all_running() would leave _proc pointing to a new (orphan-prone)
        process. close_all() must leave _proc as None.
        """
        mgr = make_manager()
        pad = await mgr.get_or_create("test")
        try:
            await pad.execute("x = 1")
            assert pad._proc is not None, "process should be alive after execution"
        finally:
            await mgr.close_all()
        assert pad._proc is None, "close_all() must not restart the worker process"


class TestScratchpadRenderNotebook:
    async def test_render_notebook_basic(self):
        """Produces markdown with code blocks and output."""
        pad = make_scratchpad(name="main")
        await pad.start()
        try:
            await pad.execute("x = 1")
            await pad.execute("print(x + 1)")
            md = pad.render_notebook()
            assert "## Scratchpad: main (2 cells)" in md
            assert "### Cell 1" in md
            assert "```python" in md
            assert "x = 1" in md
            assert "**Output:**" in md
            assert "2" in md
        finally:
            await pad.close()

    async def test_render_notebook_empty(self):
        """Empty pad returns a message."""
        pad = make_scratchpad(name="empty")
        await pad.start()
        try:
            md = pad.render_notebook()
            assert "no cells" in md.lower()
        finally:
            await pad.close()

    async def test_render_notebook_skips_empty_cells(self):
        """Whitespace-only cells are filtered out."""
        pad = make_scratchpad(name="gaps")
        await pad.start()
        try:
            await pad.execute("print('a')")
            await pad.execute("   \n  ")
            await pad.execute("print('b')")
            md = pad.render_notebook()
            assert "(2 cells)" in md
            assert "Cell 2" not in md  # whitespace cell skipped
            assert "Cell 1" in md
            assert "Cell 3" in md
        finally:
            await pad.close()

    async def test_render_notebook_truncates_long_output(self):
        """Long stdout shows 'more lines' indicator."""
        pad = make_scratchpad(name="long")
        await pad.start()
        try:
            await pad.execute("for i in range(50): print(i)")
            md = pad.render_notebook()
            assert "more lines" in md
        finally:
            await pad.close()

    async def test_render_notebook_error_summary(self):
        """Only last traceback line shown, not full trace."""
        pad = make_scratchpad(name="err")
        await pad.start()
        try:
            await pad.execute("raise ValueError('boom')")
            md = pad.render_notebook()
            assert "**Error:**" in md
            assert "ValueError: boom" in md
            # Full traceback details should NOT be present
            assert "Traceback" not in md
        finally:
            await pad.close()

    async def test_render_notebook_hides_stderr_without_error(self):
        """Warnings (stderr only, no error) are filtered out of output sections."""
        pad = make_scratchpad(name="warn")
        await pad.start()
        try:
            await pad.execute("import sys; sys.stderr.write('some warning\\n')")
            md = pad.render_notebook()
            # stderr content should NOT appear as output
            assert "**Output:**" not in md
            assert "**Error:**" not in md
        finally:
            await pad.close()

    async def test_truncate_output_lines(self):
        """Respects line limit."""
        text = "\n".join(f"line {i}" for i in range(50))
        result = LocalScratchpadRuntime._truncate_output(text, max_lines=10)
        assert "line 0" in result
        assert "line 9" in result
        assert "line 10" not in result
        assert "(40 more lines)" in result

    async def test_truncate_output_chars(self):
        """Respects char limit."""
        text = "\n".join("x" * 80 for _ in range(5))
        result = LocalScratchpadRuntime._truncate_output(text, max_lines=100, max_chars=200)
        assert "(truncated)" in result
        assert len(result) < len(text)


class TestCellMetadata:
    async def test_cell_stores_description_and_estimated_time(self):
        """execute() should store description and estimated_time on the Cell."""
        pad = make_scratchpad(name="meta")
        await pad.start()
        try:
            cell = await pad.execute(
                "print('hi')",
                description="Say hello",
                estimated_time="1s",
            )
            assert cell.description == "Say hello"
            assert cell.estimated_time == "1s"
            assert cell.stdout.strip() == "hi"
        finally:
            await pad.close()

    async def test_cell_defaults_empty_metadata(self):
        """Without arguments, description and estimated_time default to empty."""
        pad = make_scratchpad(name="defaults")
        await pad.start()
        try:
            cell = await pad.execute("print(1)")
            assert cell.description == ""
            assert cell.estimated_time == ""
        finally:
            await pad.close()

    async def test_view_shows_description_in_header(self):
        """view() should include description in the cell header."""
        pad = make_scratchpad(name="view-desc")
        await pad.start()
        try:
            await pad.execute("print(1)", description="Count to one")
            output = pad.view()
            assert "--- Cell 1: Count to one ---" in output
        finally:
            await pad.close()

    async def test_view_without_description(self):
        """view() without description falls back to plain header."""
        pad = make_scratchpad(name="view-plain")
        await pad.start()
        try:
            await pad.execute("print(1)")
            output = pad.view()
            assert "--- Cell 1 ---" in output
        finally:
            await pad.close()

    async def test_render_notebook_shows_description(self):
        """render_notebook() should include description in markdown header."""
        pad = make_scratchpad(name="nb-desc")
        await pad.start()
        try:
            await pad.execute("print(1)", description="Count to one")
            md = pad.render_notebook()
            assert "### Cell 1 \u2014 Count to one" in md
        finally:
            await pad.close()

    async def test_render_notebook_without_description(self):
        """render_notebook() without description uses plain header."""
        pad = make_scratchpad(name="nb-plain")
        await pad.start()
        try:
            await pad.execute("print(1)")
            md = pad.render_notebook()
            assert "### Cell 1" in md
            assert "\u2014" not in md
        finally:
            await pad.close()


class TestScratchpadEnvironment:
    async def test_env_vars_accessible(self, monkeypatch):
        """Secrets from .anton/.env (in os.environ) are accessible in scratchpad."""
        monkeypatch.setenv("MY_TEST_SECRET", "s3cret_value")
        pad = make_scratchpad(name="env-test")
        await pad.start()
        try:
            cell = await pad.execute(
                "import os; print(os.environ.get('MY_TEST_SECRET', 'NOT_FOUND'))"
            )
            assert cell.stdout.strip() == "s3cret_value"
        finally:
            await pad.close()

    async def test_get_llm_available_when_model_set(self):
        """get_llm() should be injected when ANTON_SCRATCHPAD_MODEL is set."""
        pad = make_scratchpad(name="llm-test", coding_model="claude-test-model")
        await pad.start()
        try:
            cell = await pad.execute("llm = get_llm(); print(llm.model)")
            assert cell.stdout.strip() == "claude-test-model"
            assert cell.error is None
        finally:
            await pad.close()

    async def test_get_llm_not_available_without_model(self):
        """get_llm() should not be in namespace when no model is configured."""
        pad = make_scratchpad(name="no-llm")
        await pad.start()
        try:
            cell = await pad.execute("get_llm()")
            assert cell.error is not None
            assert "NameError" in cell.error
        finally:
            await pad.close()

    async def test_agentic_loop_available_when_model_set(self):
        """agentic_loop() should be injected alongside get_llm()."""
        pad = make_scratchpad(name="agentic-test", coding_model="claude-test-model")
        await pad.start()
        try:
            cell = await pad.execute("print(callable(agentic_loop))")
            assert cell.stdout.strip() == "True"
            assert cell.error is None
        finally:
            await pad.close()

    async def test_agentic_loop_not_available_without_model(self):
        """agentic_loop() should not be in namespace when no model is configured."""
        pad = make_scratchpad(name="no-agentic")
        await pad.start()
        try:
            cell = await pad.execute("agentic_loop()")
            assert cell.error is not None
            assert "NameError" in cell.error
        finally:
            await pad.close()

    async def test_web_search_available_when_model_set(self):
        """web_search() should be injected alongside get_llm()."""
        pad = make_scratchpad(name="websearch-test", coding_model="claude-test-model")
        await pad.start()
        try:
            cell = await pad.execute("print(callable(web_search))")
            assert cell.stdout.strip() == "True"
            assert cell.error is None
        finally:
            await pad.close()

    async def test_web_search_not_available_without_model(self):
        """web_search() should not be in namespace when no model is configured."""
        pad = make_scratchpad(name="no-websearch")
        await pad.start()
        try:
            cell = await pad.execute("web_search('anything')")
            assert cell.error is not None
            assert "NameError" in cell.error
        finally:
            await pad.close()

    async def test_generate_object_available_when_model_set(self):
        """generate_object() should be available on the LLM wrapper."""
        pad = make_scratchpad(name="genobj-test", coding_model="claude-test-model")
        await pad.start()
        try:
            cell = await pad.execute(
                "llm = get_llm(); print(hasattr(llm, 'generate_object') and callable(llm.generate_object))"
            )
            assert cell.stdout.strip() == "True"
            assert cell.error is None
        finally:
            await pad.close()

    async def test_api_key_bridged(self, monkeypatch):
        """ANTON_ANTHROPIC_API_KEY should be bridged to ANTHROPIC_API_KEY."""
        monkeypatch.setenv("ANTON_ANTHROPIC_API_KEY", "sk-ant-test-123")
        # Remove ANTHROPIC_API_KEY if set, to test the bridge
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        pad = make_scratchpad(name="key-test", coding_model="test-model")
        await pad.start()
        try:
            cell = await pad.execute(
                "import os; print(os.environ.get('ANTHROPIC_API_KEY', 'MISSING'))"
            )
            assert cell.stdout.strip() == "sk-ant-test-123"
        finally:
            await pad.close()


class TestScratchpadVenv:
    async def test_venv_created_on_start(self):
        """Venv directory should be created when the scratchpad starts."""
        pad = make_scratchpad(name="venv-test")
        await pad.start()
        try:
            assert pad._venv_dir is not None
            assert os.path.isdir(pad._venv_dir)
            assert pad._venv_python is not None
            assert os.path.isfile(pad._venv_python)
        finally:
            await pad.close()

    async def test_venv_persisted_on_close(self):
        """Venv directory should be preserved when the scratchpad is closed."""
        pad = make_scratchpad(name="venv-close")
        await pad.start()
        venv_dir = pad._venv_dir
        assert os.path.isdir(venv_dir)
        await pad.close()
        # Venv directory persists on disk
        assert os.path.isdir(venv_dir)
        # But internal pointers are cleared
        assert pad._venv_dir is None
        assert pad._venv_python is None
        # Cleanup
        import shutil
        shutil.rmtree(venv_dir, ignore_errors=True)

    async def test_venv_persists_across_reset(self):
        """Venv should survive a reset (only the process restarts)."""
        pad = make_scratchpad(name="venv-reset")
        await pad.start()
        venv_dir = pad._venv_dir
        try:
            await pad.reset()
            assert pad._venv_dir == venv_dir
            assert os.path.isdir(venv_dir)
        finally:
            await pad.close()

    async def test_subprocess_uses_venv_python(self):
        """The subprocess should run with the venv's Python executable."""
        pad = make_scratchpad(name="venv-exec")
        await pad.start()
        try:
            cell = await pad.execute("import sys; print(sys.executable)")
            assert cell.error is None
            assert pad._venv_dir in cell.stdout.strip()
        finally:
            await pad.close()

    async def test_system_packages_available(self):
        """System site-packages should be accessible (e.g. pydantic from parent env)."""
        pad = make_scratchpad(name="venv-syspkg")
        await pad.start()
        try:
            cell = await pad.execute("import pydantic; print(pydantic.__name__)")
            assert cell.error is None
            assert cell.stdout.strip() == "pydantic"
        finally:
            await pad.close()


class TestVenvPersistence:
    """Tests for persistent venv recycling across sessions."""

    async def test_venv_recycled_on_restart(self, tmp_path):
        """Close + reopen same name → packages remembered."""
        import shutil
        venvs_base = tmp_path / "venvs"
        pad = make_scratchpad(name="recycle", _venvs_base=venvs_base)
        await pad.start()
        await pad.install_packages(["cowsay"])
        venv_dir = pad._venv_dir
        await pad.close()

        # Venv persists on disk with requirements.txt
        assert os.path.isdir(venv_dir)
        req_path = os.path.join(venv_dir, "requirements.txt")
        assert os.path.isfile(req_path)
        with open(req_path) as f:
            assert "cowsay" in f.read()

        # Reopen — should recycle the existing venv
        pad2 = make_scratchpad(name="recycle", _venvs_base=venvs_base)
        await pad2.start()
        try:
            assert "cowsay" in pad2._installed_packages
            cell = await pad2.execute("import cowsay; print('ok')")
            assert cell.error is None
            assert cell.stdout.strip() == "ok"
        finally:
            await pad2.close()
            shutil.rmtree(venvs_base, ignore_errors=True)

    async def test_venv_nuked_on_version_mismatch(self, tmp_path, monkeypatch):
        """Wrong .python_version → recreates venv."""
        import shutil
        venvs_base = tmp_path / "venvs"
        pad = make_scratchpad(name="ver-mismatch", _venvs_base=venvs_base)
        await pad.start()
        venv_dir = pad._venv_dir
        await pad.close()

        # Tamper with the .python_version file
        ver_path = os.path.join(venv_dir, ".python_version")
        with open(ver_path, "w") as f:
            f.write("2.7\n")

        # Reopen — should detect mismatch, nuke, and recreate
        pad2 = make_scratchpad(name="ver-mismatch", _venvs_base=venvs_base)
        await pad2.start()
        try:
            assert pad2._venv_dir is not None
            # The new venv should have the correct version
            with open(os.path.join(pad2._venv_dir, ".python_version")) as f:
                saved = f.read().strip()
            import sys as _sys
            assert saved == f"{_sys.version_info.major}.{_sys.version_info.minor}"
        finally:
            await pad2.close()
            shutil.rmtree(venvs_base, ignore_errors=True)

    async def test_venv_nuked_on_corruption(self, tmp_path):
        """Delete Python binary → recreates venv."""
        import shutil
        venvs_base = tmp_path / "venvs"
        pad = make_scratchpad(name="corrupt", _venvs_base=venvs_base)
        await pad.start()
        venv_dir = pad._venv_dir
        python_path = pad._venv_python
        await pad.close()

        # Delete the Python binary to simulate corruption
        os.remove(python_path)

        # Reopen — should detect corruption, nuke, and recreate
        pad2 = make_scratchpad(name="corrupt", _venvs_base=venvs_base)
        await pad2.start()
        try:
            assert pad2._venv_dir is not None
            assert pad2._venv_python is not None
            assert os.path.isfile(pad2._venv_python)
            cell = await pad2.execute("print('alive')")
            assert cell.error is None
            assert cell.stdout.strip() == "alive"
        finally:
            await pad2.close()
            shutil.rmtree(venvs_base, ignore_errors=True)

    async def test_remove_deletes_persistent_venv(self, tmp_path):
        """ScratchpadManager.remove() fully deletes the persistent venv dir."""
        import shutil
        mgr = make_manager(workspace_path=tmp_path)
        try:
            pad = await mgr.get_or_create("deleteme")
            venv_dir = pad._venv_dir
            assert os.path.isdir(venv_dir)
            await mgr.remove("deleteme")
            assert not os.path.exists(venv_dir)
        finally:
            await mgr.close_all()
            shutil.rmtree(tmp_path / ".anton", ignore_errors=True)

    async def test_requirements_saved_on_close(self, tmp_path):
        """requirements.txt is written when pad has installed packages."""
        import shutil
        venvs_base = tmp_path / "venvs"
        pad = make_scratchpad(name="req-save", _venvs_base=venvs_base)
        await pad.start()
        await pad.install_packages(["cowsay"])
        await pad.close()

        req_path = os.path.join(str(venvs_base / "req-save"), "requirements.txt")
        assert os.path.isfile(req_path)
        with open(req_path) as f:
            contents = f.read()
        assert "cowsay" in contents
        shutil.rmtree(venvs_base, ignore_errors=True)


class TestScratchpadInstall:
    async def test_install_packages_success(self):
        """install_packages should install a package into the venv."""
        pad = make_scratchpad(name="install-test")
        await pad.start()
        try:
            result = await pad.install_packages(["cowsay"])
            assert "cowsay" in result.lower() or "already satisfied" in result.lower() or "already installed" in result.lower()
            # Verify the package is importable
            cell = await pad.execute("import cowsay; print('ok')")
            assert cell.error is None
            assert cell.stdout.strip() == "ok"
        finally:
            await pad.close()

    async def test_install_empty_list(self):
        """install_packages with empty list returns a message."""
        pad = make_scratchpad(name="install-empty")
        await pad.start()
        try:
            result = await pad.install_packages([])
            assert "no packages" in result.lower()
        finally:
            await pad.close()

    async def test_install_invalid_package(self):
        """install_packages with a bogus name should report failure."""
        pad = make_scratchpad(name="install-bad")
        await pad.start()
        try:
            result = await pad.install_packages(["this-package-does-not-exist-xyz123"])
            assert "failed" in result.lower() or "error" in result.lower()
        finally:
            await pad.close()

    async def test_install_survives_reset(self):
        """Packages installed before a reset should still be available after."""
        pad = make_scratchpad(name="install-reset")
        await pad.start()
        try:
            await pad.install_packages(["cowsay"])
            await pad.reset()
            cell = await pad.execute("import cowsay; print('ok')")
            assert cell.error is None
            assert cell.stdout.strip() == "ok"
        finally:
            await pad.close()


class TestProgressAndTimeouts:
    async def test_progress_function_available_in_namespace(self):
        """progress() should be callable in scratchpad code."""
        pad = make_scratchpad(name="progress-ns")
        await pad.start()
        try:
            cell = await pad.execute("print(callable(progress))")
            assert cell.error is None
            assert cell.stdout.strip() == "True"
        finally:
            await pad.close()

    async def test_progress_resets_inactivity_timeout(self, monkeypatch):
        """Code that calls progress() frequently should survive even with a short inactivity timeout."""
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "2")
        monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "10")
        pad = make_scratchpad(name="progress-keep-alive")
        await pad.start()
        try:
            code = (
                "import time\n"
                "for i in range(3):\n"
                "    progress(f'step {i}')\n"
                "    time.sleep(1)\n"
                "print('done')\n"
            )
            cell = await pad.execute(code)
            assert cell.error is None
            assert cell.stdout.strip() == "done"
        finally:
            await pad.close()

    async def test_silence_kill_names_liveness(self, monkeypatch):
        """A worker that sends no liveness signal is killed on the silence window.

        Since ENG-578 the runtime heartbeats on a working cell's behalf, so a
        plain sleep survives; disabling the heartbeat simulates a dead/wedged
        worker and must still be killed — with the liveness wording, not the
        old "no output" story.
        """
        monkeypatch.setenv("ANTON_CELL_INACTIVITY_TIMEOUT", "2")
        monkeypatch.setenv("ANTON_CELL_TIMEOUT_DEFAULT", "60")
        monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0")
        pad = make_scratchpad(name="no-progress")
        await pad.start()
        try:
            cell = await pad.execute("import time; time.sleep(30)")
            assert cell.error is not None
            assert "liveness" in cell.error.lower()
        finally:
            await pad.close()

    async def test_execute_streaming_yields_progress(self):
        """execute_streaming() should yield progress strings and a final Cell."""
        pad = make_scratchpad(name="streaming")
        await pad.start()
        try:
            code = (
                "progress('hello')\n"
                "progress('world')\n"
                "print('result')\n"
            )
            items = []
            async for item in pad.execute_streaming(code):
                items.append(item)

            # Should have at least 2 progress strings and 1 Cell
            progress_items = [i for i in items if isinstance(i, str)]
            cell_items = [i for i in items if isinstance(i, Cell)]
            assert len(progress_items) >= 2
            assert "hello" in progress_items[0]
            assert "world" in progress_items[1]
            assert len(cell_items) == 1
            assert cell_items[0].stdout.strip() == "result"
            assert cell_items[0].error is None
        finally:
            await pad.close()

    async def test_compute_timeouts_no_estimate(self):
        """No estimate should use defaults."""
        from anton.core.backends.utils import compute_timeouts as _compute_timeouts
        total, inactivity = _compute_timeouts(0)
        assert total == 120.0
        assert inactivity == 30.0

    async def test_compute_timeouts_with_estimate(self):
        """Estimate scales the total with no cap; inactivity is clamped to cell_inactivity_max (default 60)."""
        from anton.core.backends.utils import compute_timeouts as _compute_timeouts

        # Small estimate: max(10*2, 10+30) = max(20, 40) = 40
        total, inactivity = _compute_timeouts(10)
        assert total == 40.0
        assert inactivity == 30.0  # max(5, 30) = 30, under the cap

        # Medium estimate: max(60*2, 60+30) = max(120, 90) = 120
        total, inactivity = _compute_timeouts(60)
        assert total == 120.0
        assert inactivity == 30.0  # max(30, 30) = 30, under the cap

        # Large estimate: total still scales, inactivity is capped at 60
        total, inactivity = _compute_timeouts(300)
        assert total == 600.0
        assert inactivity == 60.0  # min(max(150, 30), 60) = 60

        # Very large estimate: total keeps scaling so long-but-active cells
        # can run; the silence window stays capped.
        total, inactivity = _compute_timeouts(1000)
        assert total == 2000.0
        assert inactivity == 60.0  # min(max(500, 30), 60) = 60

    async def test_compute_timeouts_inactivity_cap_is_configurable(self):
        """cell_inactivity_max bounds the silence window regardless of estimate."""
        from anton.core.backends import utils as _utils
        from anton.core.settings import CoreSettings

        # est=300 would scale inactivity to 150s without the cap; with the
        # default cap (60) it is clamped, and the cap is tunable via settings.
        total, inactivity = _utils.compute_timeouts(300)
        assert inactivity == float(CoreSettings().cell_inactivity_max)
        assert total == 600.0  # total is intentionally left uncapped

    async def test_compute_timeouts_total_max_capped_by_default(self):
        """cell_total_max defaults to 1h. Since the liveness heartbeat keeps
        deliberately-quiet cells alive (ENG-578), a deadlock or infinite loop
        beats like a working cell — this ceiling is the only bound that ends
        it, so it must be on out of the box (and generous enough for a
        throttled batch campaign)."""
        from anton.core.settings import CoreSettings
        assert CoreSettings().cell_total_max == 3600

    async def test_compute_timeouts_total_max_backstop(self, monkeypatch):
        """When set, cell_total_max bounds the total; inactivity stays capped."""
        from anton.core.backends.utils import compute_timeouts as _compute_timeouts
        monkeypatch.setenv("ANTON_CELL_TOTAL_MAX", "300")
        total, inactivity = _compute_timeouts(1000)
        assert total == 300.0  # min(2000, 300)
        assert inactivity == 60.0


class TestSampleFunction:
    async def test_sample_available_in_namespace(self):
        """sample() should be callable in scratchpad code."""
        pad = make_scratchpad(name="sample-ns")
        await pad.start()
        try:
            cell = await pad.execute("print(callable(sample))")
            assert cell.error is None
            assert cell.stdout.strip() == "True"
        finally:
            await pad.close()

    async def test_sample_dict_preview(self):
        """sample() on a dict should show keys and truncated values."""
        pad = make_scratchpad(name="sample-dict")
        await pad.start()
        try:
            cell = await pad.execute(
                "d = {'name': 'Alice', 'age': 30, 'city': 'NYC'}\n"
                "sample(d)"
            )
            assert cell.error is None
            assert "[sample:dict]" in cell.stdout
            assert "Keys (3)" in cell.stdout
            assert "'name'" in cell.stdout
            assert "'Alice'" in cell.stdout
        finally:
            await pad.close()

    async def test_sample_list_preview(self):
        """sample() on a list should show length and first/last items."""
        pad = make_scratchpad(name="sample-list")
        await pad.start()
        try:
            cell = await pad.execute(
                "data = list(range(100))\n"
                "sample(data)"
            )
            assert cell.error is None
            assert "[sample:list]" in cell.stdout
            assert "Length: 100" in cell.stdout
            assert "[0]" in cell.stdout
            assert "95 more" in cell.stdout
        finally:
            await pad.close()

    async def test_sample_string_preview(self):
        """sample() on a string should show length and a preview."""
        pad = make_scratchpad(name="sample-str")
        await pad.start()
        try:
            cell = await pad.execute(
                "s = 'hello world' * 100\n"
                "sample(s)"
            )
            assert cell.error is None
            assert "[sample:str]" in cell.stdout
            assert "Length: 1100" in cell.stdout
            assert "hello world" in cell.stdout
        finally:
            await pad.close()

    async def test_sample_full_mode(self):
        """sample(var, mode='full') should show more content."""
        pad = make_scratchpad(name="sample-full")
        await pad.start()
        try:
            cell = await pad.execute(
                "d = {f'key_{i}': i for i in range(20)}\n"
                "sample(d, mode='full')"
            )
            assert cell.error is None
            # Full mode uses json.dumps for dicts
            assert '"key_0"' in cell.stdout
            assert '"key_19"' in cell.stdout
        finally:
            await pad.close()

    async def test_sample_set(self):
        """sample() on a set should show length and items."""
        pad = make_scratchpad(name="sample-set")
        await pad.start()
        try:
            cell = await pad.execute(
                "s = {1, 2, 3, 4, 5}\n"
                "sample(s)"
            )
            assert cell.error is None
            assert "[sample:set]" in cell.stdout
            assert "Length: 5" in cell.stdout
        finally:
            await pad.close()

    async def test_sample_custom_object(self):
        """sample() on an unknown object should show type and repr."""
        pad = make_scratchpad(name="sample-obj")
        await pad.start()
        try:
            cell = await pad.execute(
                "class Foo:\n"
                "    def __init__(self): self.x = 42\n"
                "    def __repr__(self): return 'Foo(x=42)'\n"
                "sample(Foo())"
            )
            assert cell.error is None
            assert "[sample:Foo]" in cell.stdout
            assert "Foo(x=42)" in cell.stdout
        finally:
            await pad.close()

    async def test_sample_bytes(self):
        """sample() on bytes should show length and preview."""
        pad = make_scratchpad(name="sample-bytes")
        await pad.start()
        try:
            cell = await pad.execute("sample(b'hello world')")
            assert cell.error is None
            assert "[sample:bytes]" in cell.stdout
            assert "Length: 11 bytes" in cell.stdout
        finally:
            await pad.close()

    async def test_sample_named(self):
        """sample() with _name parameter should include the label."""
        pad = make_scratchpad(name="sample-named")
        await pad.start()
        try:
            cell = await pad.execute(
                "x = [1, 2, 3]\n"
                "sample(x, _name='my_list')"
            )
            assert cell.error is None
            assert "my_list" in cell.stdout
            assert "list" in cell.stdout
        finally:
            await pad.close()

    async def test_sample_empty_dict(self):
        """sample() on an empty dict should not crash."""
        pad = make_scratchpad(name="sample-empty")
        await pad.start()
        try:
            cell = await pad.execute("sample({})")
            assert cell.error is None
            assert "Keys (0)" in cell.stdout
        finally:
            await pad.close()

    async def test_sample_empty_list(self):
        """sample() on an empty list should not crash."""
        pad = make_scratchpad(name="sample-empty-list")
        await pad.start()
        try:
            cell = await pad.execute("sample([])")
            assert cell.error is None
            assert "Length: 0" in cell.stdout
        finally:
            await pad.close()


class TestSessionPersistence:
    """ENG-1124 — the namespace must survive the pad process being replaced.

    In the product a new `ChatSession` (and so a new `ScratchpadManager`) is built for
    every user turn, which spawns a brand-new interpreter for the same pad name. Closing
    the pad and reopening it under the same name is that turn boundary.
    """

    async def test_namespace_survives_restart(self, tmp_path, monkeypatch):
        """Variables, imports and agent-defined functions survive a pad restart."""
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        venvs_base = tmp_path / "venvs"
        pad = make_scratchpad(name="persist", _venvs_base=venvs_base, session_id="c")
        await pad.start()
        cell = await pad.execute(
            "import json\n"
            "master_df = {'a': [1, 2, 3]}\n"
            "def style_title(x):\n"
            "    return f'<b>{x}</b>'\n"
            "print('built')\n"
        )
        assert cell.error is None
        await pad.close()

        pad2 = make_scratchpad(name="persist", _venvs_base=venvs_base, session_id="c")
        await pad2.start()
        try:
            # This is the exact shape that failed for the reporting customer:
            # `NameError: name 'master_df' is not defined` on the first cell of the
            # next turn, which forced a full rebuild.
            cell = await pad2.execute(
                "print(len(master_df['a']), style_title('x'), json.dumps([1]))"
            )
            assert cell.error is None, cell.error
            assert cell.stdout.strip() == "3 <b>x</b> [1]"
        finally:
            await pad2.cleanup()

    async def test_namespace_isolated_per_pad_name(self, tmp_path, monkeypatch):
        """A different pad name must not see another pad's namespace."""
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        venvs_base = tmp_path / "venvs"
        pad = make_scratchpad(name="pad-one", _venvs_base=venvs_base, session_id="c")
        await pad.start()
        await pad.execute("secret = 'one'")
        await pad.close()

        other = make_scratchpad(name="pad-two", _venvs_base=venvs_base, session_id="c")
        await other.start()
        try:
            cell = await other.execute("print('secret' in dir())")
            assert cell.stdout.strip() == "False"
        finally:
            await other.cleanup()

    async def test_oversized_namespace_is_skipped_and_reported(self, tmp_path, monkeypatch):
        """Over the cap we skip the write and say so on `logs` — never on `error`.

        `error` would feed the consecutive-error circuit breaker and the resilience
        nudge, turning a snapshot problem into an apparent cell failure.
        """
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        monkeypatch.setenv("ANTON_SCRATCHPAD_SESSION_MAX_BYTES", "2048")
        venvs_base = tmp_path / "venvs"
        pad = make_scratchpad(name="too-big", _venvs_base=venvs_base, session_id="c")
        await pad.start()
        try:
            cell = await pad.execute("blob = 'x' * 200000\nprint('made')")
            # Reported, and NEVER on `error` — that would feed the circuit breaker.
            assert cell.error is None
            assert "NOT saved" in cell.logs and "cap" in cell.logs
        finally:
            await pad.cleanup()

    async def test_no_session_path_reports_instead_of_pretending(self, tmp_path, monkeypatch):
        """Persistence on but no path → say so, don't silently no-op (the ENG-1124 bug)."""
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        venvs_base = tmp_path / "venvs"
        pad = make_scratchpad(name="nopath", _venvs_base=venvs_base, session_id="c")
        # Simulate an unwritable snapshot dir — start() then leaves the env var unset.
        monkeypatch.setattr(pad, "_session_snapshot_path", lambda **kw: None)
        await pad.start()
        try:
            cell = await pad.execute("print('ran')")
            assert cell.error is None
            assert cell.stdout.strip() == "ran"
            assert "ANTON_SCRATCHPAD_SESSION_PATH is unset" in cell.logs
        finally:
            await pad.cleanup()

    async def test_namespace_isolated_per_session_id(self, tmp_path, monkeypatch):
        """Same pad name in two conversations must not share state.

        Without conversation scoping, two conversations in one workspace that both use
        the pad name the agent likes (`main`, `analysis`, …) would read each other's
        variables — wrong, and a confidentiality problem when a datasource was only
        enabled in one of them.
        """
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        venvs_base = tmp_path / "venvs"
        first = make_scratchpad(name="main", _venvs_base=venvs_base, session_id="conv-a")
        await first.start()
        await first.execute("secret = 'from-conv-a'")
        await first.close()

        second = make_scratchpad(name="main", _venvs_base=venvs_base, session_id="conv-b")
        await second.start()
        try:
            cell = await second.execute("print('secret' in dir())")
            assert cell.stdout.strip() == "False"
        finally:
            await second.cleanup()

        # ...and conversation A still has its own state on a later turn.
        again = make_scratchpad(name="main", _venvs_base=venvs_base, session_id="conv-a")
        await again.start()
        try:
            cell = await again.execute("print(secret)")
            assert cell.error is None, cell.error
            assert cell.stdout.strip() == "from-conv-a"
        finally:
            await again.cleanup()

    def test_snapshot_path_contains_a_traversing_pad_name(self, tmp_path):
        """A model-chosen pad name flows into a path, so it must not escape the root."""
        pad = make_scratchpad(
            name="../../etc/passwd", _venvs_base=tmp_path / "venvs", session_id="conv"
        )
        path = pad._session_snapshot_path()
        assert path is not None
        root = (tmp_path / "venvs").parent / "scratchpad-sessions"
        assert str(path).startswith(str(root.resolve()))
        assert ".." not in path.parts

    async def test_reset_still_clears_state(self, tmp_path, monkeypatch):
        """`reset` is documented as "clearing all state" — persistence must not defeat it.

        Regression guard: with a snapshot on disk, restarting the process would reload it
        and `reset` would silently stop resetting anything.
        """
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        pad = make_scratchpad(
            name="resettable", _venvs_base=tmp_path / "venvs", session_id="conv"
        )
        await pad.start()
        try:
            await pad.execute("kept = 'before reset'")
            cell = await pad.execute("print(kept)")
            assert cell.stdout.strip() == "before reset"

            await pad.reset()

            cell = await pad.execute("print('kept' in dir())")
            assert cell.stdout.strip() == "False"
        finally:
            await pad.cleanup()

    async def test_cleanup_removes_the_snapshot(self, tmp_path, monkeypatch):
        """`remove` deletes the venv; it must not leave the namespace pickle behind."""
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        pad = make_scratchpad(
            name="disposable", _venvs_base=tmp_path / "venvs", session_id="conv"
        )
        await pad.start()
        await pad.execute("x = 1")
        snapshot = pad._session_snapshot_path()
        assert snapshot is not None and snapshot.exists()
        await pad.cleanup()
        assert not snapshot.exists()

    async def test_agent_variable_shadowing_a_helper_name_survives(self, tmp_path, monkeypatch):
        """Adversarial: `sample` is both an injected helper and a plausible variable.

        Excluding helpers from the snapshot *by name* silently destroyed the agent's
        data — `sample = df.sample(100)` came back as `<function sample>` on the next
        turn, with no error. Two things are needed: exclude helpers by identity (so a
        rebound name is treated as data), and inject with `setdefault` (because the
        injections run after the snapshot is loaded and would otherwise clobber it).
        """
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        venvs_base = tmp_path / "venvs"
        pad = make_scratchpad(name="shadow", _venvs_base=venvs_base, session_id="c")
        await pad.start()
        await pad.execute("sample = [1, 2, 3]\nprogress = 0.5")
        await pad.close()

        pad2 = make_scratchpad(name="shadow", _venvs_base=venvs_base, session_id="c")
        await pad2.start()
        try:
            cell = await pad2.execute("print(sample, progress)")
            assert cell.error is None, cell.error
            assert cell.stdout.strip() == "[1, 2, 3] 0.5"
        finally:
            await pad2.cleanup()

    async def test_helpers_still_injected_for_a_pad_that_never_rebound_them(
        self, tmp_path, monkeypatch
    ):
        """The other direction: `setdefault` must not stop the helpers being injected."""
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        pad = make_scratchpad(name="fresh", _venvs_base=tmp_path / "venvs", session_id="c")
        await pad.start()
        try:
            cell = await pad.execute("print(callable(sample), callable(progress))")
            assert cell.error is None, cell.error
            assert cell.stdout.strip() == "True True"
        finally:
            await pad.cleanup()

    async def test_pad_names_that_sanitise_alike_do_not_share_a_snapshot(
        self, tmp_path, monkeypatch
    ):
        """Adversarial: `'my pad'` and `'my_pad'` both sanitise to `my_pad`.

        Without a digest in the filename they shared one snapshot, so one pad loaded
        the other's namespace — the same cross-contamination the per-conversation
        scoping exists to prevent, just within a conversation.
        """
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        venvs_base = tmp_path / "venvs"
        a = make_scratchpad(name="my pad", _venvs_base=venvs_base, session_id="c")
        await a.start()
        await a.execute("who = 'space'")
        await a.close()

        b = make_scratchpad(name="my_pad", _venvs_base=venvs_base, session_id="c")
        await b.start()
        try:
            cell = await b.execute("print('who' in dir())")
            assert cell.stdout.strip() == "False", "distinct pads must not share a snapshot"
        finally:
            await b.cleanup()

    async def test_oversized_snapshot_aborts_the_write_instead_of_finishing_it(
        self, tmp_path, monkeypatch
    ):
        """The cap must bound the COST, not just what we keep.

        Writing the whole pickle and then deleting it meant a huge namespace paid a
        full serialise + write on every cell — the exact cost the cap exists to avoid.
        """
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        monkeypatch.setenv("ANTON_SCRATCHPAD_SESSION_MAX_BYTES", "4096")
        pad = make_scratchpad(name="huge", _venvs_base=tmp_path / "venvs", session_id="c")
        await pad.start()
        try:
            cell = await pad.execute("blob = 'x' * 500000\nprint('made')")
            assert cell.error is None
            assert "NOT saved" in cell.logs and "cap" in cell.logs
            snapshot = pad._session_snapshot_path()
            assert snapshot is not None
            # No snapshot, and no abandoned temp file left behind.
            assert not snapshot.exists()
            assert list(snapshot.parent.glob("*.tmp")) == []
        finally:
            await pad.cleanup()

    async def test_one_unpicklable_object_does_not_lose_the_rest(self, tmp_path, monkeypatch):
        """A live DB connection must not take the whole namespace down with it.

        Pickling is all-or-nothing per file, so before this a single unpicklable value
        lost everything. The objects that fail are exactly the ones long stateful tasks
        hold — `sqlite3.Connection`, sockets (SMTP for a mail campaign), generators — so
        the workloads that most need persistence were the ones getting none of it.
        """
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        venvs_base = tmp_path / "venvs"
        pad = make_scratchpad(name="withconn", _venvs_base=venvs_base, session_id="c")
        await pad.start()
        cell = await pad.execute(
            "import sqlite3\n"
            "conn = sqlite3.connect(':memory:')\n"
            "master_df = {'a': [1, 2, 3]}\n"
            "rates = 0.458\n"
        )
        assert cell.error is None
        # Reported, naming the casualty, and never on `error`.
        assert "conn" in cell.logs and "could not be preserved" in cell.logs
        await pad.close()

        pad2 = make_scratchpad(name="withconn", _venvs_base=venvs_base, session_id="c")
        await pad2.start()
        try:
            cell = await pad2.execute(
                "print(master_df['a'], rates, 'conn' in dir())"
            )
            assert cell.error is None, cell.error
            assert cell.stdout.strip() == "[1, 2, 3] 0.458 False"
        finally:
            await pad2.cleanup()

    async def test_unpicklable_warning_is_not_repeated_every_cell(self, tmp_path, monkeypatch):
        """Told once, not on every cell — and the per-key scan runs once, not per cell."""
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        pad = make_scratchpad(name="quiet", _venvs_base=tmp_path / "venvs", session_id="c")
        await pad.start()
        try:
            first = await pad.execute("import socket\nsck = socket.socket()\nkeep = 1")
            assert "could not be preserved" in first.logs
            second = await pad.execute("print('again')")
            assert "could not be preserved" not in (second.logs or "")
        finally:
            await pad.cleanup()

    async def test_rebinding_an_unpicklable_name_to_data_is_retried(self, tmp_path, monkeypatch):
        """The skip is keyed on the object, not the name — a rebind must be persisted."""
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        venvs_base = tmp_path / "venvs"
        pad = make_scratchpad(name="rebind", _venvs_base=venvs_base, session_id="c")
        await pad.start()
        await pad.execute("import socket\nthing = socket.socket()")
        await pad.execute("thing = 'now just a string'")
        await pad.close()

        pad2 = make_scratchpad(name="rebind", _venvs_base=venvs_base, session_id="c")
        await pad2.start()
        try:
            cell = await pad2.execute("print(thing)")
            assert cell.error is None, cell.error
            assert cell.stdout.strip() == "now just a string"
        finally:
            await pad2.cleanup()

    async def test_no_persistence_without_a_session_id(self, tmp_path, monkeypatch):
        """A runtime with no conversation scope must not persist anywhere.

        There used to be a shared `_no-session` fallback bucket. That is a
        confidentiality boundary, not a convenience: cowork-server's transient
        `CredentialProbe` builds a `ChatSession` with **no** session id and parses `DS_*`
        datasource credentials in the scratchpad, and `ANTON_SCRATCHPAD_PERSIST_SESSION`
        is process-global — so a probe inherits it once any normal chat has enabled it.
        With a shared bucket those credentials reach disk on a predictable path and a
        later probe reusing the pad name reloads them.
        """
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        venvs_base = tmp_path / "venvs"
        pad = make_scratchpad(name="unscoped", _venvs_base=venvs_base)  # no session_id
        await pad.start()
        try:
            cell = await pad.execute("secret = 'DS_POSTGRES_PROD__PASSWORD'")
            assert cell.error is None
            # Says so rather than silently pretending to persist.
            assert "ANTON_SCRATCHPAD_SESSION_PATH is unset" in cell.logs
            assert pad._session_snapshot_path() is None
            # And nothing was written anywhere under the snapshot tree.
            sessions = venvs_base.parent / "scratchpad-sessions"
            assert not sessions.exists() or list(sessions.rglob("*.pkl")) == []
        finally:
            await pad.cleanup()

    async def test_a_non_path_safe_session_id_is_refused(self, tmp_path, monkeypatch):
        """`_safe_segment` is not injective, so refuse rather than transform.

        `tenant/a` and `tenant_a` would sanitise to one directory. Transforming the id
        would also break the path cowork-server computes when it prunes, so the id must
        be path-safe as supplied. A UUID — what every real host passes — is unchanged.
        """
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        for bad in ["tenant/a", "../escape", ".hidden", "a b"]:
            pad = make_scratchpad(name="p", _venvs_base=tmp_path / "venvs", session_id=bad)
            assert pad._session_snapshot_path() is None, bad
        ok = make_scratchpad(
            name="p", _venvs_base=tmp_path / "venvs",
            session_id="e08a7ebd-e83c-4353-b0b9-550280b5bdd0",
        )
        assert ok._session_snapshot_path() is not None

    async def test_value_survives_repeated_restarts_not_just_one(self, tmp_path, monkeypatch):
        """Regression: the helper-identity fix worked for exactly ONE restart.

        `_INJECTED_HELPERS` was built from `namespace` *after* injection, so on the first
        restore it recorded the agent's own restored value as though it were our helper —
        and the next dump then excluded it. `sample` came back correct on turn 2 and was
        `<function sample>` from turn 3 on. Caught in review; my original test only did
        one restart.
        """
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        venvs_base = tmp_path / "venvs"
        first = make_scratchpad(name="many", _venvs_base=venvs_base, session_id="c")
        await first.start()
        await first.execute("sample = [1, 2, 3]")
        await first.close()

        for _ in range(3):
            pad = make_scratchpad(name="many", _venvs_base=venvs_base, session_id="c")
            await pad.start()
            cell = await pad.execute("print(repr(sample))")
            await pad.close()
            assert cell.stdout.strip() == "[1, 2, 3]", cell.stdout


_HELPER_MODULE_SRC = (
    "class Campaign:\n"
    "    def __init__(self, name):\n"
    "        self.name = name\n"
    "    def send(self):\n"
    "        return 'sent ' + self.name\n"
)


def _write_and_import_helper(module: str = "campaign_engine") -> str:
    """Cell source that authors a local module, imports it, and binds an instance.

    This is the shape ENG-1124's dill probe never covered: its agent-defined functions
    were declared *inside* the exec namespace (pickled by value, always fine), never in
    a .py file the agent wrote. dill stores objects from a real module by REFERENCE to
    that module, which is the whole bug.
    """
    return (
        "import sys, os\n"
        f"open({module + '.py'!r}, 'w').write({_HELPER_MODULE_SRC!r})\n"
        # The agent adds cwd itself — Python never does. This works for the rest of
        # THIS process and is gone by the time the next one loads the snapshot.
        "sys.path.insert(0, os.getcwd())\n"
        f"import {module}\n"
        f"c = {module}.Campaign('august')\n"
        "rates = 0.458\n"
    )


class TestAgentAuthoredModuleSnapshots:
    """ENG-1366 — a namespace holding objects from an agent-written .py file.

    Measured at 8% of cells for one production user, costing a cache-cold reload each
    time. It never surfaced as an error because the agent absorbs the loss by re-reading
    from disk, which is why it survived the fix that was meant to end it.
    """

    async def test_namespace_survives_an_agent_written_local_module(
        self, tmp_path, monkeypatch
    ):
        """The reported bug: one agent-authored helper discarded the whole namespace."""
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        venvs_base = tmp_path / "venvs"
        workspace = tmp_path / "ws"
        workspace.mkdir()

        pad = make_scratchpad(
            name="helpermod",
            _venvs_base=venvs_base,
            session_id="c",
            workspace_path=workspace,
        )
        await pad.start()
        cell = await pad.execute(_write_and_import_helper())
        assert cell.error is None, cell.error
        await pad.close()
        assert (workspace / "campaign_engine.py").exists()

        pad2 = make_scratchpad(
            name="helpermod",
            _venvs_base=venvs_base,
            session_id="c",
            workspace_path=workspace,
        )
        await pad2.start()
        try:
            cell = await pad2.execute("print(c.send(), rates)")
            # Before the fix: `logs` carried "Failed to load scratchpad session" with
            # ModuleNotFoundError, and BOTH names were gone — not just the helper.
            assert "Failed to load scratchpad session" not in (cell.logs or "")
            assert cell.error is None, cell.error
            assert cell.stdout.strip() == "sent august 0.458", cell.stdout
        finally:
            await pad2.cleanup()

    async def test_an_unresolvable_value_drops_only_itself(self, tmp_path, monkeypatch):
        """When a reference genuinely cannot be rebuilt, the rest must still load.

        The helper file is deleted between turns, so no `sys.path` entry can save it.
        This is the case the per-value snapshot format exists for: a pickle stream is a
        sequential program, so a single unresolvable reference used to kill everything
        after it too.
        """
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        venvs_base = tmp_path / "venvs"
        workspace = tmp_path / "ws"
        workspace.mkdir()

        pad = make_scratchpad(
            name="goneaway",
            _venvs_base=venvs_base,
            session_id="c",
            workspace_path=workspace,
        )
        await pad.start()
        cell = await pad.execute(_write_and_import_helper())
        assert cell.error is None, cell.error
        await pad.close()
        (workspace / "campaign_engine.py").unlink()

        pad2 = make_scratchpad(
            name="goneaway",
            _venvs_base=venvs_base,
            session_id="c",
            workspace_path=workspace,
        )
        await pad2.start()
        try:
            cell = await pad2.execute("print(rates, 'c' in dir())")
            assert cell.error is None, cell.error
            # The unrelated variable survived; only the helper-backed one is gone.
            assert cell.stdout.strip() == "0.458 False", cell.stdout
            # …and the loss is NAMED, on `logs` and never on `error` (a snapshot
            # problem must not feed the consecutive-error circuit breaker).
            logs = cell.logs or ""
            assert "could not be rebuilt" in logs, logs
            # Match the rendered name list, not a bare "c" — the letter appears in
            # the prose of the note itself, so a substring check would pass even if
            # nothing were named at all.
            named = logs.split("now undefined: ", 1)[1].split(".", 1)[0]
            assert sorted(n.strip() for n in named.split(",")) == [
                "c",
                "campaign_engine",
            ], named
            assert "ModuleNotFoundError" in logs, logs
            assert cell.error is None
        finally:
            await pad2.cleanup()

    async def test_the_workspace_is_not_left_on_syspath(self, tmp_path, monkeypatch):
        """The import path is widened for the load only, not for the agent's cells.

        Guards the load-scoped design against drifting into a process-wide one: the
        workspace is the agent's own directory and may hold files named after stdlib
        modules, so leaving it on `sys.path` would change import resolution for every
        subsequent cell.
        """
        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        venvs_base = tmp_path / "venvs"
        workspace = tmp_path / "ws"
        workspace.mkdir()

        pad = make_scratchpad(
            name="scoped",
            _venvs_base=venvs_base,
            session_id="c",
            workspace_path=workspace,
        )
        await pad.start()
        assert (await pad.execute(_write_and_import_helper())).error is None
        await pad.close()

        pad2 = make_scratchpad(
            name="scoped",
            _venvs_base=venvs_base,
            session_id="c",
            workspace_path=workspace,
        )
        await pad2.start()
        try:
            cell = await pad2.execute(
                "import sys, os\n"
                "here = os.path.realpath(os.getcwd())\n"
                "on_path = any(os.path.realpath(p) == here for p in sys.path)\n"
                # Restored AND not left on the path — the pair is the point.
                "print(c.send(), on_path)\n"
            )
            assert cell.error is None, cell.error
            assert cell.stdout.strip() == "sent august False", cell.stdout
        finally:
            await pad2.cleanup()

    async def test_the_snapshot_is_not_a_bare_dict(self, tmp_path, monkeypatch):
        """An anton predating this format must REPORT, not silently load junk.

        The older loader accepts any dict in the session file as the namespace itself.
        A dict envelope would therefore hand it `{'values': ..., '__anton_snapshot__':
        2}` as the agent's variables — every real name gone, nothing said — which is
        the exact silent no-op ENG-1124 exists to end, reachable by a rollback or an
        older desktop build resuming the conversation. A non-dict trips that loader's
        `isinstance(ns, dict)` guard so it starts fresh AND reports.

        Verified against the real staging loader at review time; this pins it so the
        envelope cannot drift back to a dict.
        """
        import dill

        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        pad = make_scratchpad(
            name="notadict", _venvs_base=tmp_path / "venvs", session_id="c"
        )
        await pad.start()
        try:
            assert (await pad.execute("kept = 1")).error is None
            snapshot = pad._session_snapshot_path()
            assert snapshot is not None and snapshot.exists()
            written = dill.loads(snapshot.read_bytes())
            assert not isinstance(written, dict), (
                "a dict snapshot is silently mistaken for a namespace by older anton"
            )
        finally:
            await pad.cleanup()

    async def test_a_snapshot_in_the_old_format_still_loads(self, tmp_path, monkeypatch):
        """A conversation in flight across the upgrade keeps its state.

        The pre-ENG-1366 snapshot is one stream holding real objects; the new one is an
        envelope of individually-pickled values. The loader must read both, or the
        release costs every open conversation a cold turn.
        """
        import dill

        monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "true")
        pad = make_scratchpad(
            name="legacy", _venvs_base=tmp_path / "venvs", session_id="c"
        )
        snapshot = pad._session_snapshot_path(create=True)
        assert snapshot is not None
        snapshot.write_bytes(dill.dumps({"carried_over": [1, 2, 3]}))

        await pad.start()
        try:
            cell = await pad.execute("print(carried_over)")
            assert cell.error is None, cell.error
            assert cell.stdout.strip() == "[1, 2, 3]", cell.stdout
        finally:
            await pad.cleanup()
