from __future__ import annotations

import asyncio
import os

import pytest

import anton.scratchpad as scratchpad_module
from anton.scratchpad import Cell, Scratchpad, ScratchpadManager


class TestScratchpadBasicExecution:
    async def test_basic_execution(self):
        """print(42) should return '42' in stdout."""
        pad = Scratchpad(name="test")
        await pad.start()
        try:
            cell = await pad.execute("print(42)")
            assert cell.stdout.strip() == "42"
            assert cell.error is None
        finally:
            await pad.close()

    async def test_state_persists(self):
        """Variable from cell 1 should be available in cell 2."""
        pad = Scratchpad(name="test")
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
        pad = Scratchpad(name="test")
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
        pad = Scratchpad(name="test")
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
        pad = Scratchpad(name="test")
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
        pad = Scratchpad(name="empty")
        await pad.start()
        try:
            output = pad.view()
            assert "empty" in output.lower()
        finally:
            await pad.close()


class TestScratchpadReset:
    async def test_reset_clears_state(self):
        """Variables should be gone after reset."""
        pad = Scratchpad(name="test")
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


class TestScratchpadEdgeCases:
    async def test_timeout_kills_process(self, monkeypatch):
        """Long-running code triggers timeout."""
        monkeypatch.setattr(scratchpad_module, "_CELL_TIMEOUT_DEFAULT", 1)
        monkeypatch.setattr(scratchpad_module, "_CELL_INACTIVITY_TIMEOUT", 1)
        pad = Scratchpad(name="test")
        await pad.start()
        try:
            cell = await pad.execute("import time; time.sleep(60)")
            assert cell.error is not None
            assert "timed out" in cell.error.lower() or "inactivity" in cell.error.lower()
        finally:
            await pad.close()

    async def test_output_truncation(self):
        """Large output should be capped at 10KB when read through Cell."""
        pad = Scratchpad(name="test")
        await pad.start()
        try:
            # The raw cell captures full stdout; truncation happens at the
            # chat handler level. Here we just verify we get the full output.
            cell = await pad.execute("print('x' * 20000)")
            assert len(cell.stdout) >= 20000
            assert cell.error is None
        finally:
            await pad.close()

    async def test_dead_process_detected(self):
        """If process is dead, execute reports it."""
        pad = Scratchpad(name="test")
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
        pad = Scratchpad(name="test")
        await pad.start()
        try:
            cell = await pad.execute("import sys; sys.stderr.write('warn\\n')")
            assert "warn" in cell.stderr
        finally:
            await pad.close()


class TestScratchpadManager:
    async def test_get_or_create(self):
        """Auto-creates a scratchpad on first access."""
        mgr = ScratchpadManager()
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
        mgr = ScratchpadManager()
        try:
            await mgr.get_or_create("beta")
            result = await mgr.remove("beta")
            assert "beta" in result
            assert "beta" not in mgr.list_pads()
        finally:
            await mgr.close_all()

    async def test_remove_nonexistent(self):
        """remove() on unknown name returns a message."""
        mgr = ScratchpadManager()
        result = await mgr.remove("nope")
        assert "nope" in result

    async def test_close_all(self):
        """close_all() cleans up everything."""
        mgr = ScratchpadManager()
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
        mgr = ScratchpadManager()
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
        pad = Scratchpad(name="main")
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
        pad = Scratchpad(name="empty")
        await pad.start()
        try:
            md = pad.render_notebook()
            assert "no cells" in md.lower()
        finally:
            await pad.close()

    async def test_render_notebook_skips_empty_cells(self):
        """Whitespace-only cells are filtered out."""
        pad = Scratchpad(name="gaps")
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
        pad = Scratchpad(name="long")
        await pad.start()
        try:
            await pad.execute("for i in range(50): print(i)")
            md = pad.render_notebook()
            assert "more lines" in md
        finally:
            await pad.close()

    async def test_render_notebook_error_summary(self):
        """Only last traceback line shown, not full trace."""
        pad = Scratchpad(name="err")
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
        pad = Scratchpad(name="warn")
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
        result = Scratchpad._truncate_output(text, max_lines=10)
        assert "line 0" in result
        assert "line 9" in result
        assert "line 10" not in result
        assert "(40 more lines)" in result

    async def test_truncate_output_chars(self):
        """Respects char limit."""
        text = "\n".join("x" * 80 for _ in range(5))
        result = Scratchpad._truncate_output(text, max_lines=100, max_chars=200)
        assert "(truncated)" in result
        assert len(result) < len(text)


class TestCellMetadata:
    async def test_cell_stores_description_and_estimated_time(self):
        """execute() should store description and estimated_time on the Cell."""
        pad = Scratchpad(name="meta")
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
        pad = Scratchpad(name="defaults")
        await pad.start()
        try:
            cell = await pad.execute("print(1)")
            assert cell.description == ""
            assert cell.estimated_time == ""
        finally:
            await pad.close()

    async def test_view_shows_description_in_header(self):
        """view() should include description in the cell header."""
        pad = Scratchpad(name="view-desc")
        await pad.start()
        try:
            await pad.execute("print(1)", description="Count to one")
            output = pad.view()
            assert "--- Cell 1: Count to one ---" in output
        finally:
            await pad.close()

    async def test_view_without_description(self):
        """view() without description falls back to plain header."""
        pad = Scratchpad(name="view-plain")
        await pad.start()
        try:
            await pad.execute("print(1)")
            output = pad.view()
            assert "--- Cell 1 ---" in output
        finally:
            await pad.close()

    async def test_render_notebook_shows_description(self):
        """render_notebook() should include description in markdown header."""
        pad = Scratchpad(name="nb-desc")
        await pad.start()
        try:
            await pad.execute("print(1)", description="Count to one")
            md = pad.render_notebook()
            assert "### Cell 1 \u2014 Count to one" in md
        finally:
            await pad.close()

    async def test_render_notebook_without_description(self):
        """render_notebook() without description uses plain header."""
        pad = Scratchpad(name="nb-plain")
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
        pad = Scratchpad(name="env-test")
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
        pad = Scratchpad(name="llm-test", _coding_model="claude-test-model")
        await pad.start()
        try:
            cell = await pad.execute("llm = get_llm(); print(llm.model)")
            assert cell.stdout.strip() == "claude-test-model"
            assert cell.error is None
        finally:
            await pad.close()

    async def test_get_llm_not_available_without_model(self):
        """get_llm() should not be in namespace when no model is configured."""
        pad = Scratchpad(name="no-llm")
        await pad.start()
        try:
            cell = await pad.execute("get_llm()")
            assert cell.error is not None
            assert "NameError" in cell.error
        finally:
            await pad.close()

    async def test_agentic_loop_available_when_model_set(self):
        """agentic_loop() should be injected alongside get_llm()."""
        pad = Scratchpad(name="agentic-test", _coding_model="claude-test-model")
        await pad.start()
        try:
            cell = await pad.execute("print(callable(agentic_loop))")
            assert cell.stdout.strip() == "True"
            assert cell.error is None
        finally:
            await pad.close()

    async def test_agentic_loop_not_available_without_model(self):
        """agentic_loop() should not be in namespace when no model is configured."""
        pad = Scratchpad(name="no-agentic")
        await pad.start()
        try:
            cell = await pad.execute("agentic_loop()")
            assert cell.error is not None
            assert "NameError" in cell.error
        finally:
            await pad.close()

    async def test_generate_object_available_when_model_set(self):
        """generate_object() should be available on the LLM wrapper."""
        pad = Scratchpad(name="genobj-test", _coding_model="claude-test-model")
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
        pad = Scratchpad(name="key-test", _coding_model="test-model")
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
        pad = Scratchpad(name="venv-test")
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
        pad = Scratchpad(name="venv-close")
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
        pad = Scratchpad(name="venv-reset")
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
        pad = Scratchpad(name="venv-exec")
        await pad.start()
        try:
            cell = await pad.execute("import sys; print(sys.executable)")
            assert cell.error is None
            assert pad._venv_dir in cell.stdout.strip()
        finally:
            await pad.close()

    async def test_system_packages_available(self):
        """System site-packages should be accessible (e.g. pydantic from parent env)."""
        pad = Scratchpad(name="venv-syspkg")
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
        pad = Scratchpad(name="recycle", _venvs_base=venvs_base)
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
        pad2 = Scratchpad(name="recycle", _venvs_base=venvs_base)
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
        pad = Scratchpad(name="ver-mismatch", _venvs_base=venvs_base)
        await pad.start()
        venv_dir = pad._venv_dir
        await pad.close()

        # Tamper with the .python_version file
        ver_path = os.path.join(venv_dir, ".python_version")
        with open(ver_path, "w") as f:
            f.write("2.7\n")

        # Reopen — should detect mismatch, nuke, and recreate
        pad2 = Scratchpad(name="ver-mismatch", _venvs_base=venvs_base)
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
        pad = Scratchpad(name="corrupt", _venvs_base=venvs_base)
        await pad.start()
        venv_dir = pad._venv_dir
        python_path = pad._venv_python
        await pad.close()

        # Delete the Python binary to simulate corruption
        os.remove(python_path)

        # Reopen — should detect corruption, nuke, and recreate
        pad2 = Scratchpad(name="corrupt", _venvs_base=venvs_base)
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
        venvs_base = tmp_path / "venvs"
        mgr = ScratchpadManager(workspace_path=tmp_path)
        # Override base to use our tmp dir
        mgr._venvs_base = venvs_base
        try:
            pad = await mgr.get_or_create("deleteme")
            venv_dir = pad._venv_dir
            assert os.path.isdir(venv_dir)
            await mgr.remove("deleteme")
            assert not os.path.exists(venv_dir)
        finally:
            await mgr.close_all()
            shutil.rmtree(venvs_base, ignore_errors=True)

    async def test_requirements_saved_on_close(self, tmp_path):
        """requirements.txt is written when pad has installed packages."""
        import shutil
        venvs_base = tmp_path / "venvs"
        pad = Scratchpad(name="req-save", _venvs_base=venvs_base)
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
        pad = Scratchpad(name="install-test")
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
        pad = Scratchpad(name="install-empty")
        await pad.start()
        try:
            result = await pad.install_packages([])
            assert "no packages" in result.lower()
        finally:
            await pad.close()

    async def test_install_invalid_package(self):
        """install_packages with a bogus name should report failure."""
        pad = Scratchpad(name="install-bad")
        await pad.start()
        try:
            result = await pad.install_packages(["this-package-does-not-exist-xyz123"])
            assert "failed" in result.lower() or "error" in result.lower()
        finally:
            await pad.close()

    async def test_install_survives_reset(self):
        """Packages installed before a reset should still be available after."""
        pad = Scratchpad(name="install-reset")
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
        pad = Scratchpad(name="progress-ns")
        await pad.start()
        try:
            cell = await pad.execute("print(callable(progress))")
            assert cell.error is None
            assert cell.stdout.strip() == "True"
        finally:
            await pad.close()

    async def test_progress_resets_inactivity_timeout(self, monkeypatch):
        """Code that calls progress() frequently should survive even with a short inactivity timeout."""
        monkeypatch.setattr(scratchpad_module, "_CELL_INACTIVITY_TIMEOUT", 2)
        monkeypatch.setattr(scratchpad_module, "_CELL_TIMEOUT_DEFAULT", 10)
        pad = Scratchpad(name="progress-keep-alive")
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

    async def test_inactivity_timeout_kills_without_progress(self, monkeypatch):
        """Code that sleeps without progress() calls should be killed by inactivity timeout."""
        monkeypatch.setattr(scratchpad_module, "_CELL_INACTIVITY_TIMEOUT", 2)
        monkeypatch.setattr(scratchpad_module, "_CELL_TIMEOUT_DEFAULT", 60)
        pad = Scratchpad(name="no-progress")
        await pad.start()
        try:
            cell = await pad.execute("import time; time.sleep(30)")
            assert cell.error is not None
            assert "inactivity" in cell.error.lower()
        finally:
            await pad.close()

    async def test_execute_streaming_yields_progress(self):
        """execute_streaming() should yield progress strings and a final Cell."""
        pad = Scratchpad(name="streaming")
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
        from anton.scratchpad import _compute_timeouts
        total, inactivity = _compute_timeouts(0)
        assert total == 120.0
        assert inactivity == 30.0

    async def test_compute_timeouts_with_estimate(self):
        """Estimate should scale total timeout and inactivity with no hard cap."""
        from anton.scratchpad import _compute_timeouts

        # Small estimate: max(10*2, 10+30) = max(20, 40) = 40
        total, inactivity = _compute_timeouts(10)
        assert total == 40.0
        assert inactivity == 30.0  # max(5, 30) = 30

        # Medium estimate: max(60*2, 60+30) = max(120, 90) = 120
        total, inactivity = _compute_timeouts(60)
        assert total == 120.0
        assert inactivity == 30.0  # max(30, 30) = 30

        # Large estimate: max(300*2, 300+30) = max(600, 330) = 600
        total, inactivity = _compute_timeouts(300)
        assert total == 600.0
        assert inactivity == 150.0  # max(150, 30) = 150

        # Very large estimate: scales with estimate
        total, inactivity = _compute_timeouts(1000)
        assert total == 2000.0
        assert inactivity == 500.0  # max(500, 30) = 500


class TestSampleFunction:
    async def test_sample_available_in_namespace(self):
        """sample() should be callable in scratchpad code."""
        pad = Scratchpad(name="sample-ns")
        await pad.start()
        try:
            cell = await pad.execute("print(callable(sample))")
            assert cell.error is None
            assert cell.stdout.strip() == "True"
        finally:
            await pad.close()

    async def test_sample_dict_preview(self):
        """sample() on a dict should show keys and truncated values."""
        pad = Scratchpad(name="sample-dict")
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
        pad = Scratchpad(name="sample-list")
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
        pad = Scratchpad(name="sample-str")
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
        pad = Scratchpad(name="sample-full")
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
        pad = Scratchpad(name="sample-set")
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
        pad = Scratchpad(name="sample-obj")
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
        pad = Scratchpad(name="sample-bytes")
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
        pad = Scratchpad(name="sample-named")
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
        pad = Scratchpad(name="sample-empty")
        await pad.start()
        try:
            cell = await pad.execute("sample({})")
            assert cell.error is None
            assert "Keys (0)" in cell.stdout
        finally:
            await pad.close()

    async def test_sample_empty_list(self):
        """sample() on an empty list should not crash."""
        pad = Scratchpad(name="sample-empty-list")
        await pad.start()
        try:
            cell = await pad.execute("sample([])")
            assert cell.error is None
            assert "Length: 0" in cell.stdout
        finally:
            await pad.close()
