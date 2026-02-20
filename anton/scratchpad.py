"""Scratchpad — persistent Python subprocess for stateful, notebook-like execution."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import venv
from dataclasses import dataclass, field
from pathlib import Path

_CELL_TIMEOUT = 30
_INSTALL_TIMEOUT = 120
_MAX_OUTPUT = 10_000

_BOOT_SCRIPT = r'''
import io
import json
import os
import sys
import traceback

_CELL_DELIM = "__ANTON_CELL_END__"
_RESULT_START = "__ANTON_RESULT__"
_RESULT_END = "__ANTON_RESULT_END__"

# Persistent namespace across cells
namespace = {"__builtins__": __builtins__}

# --- Inject get_llm() for LLM access from scratchpad code ---
_scratchpad_model = os.environ.get("ANTON_SCRATCHPAD_MODEL", "")
if _scratchpad_model:
    try:
        import asyncio as _llm_asyncio

        _scratchpad_provider_name = os.environ.get("ANTON_SCRATCHPAD_PROVIDER", "anthropic")
        if _scratchpad_provider_name == "openai":
            from anton.llm.openai import OpenAIProvider as _ProviderClass
        else:
            from anton.llm.anthropic import AnthropicProvider as _ProviderClass

        _llm_provider = _ProviderClass()  # reads API key from env
        _llm_model = _scratchpad_model

        class _ScratchpadLLM:
            """Sync LLM wrapper for scratchpad use. Mirrors SkillLLM interface."""

            @property
            def model(self):
                return _llm_model

            def complete(self, *, system, messages, tools=None, tool_choice=None, max_tokens=4096):
                """Call the LLM synchronously. Returns an LLMResponse."""
                return _llm_asyncio.run(_llm_provider.complete(
                    model=_llm_model,
                    system=system,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=max_tokens,
                ))

            def generate_object(self, schema_class, *, system, messages, max_tokens=4096):
                """Generate a structured object matching a Pydantic model.

                Uses tool_choice to force the LLM to return structured data.
                Supports single models and list[Model].

                Args:
                    schema_class: A Pydantic BaseModel subclass, or list[Model].
                    system: System prompt.
                    messages: Conversation messages.
                    max_tokens: Max tokens for the LLM call.

                Returns:
                    An instance of schema_class (or a list of instances).
                """
                from pydantic import BaseModel as _BaseModel

                is_list = hasattr(schema_class, "__origin__") and schema_class.__origin__ is list
                if is_list:
                    inner_class = schema_class.__args__[0]

                    class _ArrayWrapper(_BaseModel):
                        items: list[inner_class]

                    schema = _ArrayWrapper.model_json_schema()
                    tool_name = f"{inner_class.__name__}_array"
                else:
                    schema = schema_class.model_json_schema()
                    tool_name = schema_class.__name__

                tool = {
                    "name": tool_name,
                    "description": f"Generate structured output matching the {tool_name} schema.",
                    "input_schema": schema,
                }

                response = self.complete(
                    system=system,
                    messages=messages,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool_name},
                    max_tokens=max_tokens,
                )

                if not response.tool_calls:
                    raise ValueError("LLM did not return structured output.")

                import json as _json
                raw = response.tool_calls[0].input

                if is_list:
                    wrapper = _ArrayWrapper.model_validate(raw)
                    return wrapper.items
                return schema_class.model_validate(raw)

        _scratchpad_llm_instance = _ScratchpadLLM()

        def get_llm():
            """Get a pre-configured LLM client. No API keys needed."""
            return _scratchpad_llm_instance

        def agentic_loop(*, system, user_message, tools, handle_tool, max_turns=10, max_tokens=4096):
            """Run a synchronous LLM tool-call loop.

            The LLM reasons, calls tools via handle_tool(name, inputs) -> str,
            and iterates until it produces a final text response.

            Args:
                system: System prompt for the LLM.
                user_message: Initial user message.
                tools: Tool definitions (Anthropic tool schema format).
                handle_tool: Callback (tool_name, tool_input) -> result_string.
                max_turns: Safety limit on LLM round-trips (default 10).
                max_tokens: Max tokens per LLM call.

            Returns:
                The final text response from the LLM.
            """
            llm = get_llm()
            messages = [{"role": "user", "content": user_message}]

            response = None
            for _ in range(max_turns):
                response = llm.complete(
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                )

                if not response.tool_calls:
                    return response.content

                # Build assistant message with text + tool_use blocks
                assistant_content = []
                if response.content:
                    assistant_content.append({"type": "text", "text": response.content})
                for tc in response.tool_calls:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    })
                messages.append({"role": "assistant", "content": assistant_content})

                # Execute each tool and collect results
                tool_results = []
                for tc in response.tool_calls:
                    try:
                        result = handle_tool(tc.name, tc.input)
                    except Exception as exc:
                        result = f"Error: {exc}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})

            # Hit max_turns
            return response.content if response else ""

        namespace["get_llm"] = get_llm
        namespace["agentic_loop"] = agentic_loop
    except Exception:
        pass  # LLM not available — not fatal (e.g. anthropic not installed)

# --- Inject run_skill() if skill dirs are available ---
_skill_dirs_raw = os.environ.get("ANTON_SKILL_DIRS", "")
if _skill_dirs_raw:
    try:
        import importlib.util
        _skill_dirs = [d for d in _skill_dirs_raw.split(os.pathsep) if d]
        _registry = {}

        for skills_dir in _skill_dirs:
            from pathlib import Path as _Path
            skills_path = _Path(skills_dir)
            if not skills_path.is_dir():
                continue
            for skill_file in sorted(skills_path.glob("*/skill.py")):
                module_name = f"anton_skill_{skill_file.parent.name}"
                spec = importlib.util.spec_from_file_location(module_name, skill_file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    continue
                for attr_name in dir(module):
                    obj = getattr(module, attr_name)
                    info = getattr(obj, "_skill_info", None)
                    if info is not None and hasattr(info, "name"):
                        _registry[info.name] = info

        def run_skill(name, **kwargs):
            """Run an Anton skill by name. Returns the skill's output."""
            import asyncio as _asyncio
            _skill = _registry.get(name)
            if _skill is None:
                raise ValueError(f"Unknown skill: {name}. Available: {list(_registry.keys())}")
            result = _asyncio.run(_skill.execute(**kwargs))
            return result.output

        namespace["run_skill"] = run_skill
    except Exception:
        pass  # Skills not available — not fatal

# Read-execute loop
_real_stdout = sys.stdout
_real_stdin = sys.stdin

while True:
    lines = []
    try:
        for line in _real_stdin:
            stripped = line.rstrip("\n")
            if stripped == _CELL_DELIM:
                break
            lines.append(line)
        else:
            # EOF — parent closed stdin
            break
    except EOFError:
        break

    code = "".join(lines)
    if not code.strip():
        result = {"stdout": "", "stderr": "", "error": None}
        _real_stdout.write(_RESULT_START + "\n")
        _real_stdout.write(json.dumps(result) + "\n")
        _real_stdout.write(_RESULT_END + "\n")
        _real_stdout.flush()
        continue

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    error = None

    sys.stdout = out_buf
    sys.stderr = err_buf
    try:
        compiled = compile(code, "<scratchpad>", "exec")
        exec(compiled, namespace)
    except Exception:
        error = traceback.format_exc()
    finally:
        sys.stdout = _real_stdout
        sys.stderr = sys.__stderr__

    result = {
        "stdout": out_buf.getvalue(),
        "stderr": err_buf.getvalue(),
        "error": error,
    }
    _real_stdout.write(_RESULT_START + "\n")
    _real_stdout.write(json.dumps(result) + "\n")
    _real_stdout.write(_RESULT_END + "\n")
    _real_stdout.flush()
'''

_CELL_DELIM = "__ANTON_CELL_END__"
_RESULT_START = "__ANTON_RESULT__"
_RESULT_END = "__ANTON_RESULT_END__"


@dataclass
class Cell:
    code: str
    stdout: str
    stderr: str
    error: str | None


@dataclass
class Scratchpad:
    name: str
    cells: list[Cell] = field(default_factory=list)
    _proc: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _boot_path: str | None = field(default=None, repr=False)
    _skill_dirs: list[Path] = field(default_factory=list, repr=False)
    _coding_provider: str = field(default="anthropic", repr=False)
    _coding_model: str = field(default="", repr=False)
    _coding_api_key: str = field(default="", repr=False)
    _venv_dir: str | None = field(default=None, repr=False)
    _venv_python: str | None = field(default=None, repr=False)

    def _ensure_venv(self) -> None:
        """Create a lightweight per-scratchpad venv (idempotent).

        Uses system_site_packages=True so the real system packages are visible.
        If we're running inside a parent venv, we also drop a .pth file so the
        parent venv's site-packages are visible in the child.
        """
        if self._venv_dir is not None:
            return
        self._venv_dir = tempfile.mkdtemp(prefix="anton_venv_")
        venv.create(self._venv_dir, system_site_packages=True, with_pip=False)
        # Resolve the venv python path
        bin_dir = os.path.join(self._venv_dir, "bin")
        if sys.platform == "win32":
            bin_dir = os.path.join(self._venv_dir, "Scripts")
        self._venv_python = os.path.join(bin_dir, "python")

        # If running inside a parent venv, make its packages visible via .pth file
        if sys.prefix != sys.base_prefix:
            import site as _site
            parent_site = _site.getsitepackages()
            # Find the child venv's site-packages to place the .pth file
            child_lib = os.path.join(self._venv_dir, "lib")
            child_site = None
            for dirpath, dirnames, _ in os.walk(child_lib):
                if "site-packages" in dirnames:
                    child_site = os.path.join(dirpath, "site-packages")
                    break
            if child_site and parent_site:
                pth_path = os.path.join(child_site, "_parent_venv.pth")
                with open(pth_path, "w") as f:
                    for sp in parent_site:
                        f.write(sp + "\n")

    async def start(self) -> None:
        """Write the boot script to a temp file and launch the subprocess."""
        self._ensure_venv()

        fd, path = tempfile.mkstemp(suffix=".py", prefix="anton_scratchpad_")
        os.write(fd, _BOOT_SCRIPT.encode())
        os.close(fd)
        self._boot_path = path

        env = os.environ.copy()
        if self._skill_dirs:
            env["ANTON_SKILL_DIRS"] = os.pathsep.join(str(d) for d in self._skill_dirs)
        if self._coding_model:
            env["ANTON_SCRATCHPAD_MODEL"] = self._coding_model
        if self._coding_provider:
            env["ANTON_SCRATCHPAD_PROVIDER"] = self._coding_provider
        # Ensure the SDKs can find API keys under their expected names.
        # Anton stores them as ANTON_*_API_KEY; the SDKs expect *_API_KEY.
        if "ANTHROPIC_API_KEY" not in env and "ANTON_ANTHROPIC_API_KEY" in env:
            env["ANTHROPIC_API_KEY"] = env["ANTON_ANTHROPIC_API_KEY"]
        if "OPENAI_API_KEY" not in env and "ANTON_OPENAI_API_KEY" in env:
            env["OPENAI_API_KEY"] = env["ANTON_OPENAI_API_KEY"]
        # If settings provided an explicit API key (e.g. from ~/.anton/.env or
        # Pydantic settings), inject it so the subprocess SDK can authenticate.
        if self._coding_api_key:
            sdk_key = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
            }.get(self._coding_provider, "")
            if sdk_key and sdk_key not in env:
                env[sdk_key] = self._coding_api_key
        # Ensure the anton package is importable in the subprocess (needed for
        # get_llm and skill loading). The boot script runs from a temp file, so
        # the project root isn't on sys.path by default.
        _anton_root = str(Path(__file__).resolve().parent.parent)
        python_path = env.get("PYTHONPATH", "")
        if _anton_root not in python_path:
            env["PYTHONPATH"] = _anton_root + (os.pathsep + python_path if python_path else "")

        self._proc = await asyncio.create_subprocess_exec(
            self._venv_python, path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

    async def execute(self, code: str) -> Cell:
        """Send code to the subprocess, read the JSON result, return a Cell."""
        if self._proc is None or self._proc.returncode is not None:
            # Process died — auto-note
            cell = Cell(
                code=code,
                stdout="",
                stderr="",
                error="Scratchpad process is not running. Use reset to restart.",
            )
            self.cells.append(cell)
            return cell

        payload = code + "\n" + _CELL_DELIM + "\n"
        self._proc.stdin.write(payload.encode())  # type: ignore[union-attr]
        await self._proc.stdin.drain()  # type: ignore[union-attr]

        try:
            result_data = await asyncio.wait_for(
                self._read_result(), timeout=_CELL_TIMEOUT
            )
        except asyncio.TimeoutError:
            self._proc.kill()
            await self._proc.wait()
            cell = Cell(
                code=code,
                stdout="",
                stderr="",
                error=f"Cell timed out after {_CELL_TIMEOUT}s. Process killed — state lost. Use reset to restart.",
            )
            self.cells.append(cell)
            return cell

        cell = Cell(
            code=code,
            stdout=result_data.get("stdout", ""),
            stderr=result_data.get("stderr", ""),
            error=result_data.get("error"),
        )
        self.cells.append(cell)
        return cell

    async def _read_result(self) -> dict:
        """Read lines from stdout until we get the result delimiters."""
        lines: list[str] = []
        in_result = False
        while True:
            raw = await self._proc.stdout.readline()  # type: ignore[union-attr]
            if not raw:
                # Process exited
                return {"stdout": "", "stderr": "", "error": "Process exited unexpectedly."}
            line = raw.decode().rstrip("\n")
            if line == _RESULT_START:
                in_result = True
                continue
            if line == _RESULT_END:
                break
            if in_result:
                lines.append(line)
        return json.loads("\n".join(lines))

    def view(self) -> str:
        """Format all cells with their outputs."""
        if not self.cells:
            return f"Scratchpad '{self.name}' is empty."

        parts: list[str] = []
        for i, cell in enumerate(self.cells):
            parts.append(f"--- Cell {i + 1} ---")
            parts.append(cell.code)
            if cell.stdout:
                parts.append(f"[stdout]\n{cell.stdout}")
            if cell.stderr:
                parts.append(f"[stderr]\n{cell.stderr}")
            if cell.error:
                parts.append(f"[error]\n{cell.error}")
            if not cell.stdout and not cell.stderr and not cell.error:
                parts.append("(no output)")
        return "\n".join(parts)

    @staticmethod
    def _truncate_output(text: str, max_lines: int = 20, max_chars: int = 2000) -> str:
        """Truncate output to *max_lines* / *max_chars*, whichever is shorter."""
        lines = text.split("\n")
        # Apply line limit
        if len(lines) > max_lines:
            kept = "\n".join(lines[:max_lines])
            remaining = len(lines) - max_lines
            return kept + f"\n... ({remaining} more lines)"
        # Apply char limit (don't cut mid-line)
        if len(text) > max_chars:
            total = 0
            kept_lines: list[str] = []
            for line in lines:
                if total + len(line) + 1 > max_chars and kept_lines:
                    break
                kept_lines.append(line)
                total += len(line) + 1
            return "\n".join(kept_lines) + "\n... (truncated)"
        return text

    def render_notebook(self) -> str:
        """Return a clean markdown notebook-style summary of all cells."""
        # Filter out empty/whitespace-only cells
        numbered: list[tuple[int, Cell]] = []
        idx = 0
        for cell in self.cells:
            idx += 1
            if not cell.code.strip():
                continue
            numbered.append((idx, cell))

        if not numbered:
            return f"Scratchpad '{self.name}' has no cells."

        parts: list[str] = [f"## Scratchpad: {self.name} ({len(numbered)} cells)"]

        for i, (num, cell) in enumerate(numbered):
            parts.append(f"\n### Cell {num}")
            parts.append(f"```python\n{cell.code}\n```\n")

            if cell.error:
                # Show only the last traceback line
                last_line = cell.error.strip().split("\n")[-1]
                parts.append(f"**Error:** `{last_line}`")
            elif cell.stdout:
                truncated = self._truncate_output(cell.stdout.rstrip("\n"))
                parts.append(f"**Output:**\n```\n{truncated}\n```\n")

            if i < len(numbered) - 1:
                parts.append("---")

        return "\n".join(parts)

    async def _stop_process(self) -> None:
        """Kill the subprocess and delete the boot script, but keep the venv."""
        if self._proc is not None and self._proc.returncode is None:
            try:
                self._proc.kill()
                await self._proc.wait()
            except ProcessLookupError:
                pass
        self._proc = None
        if self._boot_path is not None:
            try:
                os.unlink(self._boot_path)
            except OSError:
                pass
            self._boot_path = None

    async def reset(self) -> None:
        """Kill the process, clear cells, restart. Venv (and installed packages) survive."""
        await self._stop_process()
        self.cells.clear()
        await self.start()

    async def close(self) -> None:
        """Kill the process and clean up the boot script temp file and venv."""
        await self._stop_process()
        if self._venv_dir is not None:
            try:
                shutil.rmtree(self._venv_dir)
            except OSError:
                pass
            self._venv_dir = None
            self._venv_python = None

    async def install_packages(self, packages: list[str]) -> str:
        """Install packages into the scratchpad's venv via pip."""
        if not packages:
            return "No packages specified."
        self._ensure_venv()
        proc = await asyncio.create_subprocess_exec(
            self._venv_python, "-m", "pip", "install", "--no-input", *packages,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_INSTALL_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Install timed out after {_INSTALL_TIMEOUT}s."
        output = stdout.decode()
        if proc.returncode != 0:
            return f"Install failed (exit {proc.returncode}):\n{output}"
        return output


class ScratchpadManager:
    """Manages named scratchpad instances."""

    def __init__(
        self,
        skill_dirs: list[Path] | None = None,
        coding_provider: str = "anthropic",
        coding_model: str = "",
        coding_api_key: str = "",
    ) -> None:
        self._pads: dict[str, Scratchpad] = {}
        self._skill_dirs: list[Path] = skill_dirs or []
        self._coding_provider: str = coding_provider
        self._coding_model: str = coding_model
        self._coding_api_key: str = coding_api_key
        self._available_packages: list[str] = self.probe_packages()

    @staticmethod
    def probe_packages() -> list[str]:
        """Return sorted list of installed package distribution names."""
        from importlib.metadata import distributions

        return sorted({d.metadata["Name"] for d in distributions()})

    async def get_or_create(self, name: str) -> Scratchpad:
        """Return existing pad or create + start a new one."""
        if name not in self._pads:
            pad = Scratchpad(
                name=name,
                _skill_dirs=self._skill_dirs,
                _coding_provider=self._coding_provider,
                _coding_model=self._coding_model,
                _coding_api_key=self._coding_api_key,
            )
            await pad.start()
            self._pads[name] = pad
        return self._pads[name]

    async def remove(self, name: str) -> str:
        """Kill and delete a scratchpad."""
        pad = self._pads.pop(name, None)
        if pad is None:
            return f"No scratchpad named '{name}'."
        await pad.close()
        return f"Scratchpad '{name}' removed."

    def list_pads(self) -> list[str]:
        return list(self._pads.keys())

    async def close_all(self) -> None:
        """Cleanup all scratchpads on session end."""
        for pad in self._pads.values():
            await pad.close()
        self._pads.clear()
