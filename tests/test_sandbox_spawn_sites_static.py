"""No process spawn in anton/ skips the sandbox launcher unnoticed.

Every reference to an API that starts a process is listed here by (file,
enclosing function, API), counted, with the reason it may exist. The test
fails when the found sites and the list differ, so a new spawn needs a
deliberate entry.

A reference counts whether it is called or passed along, so a spawn API
handed to `asyncio.to_thread`, `run_in_executor` or `functools.partial`, or
bound to a local name, is found where it is named. A site whose kind is
LAUNCHER runs agent code (a person's venv Python, or a program it can reach)
and must route through `CellSandbox.launch()` when a policy is set, so its
enclosing function must call `.launch(`. A site whose kind is REFUSED is
never reached under a policy: the browser and LibreOffice checks take a
required `cell_sandbox` and return before the spawn when it is set, and the
backend launcher reads the policy off its pool first. The other kinds run
only in the CLI, only in the hosted cloud pod, or run no agent code.

Names are resolved through import aliases (`import subprocess as _sp`,
`from asyncio import subprocess`), so a rename cannot hide one. Any
`.subprocess_exec` or `.subprocess_shell` counts whatever its receiver (an
event loop), and `getattr(<os|subprocess|asyncio>, name)` counts unless
`name` is a string that names no spawn API.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

ANTON_ROOT = Path(__file__).resolve().parent.parent / "anton"

LAUNCHER = "launcher"  # through CellSandbox.launch() under a policy
REFUSED = "refused"  # skipped by its caller under a policy
CLI = "cli"  # the terminal app only; no host builds a policy there
POD = "pod"  # the hosted cloud pod only, where the sandbox stays off
HOST = "host"  # runs no agent code

_SPAWN_MODULE_ATTRS = {
    "subprocess": {"run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput"},
    "os": {
        "system", "popen", "fork", "forkpty", "posix_spawn", "posix_spawnp",
        "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe",
        "spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe",
    },
    "asyncio": {"create_subprocess_exec", "create_subprocess_shell"},
    "pty": {"spawn", "fork"},
    "multiprocessing": {"Process", "Pool", "get_context"},
    "concurrent.futures": {"ProcessPoolExecutor"},
    "webbrowser": {"open", "open_new", "open_new_tab"},
}
# Event-loop methods that start a process, whatever object they are read from.
_LOOP_SPAWN_ATTRS = {"subprocess_exec", "subprocess_shell"}
# Modules whose attributes a dynamic getattr could turn into a spawn.
_GETATTR_MODULES = {"os", "subprocess", "asyncio"}

#: (path under anton/, enclosing function, call, kind, reason). One entry per
#: occurrence.
ALLOWLIST = (
    ("core/backends/local.py", "LocalScratchpadRuntime._create_venv", "subprocess.run", HOST,
     "uv venv on the server's own interpreter; never reached under a policy, where _uv() is None"),
    ("core/backends/local.py", "LocalScratchpadRuntime._verify_venv_python", "subprocess.run", HOST,
     "the health check with no policy; a policy returns earlier through the sandboxed check"),
    ("core/backends/local.py", "LocalScratchpadRuntime._verify_venv_python_sandboxed", "subprocess.run", LAUNCHER,
     "the venv health check under a policy"),
    ("core/backends/local.py", "LocalScratchpadRuntime._add_windows_firewall_rule", "subprocess.run", HOST,
     "netsh on Windows, which has no sandbox"),
    ("core/backends/local.py", "LocalScratchpadRuntime._start_process.spawn", "asyncio.create_subprocess_exec", LAUNCHER,
     "the cell, with no policy"),
    ("core/backends/local.py", "LocalScratchpadRuntime._start_process.spawn", "asyncio.create_subprocess_exec", LAUNCHER,
     "the cell, through the launcher"),
    ("core/backends/local.py", "LocalScratchpadRuntime.install_packages", "asyncio.create_subprocess_exec", LAUNCHER,
     "package installs, with no policy"),
    ("core/backends/local.py", "LocalScratchpadRuntime.install_packages", "asyncio.create_subprocess_exec", LAUNCHER,
     "package installs, through the launcher"),
    ("core/backends/sandbox.py", "self_test", "subprocess.run", LAUNCHER,
     "the startup self-test's child, through the launcher"),
    ("core/backends/landlock_launcher.py", "_exec", "os.execve", HOST,
     "the launcher replacing itself with the target after the lock"),
    ("core/tools/generate_artifact/verifiers.py", "_spawn_check", "asyncio.create_subprocess_exec", LAUNCHER,
     "the verifier's compile and import checks, with no policy"),
    ("core/tools/generate_artifact/verifiers.py", "_spawn_check", "asyncio.create_subprocess_exec", LAUNCHER,
     "the verifier's compile and import checks, through the launcher"),
    ("core/artifacts/backend_launcher.py", "launch_artifact_backend", "asyncio.create_subprocess_exec", REFUSED,
     "app backends; launch_artifact_backend returns BACKENDS_OFF_MESSAGE under a policy first"),
    ("core/artifacts/html_lint.py", "_lint_target", "subprocess.run", REFUSED,
     "the headless-browser lint; lint_html and lint_url return None under a policy first"),
    ("core/artifacts/xlsx_office_check.py", "_convert", "subprocess.Popen", REFUSED,
     "the LibreOffice check; check_xlsx_via_office and check_opens_via_office return None under a policy first"),
    ("cli.py", "_reexec", "os.execv", CLI, "the CLI re-executing itself after an update"),
    ("cli.py", "_ensure_dependencies", "subprocess.run", CLI, "the CLI installing its own dependencies"),
    ("cli.py", "_ensure_terms_consent", "os.system", CLI, "clearing the terminal"),
    ("cli.py", "_render_terms_screen", "os.system", CLI, "clearing the terminal"),
    ("chat.py", "_agent_zero", "os.system", CLI, "clearing the terminal"),
    ("clipboard.py", "_grab_text", "subprocess.run", CLI, "reading the terminal clipboard (pbpaste)"),
    ("clipboard.py", "_grab_text", "subprocess.run", CLI, "reading the terminal clipboard (PowerShell)"),
    ("clipboard.py", "_grab_text", "subprocess.run", CLI, "reading the terminal clipboard (wl-paste)"),
    ("clipboard.py", "_grab_text", "subprocess.run", CLI, "reading the terminal clipboard (xclip)"),
    ("utils/clipboard.py", "ensure_clipboard.<lambda>", "subprocess.run", CLI,
     "installing Pillow into the CLI's own interpreter with uv"),
    ("utils/clipboard.py", "ensure_clipboard.<lambda>", "subprocess.run", CLI,
     "installing Pillow into the CLI's own interpreter with pip"),
    ("chat.py", "_handle_publish", "webbrowser.open", CLI, "opening a published page in the terminal user's browser"),
    ("chat.py", "_handle_publish", "webbrowser.open", CLI, "opening a published page in the terminal user's browser"),
    ("chat.py", "_handle_remote", "webbrowser.open", CLI, "opening the remote sign-in page"),
    ("cli.py", "_setup_minds", "webbrowser.open", CLI, "opening the sign-in page during CLI setup"),
    ("tools.py", "handle_publish_or_preview", "webbrowser.open", CLI,
     "the CLI's publish tool; cowork-server registers its own publish handler"),
    ("tools.py", "handle_publish_or_preview", "webbrowser.open", CLI,
     "the CLI's publish tool; cowork-server registers its own publish handler"),
    ("demo_data/nvda_btc_scratchpad_backup.py", "<module>", "webbrowser.open", CLI,
     "the CLI demo's sample script, run by the terminal demo"),
    ("updater.py", "_check_and_update", "subprocess.run", CLI, "the CLI self-updater"),
    ("updater.py", "_read_installed_anton_version", "subprocess.run", CLI, "the CLI self-updater"),
    ("cloud_turn/launcher.py", "run_turn_directly", "os.execv", POD, "the hosted pod's entrypoint"),
    ("cloud_turn/resident.py", "_Resident._fork", "os.fork", POD, "the hosted pod's resident server"),
)


def _aliases(tree: ast.AST) -> tuple[dict[str, str], dict[str, str]]:
    """Local names for imported modules and imported names, mapped to their dotted names."""
    modules: dict[str, str] = {}
    callables: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    modules[alias.asname] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    modules[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                callables[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return modules, callables


def _dotted(node: ast.expr, modules: dict[str, str], callables: dict[str, str]) -> list[str]:
    """Every dotted name ``node`` can read, resolved through import aliases.

    The alias maps cover the whole file, so a name imported both as a module
    and as a name (`import subprocess` at the top, `from asyncio import
    subprocess` in one function) gives both readings.
    """
    if isinstance(node, ast.Name):
        return [name for name in (callables.get(node.id), modules.get(node.id)) if name]
    if isinstance(node, ast.Attribute):
        return [f"{owner}.{node.attr}" for owner in _dotted(node.value, modules, callables)]
    return []


def _api_label(dotted: str) -> str | None:
    """The label of the spawn API ``dotted`` names, or None when it names none."""
    module, _, attr = dotted.rpartition(".")
    # `asyncio.subprocess.create_subprocess_exec` is asyncio's own.
    if module == "asyncio.subprocess":
        module = "asyncio"
    if attr in _SPAWN_MODULE_ATTRS.get(module, ()):
        return f"{module}.{attr}"
    return None


def _spawn_label(node: ast.expr, modules: dict[str, str], callables: dict[str, str]) -> str | None:
    """The spawn API an ``ast.Name`` or ``ast.Attribute`` refers to, or None."""
    if isinstance(node, ast.Attribute) and node.attr in _LOOP_SPAWN_ATTRS:
        return f"loop.{node.attr}"
    for dotted in _dotted(node, modules, callables):
        label = _api_label(dotted)
        if label is not None:
            return label
    return None


def _getattr_label(node: ast.Call, modules: dict[str, str], callables: dict[str, str]) -> str | None:
    """The spawn a ``getattr(<module>, name)`` call can reach, or None."""
    if not (isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 2):
        return None
    name = node.args[1]
    for owner in _dotted(node.args[0], modules, callables):
        if owner.split(".")[0] not in _GETATTR_MODULES:
            continue
        if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
            return f"getattr({owner})"
        label = _api_label(f"{owner}.{name.value}")
        if label is not None:
            return label
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel: str, modules: dict[str, str], callables: dict[str, str]) -> None:
        self.rel = rel
        self._modules = modules
        self._callables = callables
        self._stack: list[str] = []
        self.found: list[tuple[str, str, str]] = []
        self.launching: set[str] = set()

    def _enclosing(self) -> str:
        return ".".join(self._stack) or "<module>"

    def _scope(self, name: str, node: ast.AST) -> None:
        self._stack.append(name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "launch"
            for n in ast.walk(node)
        ):
            self.launching.add(self._enclosing())
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scope(node.name, node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scope(node.name, node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope(node.name, node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._scope("<lambda>", node)

    def _record(self, label: str) -> None:
        self.found.append((self.rel, self._enclosing(), label))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        label = _spawn_label(node, self._modules, self._callables)
        if label is not None:
            # Its inner names are the same reference.
            self._record(label)
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            label = _spawn_label(node, self._modules, self._callables)
            if label is not None:
                self._record(label)

    def visit_Call(self, node: ast.Call) -> None:
        label = _getattr_label(node, self._modules, self._callables)
        if label is not None:
            self._record(label)
        self.generic_visit(node)


def _scan() -> tuple[Counter[tuple[str, str, str]], set[tuple[str, str]]]:
    found: Counter[tuple[str, str, str]] = Counter()
    launching: set[tuple[str, str]] = set()
    for path in sorted(ANTON_ROOT.rglob("*.py")):
        rel = str(path.relative_to(ANTON_ROOT))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _Visitor(rel, *_aliases(tree))
        visitor.visit(tree)
        found.update(visitor.found)
        launching.update((rel, fn) for fn in visitor.launching)
    return found, launching


def _render(counts: Counter[tuple[str, str, str]]) -> str:
    return "\n".join(
        f"  {rel}: {fn}: {label}" + (f" (x{n})" if n > 1 else "") for (rel, fn, label), n in sorted(counts.items())
    )


def test_every_process_spawn_in_anton_is_listed():
    found, _ = _scan()
    allowed = Counter((rel, fn, label) for rel, fn, label, _kind, _reason in ALLOWLIST)
    new = found - allowed
    assert not new, (
        "New process spawn(s) in anton/. If it runs agent code, route it through "
        "CellSandbox.launch() under a policy, then list it here with its kind and reason:\n" + _render(new)
    )
    gone = allowed - found
    assert not gone, "Listed spawn site(s) no longer exist; remove them from ALLOWLIST:\n" + _render(gone)


def test_every_site_that_runs_agent_code_calls_the_launcher():
    _, launching = _scan()
    skipped = sorted(
        {(rel, fn) for rel, fn, _label, kind, _reason in ALLOWLIST if kind == LAUNCHER} - launching
    )
    assert not skipped, f"These sites run agent code without calling CellSandbox.launch(): {skipped}"


def _found(source: str) -> list[tuple[str, str, str]]:
    tree = ast.parse(source)
    visitor = _Visitor("x.py", *_aliases(tree))
    visitor.visit(tree)
    return visitor.found


def test_the_scan_sees_through_an_import_alias():
    assert _found(
        "import subprocess as _sp\nfrom asyncio import create_subprocess_exec as go\n"
        "def f():\n    _sp.run(['x'])\n    go('x')\n"
    ) == [("x.py", "f", "subprocess.run"), ("x.py", "f", "asyncio.create_subprocess_exec")]


@pytest.mark.parametrize(
    ("source", "label"),
    [
        ("import asyncio, subprocess\nasync def f(v):\n    await asyncio.to_thread(subprocess.run, [v])\n",
         "subprocess.run"),
        ("import subprocess\nasync def f(loop, v):\n    await loop.run_in_executor(None, subprocess.run, [v])\n",
         "subprocess.run"),
        ("import functools, subprocess\ndef f(v):\n    return functools.partial(subprocess.Popen, [v])\n",
         "subprocess.Popen"),
        ("import subprocess\ndef f(v):\n    run = subprocess.run\n    run([v])\n", "subprocess.run"),
        ("from subprocess import run\ndef f(v):\n    go = run\n", "subprocess.run"),
        ("async def f(loop, v):\n    await loop.subprocess_exec(object, v)\n", "loop.subprocess_exec"),
        ("async def f(v):\n    await self.loop.subprocess_shell(object, v)\n", "loop.subprocess_shell"),
        ("from asyncio import subprocess as asp\nasync def f(v):\n    await asp.create_subprocess_exec(v)\n",
         "asyncio.create_subprocess_exec"),
        ("from asyncio import subprocess\nasync def f(v):\n    await subprocess.create_subprocess_shell(v)\n",
         "asyncio.create_subprocess_shell"),
        ("import subprocess\ndef g():\n    from asyncio import subprocess\ndef f(v):\n    subprocess.run([v])\n",
         "subprocess.run"),
        ("import subprocess\ndef f(v):\n    getattr(subprocess, 'run')([v])\n", "subprocess.run"),
        ("import os\ndef f(name, v):\n    getattr(os, name)(v)\n", "getattr(os)"),
        ("import multiprocessing\ndef f():\n    multiprocessing.get_context('fork').Pool(2)\n",
         "multiprocessing.get_context"),
        ("from concurrent.futures import ProcessPoolExecutor\ndef f():\n    ProcessPoolExecutor()\n",
         "concurrent.futures.ProcessPoolExecutor"),
        ("import concurrent.futures\ndef f():\n    concurrent.futures.ProcessPoolExecutor()\n",
         "concurrent.futures.ProcessPoolExecutor"),
    ],
)
def test_the_scan_sees_a_spawn_api_passed_along_or_renamed(source, label):
    assert _found(source) == [("x.py", "f", label)]


def test_the_scan_ignores_a_getattr_that_names_no_spawn_api():
    assert _found("import os\ndef f():\n    return getattr(os, 'O_NOFOLLOW', 0)\n") == []
