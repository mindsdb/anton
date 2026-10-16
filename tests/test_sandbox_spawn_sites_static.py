"""No process spawn in anton/ skips the sandbox launcher unnoticed.

Every call that starts a process is listed here by (file, enclosing function,
call), counted, with the reason it may exist. The test fails when the found
sites and the list differ, so a new spawn needs a deliberate entry.

A site whose kind is LAUNCHER runs agent code (a person's venv Python, or a
program it can reach) and must route through `CellSandbox.launch()` when a
policy is set, so its enclosing function must call `.launch(`. A site whose
kind is REFUSED is skipped under a policy by its caller. The other kinds run
only in the CLI, only in the hosted cloud pod, or run no agent code.

Calls are matched through import aliases (`import subprocess as _sp`), so a
local rename cannot hide one.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

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
    "multiprocessing": {"Process", "Pool"},
    "webbrowser": {"open", "open_new", "open_new_tab"},
}

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
    ("core/backends/local.py", "LocalScratchpadRuntime.start.spawn", "asyncio.create_subprocess_exec", LAUNCHER,
     "the cell, with no policy"),
    ("core/backends/local.py", "LocalScratchpadRuntime.start.spawn", "asyncio.create_subprocess_exec", LAUNCHER,
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
     "the headless-browser lint; tool_handlers and the generator skip it under a policy"),
    ("core/artifacts/xlsx_office_check.py", "_convert", "subprocess.Popen", REFUSED,
     "the LibreOffice check; tool_handlers skips it under a policy"),
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
    """Local names for imported modules and callables, mapped to their real names."""
    modules: dict[str, str] = {}
    callables: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules[alias.asname or alias.name.split(".")[0]] = alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                callables[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return modules, callables


def _spawn_label(fn: ast.expr, modules: dict[str, str], callables: dict[str, str]) -> str | None:
    if isinstance(fn, ast.Attribute):
        owner = fn.value
        # `asyncio.subprocess.create_subprocess_exec` and friends
        if isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name):
            root = modules.get(owner.value.id, owner.value.id)
            if root == "asyncio" and fn.attr in _SPAWN_MODULE_ATTRS["asyncio"]:
                return f"asyncio.{fn.attr}"
        if isinstance(owner, ast.Name):
            module = modules.get(owner.id, owner.id)
            if fn.attr in _SPAWN_MODULE_ATTRS.get(module, ()):
                return f"{module}.{fn.attr}"
        return None
    if isinstance(fn, ast.Name) and fn.id in callables:
        module, _, attr = callables[fn.id].rpartition(".")
        root = module.split(".")[0]
        if attr in _SPAWN_MODULE_ATTRS.get(root, ()):
            return f"{root}.{attr}"
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel: str, modules: dict[str, str], callables: dict[str, str]) -> None:
        self.rel = rel
        self._modules = modules
        self._callables = callables
        self._stack: list[str] = []
        self._launch_calls: list[bool] = []
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

    def visit_Call(self, node: ast.Call) -> None:
        label = _spawn_label(node.func, self._modules, self._callables)
        if label is not None:
            self.found.append((self.rel, self._enclosing(), label))
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


def test_the_scan_sees_through_an_import_alias():
    tree = ast.parse("import subprocess as _sp\nfrom asyncio import create_subprocess_exec as go\n"
                     "def f():\n    _sp.run(['x'])\n    go('x')\n")
    visitor = _Visitor("x.py", *_aliases(tree))
    visitor.visit(tree)
    assert visitor.found == [("x.py", "f", "subprocess.run"), ("x.py", "f", "asyncio.create_subprocess_exec")]
