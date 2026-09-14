"""Pin what the yolo engine is allowed to depend on.

`agent.py` imports anton's `LLMClient` directly — a yolo run should get
the configured provider, the coding-model split, forced `tool_choice`,
pydantic validation and turn tracing, and the handler should pass the
same client the rest of the agent already holds. There is nothing to
abstract there.

`patch.py` and `workspace.py` are different. They are pure functions over
strings and paths, and that is exactly why the engine is provable in
milliseconds instead of needing a provider. That property is easy to lose
to one convenience import, so it is pinned here.
"""

from __future__ import annotations

import ast
import pathlib

# Pure: these must reach for nothing at all.
ENGINE = ["patch.py", "workspace.py"]
# Builds on the engine, and on nothing else in anton.
ENGINE_ADJACENT = ["data.py"]
YOLO = pathlib.Path(__file__).resolve().parents[1] / "anton" / "core" / "yolo"


def imports_of(filename: str) -> set[str]:
    tree = ast.parse((YOLO / filename).read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_the_engine_imports_only_the_standard_library():
    """No anton, no pydantic, no SDKs. If this fails, ask whether the
    import is worth giving up a test suite that runs without a provider."""
    allowed = {
        "__future__", "contextlib", "dataclasses", "datetime", "json",
        "pathlib", "re", "signal", "threading", "typing",
    }
    for filename in ENGINE:
        assert imports_of(filename) <= allowed, (
            f"{filename} imports {sorted(imports_of(filename) - allowed)}; "
            "the engine is meant to be pure"
        )


def test_the_engine_adjacent_modules_reach_no_further_than_the_engine():
    """data.py may build on workspace.py. It may not reach into anton's
    LLM layer, its settings, or its session — that would make the data
    format untestable without a running agent."""
    import ast

    for filename in ENGINE_ADJACENT:
        tree = ast.parse((YOLO / filename).read_text())
        for node in ast.walk(tree):
            module = getattr(node, "module", None)
            if isinstance(node, ast.ImportFrom) and module and module.startswith("anton"):
                assert module.startswith("anton.core.yolo"), (
                    f"{filename} imports {module}; engine-adjacent code may only "
                    "build on the yolo package itself"
                )


def test_the_engine_runs_with_no_provider_configured(tmp_path):
    """The point of the rule above, demonstrated rather than asserted."""
    from anton.core.yolo.patch import parse_patch
    from anton.core.yolo.workspace import Workspace

    workspace = Workspace(tmp_path)
    workspace.write("f.txt", "before\n")
    [file_patch] = parse_patch("--- a/f.txt\n+++ b/f.txt\n@@\n-before\n+after\n")
    from anton.core.yolo.patch import apply_hunks

    assert apply_hunks(workspace.read("f.txt"), file_patch.hunks) == "after\n"


def test_the_loop_takes_the_real_client():
    """agent.py names LLMClient, so a wrong object is a type error at the
    call site rather than a surprise at runtime."""
    import inspect

    from anton.core.llm.client import LLMClient
    from anton.core.yolo import YoloEditor

    annotation = inspect.get_annotations(YoloEditor)["llm_client"]
    assert annotation is LLMClient or annotation == "LLMClient"
