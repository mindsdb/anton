#!/usr/bin/env python3
"""Decide whether a PR needs the verifier's full live verdict matrix (ENG-2863).

The `verdict-eval` workflow triggers on `anton/core/session.py` (~6k lines,
touched by most PRs) plus two LLM plumbing files. Only edits to the parts of
those files that the verifier's *judgment* depends on need the ~100-call live
matrix; everything else needs only the one-call truncation guard.

"Depends on" is decided by LINE SPANS, not by grepping changed text: each
version of each file is parsed with `ast`, the spans of the named classes,
functions and module constants are taken, and the PR's hunks (`git diff -U0`)
are intersected with them on both the old and the new side. A body-only edit
inside `_render_verify_transcript` counts (its span moved); an edit anywhere
in `turn_stream`, which merely *calls* the verifier, does not. A change to the
eval module itself, or to the small structured-output builder, always counts.

Stdlib only: the workflow runs it before dependencies are installed. Pure
functions are kept separate from git/subprocess so tests/test_verifier_eval_gate.py
can pin them without a repository.
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys

# Names whose spans decide the scope, per file. A renamed symbol here must be
# renamed in the code too, or the gate test fails — that is the point of the
# test, not an inconvenience: a marker that matches nothing turns the matrix
# off for every PR touching that code.
SPAN_MARKERS: dict[str, tuple[str, ...]] = {
    "anton/core/session.py": (
        "_VerifierVerdict",
        "_VERIFIER_TOKEN_BUDGETS",
        "_VERIFIER_NO_PREAMBLE",
        "_VERIFIER_JUDGMENT_RUBRIC",
        "_build_verify_request",
        "_render_verify_transcript",
        "_render_tool_result_content",
        "_clip_keep_cause",
    ),
    "anton/core/llm/client.py": (
        # The public entry points are 5-line delegators; the forced-tool-call
        # body they delegate to is where the verdict call actually happens
        # (tool_choice, budget, truncation classification). Review of #486
        # mutated tool_choice inside it and the first cut said "guard".
        "_generate_object_with",
        "_call_with_auth_confirmation",
        "generate_object",
        "generate_object_code",
    ),
}
# Any change at all in these files runs the full matrix.
ALWAYS_FULL: tuple[str, ...] = (
    "tests/test_verifier_verdict_live.py",
    "anton/core/llm/structured.py",
    ".github/scripts/verifier_eval_scope.py",
)
GUARD_K = "narrating_model_reaches_a_verdict"

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def marker_spans(source: str, names: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    """Map each marker name to its (first, last) line span in `source`.

    Classes and functions (including methods, so `generate_object_code` on
    `LLMClient` resolves) by name; module-level constants by assignment target.
    Names that do not resolve are simply absent — the caller decides whether
    that is an error (the gate test) or a non-match (a deleted symbol).
    """
    spans: dict[str, tuple[int, int]] = {}
    tree = ast.parse(source)
    for node in ast.walk(tree):
        name = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            name = targets[0] if len(targets) == 1 else None
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if name in names and name not in spans:
            spans[name] = (node.lineno, node.end_lineno or node.lineno)
    return spans


def parse_hunks(diff: str) -> list[tuple[int, int, int, int]]:
    """(old_start, old_len, new_start, new_len) for every hunk header."""
    out = []
    for line in diff.splitlines():
        m = _HUNK.match(line)
        if m:
            out.append(
                (
                    int(m.group(1)),
                    int(m.group(2)) if m.group(2) is not None else 1,
                    int(m.group(3)),
                    int(m.group(4)) if m.group(4) is not None else 1,
                )
            )
    return out


def _intersects(start: int, length: int, span: tuple[int, int]) -> bool:
    # A zero-length hunk side is an insertion/deletion point AT `start`; treat
    # it as touching the line it sits on so an insertion inside a span counts.
    end = start + max(length, 1) - 1
    return start <= span[1] and end >= span[0]


def touched(
    hunks: list[tuple[int, int, int, int]],
    old_spans: dict[str, tuple[int, int]],
    new_spans: dict[str, tuple[int, int]],
) -> set[str]:
    """Marker names whose span any hunk touches, on either side."""
    hit: set[str] = set()
    for old_start, old_len, new_start, new_len in hunks:
        for name, span in old_spans.items():
            if _intersects(old_start, old_len, span):
                hit.add(name)
        for name, span in new_spans.items():
            if _intersects(new_start, new_len, span):
                hit.add(name)
    return hit


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def _show(rev: str, path: str) -> str:
    if rev == "WORKTREE":
        try:
            return open(path, encoding="utf-8").read()
        except FileNotFoundError:
            return ""
    try:
        return _git("show", f"{rev}:{path}")
    except subprocess.CalledProcessError:
        return ""  # file absent at that revision


def decide(base: str, head: str) -> tuple[str, list[str]]:
    """Return ("full" | "guard", reasons).

    `base` is reduced to the MERGE BASE with `head` first. The workflow passes
    `pull_request.base.sha`, which is the base branch's *tip*, and a two-dot
    diff against the tip includes everything the base gained since the branch
    point as a reverse change. Measured on #486's review: five real
    guard-eligible PRs flipped to "full" purely because staging had gained an
    ALWAYS_FULL commit after they branched — the saving evaporating exactly
    when verifier work is active. The error is in the safe direction, but cost
    is the whole point of this script.
    """
    head_arg = [] if head == "WORKTREE" else [head]
    base = _git("merge-base", base, "HEAD" if head == "WORKTREE" else head).strip()
    changed = _git("diff", "--name-only", base, *head_arg, "--").split()
    reasons: list[str] = []
    for path in ALWAYS_FULL:
        if path in changed:
            reasons.append(f"{path} changed")
    for path, names in SPAN_MARKERS.items():
        if path not in changed:
            continue
        diff = _git("diff", "-U0", base, *head_arg, "--", path)
        old_spans = marker_spans(_show(base, path), names)
        new_spans = marker_spans(_show(head, path), names)
        for name in sorted(touched(parse_hunks(diff), old_spans, new_spans)):
            reasons.append(f"{path}: {name}")
    return ("full" if reasons else "guard"), reasons


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="base revision (sha or ref)")
    ap.add_argument("--head", required=True, help="head revision, or WORKTREE for the working tree")
    ap.add_argument("--github-output", help="path to $GITHUB_OUTPUT to append scope= and pytest_args=")
    a = ap.parse_args(argv)
    try:
        scope, reasons = decide(a.base, a.head)
    except Exception as exc:  # noqa: BLE001 — fail CLOSED, to the full matrix
        # A git error, an unparsable revision, anything unexpected: the safe
        # answer is the full matrix, never a red job and never the guard. A
        # decision step that can fail open would re-create ENG-1334.
        scope, reasons = "full", [f"scope decision failed, defaulting to full: {exc!r}"[:300]]
    print(f"verifier-eval scope: {scope}")
    for r in reasons:
        print(f"  because {r}")
    if a.github_output:
        with open(a.github_output, "a", encoding="utf-8") as fh:
            fh.write(f"scope={scope}\n")
            fh.write(f"pytest_args={'' if scope == 'full' else '-k ' + GUARD_K}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
