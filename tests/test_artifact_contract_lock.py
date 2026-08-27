"""Contract lock: every verifier rule is mentioned in a generator prompt.

The rules the generated code must satisfy live as string literals inside
`errors.append(...)` / `warnings.append(...)` and
`VerifyResult(errors=[...], warnings=[...])`. There is no registry, so this module
rebuilds one by walking the AST and compares it as a set against a hand-written
`rule -> expected marker in prompt` table.

A new, deleted or reworded rule breaks the set comparison — which is the whole
point: the table cannot silently drift away from the code.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from anton.core.tools.generate_artifact import orchestrator as orchestrator_mod
from anton.core.tools.generate_artifact import prompts
from anton.core.tools.generate_artifact import verifiers as verifiers_mod

KEY_LEN = 60
_MAX_DEPTH = 10

# Resolution sentinels. UNRESOLVED — neither static text nor an empty list could
# be extracted from the expression (fails the test). EMPTY — the expression is an
# empty list literal, so its contribution is known to be zero.
UNRESOLVED = object()
EMPTY = object()

_RULE_KWARGS = ("errors", "warnings")


def _squash(text: str) -> str:
    """Case and line breaks are insignificant. NO truncation — safe for prompt lookups."""
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize(text: str) -> str:
    """A rule key: the same plus a length limit.

    Truncation is only needed for rule keys. NEVER apply `normalize` to a whole
    prompt: it would cut it to 60 characters and every marker would "go missing".
    Use `_squash` for prompt lookups.
    """
    return _squash(text)[:KEY_LEN]


@dataclass
class ExtractResult:
    errors: set[str] = field(default_factory=set)
    warnings: set[str] = field(default_factory=set)
    unresolved: list[str] = field(default_factory=list)


def _module_assignments(tree: ast.Module) -> dict[str, ast.expr]:
    out: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if node.value is None:
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                out[t.id] = node.value
    return out


def _local_assignments(fn: ast.AST) -> dict[str, list[ast.expr]]:
    """Every assignment inside the function. A list, so ambiguity is visible:
    a name assigned twice cannot be resolved."""
    out: dict[str, list[ast.expr]] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if node.value is None:
            continue
        for t in targets:
            if isinstance(t, ast.Name):
                out.setdefault(t.id, []).append(node.value)
    return out


def _resolve_name(name: str, local, module) -> ast.expr | None:
    cands = local.get(name)
    if cands is None and name in module:
        cands = [module[name]]
    if not cands or len(cands) != 1:
        return None
    return cands[0]


def _static_head(expr: ast.expr, *, local, module, depth: int = 0):
    """The longest statically known text prefix of the expression."""
    if depth > _MAX_DEPTH:
        return UNRESOLVED
    if isinstance(expr, ast.Constant):
        return expr.value if isinstance(expr.value, str) else UNRESOLVED
    if isinstance(expr, ast.JoinedStr):
        head = ""
        for part in expr.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                head += part.value
            else:
                break
        return head or UNRESOLVED
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return _static_head(expr.left, local=local, module=module, depth=depth + 1)
    if isinstance(expr, ast.List):
        return EMPTY if not expr.elts else UNRESOLVED
    if isinstance(expr, ast.Name):
        target = _resolve_name(expr.id, local, module)
        if target is None:
            return UNRESOLVED
        return _static_head(target, local=local, module=module, depth=depth + 1)
    return UNRESOLVED


def _unwrap_list(expr: ast.expr, *, local, module) -> list[ast.expr]:
    """Elements of the list passed to VerifyResult(errors=...).

    An accumulator variable (`errors: list[str] = []`) unwraps to an empty list, so
    its contribution is zero: its items were already collected at their own
    `errors.append` calls earlier in the function.
    """
    if isinstance(expr, ast.List):
        return list(expr.elts)
    if isinstance(expr, ast.Name):
        target = _resolve_name(expr.id, local, module)
        if isinstance(target, ast.List):
            return list(target.elts)
    return [expr]


def _rule_args(call: ast.Call, *, local, module):
    """(tier, expression) for every rule this call emits."""
    fn = call.func
    if isinstance(fn, ast.Attribute) and fn.attr == "append":
        owner = fn.value
        tier = owner.attr if isinstance(owner, ast.Attribute) else getattr(owner, "id", None)
        if tier in _RULE_KWARGS and call.args:
            yield tier, call.args[0]
        return
    callee = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
    if callee == "VerifyResult":
        for kw in call.keywords:
            if kw.arg not in _RULE_KWARGS:
                continue
            for item in _unwrap_list(kw.value, local=local, module=module):
                yield kw.arg, item


def extract_rules(source: str, func_names: tuple[str, ...]) -> ExtractResult:
    """Collect the rules from the named functions of the source."""
    tree = ast.parse(source)
    module = _module_assignments(tree)
    result = ExtractResult()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.name not in func_names:
            continue
        local = _local_assignments(fn)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            for tier, expr in _rule_args(node, local=local, module=module):
                head = _static_head(expr, local=local, module=module)
                if head is EMPTY:
                    continue
                if head is UNRESOLVED:
                    result.unresolved.append(f"line {expr.lineno}")
                    continue
                getattr(result, tier).add(normalize(head))
    return result


# ── Extractor tests (synthetic sources) ──────────────────────────────────────

def test_extracts_plain_literal_append():
    src = """
def verify_x():
    errors = []
    errors.append("Plain rule text.")
    return VerifyResult(errors=errors)
"""
    r = extract_rules(src, ("verify_x",))
    assert r.errors == {normalize("Plain rule text.")}
    assert r.warnings == set()
    assert r.unresolved == []


def test_accumulator_contributes_nothing_and_does_not_fail():
    """VerifyResult(errors=errors) backed by an empty literal contributes nothing.

    This is the first `return` in verifiers.py; a naive implementation dies on it.
    """
    src = """
def verify_x():
    errors = []
    warnings = []
    return VerifyResult(errors=errors, warnings=warnings)
"""
    r = extract_rules(src, ("verify_x",))
    assert r.errors == set()
    assert r.warnings == set()
    assert r.unresolved == []


def test_extracts_constructor_literal_list():
    src = """
def verify_x():
    return VerifyResult(errors=["Infra failure happened."])
"""
    r = extract_rules(src, ("verify_x",))
    assert r.errors == {normalize("Infra failure happened.")}


def test_extracts_fstring_static_head():
    src = '''
def verify_x():
    errors = []
    errors.append(f"Absolute URL not allowed: {value!r} tail")
'''
    r = extract_rules(src, ("verify_x",))
    assert r.errors == {normalize("Absolute URL not allowed: ")}


def test_extracts_binop_left_head():
    src = """
def verify_x():
    errors = []
    errors.append("Root routes found: " + ", ".join(routes))
"""
    r = extract_rules(src, ("verify_x",))
    assert r.errors == {normalize("Root routes found: ")}


def test_resolves_local_variable_and_chained_concat():
    """The DS_* rule is built into a local variable and concatenated from four parts."""
    src = """
def _gen_verify_backend():
    msg = (
        "backend reads DS_* env keys with no matching vault "
        "connection: " + ", ".join(unmapped)
        + ". Available: " + known
    )
    verdict.errors.append(msg)
"""
    r = extract_rules(src, ("_gen_verify_backend",))
    assert r.errors == {
        normalize("backend reads DS_* env keys with no matching vault connection: ")
    }
    assert r.unresolved == []


def test_warnings_tier_is_separate():
    src = """
def verify_x():
    warnings = []
    warnings.append("Advisory only.")
"""
    r = extract_rules(src, ("verify_x",))
    assert r.errors == set()
    assert r.warnings == {normalize("Advisory only.")}


def test_ambiguous_name_is_unresolved():
    """A name assigned twice cannot be resolved — that must be visible."""
    src = """
def verify_x():
    errors = []
    msg = "first"
    msg = "second"
    errors.append(msg)
"""
    r = extract_rules(src, ("verify_x",))
    assert r.errors == set()
    assert len(r.unresolved) == 1


def test_ignores_calls_outside_target_functions():
    src = """
def other():
    errors = []
    errors.append("Not in scope.")
"""
    r = extract_rules(src, ("verify_x",))
    assert r.errors == set()


def test_ignores_non_rule_assignments():
    """`extra = ...` is not a rule and must not enter the set."""
    src = """
def verify_x():
    extra = "## Verification failed — fix these"
    errors = []
    errors.append("Real rule.")
"""
    r = extract_rules(src, ("verify_x",))
    assert r.errors == {normalize("Real rule.")}


def test_squash_does_not_truncate_but_normalize_does():
    """The two normalisations are not interchangeable.

    `normalize` applied to a whole prompt would cut it to 60 characters and "lose"
    every marker — prompt lookups must go through `_squash`.
    """
    long = "x" * 200
    assert len(_squash(long)) == 200
    assert len(normalize(long)) == KEY_LEN
    assert _squash("  A\n  B  ") == "a b"


# ── Rule table ───────────────────────────────────────────────────────────────

_VERIFIER_FUNCS = ("verify_frontend", "evaluate_backend", "verify_backend")
_ORCHESTRATOR_FUNCS = ("_gen_verify_backend", "_gen_verify_frontend")

_ARTIFACT_PATH = Path("/tmp/artifact-lock-probe")


def _prompts() -> dict[str, str]:
    """The prompts exactly as the generator receives them."""
    return {
        "html": prompts.build_subagent_system_prompt("html-app", _ARTIFACT_PATH),
        "frontend": prompts.build_frontend_system_prompt(_ARTIFACT_PATH),
        "backend_stateless": prompts.build_backend_system_prompt(
            _ARTIFACT_PATH, stateless=True
        ),
        "backend_stateful": prompts.build_backend_system_prompt(
            _ARTIFACT_PATH, stateless=False
        ),
    }


_BACKEND_BOTH = ("backend_stateless", "backend_stateful")
_FRONT_BOTH = ("html", "frontend")


@dataclass(frozen=True)
class Rule:
    tier: str                            # "errors" | "warnings"
    head: str                            # static prefix of the verifier message
    checks: tuple[tuple[str, str], ...]  # (prompt key, marker expected in it)


def _both(prompt_keys: tuple[str, ...], marker: str) -> tuple[tuple[str, str], ...]:
    return tuple((k, marker) for k in prompt_keys)


RULES: tuple[Rule, ...] = (
    # ── verify_frontend: applies to html-app and the fullstack frontend alike ──
    Rule("errors", "Frontend must be a valid HTML document with an explicit <body>...</body>.",
         _both(_FRONT_BOTH, "explicit `<body>`")),
    Rule("errors", 'Missing <meta name="viewport" content="width=device-width, initial-scale=1.0">.',
         _both(_FRONT_BOTH, 'name="viewport"')),
    Rule("errors", "Absolute URL is not allowed in fetch(): ...",
         _both(_FRONT_BOTH, "absolute URL")),
    Rule("errors", "Absolute URL is not allowed in resource references: ...",
         _both(_FRONT_BOTH, "resource reference")),
    Rule("errors", "Frontend must not use the global name window.__antonCommentsLayer.",
         _both(_FRONT_BOTH, "__antonCommentsLayer")),
    Rule("errors", "Frontend contains a mangled script tag (an underscore variant like "
                    "`<_script` or `</_script`); write plain `<script>`/`</script>`.",
         _both(_FRONT_BOTH, "underscore")),
    Rule("errors", "Frontend opens a <script> block but never closes it with </script>.",
         _both(_FRONT_BOTH, "must be closed")),
    Rule("errors", "Frontend must not use universal `* { ... !important }` rules.",
         _both(_FRONT_BOTH, "!important")),
    Rule("errors", "Frontend uses an extreme z-index (> 1000); keep it within a sane range.",
         _both(_FRONT_BOTH, "z-index")),
    Rule("warnings", "Significant blocks have no stable `id` attributes.",
         _both(_FRONT_BOTH, "stable `id`")),
    Rule("warnings", "Chart/library CDN other than ECharts detected: ",
         _both(_FRONT_BOTH, "ECharts")),
    # ── verify_frontend, fullstack only ──
    Rule("errors", 'Missing <meta name="api-base" content=""> (required for fullstack frontends).',
         (("frontend", 'name="api-base"'),)),
    Rule("errors", "Backend call must use the /api/* prefix, got: ",
         (("frontend", "/api/*"),)),
    # ── evaluate_backend ──
    Rule("errors", "backend.py must define `app` as a FastAPI instance.",
         _both(_BACKEND_BOTH, "app = FastAPI()")),
    Rule("errors", 'backend.py must define `handler = Mangum(app, lifespan="off")`.',
         _both(_BACKEND_BOTH, 'Mangum(app, lifespan="off")')),
    Rule("errors", "backend.py must define a module-level `SECRETS` dict.",
         _both(_BACKEND_BOTH, "SECRETS")),
    Rule("errors", "All API routes must live under /api/*; found root routes: ",
         _both(_BACKEND_BOTH, "/api/*")),
    Rule("errors", "backend.py must expose GET /api/health.",
         _both(_BACKEND_BOTH, "/api/health")),
    Rule("errors", "Secret copied into module-level variable `",
         _both(_BACKEND_BOTH, "point of use")),
    Rule("errors", "requirements.txt must list `",
         _both(_BACKEND_BOTH, "requirements.txt")),
    # ── evaluate_backend: STATE contract (PR #259) ──
    Rule("errors", "requirements.txt must not list `anton_state` — the STATE SDK is ",
         (("backend_stateful", "NEVER list `anton_state`"),)),
    Rule("errors", "backend.py must define a module-level `STATE = None` slot (the cloud runner overlays",
         (("backend_stateful", "STATE = None"),)),
    Rule("errors", "state_manifest.json is missing — a stateful backend must declare its STATE key schema",
         (("backend_stateful", "state_manifest.json"),)),
    Rule("errors", "state_manifest.json is invalid: ",
         (("backend_stateful", "state_manifest.json"),)),
    Rule("errors", "STATE store built at import time via `",
         (("backend_stateful", "point of use"),)),
    Rule("errors", "anton_state imported in a stateless backend — the STATE store is for fullstack-stateful-app",
         (("backend_stateless", "anton_state"),)),
    # ── orchestrator ──
    Rule("errors", "backend reads DS_* env keys with no matching vault connection: ",
         _both(_BACKEND_BOTH, "DS_<ENGINE>_<NAME>__<FIELD>")),
    Rule("errors", "No HTML entry file was written. Write static/index.html (or the html-app page).",
         (("html", "dashboard.html"), ("frontend", "static/index.html"))),
)

# Infrastructure failures: the generator cannot comply with them, so they have no
# prompt marker. Six of the seven are emitted by the VerifyResult constructor
# rather than .append — which is why the walk must cover both forms.
ALLOWLIST: tuple[str, ...] = (
    "backend.py failed to import in the venv: ",
    "backend.py was not written.",
    "Dependency install failed:\n",
    "Scratchpad venv Python is unavailable (remote runtime?).",
    "backend.py failed to compile:\n",
    "backend.py import timed out after ",
    "backend introspection produced no JSON. stderr:\n",
)


def _extract_all() -> ExtractResult:
    merged = ExtractResult()
    for module, funcs in (
        (verifiers_mod, _VERIFIER_FUNCS),
        (orchestrator_mod, _ORCHESTRATOR_FUNCS),
    ):
        src = Path(module.__file__).read_text(encoding="utf-8")
        part = extract_rules(src, funcs)
        merged.errors |= part.errors
        merged.warnings |= part.warnings
        merged.unresolved += [
            f"{Path(module.__file__).name}:{loc.removeprefix('line ')}"
            for loc in part.unresolved
        ]
    return merged


def test_no_unresolvable_rule_literals():
    """An unresolvable argument is a silent hole. Fail the test with its address."""
    found = _extract_all()
    assert not found.unresolved, (
        "unresolvable rule literal: " + ", ".join(found.unresolved)
        + " — inline the literal into the call or add the rule to the RULES table"
    )


def test_rule_table_matches_verifier_source():
    """The sets match: a new/deleted/reworded rule breaks the test."""
    found = _extract_all()
    expected_errors = {normalize(r.head) for r in RULES if r.tier == "errors"}
    expected_errors |= {normalize(a) for a in ALLOWLIST}
    expected_warnings = {normalize(r.head) for r in RULES if r.tier == "warnings"}

    assert found.errors == expected_errors, (
        f"\nin the code only: {sorted(found.errors - expected_errors)}"
        f"\nin the table only: {sorted(expected_errors - found.errors)}"
    )
    assert found.warnings == expected_warnings, (
        f"\nin the code only: {sorted(found.warnings - expected_warnings)}"
        f"\nin the table only: {sorted(expected_warnings - found.warnings)}"
    )


def test_every_rule_has_its_marker_in_the_prompt():
    """Content: the rule's marker is present in the prompt of the right artifact type."""
    rendered = _prompts()
    missing: list[str] = []
    for rule in RULES:
        for prompt_key, marker in rule.checks:
            # _squash, NOT normalize: normalize truncates to KEY_LEN and would turn
            # the prompt into its own first 60 chars — then every marker "goes missing".
            haystack = _squash(rendered[prompt_key])
            if _squash(marker) not in haystack:
                missing.append(f"{prompt_key}: {marker!r} (rule {rule.head[:40]!r})")
    assert not missing, "markers missing from prompts:\n" + "\n".join(missing)
