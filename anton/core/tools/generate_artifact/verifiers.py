"""Verifiers for the artifact-generation FSM.

`verify_frontend` and `evaluate_backend` are pure functions (deterministic,
no I/O) so they are fully unit-testable. `verify_backend` is the async glue
that installs deps in the scratchpad venv, imports the module in a subprocess
with a timeout, and folds the result into `evaluate_backend`.
"""
from __future__ import annotations

import re

from .state import VerifyResult

_FETCH_CALL = re.compile(r"""fetch\s*\(\s*(?:api\s*\(\s*)?['"]([^'"]+)['"]""")
_BARE_SCRIPT_SRC = re.compile(r"""<script[^>]*\bsrc\s*=\s*['"]([^'"]+)['"]""", re.I)
_ECHARTS_CDN = "cdn.jsdelivr.net/npm/echarts"


def verify_frontend(html: str, *, is_fullstack: bool) -> VerifyResult:
    errors: list[str] = []
    warnings: list[str] = []
    low = html.lower()

    # 1. Valid document with an explicit <body>.
    if "<body" not in low or "</body>" not in low:
        errors.append("Frontend must be a valid HTML document with an explicit <body>...</body>.")

    # 2. viewport meta (match the actual <meta> tag, not incidental JS strings).
    if not re.search(r"""<meta[^>]+name=['"]viewport['"]""", html, re.I):
        errors.append('Missing <meta name="viewport" content="width=device-width, initial-scale=1.0">.')

    # 3. api-base meta (fullstack only).
    if is_fullstack and not re.search(r"""<meta[^>]+name=['"]api-base['"]""", html, re.I):
        errors.append('Missing <meta name="api-base" content=""> (required for fullstack frontends).')

    # 4. No absolute URLs in fetch() calls or non-script resource references.
    #    Any <script src="https://..."> (a CDN library) is allowed — the task
    #    rules permit user-requested libraries, so a foreign CDN is a warning
    #    (below), never a hard error.
    for m in re.finditer(r"""fetch\s*\(\s*(?:api\s*\(\s*)?['"]?https?://""", html, re.I):
        errors.append(f"Absolute URL is not allowed in fetch(): ...{html[m.start():m.start()+60]!r}")
        break
    for m in re.finditer(r"""(?:href|src)\s*=\s*['"]https?://""", html, re.I):
        tag_start = html.rfind("<", 0, m.start())
        tag = html[tag_start:m.start()].lower()
        if tag.startswith("<script"):
            continue  # CDN library script — allowed
        errors.append(
            f"Absolute URL is not allowed in resource references: ...{html[m.start():m.start()+60]!r}"
        )
        break

    # 5. All backend calls under /api/* (fullstack only).
    if is_fullstack:
        for call in _FETCH_CALL.findall(html):
            path = call.strip()
            if path.startswith("http"):
                continue  # already flagged above
            if path.startswith("/") and not path.startswith("/api/"):
                errors.append(f"Backend call must use the /api/* prefix, got: {path!r}")
                break

    # 6. Forbidden globals / CSS.
    if "__antonCommentsLayer" in html:
        errors.append("Frontend must not use the global name window.__antonCommentsLayer.")
    if re.search(r"\*\s*\{[^}]*!important", html):
        errors.append("Frontend must not use universal `* { ... !important }` rules.")
    for m in re.finditer(r"z-index\s*:\s*(\d+)", low):
        if int(m.group(1)) > 1000:
            errors.append("Frontend uses an extreme z-index (> 1000); keep it within a sane range.")
            break

    # ── Warnings (advisory) ────────────────────────────────────────────────
    # Block-level containers without any stable id — weak anchors for the
    # comment layer. Advisory only (never fails the step).
    block_tags = re.findall(r"<(?:div|section|table|main|article)\b", low)
    if block_tags and "id=" not in low:
        warnings.append("Significant blocks have no stable `id` attributes.")
    # Charts via a non-ECharts CDN.
    for src in _BARE_SCRIPT_SRC.findall(html):
        if src.startswith("http") and _ECHARTS_CDN not in src and "echarts" not in src:
            warnings.append(f"Chart/library CDN other than ECharts detected: {src!r} (allowed only if the user asked).")
            break

    return VerifyResult(errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Backend evaluation — pure
# ---------------------------------------------------------------------------

import ast

_DS_KEY = re.compile(r"DS_[A-Z0-9_]+__[A-Z0-9_]+")
_CORE_REQS = ("fastapi", "mangum", "uvicorn")

# PEP 508: the package name is the leading letters/digits/._- run; anything
# after it (extras like `[standard]`, version specifiers, spaces) is not part
# of the name.
_REQ_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _requirement_names(requirements: str) -> set[str]:
    """PEP 503-normalized package names from requirements.txt lines.

    Accepts extras (`uvicorn[standard]`) and any version specifier
    (`fastapi>=0.100`, `pkg ~= 1.2`): the verifier must never be stricter
    than pip about valid input, or the generate→verify retry loop turns a
    perfectly good file into a guaranteed terminal failure.
    """
    names: set[str] = set()
    for raw in requirements.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = _REQ_NAME.match(line)
        if m:
            names.add(re.sub(r"[-_.]+", "-", m.group(0)).lower())
    return names


def _imports_anton_state(source: str) -> bool:
    """True if the module imports the anton_state SDK (any form)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == "anton_state" for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "anton_state":
                return True
    return False


# The functions whose module-level call means "STATE store built at import
# time". `get_store` is the conventional helper name from the backend template;
# calling it at module level defeats its whole purpose.
_STORE_BUILDERS = ("open_store", "from_backend_state", "get_store")


def _module_level_store_builds(source: str) -> list[str]:
    """Names of store-builder functions called at module level (import time).

    Mirrors `_module_level_secret_copies`: the cloud runner overlays
    `backend.STATE` after import, so a store built at import time binds to the
    local SQLite driver even in the cloud. Walks only statements outside
    function/class bodies — a call inside a route is exactly what we want.
    """
    offenders: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return offenders
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in _STORE_BUILDERS:
                offenders.append(name)
    return offenders


def _module_level_secret_copies(source: str) -> list[str]:
    """Return names of module-level vars assigned directly from SECRETS[...]."""
    offenders: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return offenders
    for node in tree.body:  # module level only
        if not isinstance(node, ast.Assign):
            continue
        val = node.value
        # match SECRETS[...] or SECRETS.get(...)
        is_secret = (
            isinstance(val, ast.Subscript)
            and isinstance(val.value, ast.Name)
            and val.value.id == "SECRETS"
        ) or (
            isinstance(val, ast.Call)
            and isinstance(val.func, ast.Attribute)
            and isinstance(val.func.value, ast.Name)
            and val.func.value.id == "SECRETS"
        )
        if is_secret:
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    offenders.append(tgt.id)
    return offenders


def evaluate_backend(
    introspection: dict,
    source: str,
    requirements: str,
    *,
    artifact_type: str = "",
    state_manifest: str | None = None,
) -> tuple[VerifyResult, list[str]]:
    """Pure contract evaluation of a generated backend.

    `artifact_type` enables the type-specific STATE checks; the empty default
    keeps the pre-stateful call shape (and its direct-call tests) intact.
    `state_manifest` is the raw text of `state_manifest.json` (None = the file
    does not exist) — read by the async glue, validated here to stay pure.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not introspection.get("import_ok"):
        errors.append(
            "backend.py failed to import in the venv: "
            + (introspection.get("import_error") or "unknown error")
        )
        # Without a successful import, route/contract data is unreliable — stop here.
        ds_keys = sorted(set(_DS_KEY.findall(source)))
        return VerifyResult(errors=errors, warnings=warnings), ds_keys

    if not introspection.get("app_ok"):
        errors.append("backend.py must define `app` as a FastAPI instance.")
    if not introspection.get("handler_ok"):
        errors.append('backend.py must define `handler = Mangum(app, lifespan="off")`.')
    if not introspection.get("secrets_ok"):
        errors.append("backend.py must define a module-level `SECRETS` dict.")

    api_routes = list(introspection.get("api_routes") or [])
    root_routes = list(introspection.get("root_routes") or [])
    if root_routes:
        errors.append(
            "All API routes must live under /api/*; found root routes: "
            + ", ".join(sorted(root_routes))
        )
    if "/api/health" not in api_routes:
        errors.append("backend.py must expose GET /api/health.")

    for name in _module_level_secret_copies(source):
        errors.append(
            f"Secret copied into module-level variable `{name}` at import time — "
            "read SECRETS[...] at point of use inside the route instead."
        )

    req_names = _requirement_names(requirements)
    for core in _CORE_REQS:
        if core not in req_names:
            # Self-evidencing message: show what WAS parsed, so a mismatch
            # between the file and this parser is visible to the retry loop.
            errors.append(
                f"requirements.txt must list `{core}`. Parsed package names: "
                + (", ".join(sorted(req_names)) or "(none)")
            )
    if "anton-state" in req_names:
        # Applies to every type: the package is not on any registry, so pip
        # fails on the line. The install step filters it defensively, but the
        # correct file simply does not carry it.
        errors.append(
            "requirements.txt must not list `anton_state` — the STATE SDK is "
            "injected at runtime; remove that line."
        )

    if artifact_type == "fullstack-stateful-app":
        if not introspection.get("state_defined"):
            errors.append(
                "backend.py must define a module-level `STATE = None` slot "
                "(the cloud runner overlays it before each request)."
            )
        if state_manifest is None:
            errors.append(
                "state_manifest.json is missing — a stateful backend must "
                "declare its STATE key schema next to backend.py."
            )
        else:
            manifest_error = _validate_state_manifest(state_manifest)
            if manifest_error:
                errors.append("state_manifest.json is invalid: " + manifest_error)
        for name in _module_level_store_builds(source):
            errors.append(
                f"STATE store built at import time via `{name}(...)` — build "
                "it at point of use inside the route instead (the cloud "
                "overlay of STATE happens after import)."
            )
    elif artifact_type == "fullstack-stateless-app":
        if _imports_anton_state(source):
            errors.append(
                "anton_state imported in a stateless backend — the STATE "
                "store is for fullstack-stateful-app only; persistence goes "
                "to external data sources here."
            )

    ds_keys = sorted(set(_DS_KEY.findall(source)))
    return VerifyResult(errors=errors, warnings=warnings), ds_keys


def _validate_state_manifest(text: str) -> str | None:
    """One-line validation error for state_manifest.json, or None if valid.

    Delegates to `anton_state.schema.StateSchema` — the same model the SDK
    loads at runtime — so the verifier can never accept a manifest the
    backend would then fail on. Imported lazily: the anton process has the
    package on its path (unlike the scratchpad venv), and the html-app path
    never needs it.
    """
    from anton_state.schema import StateSchema
    from pydantic import ValidationError

    try:
        StateSchema.model_validate_json(text)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ())) or "(root)"
        return f"{loc}: {first.get('msg', 'invalid')}"
    except ValueError as exc:  # not JSON at all
        return str(exc)
    return None


# ---------------------------------------------------------------------------
# Backend verification — async scratchpad-venv glue
# ---------------------------------------------------------------------------

import asyncio
import json
from pathlib import Path

# Runs inside the scratchpad venv. Imports the artifact backend, introspects
# `app`, and prints one JSON line. Docs routes and the StaticFiles Mount are
# excluded so only real API routes remain.
_INTROSPECT_SCRIPT = r'''
import json, sys
result = {"import_ok": False, "import_error": "", "handler_ok": False,
          "app_ok": False, "secrets_ok": False, "state_defined": False,
          "api_routes": [], "root_routes": []}
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("artifact_backend", "backend.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result["import_ok"] = True
    from fastapi import FastAPI
    from fastapi.routing import APIRoute
    from mangum import Mangum
    app = getattr(mod, "app", None)
    result["app_ok"] = isinstance(app, FastAPI)
    result["handler_ok"] = isinstance(getattr(mod, "handler", None), Mangum)
    result["secrets_ok"] = isinstance(getattr(mod, "SECRETS", None), dict)
    result["state_defined"] = hasattr(mod, "STATE")
    _DOCS = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    if result["app_ok"]:
        for r in app.routes:
            if not isinstance(r, APIRoute):
                continue  # skips Mount("/") and Starlette internals
            if r.path in _DOCS:
                continue
            (result["api_routes"] if r.path.startswith("/api/")
             else result["root_routes"]).append(r.path)
except Exception as exc:  # noqa: BLE001
    result["import_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result))
'''


def _parse_requirements(text: str) -> list[str]:
    pkgs: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        # `anton_state` is injected at runtime, never installable from a
        # registry (same filter as backend_launcher). evaluate_backend flags
        # the line as a contract error; skipping it here keeps the install
        # step alive so that error actually reaches the retry loop instead of
        # an opaque pip failure.
        pkg_name = re.split(r"[<>=!~ \[]", line, maxsplit=1)[0].strip()
        if pkg_name.replace("-", "_").lower() == "anton_state":
            continue
        pkgs.append(line)
    return pkgs


async def verify_backend(
    *,
    scratchpad_pool,
    slug: str,
    artifact_path: Path,
    import_timeout: float = 15.0,
    artifact_type: str = "",
) -> tuple[VerifyResult, list[str]]:
    backend_py = artifact_path / "backend.py"
    if not backend_py.is_file():
        return VerifyResult(errors=["backend.py was not written."]), []
    source = backend_py.read_text(encoding="utf-8")
    req_text = ""
    req_path = artifact_path / "requirements.txt"
    if req_path.is_file():
        req_text = req_path.read_text(encoding="utf-8")
    state_manifest: str | None = None
    manifest_path = artifact_path / "state_manifest.json"
    if manifest_path.is_file():
        state_manifest = manifest_path.read_text(encoding="utf-8")

    # Provision venv and install deps.
    pad = await scratchpad_pool.get_or_create(slug)
    pkgs = _parse_requirements(req_text)
    if pkgs:
        install = await pad.install_packages(pkgs)
        if isinstance(install, str) and (
            install.startswith("Install failed") or install.startswith("Install timed out")
        ):
            return VerifyResult(errors=[f"Dependency install failed:\n{install}"]), []

    venv_python = await scratchpad_pool.venv_python(slug)
    if not venv_python:
        return VerifyResult(errors=["Scratchpad venv Python is unavailable (remote runtime?)."]), []

    # py_compile first (fast syntax gate).
    proc = await asyncio.create_subprocess_exec(
        venv_python, "-m", "py_compile", "backend.py",
        cwd=str(artifact_path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, cerr = await proc.communicate()
    if proc.returncode != 0:
        return VerifyResult(errors=[f"backend.py failed to compile:\n{cerr.decode(errors='replace')}"]), \
            sorted(set(_DS_KEY.findall(source)))

    # Import + introspect in a subprocess with a timeout. The env matters:
    # `build_backend_env` puts `anton_state` on PYTHONPATH exactly like the
    # launcher does for the real backend process — without it a correct
    # stateful backend fails right here on its own SDK import.
    from anton.core.artifacts.backend_launcher import build_backend_env

    proc = await asyncio.create_subprocess_exec(
        venv_python, "-c", _INTROSPECT_SCRIPT,
        cwd=str(artifact_path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env=build_backend_env(None),
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=import_timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return VerifyResult(errors=[f"backend.py import timed out after {import_timeout}s."]), \
            sorted(set(_DS_KEY.findall(source)))

    line = (out.decode(errors="replace").strip().splitlines() or [""])[-1]
    try:
        introspection = json.loads(line)
    except json.JSONDecodeError:
        return VerifyResult(
            errors=["backend introspection produced no JSON. stderr:\n" + err.decode(errors="replace")]
        ), sorted(set(_DS_KEY.findall(source)))

    return evaluate_backend(
        introspection, source, req_text,
        artifact_type=artifact_type, state_manifest=state_manifest,
    )
