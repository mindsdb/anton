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
    introspection: dict, source: str, requirements: str
) -> tuple[VerifyResult, list[str]]:
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

    req_lines = {
        ln.split("#", 1)[0].strip().split("==")[0].strip().lower()
        for ln in requirements.splitlines()
        if ln.split("#", 1)[0].strip() and not ln.strip().startswith("-")
    }
    for core in _CORE_REQS:
        if core not in req_lines:
            errors.append(f"requirements.txt must list `{core}`.")

    ds_keys = sorted(set(_DS_KEY.findall(source)))
    return VerifyResult(errors=errors, warnings=warnings), ds_keys


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
          "app_ok": False, "secrets_ok": False, "api_routes": [], "root_routes": []}
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
        pkgs.append(line)
    return pkgs


async def verify_backend(
    *, scratchpad_pool, slug: str, artifact_path: Path, import_timeout: float = 15.0
) -> tuple[VerifyResult, list[str]]:
    backend_py = artifact_path / "backend.py"
    if not backend_py.is_file():
        return VerifyResult(errors=["backend.py was not written."]), []
    source = backend_py.read_text(encoding="utf-8")
    req_text = ""
    req_path = artifact_path / "requirements.txt"
    if req_path.is_file():
        req_text = req_path.read_text(encoding="utf-8")

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

    # Import + introspect in a subprocess with a timeout.
    proc = await asyncio.create_subprocess_exec(
        venv_python, "-c", _INTROSPECT_SCRIPT,
        cwd=str(artifact_path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
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

    return evaluate_backend(introspection, source, req_text)
