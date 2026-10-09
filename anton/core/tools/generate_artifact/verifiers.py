"""Verifiers for the artifact-generation FSM.

`verify_frontend` and `evaluate_backend` are pure functions (deterministic,
no I/O) so they are fully unit-testable. `verify_backend` is the async glue
that installs deps in the scratchpad venv, imports the module in a subprocess
with a timeout, and folds the result into `evaluate_backend`.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import re
from pathlib import Path

from anton.core.artifacts import html_lint
from anton.core.artifacts.html_lint import lint_html, lint_url
from anton.core.utils.scratchpad import install_call_failed

from .state import VerifyResult

_FETCH_CALL = re.compile(r"""fetch\s*\(\s*(?:api\s*\(\s*)?['"]([^'"]+)['"]""")
# Every literal `fetch()` target, template literals included: the `api()`
# helper is the documented shape, and a path with a parameter is written as
# `fetch(api(`/api/rooms/${code}`))` — backticks, which `_FETCH_CALL` never
# saw. A non-literal first argument (a variable) is not a match and is not
# checked: there is nothing static to compare. Group 3 catches a `+` right
# after the literal — `fetch(api('/api/rooms/' + code))` — where the literal
# is a prefix of the path, not the path: compared as written it normalises
# to `/api/rooms` and fails a contract that only has `/api/rooms/{code}`.
# Such a call is skipped like a variable.
_FETCH_TARGET = re.compile(
    r"""fetch\s*\(\s*(?:api\s*\(\s*)?(['"`])(.*?)\1(\s*\+)?""", re.S
)
_API_METHODS = ("get", "post", "put", "patch", "delete")
_HEALTH_PATH = "/api/health"


def normalise_api_path(path: str) -> str:
    """`/api/rooms/{code}/` and `/api/rooms/{id}` compare equal: parameter
    names are the contract writer's to choose, the segments are not."""
    return re.sub(r"\{[^}]*\}", "{}", str(path).rstrip("/"))


def contract_operations(api_spec: str | None) -> set[str] | None:
    """`METHOD /normalised/path` for every operation `openapi.json` defines.

    None — not an empty set — when there is no usable contract (no document,
    not JSON, no `paths`): the callers then skip the comparison instead of
    reporting every route as undeclared. `/api/health` is left out on both
    sides; it is a launcher requirement, not part of the contract.
    """
    if not api_spec:
        return None
    try:
        spec = json.loads(api_spec)
    except (TypeError, json.JSONDecodeError):
        return None
    paths = spec.get("paths") if isinstance(spec, dict) else None
    if not isinstance(paths, dict) or not paths:
        return None
    ops: set[str] = set()
    for path, methods in paths.items():
        key = normalise_api_path(path)
        if key == _HEALTH_PATH or not isinstance(methods, dict):
            continue
        for method in _API_METHODS:
            if isinstance(methods.get(method), dict):
                ops.add(f"{method.upper()} {key}")
    return ops or None


def contract_paths(operations: set[str] | None) -> set[str] | None:
    """The normalised paths behind `contract_operations`, or None with it."""
    if operations is None:
        return None
    return {op.split(" ", 1)[1] for op in operations}


def _fetch_path_key(raw: str) -> str | None:
    """The contract key a literal `fetch()` target compares under, or None
    when there is nothing to compare: an absolute URL (flagged elsewhere), a
    call outside `/api/` (flagged elsewhere), the health route."""
    target = raw.strip()
    if target.lower().startswith("http"):
        return None
    # `${API_BASE}/api/items` — the base spliced in by hand instead of `api()`.
    target = re.sub(r"^\$\{[^}]*\}", "", target)
    target = target.split("?", 1)[0]
    # `${code}` inside the path is a parameter, same as `{code}` in the contract.
    target = re.sub(r"\$\{[^}]*\}", "{}", target)
    if not target.startswith("/api/"):
        return None
    key = normalise_api_path(target)
    return None if key == _HEALTH_PATH else key
_BARE_SCRIPT_SRC = re.compile(r"""<script[^>]*\bsrc\s*=\s*['"]([^'"]+)['"]""", re.I)
# The body of a tag, scanned for attributes in a second step so that every
# `src`/`href` of one tag is seen. The body is `[^<>]`, not `[^>]`: with
# `[^>]` every `<b` in embedded data scans on to the next `>`, which is
# quadratic in the page size (12.8 s on 60 KB of `1<b,`).
_TAG_BODY = re.compile(r"<[a-z][^<>]*", re.I)
# A `src`/`href` attribute inside a tag body whose value starts with exactly
# one `/`: a path from the root. `(?<![\w-])` keeps `data-src`/`data-href` out;
# `(?![/\\])` lets a protocol-relative `//host` through (and `/\host`, which
# browsers read the same way). Quoted or not; leading spaces are tolerated
# because the browser strips them.
_ROOT_RELATIVE_ATTR = re.compile(
    r"""(?<![\w-])(?P<name>src|href)\s*=\s*['"]?\s*(?P<value>/(?![/\\])[^\s'"<>]*)""",
    re.I,
)
# A `fetch()` or `new EventSource()` whose first argument is a string literal
# from the root, i.e. not wrapped in `api()`. `(?<![\w$])` keeps `prefetch(`
# out and lets `window.fetch(` in; `re.S` for multi-line template literals.
_ROOT_RELATIVE_CALL = re.compile(
    r"""(?<![\w$])(?P<name>fetch|EventSource)\s*\(\s*(?P<quote>['"`])(?P<value>/(?![/\\]).*?)(?P=quote)""",
    re.S,
)
_ROOT_RELATIVE_SHOWN = 5
_ROOT_RELATIVE_VALUE_MAX = 80
# Libraries the design rules let a page load from the network. Matched as
# substrings of the script URL so a different CDN host or version still
# passes; anything else is advisory-flagged, since the prompt allows other
# libraries only when the user asked for them.
_ALLOWED_CDN_LIBRARIES = ("echarts", "tailwindcss")

_UNIVERSAL_IMPORTANT = re.compile(r"\*\s*\{[^}]*!important")
# Media queries whose universal `!important` blocks are legitimate practice
# rather than a global style override: the reduced-motion accessibility reset
# (`* { animation: none !important; }`) and print stylesheets. The rule this
# feeds exists to stop a page from fighting the host application's styles;
# neither of these contexts can do that. Measured 2026-08-27: the reset is
# something models emit reflexively, and flagging it failed an otherwise
# valid artifact.
_EXEMPT_MEDIA_HEADER = re.compile(
    r"@media[^{]*(?:prefers-reduced-motion|\bprint\b)[^{]*\{", re.I
)


def _exempt_media_spans(html: str) -> list[tuple[int, int]]:
    """(start, end) spans of @media blocks exempt from the `* { !important }` rule.

    The block end is found by brace counting from the header's opening brace —
    CSS inside `@media` nests rule blocks, so a non-greedy regex would stop at
    the first `}` and truncate the span.
    """
    spans: list[tuple[int, int]] = []
    for m in _EXEMPT_MEDIA_HEADER.finditer(html):
        depth = 1
        i = m.end()
        while i < len(html) and depth:
            if html[i] == "{":
                depth += 1
            elif html[i] == "}":
                depth -= 1
            i += 1
        spans.append((m.start(), i))
    return spans


def _shown_value(value: str) -> str:
    """A value for the message: one line, at most `_ROOT_RELATIVE_VALUE_MAX`
    characters, in single quotes whatever quote the page used."""
    value = re.sub(r"\s+", " ", value)
    if len(value) > _ROOT_RELATIVE_VALUE_MAX:
        value = value[: _ROOT_RELATIVE_VALUE_MAX - 1] + "…"
    return f"'{value}'"


def _root_relative_detail(html: str, *, is_fullstack: bool) -> str | None:
    """The rest of rule 4a's message after its fixed head, or None when the
    page has no root-relative path.

    Every finding goes into the one message, in page order and without
    repeats: a retry that is shown only the first one fixes only that one
    and fails again. The hints follow what was found: a relative path for
    files and links, `api()` for a fullstack page's calls and for an API
    address in an attribute, embedded data for an html-app page's calls
    (the published html-app bundle carries no file a `fetch()` names).
    """
    hits = [
        (tag.start() + m.start(), True, m)
        for tag in _TAG_BODY.finditer(html)
        for m in _ROOT_RELATIVE_ATTR.finditer(tag.group())
    ]
    hits += [(m.start(), False, m) for m in _ROOT_RELATIVE_CALL.finditer(html)]
    if not hits:
        return None
    hits.sort(key=lambda hit: hit[0])

    entries = [
        f"{m.group('name').lower()}={_shown_value(m.group('value'))}"
        if in_attr
        else f"{m.group('name')}({_shown_value(m.group('value'))})"
        for _, in_attr, m in hits
    ]
    shown = list(dict.fromkeys(entries))
    listed = ", ".join(shown[:_ROOT_RELATIVE_SHOWN])
    if len(shown) > _ROOT_RELATIVE_SHOWN:
        listed += f" and {len(shown) - _ROOT_RELATIVE_SHOWN} more"

    attr_values = [m.group("value") for _, in_attr, m in hits if in_attr]
    hints = []
    if attr_values:
        hints.append(
            "Use a relative path for files and links, e.g. logo.png, "
            "or a full https:// URL for an external page."
        )
        if is_fullstack and any(v.startswith("/api/") for v in attr_values):
            hints.append(
                "Set an API address in an attribute from JS: a.href = api('/api/export')."
            )
    if any(not in_attr for _, in_attr, _ in hits):
        hints.append(
            "Call the API through api(): fetch(api('/api/items')), "
            "new EventSource(api('/api/stream'))."
            if is_fullstack
            else "Embed the data in the page instead of fetching it."
        )
    hints.append("Moving the same root-relative path into a JS string does not fix it.")
    return (
        f"{listed}. The page is served under a path prefix, in the app preview "
        "and once published, and a path from the root drops it. " + " ".join(hints)
    )


def verify_frontend(
    html: str, *, is_fullstack: bool, api_paths: set[str] | None = None
) -> VerifyResult:
    """Static checks of the page. `api_paths` (from `contract_paths`) is the
    set of normalised `openapi.json` paths a fullstack page may call; None
    skips that comparison — html-app pages and callers without a contract."""
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

    # 4. No absolute URLs in fetch() calls. Only fetch() — a hardcoded host in
    #    a data call breaks the artifact the moment it is published, because the
    #    backend it names is the local one.
    #
    #    Absolute URLs in `href`/`src` are NOT checked at all — not an error and
    #    not a warning. The rule used to cover them and produced two live
    #    failures in a row: a dashboard built from a web article legitimately
    #    links back to its source (`<a href>`) and shows the source's images
    #    (`<img src>`), and the accepted PRD had asked for both. Worse, the
    #    check only ever saw HTML text, so the retry "fixed" it by moving the
    #    same URLs into JS strings rendered through innerHTML — identical DOM,
    #    a full regeneration burned, and the model now knows the workaround.
    #    A check that a correct artifact fails and an incorrect one passes is
    #    worse than no check.
    for m in re.finditer(r"""fetch\s*\(\s*(?:api\s*\(\s*)?['"]?https?://""", html, re.I):
        errors.append(f"Absolute URL is not allowed in fetch(): ...{html[m.start():m.start()+60]!r}")
        break

    # 4a. No root-relative path: a `src`/`href` value, or a literal `fetch()`
    #     / `new EventSource()` target, that starts with exactly one `/`,
    #     `/api/...` included. The page is served under a path prefix (the
    #     app preview's proxy route, the published host's `/_t/<…>/`) that
    #     such a path drops: a file or call gets 401 or 404, a link reloads
    #     the frame. `verify_app_live` loads the fullstack page from the
    #     backend's root, where the path works, so only this text check sees
    #     it there. This does not reopen rule 4's decision on ABSOLUTE URLs:
    #     `https://`, `//`, `data:`, `blob:`, `#`, `mailto:` and relative
    #     paths all pass. Only literal `src`/`href` values and literal call
    #     targets are checked: `srcset`, `action`, CSS `url()`, an attribute
    #     after a ">" inside an earlier quoted value, and most addresses built
    #     in JS are not seen.
    root_relative = _root_relative_detail(html, is_fullstack=is_fullstack)
    if root_relative:
        errors.append("Root-relative path is not allowed: " + root_relative)

    # 5. All backend calls under /api/* (fullstack only).
    if is_fullstack:
        for call in _FETCH_CALL.findall(html):
            path = call.strip()
            if path.startswith("http"):
                continue  # already flagged above
            if path.startswith("/") and not path.startswith("/api/"):
                errors.append(f"Backend call must use the /api/* prefix, got: {path!r}")
                break

    # 5a. Every literal backend call names a path the contract defines
    #     (fullstack only, and only when a contract is known). The backend is
    #     held to the same document from its side, so a page that passes here
    #     calls routes that exist (I-34). Reported once, with the whole
    #     contract, so the retry has what it needs to fix the call.
    if is_fullstack and api_paths is not None:
        for _quote, raw, concatenated in _FETCH_TARGET.findall(html):
            if concatenated:
                continue
            key = _fetch_path_key(raw)
            if key is None or key in api_paths:
                continue
            errors.append(
                f"fetch() calls `{raw.strip()}`, which openapi.json does not define; "
                "the contract's paths are: " + ", ".join(sorted(api_paths))
            )
            break

    # 5b. Script-tag integrity — the "JS never runs" class: an opening
    # `<script` with no closing `</script` anywhere.
    if "<script" in low and "</script" not in low:
        errors.append("Frontend opens a <script> block but never closes it with </script>.")

    # 6. Forbidden globals / CSS.
    if "__antonCommentsLayer" in html:
        errors.append("Frontend must not use the global name window.__antonCommentsLayer.")
    exempt_spans = _exempt_media_spans(html)
    for m in _UNIVERSAL_IMPORTANT.finditer(html):
        if any(start <= m.start() < end for start, end in exempt_spans):
            continue
        errors.append("Frontend must not use universal `* { ... !important }` rules.")
        break
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
    # A library loaded from a CDN the design rules do not name.
    for src in _BARE_SCRIPT_SRC.findall(html):
        if src.startswith("http") and not any(lib in src.lower() for lib in _ALLOWED_CDN_LIBRARIES):
            warnings.append(f"Library CDN other than ECharts or Tailwind detected: {src!r} (allowed only if the user asked).")
            break

    return VerifyResult(errors=errors, warnings=warnings)


def verify_frontend_live(entry: Path) -> VerifyResult | None:
    """Load the written page once in a headless browser (`html_lint`).

    The class of defects the text checks above cannot see: a script that
    throws on load, a reference to a missing local file, a renderer crash,
    a body that renders nothing. Findings become verifier errors with the
    same fixed prefixes the contract lock registers, so the retry kickoff
    carries the browser's own message (`Uncaught ReferenceError: state is
    not defined (line 4)`), which is exactly what the generator needs.

    Returns `None` when the page could not be checked at all — no browser
    configured (`ANTON_HTML_LINT_BROWSER` unset: the cloud, a bare CLI),
    a timeout, malformed runner output — as distinct from an empty verdict,
    which means it loaded cleanly. The caller then rests on the static
    verdict alone and notes the skip in the trace.

    Single-file `html-app` pages only: `lint_html` loads through `file://`,
    where a fullstack frontend's relative `/api/*` fetches cannot resolve and
    surface as `TypeError: Failed to fetch` console errors that are the
    harness's doing, not the page's (measured 2026-09-17). The fullstack
    page gets the same check from `verify_app_live`, through its running
    backend, after `run_app`. The path is resolved here as well as in
    `lint_html`: Electron's `loadFile` reads a relative path against its own
    app directory, not the cwd.
    """
    findings = lint_html(entry.resolve())
    if findings is None:
        return None
    return _browser_verdict(findings, served=False)


def verify_app_live(url: str) -> VerifyResult | None:
    """Load the fullstack page from its running backend in a headless browser.

    The counterpart of `verify_frontend_live` for `static/index.html`: served
    by the backend `run_app` just launched, the page's relative `/api/*`
    fetches resolve, so a console error is the page's own. A 4xx/5xx from the
    origin — an asset missing from `static/`, a `fetch()` to a route the
    backend does not serve — is reported as a failed request. Same `None`
    contract as the file-based check (no browser, timeout, garbage output).
    """
    findings = lint_url(url)
    if findings is None:
        return None
    return _browser_verdict(findings, served=True)


def _browser_verdict(findings, *, served: bool) -> VerifyResult:
    """Browser findings as verifier errors and warnings.

    Every literal below is a `.append(...)` on purpose: the contract lock
    reads them from this file and requires each in the generator prompts.
    `served` picks the wording for a failed request — a local file under
    file://, a URL of the page's own origin over http.
    """
    errors: list[str] = []
    warnings: list[str] = []
    for f in findings:
        if f.kind == "console_error":
            errors.append(
                f"Loaded in a headless browser, the page logged a console error: {f.detail}"
            )
        elif f.kind == "crashed":
            errors.append("Loaded in a headless browser, the page crashed the renderer process.")
        elif f.kind == "failed_request" and served:
            errors.append(
                "Loaded in a headless browser from the running backend, the page "
                f"requested a URL that failed: {f.detail}"
            )
        elif f.kind == "failed_request":
            errors.append(
                "Loaded in a headless browser, the page requested a local file that "
                f"does not exist: {f.detail}"
            )
        elif f.kind == "empty_page":
            # A warning, not an error: the runner calls this a heuristic. A
            # canvas-only page or one that fills in from a timer renders no
            # text at load, and both are legitimate.
            warnings.append(
                "Loaded in a headless browser, the page rendered no visible text or elements."
            )
    return VerifyResult(errors=errors, warnings=warnings)


def browser_check_skip_reason() -> str:
    """Why `verify_frontend_live` returned None, for the trace.

    `lint_html` folds every "could not check" into one None. A live trace
    said "ANTON_HTML_LINT_BROWSER unset" for what may
    have been a wrong path or a timeout — the three need different fixes, so
    the trace has to tell them apart. Evaluated after the fact, so a variable
    that changes between the check and this call could mislabel one run; the
    environment of a generation process does not change underneath it.
    """
    configured = os.environ.get(html_lint.BROWSER_ENV_VAR)
    if not configured:
        return f"no headless browser configured ({html_lint.BROWSER_ENV_VAR} unset)"
    if html_lint.discover_browser() is None:
        return (
            f"{html_lint.BROWSER_ENV_VAR} names no executable "
            f"(not a file, not on PATH): {configured!r}"
        )
    return (
        "browser configured but the check produced no result "
        f"(runner timeout of {html_lint.TIMEOUT_SECONDS}s, crash or malformed output)"
    )


# ---------------------------------------------------------------------------
# Backend evaluation — pure
# ---------------------------------------------------------------------------

_DS_KEY = re.compile(r"DS_[A-Z0-9_]+__[A-Z0-9_]+")
_CORE_REQS = ("fastapi", "mangum", "uvicorn")

# PEP 508: the package name is the leading letters/digits/._- run; anything
# after it (extras like `[standard]`, version specifiers, spaces) is not part
# of the name.
_REQ_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _ds_keys(source: str) -> list[str]:
    """Every distinct `DS_<PREFIX>__<FIELD>` name `source` reads, sorted."""
    return sorted(set(_DS_KEY.findall(source)))


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
    api_operations: set[str] | None = None,
) -> tuple[VerifyResult, list[str]]:
    """Pure contract evaluation of a generated backend.

    `artifact_type` enables the type-specific STATE checks; the empty default
    keeps the pre-stateful call shape (and its direct-call tests) intact.
    `state_manifest` is the raw text of `state_manifest.json` (None = the file
    does not exist) — read by the async glue, validated here to stay pure.
    `api_operations` (from `contract_operations`) is what `openapi.json`
    promises; None skips the comparison.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not introspection.get("import_ok"):
        errors.append(
            "backend.py failed to import in the venv: "
            + (introspection.get("import_error") or "unknown error")
        )
        # Without a successful import, route/contract data is unreliable — stop here.
        return VerifyResult(errors=errors, warnings=warnings), _ds_keys(source)

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

    if api_operations is not None:
        # The contract from the backend's side (I-34): every operation the
        # document promises has a route with that method and those segments.
        # A route the document does not list is only a warning — it breaks
        # nothing, and a regeneration costs rounds; a missing one is the page
        # calling a 404.
        implemented = {
            f"{method} {normalise_api_path(path)}"
            for method, _, path in (
                op.partition(" ") for op in introspection.get("api_operations") or []
            )
            if normalise_api_path(path) != _HEALTH_PATH
            and method.lower() in _API_METHODS
        }
        missing = sorted(api_operations - implemented)
        if missing:
            errors.append(
                "backend.py does not implement " + ", ".join(f"`{op}`" for op in missing)
                + " from openapi.json — every operation of the contract needs a "
                "route with that method and path (parameter names may differ)."
            )
        extra = sorted(implemented - api_operations)
        if extra:
            warnings.append(
                "backend.py adds routes that openapi.json does not define: "
                + ", ".join(f"`{op}`" for op in extra)
                + " — the frontend cannot know them."
            )

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

    return VerifyResult(errors=errors, warnings=warnings), _ds_keys(source)


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

# Runs inside the scratchpad venv. Imports the artifact backend, introspects
# `app`, and prints one JSON line. Docs routes and the StaticFiles Mount are
# excluded so only real API routes remain.
_INTROSPECT_SCRIPT = r'''
import json, sys
result = {"import_ok": False, "import_error": "", "handler_ok": False,
          "app_ok": False, "secrets_ok": False, "state_defined": False,
          "api_routes": [], "root_routes": [], "api_operations": []}
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
            if r.path.startswith("/api/"):
                for m in sorted(r.methods or ()):
                    result["api_operations"].append(f"{m} {r.path}")
except Exception as exc:  # noqa: BLE001
    result["import_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result))
'''


async def verify_backend(
    *,
    scratchpad_pool,
    slug: str,
    artifact_path: Path,
    import_timeout: float = 15.0,
    artifact_type: str = "",
    api_operations: set[str] | None = None,
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

    # Provision venv and install deps. The launcher's parser, so the verifier
    # installs exactly what the real launch will: `anton_state` is skipped
    # (injected at runtime, never installable), which keeps the install step
    # alive so `evaluate_backend`'s contract error about that line reaches the
    # retry loop instead of an opaque pip failure.
    from anton.core.artifacts.backend_launcher import build_backend_env, parse_requirements

    pad = await scratchpad_pool.get_or_create(slug)
    pkgs = parse_requirements(req_text)
    if pkgs:
        install = await pad.install_packages(pkgs)
        if isinstance(install, str) and install_call_failed(install):
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
            _ds_keys(source)

    # Import + introspect in a subprocess with a timeout. The env matters:
    # `build_backend_env` puts `anton_state` on PYTHONPATH exactly like the
    # launcher does for the real backend process — without it a correct
    # stateful backend fails right here on its own SDK import.
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
            _ds_keys(source)

    line = (out.decode(errors="replace").strip().splitlines() or [""])[-1]
    try:
        introspection = json.loads(line)
    except json.JSONDecodeError:
        return VerifyResult(
            errors=["backend introspection produced no JSON. stderr:\n" + err.decode(errors="replace")]
        ), _ds_keys(source)

    return evaluate_backend(
        introspection, source, req_text,
        artifact_type=artifact_type, state_manifest=state_manifest,
        api_operations=api_operations,
    )
