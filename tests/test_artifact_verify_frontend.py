from __future__ import annotations

from pathlib import Path

from anton.core.artifacts.html_lint import HtmlFinding
from anton.core.tools.generate_artifact import verifiers
from anton.core.tools.generate_artifact.verifiers import (
    browser_check_skip_reason,
    verify_frontend,
    verify_app_live,
    verify_frontend_live,
)

GOOD = """<!doctype html><html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="api-base" content="">
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
</head><body>
<div id="kpi-revenue"></div>
<script>
const API_BASE = document.querySelector('meta[name="api-base"]')?.content || "";
const api = (p) => `${API_BASE}${p}`;
fetch(api('/api/items'));
</script>
</body></html>"""


def test_good_fullstack_frontend_passes():
    r = verify_frontend(GOOD, is_fullstack=True)
    assert r.ok, r.errors


def test_missing_viewport_is_error():
    html = GOOD.replace('<meta name="viewport" content="width=device-width, initial-scale=1.0">', "")
    r = verify_frontend(html, is_fullstack=True)
    assert not r.ok
    assert any("viewport" in e for e in r.errors)


def test_missing_api_base_is_error_for_fullstack():
    html = GOOD.replace('<meta name="api-base" content="">', "")
    r = verify_frontend(html, is_fullstack=True)
    assert not r.ok
    assert any("api-base" in e for e in r.errors)


def test_absolute_fetch_url_is_error():
    html = GOOD.replace("fetch(api('/api/items'))", "fetch('http://localhost:8000/api/items')")
    r = verify_frontend(html, is_fullstack=True)
    assert not r.ok
    assert any("absolute" in e.lower() for e in r.errors)


def test_absolute_href_and_img_src_are_not_flagged_at_all():
    """I-17: a dashboard built from a web article links back to it and shows its images.

    Neither an error nor a warning — the accepted PRD asks for exactly these,
    and the previous rule failed two correct artifacts in a row. Verified for
    both artifact shapes because `_VERIFIER_CONTRACT` is shared.
    """
    html = GOOD.replace(
        '<div id="kpi-revenue"></div>',
        '<a href="https://habr.com/ru/articles/1074010/">Источник</a>'
        '<img src="https://habrastorage.org/getpro/habr/upload_files/a.png">'
        '<link href="https://fonts.googleapis.com/css2?family=Inter" rel="stylesheet">',
    )
    for is_fullstack in (True, False):
        r = verify_frontend(html, is_fullstack=is_fullstack)
        assert r.ok, r.errors
        assert not [w for w in r.warnings if "absolute" in w.lower() or "http" in w.lower()]


def test_absolute_fetch_url_is_still_an_error_next_to_absolute_href():
    """Dropping the href/src rule must not weaken the fetch() one."""
    html = GOOD.replace(
        "fetch(api('/api/items'))", "fetch('http://localhost:8000/api/items')"
    ).replace('<div id="kpi-revenue"></div>', '<a href="https://example.com/src">src</a>')
    r = verify_frontend(html, is_fullstack=True)
    assert not r.ok
    assert any("fetch()" in e for e in r.errors)


def test_bare_path_backend_call_is_error_for_fullstack():
    html = GOOD.replace("fetch(api('/api/items'))", "fetch('/items')")
    r = verify_frontend(html, is_fullstack=True)
    assert not r.ok
    assert any("/api/" in e for e in r.errors)


def test_missing_body_is_error():
    r = verify_frontend("<div>no body</div>", is_fullstack=False)
    assert not r.ok
    assert any("body" in e.lower() for e in r.errors)


def test_forbidden_globals_are_errors():
    html = GOOD.replace("</script>", "window.__antonCommentsLayer = 1;</script>")
    r = verify_frontend(html, is_fullstack=True)
    assert not r.ok
    assert any("__antonCommentsLayer" in e for e in r.errors)


def test_missing_ids_is_only_a_warning():
    html = GOOD.replace('<div id="kpi-revenue"></div>', "<div></div>")
    r = verify_frontend(html, is_fullstack=True)
    assert r.ok  # warning, not error
    assert r.warnings


def test_html_app_does_not_require_api_base():
    html = GOOD.replace('<meta name="api-base" content="">', "").replace("fetch(api('/api/items'));", "")
    r = verify_frontend(html, is_fullstack=False)
    assert r.ok, r.errors


def test_universal_important_at_top_level_is_error():
    html = GOOD.replace(
        "</head>", "<style>* { margin: 0 !important; }</style></head>"
    )
    r = verify_frontend(html, is_fullstack=True)
    assert not r.ok
    assert any("!important" in e for e in r.errors)


def test_universal_important_in_reduced_motion_media_is_allowed():
    """The standard accessibility reset must not fail the artifact (2026-08-27)."""
    html = GOOD.replace(
        "</head>",
        "<style>@media (prefers-reduced-motion: reduce) {\n"
        "  * { animation: none !important; transition: none !important; }\n"
        "}</style></head>",
    )
    r = verify_frontend(html, is_fullstack=True)
    assert r.ok, r.errors


def test_universal_important_in_print_media_is_allowed():
    html = GOOD.replace(
        "</head>",
        "<style>@media print { * { background: none !important; } }</style></head>",
    )
    r = verify_frontend(html, is_fullstack=True)
    assert r.ok, r.errors


def test_universal_important_in_other_media_is_still_error():
    html = GOOD.replace(
        "</head>",
        "<style>@media (max-width: 600px) { * { display: block !important; } }</style></head>",
    )
    r = verify_frontend(html, is_fullstack=True)
    assert not r.ok
    assert any("!important" in e for e in r.errors)


def test_universal_important_after_exempt_media_block_is_still_error():
    """The exemption must end where the @media block's braces end — a nested
    rule block inside the media query must not extend the span."""
    html = GOOD.replace(
        "</head>",
        "<style>@media print { .slide { display: flex !important; } }\n"
        "* { color: red !important; }</style></head>",
    )
    r = verify_frontend(html, is_fullstack=True)
    assert not r.ok
    assert any("!important" in e for e in r.errors)


def test_unclosed_script_block_is_error():
    html = GOOD.replace("</script>\n</body>", "\n</body>")
    assert "</script" in html  # the CDN script tag is still closed
    html = html.replace("</script>", "", 1).replace("</script>", "", 1)
    # now no closing tag remains anywhere
    assert "</script" not in html and "<script" in html
    r = verify_frontend(html, is_fullstack=True)
    assert not r.ok
    assert any("never closes" in e for e in r.errors)


def test_balanced_script_blocks_pass():
    r = verify_frontend(GOOD, is_fullstack=True)
    assert r.ok, r.errors


# ── verify_frontend_live: the headless-browser gate (html_lint) ──────────────

def test_live_check_is_skipped_when_no_browser_can_run(monkeypatch):
    """`None` from lint_html (no ANTON_HTML_LINT_BROWSER, timeout, garbage
    output) is "could not check", not "clean" — the caller must be able to
    tell the two apart, so it stays None here too."""
    monkeypatch.setattr(verifiers, "lint_html", lambda path: None)
    assert verify_frontend_live(Path("/tmp/x/index.html")) is None


def test_live_check_clean_page_is_an_empty_verdict(monkeypatch):
    monkeypatch.setattr(verifiers, "lint_html", lambda path: [])
    verdict = verify_frontend_live(Path("/tmp/x/index.html"))
    assert verdict is not None and verdict.ok and verdict.warnings == []


def test_live_check_maps_findings_to_errors_and_a_warning(monkeypatch):
    """Console errors, missing local files and crashes fail the step with the
    browser's own detail carried through; an empty render is advisory."""
    monkeypatch.setattr(verifiers, "lint_html", lambda path: [
        HtmlFinding(kind="console_error", detail="Uncaught ReferenceError: state is not defined (line 4)"),
        HtmlFinding(kind="failed_request", detail="file:///tmp/x/missing.js — net::ERR_FILE_NOT_FOUND"),
        HtmlFinding(kind="crashed", detail="renderer process crashed while loading the page"),
        HtmlFinding(kind="empty_page", detail="page rendered with no visible text or elements"),
    ])
    verdict = verify_frontend_live(Path("/tmp/x/index.html"))
    assert verdict is not None and not verdict.ok
    assert verdict.errors == [
        "Loaded in a headless browser, the page logged a console error: "
        "Uncaught ReferenceError: state is not defined (line 4)",
        "Loaded in a headless browser, the page requested a local file that "
        "does not exist: file:///tmp/x/missing.js — net::ERR_FILE_NOT_FOUND",
        "Loaded in a headless browser, the page crashed the renderer process.",
    ]
    assert verdict.warnings == [
        "Loaded in a headless browser, the page rendered no visible text or elements."
    ]


def test_live_check_hands_the_browser_an_absolute_path(monkeypatch):
    """Electron's loadFile resolves a relative path against its own app dir
    and reports the page as not found (seen 2026-09-17)."""
    seen: list[Path] = []

    def fake_lint(path):
        seen.append(path)
        return []

    monkeypatch.setattr(verifiers, "lint_html", fake_lint)
    verify_frontend_live(Path("relative/index.html"))
    assert seen == [Path("relative/index.html").resolve()]
    assert seen[0].is_absolute()


# ── browser_check_skip_reason: three causes, three fixes ─────────────────────

def test_skip_reason_names_the_unset_variable(monkeypatch):
    monkeypatch.delenv("ANTON_HTML_LINT_BROWSER", raising=False)
    assert browser_check_skip_reason() == (
        "no headless browser configured (ANTON_HTML_LINT_BROWSER unset)"
    )


def test_skip_reason_names_a_path_that_is_not_executable(monkeypatch, tmp_path):
    """Thirteenth live run 2026-09-17: the trace said "unset" for what could
    as well have been a wrong path — this is the case the old message hid."""
    monkeypatch.setenv("ANTON_HTML_LINT_BROWSER", str(tmp_path / "no-such-electron"))
    reason = browser_check_skip_reason()
    assert reason.startswith("ANTON_HTML_LINT_BROWSER names no executable")
    assert "no-such-electron" in reason


def test_skip_reason_blames_the_run_when_the_browser_exists(monkeypatch, tmp_path):
    fake = tmp_path / "electron"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("ANTON_HTML_LINT_BROWSER", str(fake))
    reason = browser_check_skip_reason()
    assert reason.startswith("browser configured but the check produced no result")
    assert "8s" in reason


TAILWIND_CDN = '<script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>'


def test_tailwind_cdn_is_not_flagged():
    """The design rules recommend Tailwind via CDN, so the CDN warning must not
    contradict them — it would ride the retry message next to real errors."""
    html = GOOD.replace("</head>", TAILWIND_CDN + "</head>")
    r = verify_frontend(html, is_fullstack=True)
    assert r.ok, r.errors
    assert not r.warnings, r.warnings


def test_tailwind_play_cdn_v3_is_not_flagged_either():
    html = GOOD.replace("</head>", '<script src="https://cdn.tailwindcss.com"></script></head>')
    r = verify_frontend(html, is_fullstack=True)
    assert not r.warnings, r.warnings


def test_other_library_cdn_is_still_a_warning():
    html = GOOD.replace("</head>", '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>')
    r = verify_frontend(html, is_fullstack=True)
    assert r.ok, r.errors
    assert len(r.warnings) == 1
    assert "other than ECharts or Tailwind" in r.warnings[0]
    assert "cdn.plot.ly" in r.warnings[0]


# ── the API contract from the page's side (I-34) ────────────────────────────

_PATHS = {"/api/items", "/api/rooms/{}", "/api/rooms/{}/moves"}


def _page(script: str) -> str:
    return GOOD.replace("fetch(api('/api/items'));", script)


def test_calls_inside_the_contract_pass_in_every_literal_shape():
    html = _page(
        "fetch(api('/api/items'));\n"
        'fetch(api("/api/items?limit=5"));\n'
        "fetch(api(`/api/rooms/${code}`));\n"
        "fetch(api(`/api/rooms/${code}/moves`), {method: 'POST'});\n"
        "fetch(`${API_BASE}/api/items`);\n"
        "fetch(api('/api/health'));\n"
        "fetch(api(url));\n"
    )
    r = verify_frontend(html, is_fullstack=True, api_paths=_PATHS)
    assert r.ok, r.errors


def test_a_literal_prefix_joined_with_plus_is_not_compared():
    """Review 2026-09-24 №8: `fetch(api('/api/rooms/' + code))` builds the
    contract's `/api/rooms/{code}` by concatenation. The literal alone
    normalised to `/api/rooms`, which the contract does not have, and a
    correct page failed verification. A prefix before `+` is skipped like
    a variable; the same literal on its own is still compared."""
    ok = _page("fetch(api('/api/rooms/' + code));\nfetch(api(\"/api/rooms/\" + code + '/moves'), {method: 'POST'});")
    assert verify_frontend(ok, is_fullstack=True, api_paths=_PATHS).ok

    alone = _page("fetch(api('/api/rooms/'));")
    r = verify_frontend(alone, is_fullstack=True, api_paths=_PATHS)
    assert not r.ok and any(e.startswith("fetch() calls `") for e in r.errors)


def test_a_call_outside_the_contract_is_error_and_names_the_contract():
    html = _page("fetch(api(`/api/rooms/${code}/state`));")
    r = verify_frontend(html, is_fullstack=True, api_paths=_PATHS)
    assert not r.ok
    [err] = [e for e in r.errors if e.startswith("fetch() calls `")]
    assert "`/api/rooms/${code}/state`" in err
    assert "/api/rooms/{}/moves" in err


def test_no_contract_and_html_app_skip_the_comparison():
    html = _page("fetch(api('/api/anything'));")
    assert verify_frontend(html, is_fullstack=True).ok
    assert verify_frontend(html, is_fullstack=True, api_paths=None).ok
    page = html.replace('<meta name="api-base" content="">', "")
    assert verify_frontend(page, is_fullstack=False, api_paths=_PATHS).ok


# ── verify_app_live: the served fullstack page (S-02) ───────────────────────

def test_served_check_is_skipped_when_no_browser_can_run(monkeypatch):
    monkeypatch.setattr(verifiers, "lint_url", lambda url: None)
    assert verify_app_live("http://127.0.0.1:5000") is None


def test_served_check_hands_the_url_through_unchanged(monkeypatch):
    seen: list[str] = []

    def fake(url):
        seen.append(url)
        return []
    monkeypatch.setattr(verifiers, "lint_url", fake)
    verdict = verify_app_live("http://127.0.0.1:5000")
    assert verdict is not None and verdict.ok
    assert seen == ["http://127.0.0.1:5000"]


def test_served_check_words_a_failed_request_as_a_url_not_a_local_file(monkeypatch):
    """Over http a 404 from the page's own origin is a missing asset in
    static/ or a route the backend does not serve — the message says URL,
    and the console-error / crash / empty-page wording stays shared."""
    monkeypatch.setattr(verifiers, "lint_url", lambda url: [
        HtmlFinding(kind="failed_request", detail="http://127.0.0.1:5000/api/items — HTTP 404"),
        HtmlFinding(kind="console_error", detail="Uncaught TypeError: rows is undefined (line 12)"),
        HtmlFinding(kind="crashed", detail="renderer process crashed while loading the page"),
        HtmlFinding(kind="empty_page", detail="page rendered with no visible text or elements"),
    ])
    verdict = verify_app_live("http://127.0.0.1:5000")
    assert verdict is not None and not verdict.ok
    assert verdict.errors == [
        "Loaded in a headless browser from the running backend, the page requested "
        "a URL that failed: http://127.0.0.1:5000/api/items — HTTP 404",
        "Loaded in a headless browser, the page logged a console error: "
        "Uncaught TypeError: rows is undefined (line 12)",
        "Loaded in a headless browser, the page crashed the renderer process.",
    ]
    assert verdict.warnings == [
        "Loaded in a headless browser, the page rendered no visible text or elements."
    ]
