from __future__ import annotations

import time
from pathlib import Path

import pytest

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
        '<a href="https://example.com/articles/1074010/">Source</a>'
        '<img src="https://images.example.com/upload_files/a.png">'
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
    """The standard accessibility reset must not fail the artifact."""
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
    and reports the page as not found."""
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
    """Seen live: the trace said "unset" for what could
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
    """`fetch(api('/api/rooms/' + code))` builds the
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


# ── Root-relative paths ──────────────────────────────────────────────────────

_ROOT_RELATIVE = "Root-relative path is not allowed: "
_RELATIVE_HINT = (
    "Use a relative path for files and links, e.g. logo.png, "
    "or a full https:// URL for an external page."
)
_STATIC_HINT = (
    "A fullstack page is itself served from static/, "
    "so a file in static/ is app.js, not static/app.js."
)


def _with_markup(markup: str) -> str:
    return GOOD.replace('<div id="kpi-revenue"></div>', f'<div id="kpi-revenue"></div>{markup}')


def _with_script(script: str) -> str:
    return GOOD.replace("fetch(api('/api/items'));", f"fetch(api('/api/items'));\n{script}")


def _root_relative_error(html: str, *, is_fullstack: bool) -> str:
    r = verify_frontend(html, is_fullstack=is_fullstack)
    [err] = [e for e in r.errors if e.startswith(_ROOT_RELATIVE)]
    return err


@pytest.mark.parametrize(
    "markup, shown",
    [
        ('<img src="/logo.png">', "src='/logo.png'"),
        ('<IMG SRC="/logo.png">', "src='/logo.png'"),
        ("<a href='/about'>About</a>", "href='/about'"),
        ('<a href="/">Home</a>', "href='/'"),
        ('<base href="/">', "href='/'"),
        ('<link rel="stylesheet" href="/styles.css">', "href='/styles.css'"),
        ('<script src="/app.js"></script>', "src='/app.js'"),
        ("<img src=/logo.png>", "src='/logo.png'"),
        ('<a\n  class="nav"\n  href="/docs">Docs</a>', "href='/docs'"),
        ('<svg><use xlink:href="/sprite.svg#icon"></use></svg>', "href='/sprite.svg#icon'"),
        ('<a href=" /x">X</a>', "href='/x'"),
        # HTML held in a JS string, which innerHTML turns into a real link.
        ('<script>const s = "<a href=\\"/wiki/Foo\\">Foo</a>";</script>', "href='/wiki/Foo'"),
    ],
)
def test_root_relative_src_or_href_is_error(markup, shown):
    """A path from the root drops the path prefix the page is served under."""
    for is_fullstack in (True, False):
        assert shown in _root_relative_error(_with_markup(markup), is_fullstack=is_fullstack)


def test_every_root_relative_attribute_of_one_tag_is_listed():
    """A second `src`/`href` in the same tag is reported, not only the first."""
    markup = '<svg><image href="/a.png" xlink:href="/b.png"/></svg>'
    for is_fullstack in (True, False):
        err = _root_relative_error(_with_markup(markup), is_fullstack=is_fullstack)
        assert "href='/a.png', href='/b.png'" in err


@pytest.mark.parametrize(
    "script, shown",
    [
        ("fetch('/api/items');", "fetch('/api/items')"),
        ("fetch(`/api/rooms/${code}`);", "fetch('/api/rooms/${code}')"),
        ('fetch(\n  "/data.json"\n);', "fetch('/data.json')"),
        ("window.fetch('/x');", "fetch('/x')"),
        ("self.fetch('/x');", "fetch('/x')"),
        ("const es = new EventSource('/api/stream');", "EventSource('/api/stream')"),
    ],
)
def test_root_relative_call_is_error(script, shown):
    """A literal from the root instead of `api()` misses the same prefix."""
    for is_fullstack in (True, False):
        assert shown in _root_relative_error(_with_script(script), is_fullstack=is_fullstack)


@pytest.mark.parametrize(
    "markup",
    [
        '<img src="https://images.example.com/a.png">',
        '<a href="http://example.com/">Source</a>',
        '<script src="//cdn.example.com/lib.js"></script>',
        '<img src="data:image/png;base64,iVBORw0KGgo=">',
        '<img src="blob:https://example.com/0f4c">',
        '<a href="#top">Top</a>',
        '<a href="mailto:team@example.com">Mail</a>',
        '<img src="logo.png">',
        '<img src="./logo.png">',
        '<a href="../other/index.html">Other</a>',
        '<script src="static/app.js"></script>',
        '<img data-src="/lazy.png" src="lazy.png">',
        '<a data-href="/x" href="x">X</a>',
        '<img src="/\\evil.example/a.png">',
        # Icons and the manifest load for the top-level page only.
        '<link rel="icon" href="/favicon.ico">',
        '<link rel="shortcut icon" href="/favicon.ico">',
        '<link rel="apple-touch-icon" href="/apple-touch-icon.png">',
        '<link rel="manifest" href="/site.webmanifest">',
        "<link rel=icon href=/favicon.ico>",
    ],
)
def test_allowed_src_and_href_values_pass(markup):
    for is_fullstack in (True, False):
        r = verify_frontend(_with_markup(markup), is_fullstack=is_fullstack)
        assert r.ok, r.errors


@pytest.mark.parametrize(
    "script",
    [
        "fetch(api('/api/items'));",
        "fetch(`${API_BASE}/api/items`);",
        "fetch('data.json');",
        "fetch('//cdn.example.com/x.json');",
        "fetch('/\\evil.example/x');",
        "router.prefetch('/page');",
        "cache.fetch('/users');",
        "new EventSource(api('/api/stream'));",
    ],
)
def test_allowed_calls_pass(script):
    """Other rules may still object (rule 5 to `//cdn` on a fullstack page);
    this one does not."""
    for is_fullstack in (True, False):
        r = verify_frontend(_with_script(script), is_fullstack=is_fullstack)
        assert not [e for e in r.errors if e.startswith(_ROOT_RELATIVE)], r.errors


def test_api_address_set_from_js_passes():
    html = _with_script(
        "document.getElementById('export').href = api('/api/export');"
    ).replace('<div id="kpi-revenue"></div>', '<div id="kpi-revenue"></div><a id="export">Export</a>')
    r = verify_frontend(html, is_fullstack=True)
    assert r.ok, r.errors


def test_message_lists_each_value_once_in_page_order():
    html = _with_script("fetch('/api/x');").replace(
        '<div id="kpi-revenue"></div>',
        '<div id="kpi-revenue"></div><img src="/b.png"><img src="/a.png"><img src="/b.png">',
    ).replace("</body>", '<img src="/late.png"></body>')
    err = _root_relative_error(html, is_fullstack=True)
    assert err.startswith(
        _ROOT_RELATIVE + "src='/b.png', src='/a.png', fetch('/api/x'), src='/late.png'. "
    )


def test_message_shows_twenty_values_and_counts_the_rest():
    """The page gets one retry, so the message names enough paths to fix
    a page full of copied links in one go."""
    html = _with_markup("".join(f'<img src="/{i}.png">' for i in range(22)))
    err = _root_relative_error(html, is_fullstack=False)
    assert "src='/19.png' and 2 more. " in err
    assert "'/20.png'" not in err and "'/21.png'" not in err


def test_message_keeps_values_that_differ_only_after_the_cut():
    """Repeats are dropped by the full value, not by the shortened one."""
    long = "/" + "a" * 100
    html = _with_markup(f'<img src="{long}1"><img src="{long}2">')
    err = _root_relative_error(html, is_fullstack=False)
    assert err.count("src='/") == 2


def test_message_cuts_a_long_value():
    long = "/" + "a" * 120
    err = _root_relative_error(_with_script(f"fetch(`{long}`);"), is_fullstack=True)
    shown = err[len(_ROOT_RELATIVE):].split(". The page", 1)[0]
    assert shown == "fetch('" + long[:79] + "…')"


def test_fullstack_hints_point_at_api():
    html = _with_script("fetch('/api/items');").replace(
        '<div id="kpi-revenue"></div>', '<div id="kpi-revenue"></div><a href="/api/export">Export</a>'
    )
    err = _root_relative_error(html, is_fullstack=True)
    assert _RELATIVE_HINT in err
    assert "Set an API address in an attribute from JS: a.href = api('/api/export')." in err
    assert (
        "Call the API through api(): fetch(api('/api/items')), "
        "new EventSource(api('/api/stream'))."
    ) in err
    assert "Embed the data" not in err


def test_fullstack_attribute_hint_names_the_static_folder():
    """`static/app.js` from a page served out of static/ is a 404."""
    html = _with_markup('<script src="/static/app.js"></script>')
    assert _STATIC_HINT in _root_relative_error(html, is_fullstack=True)
    assert "static/" not in _root_relative_error(
        _with_markup('<img src="/logo.png">'), is_fullstack=False
    )


def test_api_attribute_hint_only_for_an_api_address():
    err = _root_relative_error(_with_markup('<img src="/logo.png">'), is_fullstack=True)
    assert _RELATIVE_HINT in err
    assert "api(" not in err


def test_html_app_call_hint_is_to_embed_the_data():
    """The published html-app bundle carries no file a `fetch()` names."""
    err = _root_relative_error(_with_script("fetch('/data.json');"), is_fullstack=False)
    assert "Embed the data in the page instead of fetching it." in err
    assert "Use a relative path" not in err
    assert "api(" not in err


def test_html_app_page_with_an_attribute_and_a_call_gets_both_hints():
    html = _with_markup('<img src="/logo.png">').replace(
        "fetch(api('/api/items'));", "fetch('/data.json');"
    )
    err = _root_relative_error(html, is_fullstack=False)
    assert _RELATIVE_HINT in err
    assert "Embed the data in the page instead of fetching it." in err
    assert "api(" not in err


def test_message_says_moving_the_path_into_js_does_not_fix_it():
    """I-17: a retry once passed a text check by moving URLs into JS strings."""
    for is_fullstack in (True, False):
        err = _root_relative_error(_with_markup('<img src="/logo.png">'), is_fullstack=is_fullstack)
        assert err.endswith("Moving the same root-relative path into a JS string does not fix it.")


def test_large_page_with_embedded_data_is_checked_quickly():
    """The tag scan stops at the next `<` as well as `>`. With `[^>]` alone
    every `<b` in embedded data scanned on to the next `>`: 12.8 s on 60 KB,
    minutes on this page. The limit leaves room for a slow CI runner."""
    html = _with_markup("<script>var d=[" + "1<b," * 65_536 + "];</script>")  # 256 KB, no `>`
    started = time.perf_counter()
    for is_fullstack in (True, False):
        verify_frontend(html, is_fullstack=is_fullstack)
    assert time.perf_counter() - started < 10


def test_long_whitespace_after_an_attribute_name_is_checked_quickly():
    """Two `\\s*` around an optional quote backtracked quadratically: 4.5 s
    on 40 000 spaces."""
    html = _with_markup("<a src=" + " " * 40_000 + "x>")
    started = time.perf_counter()
    verify_frontend(html, is_fullstack=False)
    assert time.perf_counter() - started < 2
