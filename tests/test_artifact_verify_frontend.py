from __future__ import annotations

from anton.core.tools.generate_artifact.verifiers import verify_frontend

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
