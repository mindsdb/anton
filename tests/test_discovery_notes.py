"""Deterministic channels from the discovery phases into generation."""
from __future__ import annotations

from anton.core.tools.generate_artifact.discovery.notes import (
    WEB_EXCERPT_MAX,
    WEB_NOTES_MAX,
    render_gathering_notes,
    render_web_notes,
)


def _call(**over) -> dict:
    base = {
        "kind": "web_fetch",
        "url": "https://habr.com/ru/articles/1074010/",
        "title": "An article about dashboards",
        "excerpt": "Dashboards are useful.",
        "query": "",
    }
    base.update(over)
    return base


def test_no_calls_render_to_an_empty_string():
    assert render_web_notes([]) == ""


def test_a_fetch_renders_url_title_and_excerpt():
    out = render_web_notes([_call()])
    assert "https://habr.com/ru/articles/1074010/" in out
    assert "An article about dashboards" in out
    assert "Dashboards are useful." in out


def test_a_search_renders_its_query_not_a_url():
    out = render_web_notes([_call(kind="web_search", query="habr dashboards", url="", title="")])
    assert "habr dashboards" in out


def test_a_long_excerpt_is_capped_per_call():
    out = render_web_notes([_call(excerpt="x" * (WEB_EXCERPT_MAX * 3))])
    assert len(out) < WEB_EXCERPT_MAX * 2


def test_the_whole_section_is_capped_and_keeps_the_newest_calls():
    calls = [_call(url=f"https://example.com/{i}", excerpt="y" * 1000) for i in range(40)]
    out = render_web_notes(calls)
    assert len(out) <= WEB_NOTES_MAX + 200  # header plus the omission note
    assert "https://example.com/39" in out
    assert "https://example.com/0 " not in out


def test_a_call_without_url_or_query_is_skipped():
    assert render_web_notes([_call(kind="web_fetch", url="", query="")]) == ""


def test_the_full_page_body_never_reaches_the_notes():
    """web_notes carries pointers and a short excerpt — the 9.5KB article dump
    that phase A pulled must not ride into every generation round."""
    body = "PAGE BODY " * 5000
    out = render_web_notes([_call(excerpt=body)])
    assert len(out) < len(body) / 3


# ── render_gathering_notes ──────────────────────────────────────────────────

def test_sections_without_a_summary_are_not_dropped():
    """`summary` is required by the schema, but the renderer is built on the
    assumption that the model's JSON may not match it. Findings and
    constraints with no summary used to render as an empty string (I-46)."""
    out = render_gathering_notes({
        "summary": "",
        "data_findings": [{"source": "orders table", "shape": "504 rows"}],
        "constraints": ["no external CDN"],
    })
    assert "### Data findings" in out
    assert "- orders table\n  shape: 504 rows" in out
    assert "### Constraints\n- no external CDN" in out


def test_summary_alone_still_yields_to_legacy_notes():
    """Unchanged path: only a summary and an old-style `notes` — notes win."""
    assert render_gathering_notes({"summary": "s", "notes": "legacy body"}) == "legacy body"
    assert render_gathering_notes({"summary": "s"}) == "s"
    assert render_gathering_notes({}) == ""


def test_summary_with_sections_renders_both():
    out = render_gathering_notes({"summary": "Sum.", "open_points": ["colour?"]})
    assert out == "Sum.\n\n### Open points\n- colour?"
