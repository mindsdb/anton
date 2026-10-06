"""report_tools: escaping, structure, the filter script, reading back and editing in place."""
from __future__ import annotations

import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

from anton.core.artifacts import report_tools as rt

ROWS = [["A-100", "North", 40], ["B-200", "South", 0], ["C-300", "North", 25]]
COLS = ["Item", "Region", "Gap"]


def _page(*parts, **kw):
    return rt.page("Weekly review", *parts, **kw)


def test_page_is_a_complete_offline_document(tmp_path):
    out = rt.save(tmp_path / "r.html", _page(rt.section("Summary", rt.para("All good."))))
    text = out.read_text()
    assert text.startswith("<!doctype html>") and '<html lang="en" data-theme="light">' in text
    assert 'name="viewport"' in text and "<title>Weekly review</title>" in text
    assert "http" not in text.split("<style>")[0]
    assert rt.check(out)["title"] == "Weekly review"


def test_text_from_data_is_escaped_everywhere(tmp_path):
    evil = '<script>alert(1)</script>"&'
    page = _page(rt.section(evil, rt.para(evil), rt.bullets([evil]), rt.table([evil], [[evil]]),
                            rt.bar_chart([evil], [1], name=evil)))
    assert "<script>alert" not in page
    assert page.count("&lt;script&gt;alert(1)&lt;/script&gt;") >= 5


def test_embedded_json_cannot_close_its_script_block():
    block = rt.data("state", {"note": "</script><script>alert(1)</script>"})
    assert block.count("</script>") == 1
    assert json.loads(block[block.index(">") + 1:block.rindex("<")])["note"].startswith("</script>")


def test_nothing_is_embedded_unless_asked():
    page = _page(rt.section("Detail", rt.table(COLS, ROWS)))
    assert 'type="application/json"' not in page


def test_table_accepts_dict_rows_and_right_aligns_numbers():
    html = rt.table(COLS, [dict(zip(COLS, r)) for r in ROWS], caption="Gaps", id="gaps")
    assert '<table id="gaps"><caption>Gaps</caption>' in html
    assert '<th scope="col" class=num>Gap</th>' in html and "<td class=num>40</td>" in html


def test_table_rejects_a_row_of_the_wrong_length():
    with pytest.raises(ValueError):
        rt.table(COLS, [["only", "two"]])


def test_bar_chart_is_an_accessible_image():
    svg = rt.bar_chart(["North", "South"], [65, 0], name="Gap by region", threshold=50)
    assert 'role="img"' in svg and 'aria-label="Gap by region"' in svg and "<title>Gap by region</title>" in svg
    assert 'class="threshold"' in svg
    with pytest.raises(ValueError):
        rt.bar_chart(["x"], [-1], name="n")
    with pytest.raises(ValueError):
        rt.bar_chart(["x", "y"], [1], name="n")


def test_links_and_relative_paths(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "orders.csv").write_text("x")
    report = tmp_path / "out" / "r.html"
    href = rt.rel_link(report, tmp_path / "data" / "orders.csv")
    assert href == "../data/orders.csv"
    rt.save(report, _page(rt.section("Sources", rt.para(rt.inline("From ", rt.link(href, "orders.csv"))))))
    assert rt.check(report)["links"] == [href]
    with pytest.raises(ValueError):
        rt.link("javascript:alert(1)", "x")


def test_check_reports_structural_problems(tmp_path):
    bad = tmp_path / "bad.html"
    bad.write_text('<html><head><script src="https://cdn.example/x.js"></script></head><body>'
                   '<p id="a"></p><p id="a"></p><a href="missing.csv">m</a></body></html>')
    with pytest.raises(ValueError) as info:
        rt.check(bad)
    message = str(info.value)
    for fragment in ("no lang", "no viewport", "external resources", "duplicate ids", "missing.csv"):
        assert fragment in message


def test_check_returns_what_the_page_shows(tmp_path):
    out = rt.save(tmp_path / "r.html", _page(rt.section("Detail", rt.table(COLS, ROWS, id="detail")),
                                             rt.section("Chart", rt.bar_chart(["a"], [1], name="Chart name"))))
    seen = rt.check(out)
    assert seen["title"] == "Weekly review"
    (table,) = seen["tables"]
    assert table["id"] == "detail" and table["columns"] == COLS
    assert table["rows"] == [[str(c) for c in r] for r in ROWS]
    assert "Weekly review" in seen["text"]


def test_update_replaces_only_the_named_elements(tmp_path):
    page = _page(rt.section("Summary", rt.para("Old summary."), id="summary"),
                 rt.section("Detail", rt.table(COLS, ROWS, id="detail")),
                 rt.data("state", {"v": 1}))
    out = rt.save(tmp_path / "r.html", page)
    before = out.read_text()
    receipt = rt.update(out, {"summary": rt.inline(rt.Html("<h2>Summary</h2>"), rt.para("New <summary>.")),
                              "state": {"v": 2}})
    after = out.read_text()
    assert receipt["changed"] == ["state", "summary"]
    assert "New &lt;summary&gt;." in after and "Old summary." not in after
    assert '{"v": 2}' in after
    # Everything outside the two elements is byte-identical.
    head, tail = before.split('<section id="summary">')[0], before.split('<section id="detail-section">')[1]
    assert after.startswith(head) and tail.split('<script type="application/json"')[0] in after


def test_update_refuses_ambiguous_or_missing_ids(tmp_path):
    out = tmp_path / "r.html"
    out.write_text('<!doctype html><html lang="en"><body><p id="a">1</p><p id="a">2</p></body></html>')
    with pytest.raises(ValueError):
        rt.update(out, {"a": "x"})
    with pytest.raises(ValueError):
        rt.update(out, {"missing": "x"})
    assert out.read_text().count('id="a"') == 2


def test_update_works_on_pages_not_made_with_these_helpers(tmp_path):
    out = tmp_path / "legacy.html"
    original = ('<!DOCTYPE html>\n<html lang="en">\n<body>\n  <h1 id="title">Old</h1>\n'
                '  <div id="total" class="big">10</div><br>\n  <img src="a.png">\n</body>\n</html>\n')
    out.write_text(original)
    rt.update(out, {"title": "Planner handoff", "total": "12"})
    assert out.read_text() == original.replace(">Old<", ">Planner handoff<").replace(">10<", ">12<")


@pytest.mark.parametrize("separator", [" ", " ", "\x0c", "\x0b", "\x85", "\x1e"])
def test_update_and_insert_find_elements_after_non_newline_line_breaks(tmp_path, separator):
    # PDF text carries \x0c between pages; rt.data keeps U+2028 as is.
    out = tmp_path / "legacy.html"
    original = (f'<!DOCTYPE html>\n<html lang="en">\n<body>\n<p id="intro">Quarter{separator}summary</p>\n'
                '<section id="detail">\n<h2>Detail</h2>\n<p>old</p>\n</section>\n</body>\n</html>\n')
    out.write_text(original, encoding="utf-8")
    rt.update(out, {"detail": rt.para("new")})
    rt.insert(out, rt.para("note"), after="detail")
    expected = original.replace("\n<p>old</p>\n", "<p>new</p>").replace("</section>", "</section><p>note</p>")
    assert out.read_text(encoding="utf-8") == expected


def test_update_and_insert_keep_line_endings(tmp_path):
    out = tmp_path / "crlf.html"
    original = (b'<!DOCTYPE html>\r\n<html lang="en">\r\n<body>\r\n<div id="total">10</div>\r\n'
                b'<p id="note">x</p>\r\n</body>\r\n</html>\r\n')
    out.write_bytes(original)
    receipt = rt.update(out, {"total": "12"})
    rt.insert(out, rt.para("added"), after="note")
    assert out.read_bytes() == original.replace(b">10<", b">12<").replace(b"x</p>", b"x</p><p>added</p>")
    assert receipt["bytes_before"] == len(original)


def test_update_keeps_a_section_heading_unless_the_new_content_has_one(tmp_path):
    out = rt.save(tmp_path / "r.html", _page(rt.section("Conclusion", rt.para("Old."), id="conclusion"),
                                             rt.section("Detail", rt.para("Rows."), id="detail")))
    rt.update(out, {"conclusion": rt.para("New."), "detail": rt.inline(rt.Html("<h2>Row detail</h2>"), rt.para("R."))})
    text = out.read_text()
    assert '<section id="conclusion"><h2>Conclusion</h2><p>New.</p></section>' in text
    assert '<section id="detail"><h2>Row detail</h2><p>R.</p></section>' in text
    assert rt.check(out)["text"].count("Conclusion") == 1


def test_update_with_a_whole_element_of_the_same_id_replaces_the_element(tmp_path):
    out = rt.save(tmp_path / "r.html", _page(rt.section("Summary", rt.para("Old.")),
                                             rt.section("Detail", rt.table(COLS, ROWS[:1], id="detail"))))
    rt.update(out, {"summary-section": rt.section("Summary", rt.para("New.")),
                    "detail": rt.table(COLS, ROWS, id="detail")})
    text = out.read_text()
    assert '<section id="summary-section"><h2>Summary</h2><p>New.</p></section>' in text
    assert text.count('id="summary-section"') == 1 and "Old." not in text
    seen = rt.check(out)  # no duplicate ids, no table nested in a table
    assert [t["rows"] for t in seen["tables"]] == [[[str(c) for c in r] for r in ROWS]]


def test_update_refuses_new_content_with_the_id_twice(tmp_path):
    out = rt.save(tmp_path / "r.html", _page(rt.section("Summary", rt.para("Old."), id="s")))
    before = out.read_text()
    twice = rt.inline(rt.section("A", id="s"), rt.section("B", id="s"))
    with pytest.raises(ValueError, match="2 elements"):
        rt.update(out, {"s": twice})
    assert out.read_text() == before


def test_check_reports_markup_shown_as_text(tmp_path):
    out = rt.save(tmp_path / "r.html", _page(rt.section("Summary", rt.para("Fine."), id="summary")))
    rt.update(out, {"summary": rt.inline("<h2>Summary</h2>", rt.para("Escaped by mistake."))})
    with pytest.raises(ValueError, match="HTML tags shown as text"):
        rt.check(out)
    code = rt.save(tmp_path / "code.html", _page(rt.section("Markup", rt.Html("<pre><code>&lt;h2&gt;x&lt;/h2&gt;</code></pre>"))))
    rt.check(code)  # inside code or pre it is intended
    plain = rt.save(tmp_path / "plain.html", _page(rt.para("Stock < demand > 0; ids <P1>.")))
    rt.check(plain)


def test_insert_adds_new_parts_to_a_page_not_made_with_these_helpers(tmp_path):
    out = tmp_path / "legacy.html"
    original = ('<!DOCTYPE html>\n<html lang="en"><head><meta name="viewport" content="width=device-width">'
                '</head><body>\n<h1>Planning report</h1>\n<section id="detail"><h2>Detail</h2><table>'
                '<tr><td>1</td></tr></table></section>\n<section id="audit"><h2>Audit</h2></section>\n</body></html>\n')
    out.write_text(original)
    summary = rt.section("Decision summary", rt.para("Cover 30 <units>."), id="decision")
    receipt = rt.insert(out, summary, before="detail")
    rt.insert(out, rt.para("Checked."), after="audit")
    assert receipt["inserted"] == "before detail"
    assert out.read_text() == original.replace('<section id="detail">', summary + '<section id="detail">').replace(
        '<h2>Audit</h2></section>', '<h2>Audit</h2></section><p>Checked.</p>')
    assert "Cover 30 &lt;units&gt;." in out.read_text()
    assert [s for s in rt.check(out)["text"].split() if s in ("Decision", "Detail", "Audit")] == ["Decision", "Detail", "Audit"]


def test_insert_needs_exactly_one_existing_anchor(tmp_path):
    out = rt.save(tmp_path / "r.html", _page(rt.section("A", id="a")))
    before = out.read_text()
    for kwargs in ({}, {"before": "a", "after": "a"}, {"before": "missing"}):
        with pytest.raises(ValueError):
            rt.insert(out, rt.para("x"), **kwargs)
    assert out.read_text() == before


class _Rows(HTMLParser):
    """The filter's rows, select options, total and empty-state elements, as data for the JS test."""

    def __init__(self):
        super().__init__()
        self.rows, self.options, self.total, self.empty_hidden = [], [], None, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr" and "data-rt-key" in a:
            self.rows.append({"key": a["data-rt-key"], "value": a.get("data-rt-value"), "hidden": "hidden" in a})
        elif tag == "option":
            self.options.append(a["value"])
        elif tag == "p" and "data-rt-total" in a:
            self.total = a["data-rt-total"]
        elif tag == "p" and "data-rt-empty" in a:
            self.empty_hidden = "hidden" in a


_NODE_HARNESS = r"""
const spec = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const results = {};
for (const choice of spec.choices) {
  const rows = spec.rows.map(r => ({hidden: r.hidden,
    getAttribute: n => n === 'data-rt-key' ? r.key : r.value,
    hasAttribute: n => n === 'data-rt-value' ? r.value !== null : false}));
  const total = spec.total === null ? null : {textContent: '', getAttribute: () => spec.total};
  const empty = {hidden: false};
  let listener = null;
  const select = {selectedIndex: choice, value: spec.options[choice], addEventListener: (e, f) => { listener = f; }};
  const box = {hasAttribute: n => n === 'data-rt-hide-zero' && spec.hideZero,
    querySelector: q => q === 'select' ? select : q === '[data-rt-total]' ? total : q === '[data-rt-empty]' ? empty : null,
    querySelectorAll: () => rows};
  global.document = {querySelectorAll: () => [box]};
  eval(spec.script);
  results[spec.options[choice] || 'All'] = {visible: rows.filter(r => !r.hidden).length,
    total: total && total.textContent, empty: !empty.hidden};
}
process.stdout.write(JSON.stringify(results));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("hide_zero", [False, True])
def test_filter_script_filters_rows_and_totals(hide_zero):
    html = rt.filter_table(COLS, ROWS, key="Region", label="Region", region_name="Matching items",
                           value="Gap", total_label="Total gap", empty_text="Nothing here.", hide_zero=hide_zero)
    page = _page(html)
    assert 'role="region" aria-label="Matching items"' in page and '<label for="matching-items-filter-select">' in page
    parsed = _Rows()
    parsed.feed(html)
    # The All view is already correct in the HTML, before any script runs.
    assert [r["hidden"] for r in parsed.rows] == [False, hide_zero, False]
    assert ("Total gap: 65" in html) and parsed.empty_hidden is True
    spec = {"rows": parsed.rows, "options": parsed.options, "total": parsed.total, "hideZero": hide_zero,
            "choices": list(range(len(parsed.options))), "script": rt._FILTER_JS}
    out = json.loads(subprocess.run(["node", "-e", _NODE_HARNESS], input=json.dumps(spec), text=True,
                                    capture_output=True, check=True).stdout)
    assert out["All"] == {"visible": 2 if hide_zero else 3, "total": "Total gap: 65", "empty": False}
    assert out["North"] == {"visible": 2, "total": "Total gap: 65", "empty": False}
    assert out["South"] == {"visible": 0 if hide_zero else 1, "total": "Total gap: 0", "empty": hide_zero}


def test_filter_options_match_row_keys_for_floats_and_blanks():
    html = rt.filter_table(["Bin", "Qty"], [[2.0, 1], [None, 2], [2.5, 3], ["<b>", 4]],
                           key="Bin", label="Bin", region_name="Bins")
    parsed = _Rows()
    parsed.feed(html)
    # The script compares option values with data-rt-key exactly.
    assert parsed.options[1:] == sorted({r["key"] for r in parsed.rows}) == ["", "2", "2.5", "<b>"]
    assert '<option value="">(blank)</option>' in html and "<b>" not in html


def test_a_filter_added_to_a_page_brings_its_script_once(tmp_path):
    def table(region):
        return rt.filter_table(COLS, ROWS, key="Region", label="Region", region_name=region)

    out = rt.save(tmp_path / "r.html", _page(rt.section("Summary", rt.para("x")), rt.section("Old", id="old")))
    assert rt._FILTER_SCRIPT not in out.read_text()
    rt.insert(out, table("North items"), after="summary-section")
    rt.update(out, {"old": table("South items")})
    text = out.read_text()
    assert text.count(rt._FILTER_SCRIPT) == 1
    assert text.index(rt._FILTER_SCRIPT) > text.index("south-items-filter")  # runs after both filters
    assert text.endswith(rt._FILTER_SCRIPT + "</body></html>\n")
    rt.check(out)

    legacy = tmp_path / "legacy.html"
    legacy.write_text('<html lang="en"><body><div id="a">x</div></body></html>')
    rt.update(legacy, {"a": table("Items")})
    assert legacy.read_text().endswith(rt._FILTER_SCRIPT + "</body></html>")
    rt.insert(legacy, rt.para("note"), after="a")
    assert legacy.read_text().count(rt._FILTER_SCRIPT) == 1


def test_filter_table_needs_real_columns():
    with pytest.raises(ValueError):
        rt.filter_table(COLS, ROWS, key="Site", label="Site", region_name="r")


def test_default_section_ids_do_not_collide_with_a_same_named_table():
    page = _page(rt.section("Detail", rt.table(COLS, ROWS, id="detail")))
    assert '<section id="detail-section">' in page and '<table id="detail">' in page


def test_page_rejects_unknown_themes():
    with pytest.raises(ValueError):
        _page(theme="sepia")
    assert 'data-theme="dark"' in _page(theme="dark")


def test_importable_without_side_effects_from_a_plain_interpreter(tmp_path):
    """The scratchpad imports it from the installed package; nothing is copied into folders."""
    assert Path(rt.__file__).name == "report_tools.py"
    assert not list(tmp_path.iterdir())
