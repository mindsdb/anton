"""The artifact kit (`anton.core.artifacts.kit`) and the `generate_artifact` branch that uses it.

The kit turns the builder's data into one self-contained page: the spec (with its data)
embedded as JSON, the stylesheet, and only the runtime modules the page uses. Mistakes in
how a page is put together raise `kit.KitError` with what to change. A page saved by the
kit can be reopened (`kit.load`) and edited by block id.

`generate_artifact` sends every `html-app` to the kit's page builder
(`kit.builder.build_page`) instead of the generator pipeline, and can register a new
page itself when given a `name` instead of a `slug`.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

import anton.core.tools.generate_artifact as gen_pkg
from anton.core.artifacts import ArtifactStore, kit
from anton.core.artifacts.kit import builder, render
from anton.core.llm.provider import LLMResponse, ToolCall
from anton.core.tools import tool_handlers
from anton.core.tools.progress import ToolProgress
from anton.core.tools.tool_handlers import handle_generate_artifact

_SPEC = re.compile(r'<script type="application/json" id="ak-spec">(.*?)</script>', re.DOTALL)


def _spec_of(html: str) -> dict:
    m = _SPEC.search(html)
    assert m, "the page carries no spec"
    return json.loads(m.group(1).replace("<\\/", "</"))


def _revenue() -> pd.DataFrame:
    return pd.DataFrame({"month": ["2026-01", "2026-02", "2026-03"], "revenue": [1200.0, 1500.5, 900.0]})


# ── Building and saving a page ───────────────────────────────────────────────


def test_a_saved_page_embeds_its_spec_with_the_data(tmp_path):
    page = kit.Page("Revenue", "Up 25% in February", badges=["Q1"])
    page.kpis([kit.kpi("Revenue", 3600.5, "currency")])
    page.chart("bar", _revenue(), x="month", y="revenue", title="Revenue by month", format="currency")

    path = Path(page.save(tmp_path / "revenue", quiet=True))

    assert path == tmp_path / "revenue" / "index.html"  # a folder gets index.html
    html = path.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    spec = _spec_of(html)
    assert spec["title"] == "Revenue" and spec["badges"] == ["Q1"]
    chart = next(b for b in spec["blocks"] if b["type"] == "chart")
    # The rows themselves, straight from the DataFrame, not a description of them.
    assert spec["data"][chart["data"]] == [
        {"month": "2026-01", "revenue": 1200.0},
        {"month": "2026-02", "revenue": 1500.5},
        {"month": "2026-03", "revenue": 900.0},
    ]
    kpis = next(b for b in spec["blocks"] if b["type"] == "kpis")
    assert kpis["items"] == [{"label": "Revenue", "value": 3600.5, "format": "currency"}]


def test_markup_in_the_data_cannot_close_the_spec_script(tmp_path):
    def saved(text: str) -> str:
        page = kit.Page("Notes")
        page.text(text)
        return Path(page.save(tmp_path / str(len(text)), quiet=True)).read_text(encoding="utf-8")

    hostile = saved("a </script><script>alert(1)</script> b")
    plain = saved("a b")

    assert _spec_of(hostile)["blocks"][0]["md"] == "a </script><script>alert(1)</script> b"
    # The text closed no script early, and the no-JS rendition shows it as text.
    assert hostile.count("</script>") == plain.count("</script>")
    assert "&lt;script&gt;alert(1)" in hostile


def test_an_html_comment_opener_in_the_data_keeps_the_spec_valid_json(tmp_path):
    page = kit.Page("Notes")
    page.text("see <!-- here")
    html = Path(page.save(tmp_path, quiet=True)).read_text(encoding="utf-8")

    raw = _SPEC.search(html).group(1)
    assert "<!--" not in raw
    assert json.loads(raw)["blocks"][0]["md"] == "see <!-- here"  # what the runtime's JSON.parse sees
    assert kit.load(tmp_path).spec["blocks"][0]["md"] == "see <!-- here"


def test_save_prints_the_outline_and_warnings(tmp_path, capsys):
    page = kit.Page("Sites")
    page.table([], title="Empty")
    page.save(tmp_path, quiet=False)

    out = capsys.readouterr().out
    assert "b1 table 'Empty'" in out
    assert "WARNINGS" in out and "the data is empty" in out


def test_the_page_source_states_its_figures_without_javascript(tmp_path):
    """Before the runtime runs (or without it), the page source already reads as a report."""
    page = kit.Page("Revenue")
    page.kpis([kit.kpi("Orders", 1234, "int")])
    page.table(_revenue(), ["month", kit.col("revenue", "Revenue")], title="Detail")
    html = Path(page.save(tmp_path, quiet=True)).read_text(encoding="utf-8")

    root = html.split('<div id="ak-root">', 1)[1].split("<script>", 1)[0]
    assert "Revenue" in root and "Orders" in root and "1,234" in root
    assert "2026-02" in root


# ── Map layers ───────────────────────────────────────────────────────────────


def test_route_ends_are_stored_as_nested_lat_lon_objects():
    lanes = pd.DataFrame([{
        "lane": "RTM-NYC", "o_lat": 51.9225, "o_lon": 4.4792, "now_lat": 45.0, "now_lon": -30.0,
        "d_lat": 40.7128, "d_lon": -74.006, "teu": 120,
    }])
    page = kit.Page("Lanes")
    block = page.map(kit.routes(lanes, "o_lat", "o_lon", "d_lat", "d_lon", via_lat="now_lat", via_lon="now_lon",
                                label="lane"), title="Lanes")

    layer = block["layers"][0]
    assert layer["ends"] == ["from", "via", "to"]
    assert not any(k.endswith(("_lat", "_lon")) for k in layer)
    assert page.spec["data"][layer["data"]] == [{
        "from": {"lat": 51.9225, "lon": 4.4792},
        "via": {"lat": 45.0, "lon": -30.0},
        "to": {"lat": 40.7128, "lon": -74.006},
        "lane": "RTM-NYC", "teu": 120,
    }]


def test_points_are_stored_under_lat_lon_and_rows_without_coordinates_are_dropped():
    sites = [
        {"name": "Rotterdam", "latitude": 51.92, "longitude": 4.48},
        {"name": "Atlantis", "latitude": None, "longitude": None},
    ]
    page = kit.Page("Sites")
    block = page.map(kit.points(sites, lat="latitude", lon="longitude", label="name"))

    layer = block["layers"][0]
    assert (layer["lat"], layer["lon"]) == ("lat", "lon")
    assert page.spec["data"][layer["data"]] == [{"lat": 51.92, "lon": 4.48, "name": "Rotterdam"}]
    assert any("Atlantis" in w for w in page.warnings)


def test_a_latitude_out_of_range_is_refused():
    with pytest.raises(kit.KitError, match="out of range.*swapped"):
        kit.Page("x").map(kit.points([{"name": "a", "lat": 4.48, "lon": 151.9}, {"name": "b", "lat": 151.9, "lon": 4.48}],
                                     label="name"))


# ── Mistakes come back as KitError with what to change ───────────────────────


def test_a_column_of_timestamps_keeps_its_times_of_day():
    """Midnight is written with its time when the rest of the column has times, so the column
    reads as times throughout (not one bare date among hours)."""
    df = pd.DataFrame({"at": pd.date_range("2026-03-14", periods=3, freq="h"), "n": [1, 2, 3]})

    from anton.core.artifacts.kit.data import records

    rows = records(df)

    assert [r["at"] for r in rows] == ["2026-03-14 00:00:00", "2026-03-14 01:00:00", "2026-03-14 02:00:00"]


def test_a_column_of_dates_stays_dates():
    from anton.core.artifacts.kit.data import records

    rows = records(pd.DataFrame({"day": pd.date_range("2026-03-14", periods=2, freq="D")}))

    assert [r["day"] for r in rows] == ["2026-03-14", "2026-03-15"]


def test_profile_shows_times_of_day(capsys):
    df = pd.DataFrame({"opened": [pd.Timestamp("2026-03-14 06:40"), pd.Timestamp("2026-03-14 19:20")]})

    kit.profile(df, "incidents")

    assert "opened: datetime (with times of day), 2026-03-14 06:40 .. 2026-03-14 19:20" in capsys.readouterr().out


def test_a_date_format_over_timed_values_is_a_warning(tmp_path):
    df = pd.DataFrame({"opened": [pd.Timestamp("2026-03-14 06:40")], "id": ["INC-1"]})
    page = kit.Page("Incidents")
    page.table(df, ["id", kit.col("opened", "Opened", "date")], title="Open")

    page.save(tmp_path)

    assert any("times of day" in w and "'opened'" in w for w in page.warnings)


def test_a_chart_leaves_its_orientation_to_the_runtime_unless_told():
    df = pd.DataFrame({"team": ["A", "B"], "n": [1, 2]})
    page = kit.Page("t")

    auto = page.chart("bar", df, x="team", y="n")
    upright = page.chart("bar", df, x="team", y="n", horizontal=False)

    assert auto.get("horizontal") is None and upright["horizontal"] is False


def test_a_figure_stated_in_text_that_no_figure_shows_is_a_warning(tmp_path):
    df = pd.DataFrame({"id": ["A", "B", "C", "D"], "status": ["Late", "Late", "On time", "Late"],
                       "value": [402350.0, 1000000.0, 250000.0, 22.5]})
    page = kit.Page("Orders", "Three late orders worth $1.4M; 75% of orders are late")
    page.kpis([kit.kpi("Value at risk", 1402350, "currency_compact")])
    page.text("Since 2026-03-04 09:30, over 13 weeks (Q4 FY24), order D is 22.5 units; the total is $1.40M.")
    page.callout("Exposure is $1.8M.")
    page.table(df, ["id", "status", "value"])

    page.save(tmp_path)

    [warning] = [w for w in page.warnings if "text states" in w]
    assert "$1.8M" in warning
    for fine in ("$1.4M", "75%", "22.5", "13", "2026", "09:30"):
        assert fine not in warning.split("states")[1].split("which")[0]


def test_numbers_in_text_are_read_as_the_page_language_writes_them(tmp_path):
    df = pd.DataFrame({"site": ["Lyon", "Nice"], "droits": [91244.0, 1149.0]})
    page = kit.Page("Droits", "91 244 $ de droits à Lyon et 1 149 $ à Nice", lang="fr")
    page.table(df, ["site", "droits"])

    page.save(tmp_path)

    assert not [w for w in page.warnings if "text states" in w]


def test_kit_errors_are_value_errors():
    assert issubclass(kit.KitError, ValueError)


def test_an_unknown_chart_kind_names_the_kinds_and_the_escape_hatch():
    with pytest.raises(kit.KitError) as err:
        kit.Page("x").chart("radar", _revenue(), x="month", y="revenue")
    message = str(err.value)
    assert "'radar'" in message and "line" in message and "page.echarts" in message


def test_a_missing_field_names_the_fields_the_data_has():
    with pytest.raises(kit.KitError) as err:
        kit.Page("x").chart("line", _revenue(), x="month", y="revenu", title="Revenue")
    message = str(err.value)
    assert "'revenu'" in message
    assert "Fields: month, revenue" in message


def test_a_missing_table_column_is_refused():
    with pytest.raises(kit.KitError, match="no field 'cost'"):
        kit.Page("x").table(_revenue(), ["month", "cost"])


@pytest.mark.parametrize("make", [
    lambda: kit.kpi("Revenue", 1, "dollars"),
    lambda: kit.col("revenue", "Revenue", "dollars"),
    lambda: kit.Page("x", formats={"revenue": "dollars"}),
    lambda: kit.Page("x").chart("bar", _revenue(), x="month", y="revenue", format="dollars"),
])
def test_an_unknown_format_lists_the_formats(make):
    with pytest.raises(kit.KitError) as err:
        make()
    message = str(err.value)
    assert "'dollars'" in message and "currency" in message and "percent" in message


def test_a_non_numeric_y_is_refused():
    with pytest.raises(kit.KitError, match="must be numeric"):
        kit.Page("x").chart("bar", [{"m": "a", "v": "1,200"}], x="m", y="v")


@pytest.mark.parametrize("js", [
    "fetch('/api/data').then(r => r.json())",
    "const x = new XMLHttpRequest();",
    "fetch ('https://example.com')",
])
def test_custom_code_may_not_load_data_when_the_page_is_viewed(js):
    page = kit.Page("x")
    with pytest.raises(kit.KitError) as err:
        page.html("<div></div>", js=js)
    assert "page.dataset" in str(err.value)
    assert page.spec["blocks"] == []


def test_custom_code_that_reads_embedded_data_is_accepted():
    page = kit.Page("x")
    name = page.dataset(_revenue(), "revenue")
    page.html("<div></div>", js=f"el.textContent = AK.data('{name}').length;")
    assert page.spec["data"]["revenue"][0]["month"] == "2026-01"


# ── Reopening and editing a saved page ───────────────────────────────────────


def _saved_page(folder: Path) -> kit.Page:
    page = kit.Page("Revenue", "Q1")
    page.text("Intro")
    page.table(_revenue(), title="Detail")
    page.html('<div class="x"></div>', js="el.textContent = 'hi';", css=".x { color: red; }", title="Custom")
    page.save(folder, quiet=True)
    return page


def test_load_round_trips_a_saved_page(tmp_path):
    page = _saved_page(tmp_path)
    loaded = kit.load(tmp_path)

    assert type(loaded) is kit.Page
    assert loaded.spec == page.spec
    # Custom code and styles come back too: saving it unchanged writes the same page.
    assert loaded.html_text() == page.html_text()


def test_a_loaded_page_is_edited_by_block_id(tmp_path):
    _saved_page(tmp_path)
    page = kit.load(tmp_path)
    intro, table, custom = (b["id"] for b in page.spec["blocks"])

    with page.after(intro):
        page.callout("Revenue fell in March.", tone="warn")
    page.update(table, title="Monthly detail")
    page.remove(intro)
    page.save(tmp_path, quiet=True)

    again = kit.load(tmp_path)
    blocks = again.spec["blocks"]
    assert [b["type"] for b in blocks] == ["callout", "table", "html"]
    assert blocks[0]["md"] == "Revenue fell in March."
    assert again.block(table)["title"] == "Monthly detail"
    assert again.block(custom)["title"] == "Custom"
    assert "el.textContent = 'hi';" in again.html_text()
    assert ".x { color: red; }" in again.html_text()
    with pytest.raises(kit.KitError, match="outline"):
        again.block(intro)


def test_a_deck_round_trips_as_a_deck(tmp_path):
    deck = kit.Deck("Board update")
    with deck.slide("Revenue", layout="content"):
        deck.chart("line", _revenue(), x="month", y="revenue")
    deck.save(tmp_path, quiet=True)

    loaded = kit.load(tmp_path)
    assert isinstance(loaded, kit.Deck)
    assert loaded.spec == deck.spec


def test_load_refuses_a_page_the_kit_did_not_make(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><p>hand-made</p>", encoding="utf-8")
    with pytest.raises(kit.KitError, match="not made with the kit"):
        kit.load(tmp_path)


# ── Only the runtime the page uses ───────────────────────────────────────────


def _modules_in(html: str) -> set[str]:
    return {name for name in render.ORDER if render._asset(name) in html}


def test_a_page_inlines_only_the_runtime_modules_it_uses():
    page = kit.Page("Revenue")
    page.kpis([kit.kpi("Revenue", 1, "currency")])
    page.chart("bar", _revenue(), x="month", y="revenue")

    assert _modules_in(page.html_text()) == {"core.js", "blocks.js", "chart.js"}


def test_a_map_nested_in_tabs_brings_the_geo_runtime():
    page = kit.Page("Sites")
    tabs = page.tabs(["Map", "Table"])
    with tabs["Map"], page.section("Where"):
        page.map(kit.points([{"name": "Rotterdam", "lat": 51.92, "lon": 4.48}], label="name"))
    with tabs["Table"]:
        page.table([{"name": "Rotterdam"}])

    assert _modules_in(page.html_text()) == {"core.js", "blocks.js", "geo.js", "table.js"}


def test_a_deck_brings_the_deck_runtime():
    deck = kit.Deck("Update")
    with deck.slide("Hello", layout="title"):
        deck.text("Hi")
    assert _modules_in(deck.html_text()) == {"core.js", "blocks.js", "deck.js"}


# ── Offline geocoding ────────────────────────────────────────────────────────


def test_geocode_finds_a_city_in_its_country():
    pytest.importorskip("geonamescache")
    lat, lon = kit.geocode("Rotterdam, Netherlands")
    assert abs(lat - 51.92) < 0.1 and abs(lon - 4.48) < 0.1


def test_geocode_refuses_an_ambiguous_name_rather_than_guessing():
    pytest.importorskip("geonamescache")
    assert kit.geocode("Valencia") is None  # Spain or Venezuela
    lat, lon = kit.geocode("Valencia, Spain")
    assert abs(lat - 39.47) < 0.1 and abs(lon + 0.38) < 0.1
    assert kit.geocode("Nowhereville, Spain") is None


# ── generate_artifact sends pages to the kit's page builder ──────────────────


def _session(tmp_path: Path):
    return SimpleNamespace(_workspace=SimpleNamespace(artifacts_dir=tmp_path / "artifacts"))


async def _drain(session, tc_input):
    progress, result = [], None
    async for item in handle_generate_artifact(session, tc_input):
        if isinstance(item, ToolProgress):
            progress.append(item.text)
        else:
            result = item
    return progress, result


@pytest.fixture
def fake_builder(monkeypatch):
    """`build_page` stood in for: records its arguments, reports progress, saves a page.
    The generator pipeline must not run for a page."""
    calls: list[dict] = []

    async def build_page(session, *, slug, folder, progress=None, **kw):
        calls.append({"session": session, "slug": slug, "folder": folder, **kw})
        progress.put_nowait("Reading the request")
        progress.put_nowait("Building the page")
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / "index.html").write_text("<!doctype html>", encoding="utf-8")
        result = {"status": "generated", "slug": slug, "page": str(Path(folder) / "index.html"),
                  "summary": "Revenue was $3.6K in Q1.", "rounds": 2}
        calls[-1]["returned"] = result
        return result

    async def no_pipeline(**kw):
        raise AssertionError("a page must not go through the generator pipeline")

    monkeypatch.setattr(builder, "build_page", build_page)
    monkeypatch.setattr(gen_pkg, "generate", no_pipeline)
    return calls


async def test_an_html_app_is_built_by_the_page_builder(tmp_path, fake_builder):
    store = ArtifactStore(tmp_path / "artifacts")
    slug = store.create(name="Revenue", description="d", type="html-app", primary="index.html").slug
    session = _session(tmp_path)

    progress, result = await _drain(session, {
        "slug": slug, "user_request": "a revenue page", "user_language": "English",
        "agent_understanding": "Q1 revenue by month", "known_data": "Jan 1200, Feb 1500.5",
    })

    [call] = fake_builder
    assert call["session"] is session
    assert call["slug"] == slug and Path(call["folder"]) == store.folder_for(slug)
    assert (call["user_request"], call["user_language"]) == ("a revenue page", "English")
    assert (call["agent_understanding"], call["known_data"]) == ("Q1 revenue by month", "Jan 1200, Feb 1500.5")
    assert progress == ["Reading the request", "Building the page"]
    # The builder's result, as `result_text` words it for the agent, with a success verdict.
    assert result.ok is True and result.content == builder.result_text(call["returned"])
    payload = json.loads(result.content)
    assert payload["status"] == "generated" and payload["summary"] == "Revenue was $3.6K in Q1."
    assert payload["instruction"]


async def test_a_name_without_a_slug_registers_a_new_page(tmp_path, fake_builder):
    session = _session(tmp_path)

    _, result = await _drain(session, {
        "name": "Sales map", "description": "Sites by revenue", "user_request": "map my sites",
        "user_language": "English", "agent_understanding": "sites on a map",
    })

    store = ArtifactStore(tmp_path / "artifacts")
    [artifact] = store.list()
    assert (artifact.name, artifact.description, artifact.type) == ("Sales map", "Sites by revenue", "html-app")
    assert artifact.primary == "index.html"
    [call] = fake_builder
    assert call["slug"] == artifact.slug
    assert Path(call["folder"]) == store.folder_for(artifact.slug)
    assert json.loads(result.content)["slug"] == artifact.slug


async def test_a_failed_page_build_tells_the_agent_not_to_claim_a_page(tmp_path, monkeypatch):
    async def build_page(session, *, slug, folder, progress=None, **kw):
        return {"status": "failed", "slug": slug, "page": None, "summary": "The source had no Q1 rows.", "rounds": 10}

    monkeypatch.setattr(builder, "build_page", build_page)
    slug = ArtifactStore(tmp_path / "artifacts").create(name="Revenue", description="d", type="html-app").slug
    _, result = await _drain(_session(tmp_path), {
        "slug": slug, "user_request": "a revenue page", "user_language": "English",
        "agent_understanding": "Q1 revenue",
    })

    assert result.ok is False
    payload = json.loads(result.content)
    assert payload["status"] == "failed" and payload["page"] is None
    assert "do not claim a page was built" in payload["instruction"]


# ── The page builder's loop ──────────────────────────────────────────────────


async def test_the_builder_finishes_only_once_a_page_is_saved(tmp_path, monkeypatch):
    """`finish` before the page exists is refused; the scratchpad cell that saves it,
    then `finish`, ends the build with the summary. Every model call goes through the
    session's client, with the kit reference in the system prompt."""
    folder = tmp_path / "artifacts" / "revenue"

    async def scratchpad(session, tc_input):
        page = kit.Page("Revenue")
        page.chart("bar", _revenue(), x="month", y="revenue")
        page.save(folder, quiet=True)
        return "Saved"

    monkeypatch.setattr(tool_handlers, "handle_scratchpad", scratchpad)
    plan = AsyncMock(side_effect=[
        LLMResponse(content="", tool_calls=[ToolCall(id="t1", name="finish", input={"summary": "too early"})]),
        LLMResponse(content="", tool_calls=[ToolCall(id="t2", name="scratchpad", input={
            "action": "exec", "name": "page", "code": "...", "one_line_description": "Building the page"})]),
        LLMResponse(content="", tool_calls=[ToolCall(id="t3", name="finish", input={"summary": "Q1 revenue $3.6K"})]),
    ])
    session = SimpleNamespace(_llm=SimpleNamespace(plan=plan), history=[])
    progress: asyncio.Queue = asyncio.Queue()
    result = await builder.build_page(
        session, slug="revenue", folder=folder, user_request="a revenue page", user_language="English",
        agent_understanding="Q1 revenue", progress=progress,
    )

    assert result == {"status": "generated", "slug": "revenue", "page": str(folder / "index.html"),
                      "summary": "Q1 revenue $3.6K", "rounds": 3}
    first = plan.call_args_list[0].kwargs
    assert builder.KIT_REFERENCE in first["system"]
    assert {t["name"] for t in first["tools"]} == {"scratchpad", "recall_skill", "finish"}
    results = {block["tool_use_id"]: block["content"]
               for msg in plan.call_args_list[-1].kwargs["messages"] if isinstance(msg["content"], list)
               for block in msg["content"] if block.get("type") == "tool_result"}
    assert "Not finished" in results["t1"]
    assert results["t2"] == "Saved" and results["t3"] == "ok"
    lines = []
    while not progress.empty():
        lines.append(progress.get_nowait())
    assert "Building the page" in lines and lines[-1] == "Page ready"


async def test_a_build_that_never_saves_a_page_fails(tmp_path):
    plan = AsyncMock(return_value=LLMResponse(content="I could not reach the data.", tool_calls=[]))
    session = SimpleNamespace(_llm=SimpleNamespace(plan=plan), history=[])
    result = await builder.build_page(
        session, slug="s", folder=tmp_path / "s", user_request="a page", user_language="English",
    )

    assert result["status"] == "failed" and result["page"] is None
    assert result["summary"] == "I could not reach the data."
    assert plan.await_count <= builder.MAX_ROUNDS
