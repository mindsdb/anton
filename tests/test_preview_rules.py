"""What the generator prompts tell the model about the app preview."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.core.tools.generate_artifact import prompts, verifiers


def test_the_generator_prompts_pin_the_versions_the_check_carries():
    """A floating `@4`/`@5` gets whatever jsDelivr serves today, which the
    check's mirror does not hold, so the page could only be unverified."""
    for prompt in (
        prompts.build_subagent_system_prompt(Path("/tmp/a")),
        prompts.build_frontend_system_prompt(Path("/tmp/a")),
    ):
        assert "https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.3.3" in prompt
        assert "https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js" in prompt
        assert "@tailwindcss/browser@4\"" not in prompt and "echarts@5/" not in prompt


def test_the_verifier_contract_names_the_external_resource_warning():
    for prompt in (
        prompts.build_subagent_system_prompt(Path("/tmp/a")),
        prompts.build_frontend_system_prompt(Path("/tmp/a")),
    ):
        assert "the app preview does not load it" in " ".join(prompt.split())


def test_every_view_starts_through_the_helper_after_the_browser_gate():
    """A per-view `try`/`catch` that only wrote "This view failed to load"
    into the map panel kept the error that broke the map out of the browser
    check, which counts console errors: the catch logged none. Both frontend
    prompts now prescribe one helper for every view's setup, after the
    browser-gate bullet, and keep the storage helper, whose throw is expected
    in the preview frame, off `console.error`."""
    for prompt in (
        prompts.build_subagent_system_prompt(Path("/tmp/a")),
        prompts.build_frontend_system_prompt(Path("/tmp/a")),
    ):
        gate = prompt.index("Where a headless browser is available")
        rule = prompt.index(prompts.VIEW_HELPER_JS)
        storage = prompt.index("stays silent or uses `console.warn`")
        assert gate < rule < storage < prompt.index("Not checked by the verifier")
        assert "Start every view through this one helper" in prompt
        assert "Await or return every promise the setup starts" in prompt


def test_a_tabs_views_start_through_the_helper_too():
    """The tab rule starts a tab's charts on its first visit, from the click
    handler, where an uncaught throw stops that tab's other views. It starts
    each by its title: a setup function handed over by reference throws a
    `ReferenceError` outside the helper when its script never parsed."""
    for prompt in (
        prompts.build_subagent_system_prompt(Path("/tmp/a")),
        prompts.build_frontend_system_prompt(Path("/tmp/a")),
    ):
        tab_rule = prompt.index("Multi-tab dashboards")
        helper = prompt.index("function runView(name, el)")
        # The tab rule sits in the design rules, after the contract that
        # defines the helper it names.
        assert helper < tab_rule
        assert (
            "`initChartsFor(name)` starts each view on that tab by its title with `runView(title, el)`"
            in " ".join(prompt.split())
        )


def test_each_views_code_goes_in_a_script_of_its_own_and_starts_by_name():
    """A browser parses a `<script>` element as a whole, so a syntax error in a
    page's one script stopped every view. Both frontend prompts put the helper
    alone first, each view's setup in a script of its own, and start each view
    by its title, so the helper fails a view whose script never ran by name."""
    for prompt in (
        prompts.build_subagent_system_prompt(Path("/tmp/a")),
        prompts.build_frontend_system_prompt(Path("/tmp/a")),
    ):
        flat = " ".join(prompt.split())
        for rule in (
            "Put the helper alone in the page's first inline `<script>`",
            "Give each view an inline `<script>` of its own that holds only its setup",
            "`<script>defineView('Shipment map', (el) => { /* draw the map */ });</script>`",
            "The page's last script starts each view by its title once the page has parsed",
            "`addEventListener('DOMContentLoaded', () => { runView('Shipment map', document.getElementById('world-map')); });`",
            "A `<script type=\"module\">` runs after every classic script, so a view it registers exists only by then",
            "Give each view a title no other view on the page uses",
            "Keep the functions the page's HTML calls, such as a tab's `showPage`, outside that listener",
            "Declare nothing else at a view script's top level: every function and constant the view uses goes inside its setup",
            "a syntax error in one view's script stops only that view",
            "Never put two views' code in one script, and never pass a setup function to `runView`",
            "Inline your own CSS and all JS, each view's code in a `<script>` of its own",
        ):
            assert rule in flat, rule
        # The by-reference example is gone: it is the call that escapes the helper.
        assert "drawMap" not in prompt
    assert "your own JS in inline `<script>` elements, each view's code in a script of its own" in " ".join(
        prompts.build_frontend_system_prompt(Path("/tmp/a")).split()
    )


def test_the_helper_names_the_failed_view_first_in_both_channels():
    """The render check cuts a finding at 300 characters, and an `Error`
    logged as an argument renders as its whole stack, so a view name logged
    after it can fall past the cut. The helper builds both messages as
    strings that start with the view's name."""
    helper = prompts.VIEW_HELPER_JS
    assert "console.error(name + ' failed: ' + why)" in helper
    assert "reportError(new Error(name + ': ' + why))" in helper
    assert "el.setAttribute('data-view-failed', name)" in helper


_GENERATOR_PROMPTS = (
    prompts.build_subagent_system_prompt(Path("/tmp/a")),
    prompts.build_frontend_system_prompt(Path("/tmp/a")),
)
_MAP_PACKAGES = ("d3@7.9.0", "topojson-client@3.1.0", "world-atlas@2.0.2", "globe.gl@2.34.2")


def _skill() -> str:
    path = Path(__file__).resolve().parents[1] / "anton/core/memory/builtin_skills/build-html-dashboard/SKILL.md"
    return path.read_text(encoding="utf-8")


def test_the_generator_prompts_carry_the_tested_map_and_globe_recipes():
    """A world-map request goes to generate_artifact, which recalls no skill.
    Told that Tailwind and ECharts were the only resources a page may load,
    and given no recipe, the generator had no tested way to draw a map:
    ECharts ships no world outline. The spec writer's fixed stack names the
    same packages, or spec.md forbids what the generator is told to use."""
    for prompt in _GENERATOR_PROMPTS:
        flat = " ".join(prompt.split())
        for package in _MAP_PACKAGES:
            assert package in flat, package
        assert "echarts.registerMap('world', countries)" in flat
        assert ".polygonsData(countries.features)" in flat
        assert "d3.json(WORLD_ATLAS)" in flat
        assert "the recipe packages under MAPS AND GLOBES" in flat
        assert "for a map or globe, the recipe packages and world atlas named there" in flat
    spec = " ".join(prompts.build_tech_spec_instruction(SimpleNamespace(prd="", artifact_type="html-app")).split())
    for package in _MAP_PACKAGES:
        assert package in spec, package


def test_the_generator_prompts_state_the_preview_policy():
    for prompt in _GENERATOR_PROMPTS:
        flat = " ".join(prompt.split())
        assert "no `eval` or `new Function`, and no web workers" in flat
        assert "no external stylesheet or web font, no external image or map tile" in flat
        assert "Leaflet, raster map tiles, MapLibre" in flat
        assert "so the page renders on every deployment" in flat


def test_the_map_recipe_passes_the_static_check_it_is_written_for():
    """The recipe's own lines, put into a page, pass `verify_frontend`; the
    same atlas loaded with `fetch()` fails it, which is why the recipe uses
    `d3.json`."""
    rules = prompts._DESIGN_RULES
    bullet = rules[rules.index("- MAPS AND GLOBES"):rules.index("- d3 into an")]
    code = re.findall(r"^\s*`(.+)`$", bullet, re.M)
    tags = [line for line in code if line.startswith("<script")]
    script = [line for line in code if not line.startswith("<")]
    assert len(tags) == 2 and len(script) == 2, code
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        + "".join(tags)
        + '</head><body><div id="world-map"></div><script>\n'
        + "\n".join(script)
        + "\n</script></body></html>"
    )
    result = verifiers.verify_frontend(page, is_fullstack=False)
    assert (result.errors, result.warnings) == ([], [])
    fetched = page.replace("d3.json(WORLD_ATLAS)",
                           "fetch('https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-110m.json')")
    assert verifiers.verify_frontend(fetched, is_fullstack=False).errors


def test_the_skill_loads_the_atlas_the_way_the_check_allows():
    """The skill's item 3 said generate_artifact's checks allowed a literal
    `fetch()` of the world atlas; they never did."""
    skill = _skill()
    assert "with one exception: a map may fetch" not in skill
    assert "with `d3.json(...)` (item 9), not with `fetch()`" in skill
    assert "loading `countries-110m.json` with `d3.json` (item 3)" in skill


def test_the_skill_says_which_preview_blocks_what():
    """Desktop navigates the preview frame, so the page there inherits no
    content security policy and a file next to it loads. The skill keeps the
    rules for every deployment and says the web preview is the one that
    blocks."""
    skill = _skill()
    for flat_claim in ("never loads there", "never shows there", "never reaches it",
                       "simply does not render", "do not render in the preview."):
        assert flat_claim not in skill, flat_claim
    for reason in (
        "the web app preview shows only the page itself, so a separate file next to it",
        "The web app preview runs the page in a sandboxed frame under a content security policy",
        "Build to that policy so the page renders on every deployment",
        "break that policy, so they do not render in the web app preview",
        "The web app preview loads only the page itself, so a separate file does not show there",
        "The web app preview loads only the page itself: a separate file does not reach it there",
    ):
        assert reason in skill, reason


def test_the_skill_carries_the_builders_helper_word_for_word():
    """A hand-built page follows the skill, a built one the builder's prompt:
    one helper text, so a page fails a view the same way either way. The
    skill's build steps give each view a script of its own, and its tab rule
    starts every view by its title."""
    skill = _skill()
    flat = " ".join(skill.split())
    assert prompts.VIEW_HELPER_JS in skill
    for rule in (
        "Put the helper alone in the page's first inline `<script>`",
        "Write each view's setup as its own string, one `defineView(...)` call per view",
        "for js in [js_helper, data_js, js_shared, *view_js, js_start]",
        "Never join two views' code into one `<script>`",
        "the `runView` calls for the views shown on load go inside `addEventListener('DOMContentLoaded', ...)`, and the tab functions stay outside it",
        "A `<script type=\"module\">` runs after every classic script, so a view it registers exists only by then",
        "Keep the functions the page's HTML calls, such as a tab's `showPage`, outside that listener",
        "Give each view a title no other view on the page uses",
        "Declare nothing else at a view script's top level: every function and constant the view uses goes inside its setup",
        "`initChartsFor(name)` starts each view on that tab by its title with `runView(title, el)` (item 10)",
    ):
        assert rule in flat, rule
    assert "{data_js}{js_charts}" not in skill
    assert "catch (e) { console.error(e); el.textContent" not in skill


# Runs each script the way a browser runs a page's `<script>` elements: each
# is parsed on its own in one shared global scope, and one that does not parse
# never runs. `node:vm` compiles each before it runs any line of it, as the
# browser does. A timer lets a rejected promise settle before the report.
_HELPER_HARNESS = r"""
const vm = require('node:vm');
const errors = [], reported = [], unparsed = [];
const element = (id) => ({id, attrs: {}, textContent: 'chart',
  setAttribute(k, v) { this.attrs[k] = v; }});
const elements = {map: element('world-map'), globe: element('globe'), chart: element('trend-chart'),
  atlas: element('atlas'), odd: element('odd'), sales: element('sales-trend'), ops: element('ops-trend'),
  lanes: element('lanes'), ports: element('ports')};
const page = vm.createContext({
  elements,
  console: {error: (...args) => errors.push(args)},
  reportError: (err) => reported.push(String(err && err.message)),
});
const scripts = [
  process.argv[1],
  "defineView('Shipment map', () => { undefinedThing.draw(); });",
  "defineView('Globe', (el) => { el.textContent = 'drawn'; ) });",
  "defineView('Trend chart', (el) => { el.textContent = 'drawn'; });",
  "defineView('Atlas', async () => { await null; throw new Error('atlas fetch failed'); });",
  "defineView('Missing panel', (el) => { el.textContent = 'x'; });",
  "defineView('Trend', (el) => { el.textContent = 'sales trend'; });",
  "defineView('Trend', (el) => { el.textContent = 'ops trend'; });",
  "defineView('Lanes', (el) => { function render(e) { e.textContent = 'lanes drawn'; } render(el); });",
  "defineView('Ports', (el) => { function render(e) { e.textContent = 'ports drawn'; } render(el); });",
  "runView('Shipment map', elements.map); runView('Globe', elements.globe);" +
  " runView('Trend chart', elements.chart); runView('Atlas', elements.atlas);" +
  " runView('Missing panel', null); runView('constructor', elements.odd);" +
  " runView('Trend', elements.sales); runView('Trend', elements.ops);" +
  " runView('Lanes', elements.lanes); runView('Ports', elements.ports); after = 'script went on';",
];
for (const source of scripts) {
  try { vm.runInContext(source, page); } catch (err) { unparsed.push(err.name); }
}
const state = (el) => [el.attrs['data-view-failed'] || null, el.textContent];
setTimeout(() => process.stdout.write(JSON.stringify({
  errors, reported, unparsed, after: page.after,
  map: state(elements.map), globe: state(elements.globe), chart: state(elements.chart),
  atlas: state(elements.atlas), odd: state(elements.odd), sales: state(elements.sales), ops: state(elements.ops),
  lanes: state(elements.lanes), ports: state(elements.ports),
})), 20);
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_helper_keeps_a_failed_view_to_itself():
    """A view whose script does not parse, a throw, a rejected promise and a
    missing element each fail only their own view: the starting script goes
    on, the healthy view draws, and each failure is marked, worded and
    reported with the view's name first. The view whose script never ran has
    no setup under its name, so the helper fails it by name instead of a
    `ReferenceError` stopping the views started after it."""
    out = json.loads(subprocess.run(
        ["node", "-e", _HELPER_HARNESS, prompts.VIEW_HELPER_JS],
        text=True, capture_output=True, check=True,
    ).stdout)
    assert out["unparsed"] == ["SyntaxError"]
    assert out["after"] == "script went on"
    assert out["chart"] == [None, "drawn"]
    assert out["map"] == ["Shipment map", "Shipment map could not load."]
    assert out["globe"] == ["Globe", "Globe could not load."]
    assert out["atlas"] == ["Atlas", "Atlas could not load."]
    # A title that names an `Object` property finds no setup it never registered.
    assert out["odd"] == ["constructor", "constructor could not load."]
    assert [args[0].split(":")[0] for args in out["errors"]] == [
        "Shipment map failed", "Globe failed", "Missing panel failed", "constructor failed",
        "Trend failed", "Trend failed", "Atlas failed",
    ]
    # Two scripts registered "Trend": both views fail by name, neither draws the other's chart.
    assert out["sales"] == ["Trend", "Trend could not load."]
    assert out["ops"] == ["Trend", "Trend could not load."]
    assert out["errors"][4] == ["Trend failed: two scripts defined this view"]
    # Each setup keeps its own `render` inside it, so neither view draws the other's chart.
    assert out["lanes"] == [None, "lanes drawn"]
    assert out["ports"] == [None, "ports drawn"]
    assert out["errors"][1] == ["Globe failed: no script defined this view"]
    assert all(len(args) == 1 for args in out["errors"])
    assert out["reported"][0].startswith("Shipment map: ")
    assert out["reported"][1] == "Globe: no script defined this view"
    assert out["reported"][6] == "Atlas: atlas fetch failed"
