"""What the generator prompts tell the model about the app preview."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from anton.core.tools.generate_artifact import prompts


def test_the_generator_prompts_pin_the_versions_the_check_carries():
    """A floating `@4`/`@5` gets whatever jsDelivr serves today, which the
    check's mirror does not hold, so the page could only be unverified. The
    prompts name each package the check carries at its exact version."""
    for prompt in (
        prompts.build_subagent_system_prompt(Path("/tmp/a")),
        prompts.build_frontend_system_prompt(Path("/tmp/a")),
    ):
        for package in prompts.CHECKED_LIBRARIES:
            assert re.fullmatch(r"@?[\w./-]+@\d+\.\d+\.\d+", package), package
            assert f"`{package}`" in prompt, package


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
    handler, where an uncaught throw stops that tab's other views."""
    for prompt in (
        prompts.build_subagent_system_prompt(Path("/tmp/a")),
        prompts.build_frontend_system_prompt(Path("/tmp/a")),
    ):
        tab_rule = prompt.index("drawn when it\n  is first shown")
        helper = prompt.index("function runView(name, el, setup)")
        # The tab rule sits in the design rules, after the contract that
        # defines the helper it names.
        assert helper < tab_rule
        assert "and starts through `runView` too" in prompt


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


def _skill() -> str:
    path = Path(__file__).resolve().parents[1] / "anton/core/memory/builtin_skills/build-html-dashboard/SKILL.md"
    return path.read_text(encoding="utf-8")


def test_the_generator_prompts_point_maps_at_published_geometry():
    """ECharts ships no world outline and a hand-typed one is wrong, so a map
    draws published geometry the check carries. No recipe is prescribed."""
    for prompt in _GENERATOR_PROMPTS:
        flat = " ".join(prompt.split())
        assert "never from coordinates typed by hand" in flat
        assert "`world-atlas@2.0.2`" in flat and "`topojson-client@3.1.0`" in flat


def test_the_generator_prompts_state_the_preview_policy():
    for prompt in _GENERATOR_PROMPTS:
        flat = " ".join(prompt.split())
        assert "no `eval` or `new Function`, and no web workers" in flat
        assert "no external stylesheet or web font, no external image or map tile" in flat
        assert "Leaflet, raster map tiles, MapLibre" in flat
        assert "so the page renders on every deployment" in flat


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
    skill's build step and tab rule start every view through it."""
    skill = _skill()
    assert prompts.VIEW_HELPER_JS in skill
    assert "Define the `runView` helper from item 10 first, and start every view through it" in skill
    assert "`initChartsFor(name)` starts each view on that tab through `runView` (item 10)" in skill
    assert "catch (e) { console.error(e); el.textContent" not in skill


# Runs the helper against stub elements. `queueMicrotask` and a timer let a
# rejected promise settle before the report.
_HELPER_HARNESS = r"""
const errors = [], reported = [];
const element = (id) => ({id, attrs: {}, textContent: 'chart',
  setAttribute(k, v) { this.attrs[k] = v; }});
global.console = {error: (...args) => errors.push(args)};
global.reportError = (err) => reported.push(err instanceof Error ? err.message : String(err));
eval(process.argv[1] + `
const map = element('world-map'), globe = element('globe'), chart = element('trend-chart');
runView('Shipment map', map, () => { undefinedThing.draw(); });
runView('Globe', globe, async () => { await null; throw new Error('atlas fetch failed'); });
runView('Trend chart', chart, (el) => { el.textContent = 'drawn'; });
runView('Missing panel', null, (el) => { el.textContent = 'x'; });
let after = 'script went on';
setTimeout(() => process.stdout.write(JSON.stringify({
  errors, reported, after,
  map: [map.attrs['data-view-failed'] || null, map.textContent],
  globe: [globe.attrs['data-view-failed'] || null, globe.textContent],
  chart: [chart.attrs['data-view-failed'] || null, chart.textContent],
})), 20);
`);
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_helper_keeps_a_failed_view_to_itself():
    """A throw, a rejected promise and a missing element each fail only
    their own view: the script goes on, the healthy view draws, and each
    failure is marked, worded and reported with the view's name first."""
    out = json.loads(subprocess.run(
        ["node", "-e", _HELPER_HARNESS, prompts.VIEW_HELPER_JS],
        text=True, capture_output=True, check=True,
    ).stdout)
    assert out["after"] == "script went on"
    assert out["chart"] == [None, "drawn"]
    assert out["map"] == ["Shipment map", "Shipment map could not load."]
    assert out["globe"] == ["Globe", "Globe could not load."]
    assert [args[0].split(":")[0] for args in out["errors"]] == ["Shipment map failed", "Missing panel failed", "Globe failed"]
    assert all(len(args) == 1 for args in out["errors"])
    assert out["reported"][0].startswith("Shipment map: ")
    assert out["reported"][2] == "Globe: atlas fetch failed"
