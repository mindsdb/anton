"""A published page that downloads the map recipe's World Atlas file carries it, so its map draws
where the published link's viewer blocks the download, as MindsHub's does with connect-src 'self'."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from anton.core.artifacts import world_atlas
from anton.publisher import _zip_fullstack, _zip_html

COUNTRIES = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-110m.json"
LAND = "https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/land-110m.json"
# The recipe's page, as generate_artifact's prompts write it.
RECIPE_PAGE = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script></head><body><svg id="map"></svg>'
    f"<script>const WORLD_ATLAS = '{COUNTRIES}';"
    "d3.json(WORLD_ATLAS).then((world) => draw(topojson.feature(world, world.objects.countries)));</script></body></html>"
)


def carried(html: str) -> dict[str, object]:
    """The files the page carries, by address, from the script element that holds them."""
    start = html.index(f'<script type="application/json" id="{world_atlas.MARKER}">')
    data = html[html.index(">", start) + 1 : html.index("</script>", start)]
    return json.loads(data)


def package_file(name: str) -> object:
    return json.loads((Path(world_atlas.__file__).parent / name).read_text(encoding="utf-8"))


def test_a_page_that_downloads_the_atlas_carries_it_before_its_own_scripts():
    page = world_atlas.embed(RECIPE_PAGE)

    assert carried(page) == {COUNTRIES: package_file("countries-110m.json")}
    head = page.index("<head>") + len("<head>")
    assert page[head:].startswith("<!-- World Atlas 2.0.2")
    assert page.index(f'<script type="application/json" id="{world_atlas.MARKER}">') > head
    answer = page.index("window.fetch=")
    assert answer < page.index("d3.min.js") < page.index("d3.json(WORLD_ATLAS)")


def test_both_atlas_files_a_page_names_are_carried():
    page = world_atlas.embed(f"<html><body><script>Promise.all([d3.json('{COUNTRIES}'), d3.json('{LAND}')])</script></body></html>")

    assert carried(page) == {COUNTRIES: package_file("countries-110m.json"), LAND: package_file("land-110m.json")}
    assert page.index("window.fetch=") < page.index("Promise.all")


@pytest.mark.parametrize(
    "html",
    [
        "<html><body><script>echarts.init(el)</script></body></html>",
        # Another resolution, or a floating version, isn't one the package carries.
        "<script>d3.json('https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-50m.json')</script>",
        "<script>d3.json('https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json')</script>",
    ],
)
def test_a_page_that_names_no_carried_file_stays_as_it_is(html):
    assert world_atlas.embed(html) == html


def test_a_page_that_already_carries_the_atlas_stays_as_it_is():
    once = world_atlas.embed(RECIPE_PAGE)

    assert world_atlas.embed(once) == once


@pytest.mark.parametrize(
    "html, before",
    [
        (f"<html><body><script>d3.json('{COUNTRIES}')</script></body></html>", "<body>"),
        (f"<!DOCTYPE html><script>d3.json('{COUNTRIES}')</script>", "<script>d3.json"),
        (f"<script>d3.json('{COUNTRIES}')</script>", "<script>d3.json"),
    ],
)
def test_a_page_without_a_head_gets_the_atlas_before_its_first_script(html, before):
    page = world_atlas.embed(html)

    assert page.index(world_atlas.MARKER) < page.index("window.fetch=") < page.index(before)


def test_no_script_end_tag_inside_the_carried_data():
    page = world_atlas.embed(RECIPE_PAGE)
    start = page.index(f'id="{world_atlas.MARKER}">')

    assert page.index("</script>", start) > start + 100_000


def test_a_page_that_carries_the_atlas_carries_its_license():
    """The ISC license asks for its notice in all copies, and each carrying page is one."""
    page = world_atlas.embed(RECIPE_PAGE)
    license_text = (Path(world_atlas.__file__).parent / "LICENSE").read_text(encoding="utf-8")
    notice = page[page.index("<!-- World Atlas 2.0.2") : page.index("-->") + len("-->")]

    assert license_text in notice
    assert page.index("-->") < page.index(world_atlas.MARKER)


def test_a_published_single_page_carries_the_atlas(tmp_path):
    (tmp_path / "index.html").write_text(RECIPE_PAGE, encoding="utf-8")

    with zipfile.ZipFile(io.BytesIO(_zip_html(tmp_path / "index.html"))) as bundle:
        published = bundle.read("index.html").decode("utf-8")

    assert carried(published) == {COUNTRIES: package_file("countries-110m.json")}


def test_a_published_page_is_the_same_bytes_each_time(tmp_path):
    """The publisher compares a bundle's md5 with the one it published, so the embed adds nothing
    that changes between builds."""
    (tmp_path / "index.html").write_text(RECIPE_PAGE, encoding="utf-8")

    assert _zip_html(tmp_path / "index.html") == _zip_html(tmp_path / "index.html")


def test_a_published_fullstack_page_carries_the_atlas(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text(RECIPE_PAGE, encoding="utf-8")
    (tmp_path / "metadata.json").write_text('{"type": "fullstack"}', encoding="utf-8")

    zipped, _included = _zip_fullstack(tmp_path)
    with zipfile.ZipFile(io.BytesIO(zipped)) as bundle:
        pages = [name for name in bundle.namelist() if name.endswith("index.html")]
        assert pages
        assert all(world_atlas.MARKER in bundle.read(name).decode("utf-8") for name in pages)
