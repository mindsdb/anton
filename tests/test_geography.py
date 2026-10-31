"""Static geography evidence, separate from browser loading or geometry accuracy."""

import pytest

from anton.core.artifacts.geography import GeographyRequirement, check_geography, classify_request
from anton.core.tools.generate_artifact import component_page


@pytest.mark.parametrize("user_request,kind,required", [
    ("Build a world map of sales by country", "map", True),
    ("Show a map of Europe", "map", True),
    ("Plot cities on a map using latitude and longitude", "map", True),
    ("Make a map of France", "map", True),
    ("Show an interactive Earth globe", "globe", True),
    ("A world map and a rotating globe", "map_and_globe", True),
    ("Add a map view", "map", False),
])
def test_classifies_only_confident_earth_requests_as_required(user_request, kind, required):
    assert classify_request(user_request) == GeographyRequirement(kind, required)


@pytest.mark.parametrize("user_request", [
    "Create a roadmap", "A mind map of the organization", "Create a heatmap",
    "Explain records.map(fn)", "Draw a network schematic", "Draw a floor plan",
    "A fictional world map", "A schematic map of Earth", "Map our business process",
    "Show country sales in a table", "Do not include a map", "Make a snow globe",
])
def test_non_geographic_or_explicitly_excluded_requests_have_no_requirement(user_request):
    assert classify_request(user_request) is None


MAP = GeographyRequirement("map")
GLOBE = GeographyRequirement("globe")
BOTH = GeographyRequirement("map_and_globe")
ATLAS = """
const world = await d3.json('https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-110m.json');
const countries = topojson.feature(world, world.objects.countries);
"""
D3 = """
const projection = d3.geoNaturalEarth1().fitSize([800, 400], countries);
const path = d3.geoPath(projection);
d3.select('svg').selectAll('path').data(countries.features).join('path').attr('d', path);
"""
ECHARTS = """
echarts.registerMap('world', countries);
const chart = echarts.init(document.getElementById('map'));
chart.setOption({series:[{type:'map', map:'world'}]});
"""
GLOBE_POLYGONS = "const globe = Globe()(document.getElementById('globe')).polygonsData(countries.features);"


def page(script):
    return f"<html><body><svg></svg><script>{script}</script></body></html>"


@pytest.mark.parametrize("script,requirement", [
    (ATLAS + D3, MAP), (ATLAS + ECHARTS, MAP), (ATLAS + GLOBE_POLYGONS, GLOBE),
    (ATLAS + D3 + GLOBE_POLYGONS, BOTH),
    ("Globe()(document.getElementById('globe')).globeImageUrl('https://example.test/earth.jpg');", GLOBE),
    (ATLAS + ECHARTS.replace("series:[{type:'map', map:'world'}]", "geo:{map:'world'}"), MAP),
    ("const countries = await d3.json('/supplied-boundaries.geojson');" + D3, MAP),
    ("const countries = {type:'FeatureCollection', features:[{type:'Feature', geometry:{type:'Polygon', coordinates: userCoordinates}}]};" + D3, MAP),
])
def test_supported_renderers_require_data_and_rendering_evidence(script, requirement):
    result = check_geography(page(script), requirement)
    assert result.errors == []
    assert result.warnings == []


@pytest.mark.parametrize("content,requirement", [
    ("<svg><path d='M 0 0 L 70 30 L 5 60 Z'/></svg>", MAP),
    ("<div>World map coming soon</div>", MAP),
    (page("Globe()(document.getElementById('globe'));"), GLOBE),
    (page("Globe()(node).globeImageUrl(null).polygonsData([]);"), GLOBE),
    (page("const sphere = new THREE.Mesh(new THREE.SphereGeometry(), material); scene.add(sphere);"), GLOBE),
    (page(ATLAS), MAP), (page(D3), MAP),
    (page("echarts.registerMap('world', countries);"), MAP),
    (page("chart.setOption({series:[{type:'map',map:'world'}]});"), MAP),
    (page(ATLAS + D3), BOTH),
    ("<script src='d3.js'></script><script src='globe.gl.js'></script>", BOTH),
    (page("/* " + ATLAS + D3 + " */"), MAP),
    (page("// " + D3.replace("\n", " ")), MAP),
    ("<!-- " + page(ATLAS + D3) + " -->", MAP),
    ("<pre>" + ATLAS + D3 + "</pre>", MAP),
    (page("const recipe = `" + ATLAS + D3 + "`; document.body.textContent = recipe;"), MAP),
    ("<script type='application/json'>" + ATLAS + D3 + "</script>", MAP),
])
def test_missing_geography_returns_actionable_error_without_banning_svg(content, requirement):
    result = check_geography(content, requirement)
    assert len(result.errors) == 1
    assert "browser" in result.errors[0].lower()
    assert "atlas" in result.errors[0].lower()
    assert "preserve" in result.errors[0].lower()
    assert "SVG is forbidden" not in result.errors[0]
    assert result.warnings == []


def test_out_of_scope_is_a_noop_and_ambiguous_request_is_advisory():
    assert check_geography("<div>No map</div>", None).errors == []
    result = check_geography("<div>No map</div>", GeographyRequirement("map", required=False))
    assert result.errors == []
    assert len(result.warnings) == 1


def test_custom_geographic_renderer_is_advisory_rather_than_rejected():
    result = check_geography(page("const viewer = new Cesium.Viewer('earth');"), GLOBE)
    assert result.errors == []
    assert "custom" in result.warnings[0].lower()


@pytest.mark.parametrize("kind", ["map", "globe"])
def test_component_runtime_accepts_the_requested_supported_section(kind):
    html = component_page.document(f"await CoworkPage({{title:'Places',sections:[{{type:'{kind}',points:[]}}]}});")
    result = check_geography(html, GeographyRequirement(kind))
    assert result.errors == []
    assert result.warnings == []


def test_component_runtime_library_bundle_does_not_satisfy_a_missing_map():
    html = component_page.document("await CoworkPage({title:'Places',sections:[{type:'text',text:'No geography'}]});")
    assert check_geography(html, MAP).errors


def test_component_chart_can_use_its_automatically_registered_world_geometry():
    html = component_page.document("await CoworkPage({title:'Places',sections:[{type:'chart',option:{geo:{map:'world'}}}]});")
    assert check_geography(html, MAP).errors == []


def test_json_property_keys_are_accepted_for_supplied_geojson_and_options():
    data = 'const countries = {"type":"FeatureCollection", "features":[{"type":"Feature", "geometry":{"type":"Polygon", "coordinates":suppliedCoordinates}}]};'
    assert check_geography(page(data + D3), MAP).errors == []
    chart = ECHARTS.replace("type:'map', map:'world'", '"type":"map", "map":"world"')
    assert check_geography(page(ATLAS + chart), MAP).errors == []


def test_echarts_must_render_the_map_it_registered():
    content = page(ATLAS + ECHARTS.replace("registerMap('world'", "registerMap('different'"))
    assert check_geography(content, MAP).errors


def test_decorative_svg_path_is_not_made_geographic_by_unused_d3_calls():
    content = page(ATLAS + D3.replace(".attr('d', path)", ".attr('d', 'M 0 0 L 70 30 Z')"))
    assert check_geography(content, MAP).errors


def test_unused_template_script_is_not_rendering_evidence():
    assert check_geography("<template>" + page(ATLAS + D3) + "</template>", MAP).errors


def test_d3_orthographic_globe_with_real_geometry_is_supported():
    html = page(ATLAS + D3.replace('geoNaturalEarth1', 'geoOrthographic'))
    assert check_geography(html, GLOBE).errors == []


def test_empty_inline_geojson_does_not_make_a_blank_globe_valid():
    html = page("const countries = {type:'FeatureCollection', features:[]};" + GLOBE_POLYGONS)
    assert check_geography(html, GLOBE).errors


def test_unknown_globe_texture_is_advisory_not_certified_as_earth():
    html = page("Globe()(node).globeImageUrl('/supplied-texture.png');")
    result = check_geography(html, GLOBE)
    assert result.errors == []
    assert result.warnings


def test_d3_orthographic_projection_can_satisfy_a_map_request():
    html = page(ATLAS + D3.replace('geoNaturalEarth1', 'geoOrthographic'))
    assert check_geography(html, MAP).errors == []


def test_canonical_observed_data_binding_preserves_supplied_geojson_support():
    from anton.core.tools.generate_artifact import observed_data

    geometry = {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[0, 0], [1, 0], [0, 1], [0, 0]]]}}
    ]}
    key = observed_data.identity(geometry)
    html = page(f'const countries = window.ARTIFACT_DATA["{key}"];' + D3)
    bound = observed_data.bind(html, {key: geometry})
    result = check_geography(bound, MAP)
    assert result.errors == []
    assert result.warnings == []


def test_json_recipe_strings_cannot_supply_data_or_rendering_evidence():
    from anton.core.tools.generate_artifact import observed_data

    recipe = {'recipe': ATLAS + D3}
    html = observed_data.bind(page(D3), {observed_data.identity(recipe): recipe})
    assert check_geography(html, MAP).errors


def test_unused_json_geometry_does_not_satisfy_a_map():
    html = '<script type="application/json" id="geo">{"type":"Polygon","coordinates":[[[0,0],[1,0],[0,1],[0,0]]]}</script>'
    assert check_geography(html + page(D3), MAP).errors


def test_echarts_variable_map_name_is_advisory_instead_of_a_false_refusal():
    script = ATLAS + "const mapName = 'world';" + ECHARTS.replace("registerMap('world'", "registerMap(mapName").replace("map:'world'", "map:mapName")
    result = check_geography(page(script), MAP)
    assert result.errors == []
    assert result.warnings


def test_d3_path_accessor_is_supported():
    html = page(ATLAS + D3.replace(".attr('d', path)", ".attr('d', (feature) => path(feature))"))
    assert check_geography(html, MAP).errors == []


def test_d3_serialized_json_source_is_advisory_without_parsing_code_strings():
    script = "const countries = JSON.parse(serializedGeoJSON);" + D3
    result = check_geography(page(script), MAP)
    assert result.errors == []
    assert result.warnings


@pytest.mark.parametrize('user_request', [
    'Do not show a world map; use a table of countries instead.',
    'Show all countries in a table instead of a map.',
    'Create a map of France as an explicitly requested schematic.',
])
def test_negated_maps_and_explicit_schematics_do_not_become_requirements(user_request):
    assert classify_request(user_request) is None


def test_excluded_flat_map_does_not_expand_an_earth_globe_requirement():
    assert classify_request('Show an Earth globe, not a flat world map.') == GLOBE


@pytest.mark.parametrize('program', [
    "const mapType='map'; await CoworkPage({sections:[{type:mapType, points:[]}]});",
    "const mapName='world'; await CoworkPage({sections:[{type:'chart', option:{geo:{map:mapName}}}]});",
])
def test_dynamic_component_geography_is_advisory(program):
    result = check_geography(component_page.document(program), MAP)
    assert result.errors == []
    assert result.warnings


def test_named_d3_import_with_rendering_chain_is_advisory():
    script = "import {json,geoNaturalEarth1,geoPath,select} from 'https://cdn.jsdelivr.net/npm/d3@7.9.0/+esm';" + ATLAS + D3
    script = script.replace('d3.', '')
    result = check_geography(page(script).replace('<script>', '<script type="module">'), MAP)
    assert result.errors == []
    assert result.warnings


def test_named_d3_import_alone_is_not_rendering_evidence():
    script = "import {json,geoNaturalEarth1,geoPath,select} from 'https://cdn.jsdelivr.net/npm/d3@7.9.0/+esm';"
    assert check_geography(page(script), MAP).errors


def test_embedded_globe_texture_remains_advisory():
    result = check_geography(page("Globe()(node).globeImageUrl('data:image/png;base64,AAAA');"), GLOBE)
    assert result.errors == []
    assert result.warnings


def test_non_geographic_json_with_object_type_field_is_safe_to_inspect():
    from anton.core.tools.generate_artifact import observed_data

    data = {'type': {'name': 'sales'}, 'rows': [1, 2]}
    html = observed_data.bind(page(D3), {observed_data.identity(data): data})
    assert check_geography(html, MAP).errors
