"""Browser behavior of the geography add-on, without an external map service."""

import runpy
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

JS = Path(__file__).resolve().parents[1] / "anton/core/tools/generate_artifact/presentation_geography.js"
COMPONENT = JS.with_name("component_page.py")


def test_regional_d3_projection_spreads_nearby_points_across_the_map():
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content("<main></main>")
            page.add_script_tag(path=str(JS.with_name("presentation.js")))
            positions = page.evaluate("""() => {
              window.d3 = {geoNaturalEarth1() {
                let scale = 1, center = [0, 0], translate = [300, 150];
                const projection = pair => [translate[0] + (pair[0]-center[0])*scale,
                  translate[1] - (pair[1]-center[1])*scale];
                projection.scale = value => value === undefined ? scale : (scale = value, projection);
                projection.center = value => value === undefined ? center : (center = value, projection);
                projection.translate = value => value === undefined ? translate : (translate = value, projection);
                projection.fitExtent = (extent, feature) => {
                  const coords = feature.type === 'MultiPoint' ? feature.coordinates : [[-180,-90],[180,90]];
                  const xs = coords.map(pair => pair[0]), ys = coords.map(pair => pair[1]);
                  center = [(Math.min(...xs)+Math.max(...xs))/2,(Math.min(...ys)+Math.max(...ys))/2];
                  scale = Math.min((extent[1][0]-extent[0][0])/Math.max(1,Math.max(...xs)-Math.min(...xs)),
                    (extent[1][1]-extent[0][1])/Math.max(1,Math.max(...ys)-Math.min(...ys)));
                  translate = [(extent[0][0]+extent[1][0])/2,(extent[0][1]+extent[1][1])/2];
                  return projection;
                };
                return projection;
              }};
              const land = {type:'FeatureCollection',features:[]};
              const section = {points:[{lon:-80,lat:43},{lon:-77.5,lat:44}]};
              const regional = CoworkMapProjection(section,land,600,300);
              const world = CoworkMapProjection({points:[]},land,600,300);
              return {regional:Math.abs(regional([-80,43])[0]-regional([-77.5,44])[0]),
                world:Math.abs(world([-80,43])[0]-world([-77.5,44])[0])};
            }""")
            assert positions["regional"] > 50
            assert positions["regional"] > positions["world"] * 5
        finally:
            browser.close()


@pytest.mark.parametrize("width", [390, 1440])
def test_geographic_labels_and_legend_at_phone_and_desktop_width(width):
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": width, "height": 844})
            page.set_content('<section id="geo"><svg width="320" height="240"><g id="labels"></g></svg></section>')
            page.add_script_tag(path=str(JS))
            result = page.evaluate("""() => {
              const svg = document.querySelector('svg'), element = document.getElementById('labels');
              const wrap = node => ({
                attr(name, value) { node.setAttribute(name, value); return this; },
                text(value) { node.textContent = value; return this; },
                node() { return node; }
              });
              const group = {
                append(tag) { const n = document.createElementNS('http://www.w3.org/2000/svg', tag); element.append(n); return wrap(n); },
                insert(tag, before) { const n = document.createElementNS('http://www.w3.org/2000/svg', tag); element.insertBefore(n, before()); return wrap(n); }
              };
              const points = [
                {lon:40,lat:40,label:'Northwest Ambulatory Center'},
                {lon:41,lat:41,label:'Central Emergency Department'},
                {lon:42,lat:42,label:'Southside Children’s Hospital'}
              ];
              CoworkMapLabels(group, points, pair => [120 + pair[0]-40, 80 + pair[1]-40], 320, 240);
              CoworkGeoLegend(document.getElementById('geo'), {legend:[{label:'Emergency',color:'#e11d48'}]});
              const labels = [...element.querySelectorAll('text')].map(n => ({
                text:n.textContent, x:+n.getAttribute('x'), y:+n.getAttribute('y'), w:n.getComputedTextLength()
              }));
              return {labels, legend: document.querySelector('.geo-legend')?.textContent,
                      swatch: getComputedStyle(document.querySelector('.legend-key')).backgroundColor};
            }""")
            assert result["legend"] == "Emergency"
            assert result["swatch"] == "rgb(225, 29, 72)"
            labels = result["labels"]
            assert len(labels) == 3
            assert all(8 <= label["x"] and label["x"] + label["w"] <= 312 for label in labels)
            assert all(not (a["x"] < b["x"] + b["w"] + 5 and b["x"] < a["x"] + a["w"] + 5
                                and abs(a["y"] - b["y"]) < 16)
                       for i, a in enumerate(labels) for b in labels[i + 1:])
        finally:
            browser.close()


@pytest.mark.parametrize("reduced", [False, True])
def test_route_motion_and_rotation_only_when_requested_and_allowed(reduced):
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(reduced_motion="reduce" if reduced else "no-preference")
            page.set_content('<div id="geo"><svg><path class="route"></path></svg></div>')
            page.add_script_tag(path=str(JS))
            result = page.evaluate("""() => {
              const node = document.getElementById('geo'), controls = {autoRotate:false};
              const globe = {controls:() => controls};
              CoworkGeoMotion(node, {}, globe);
              const defaultRotation = controls.autoRotate;
              CoworkGeoMotion(node, {routeMotion:true});
              const routeMoves = node.querySelector('path').classList.contains('route-motion');
              CoworkGeoMotion(node, {autoRotate:true,rotationSpeed:.5}, globe);
              return {defaultRotation,routeMoves,rotation:controls.autoRotate};
            }""")
            assert result == {"defaultRotation": False, "routeMoves": not reduced,
                              "rotation": not reduced}
        finally:
            browser.close()


@pytest.mark.parametrize("reduced", [False, True])
def test_echarts_map_colors_labels_and_requested_motion(reduced):
    document = runpy.run_path(str(COMPONENT))["document"]
    html = document("""await CoworkPage({title:'Hospital referrals',sections:[{
      type:'map',title:'Referrals',library:'echarts',routeMotion:true,
      points:[{name:'North ward',label:'North ward',lat:40,lon:20,color:'#e11d48'}],
      routes:[{from:[20,40],to:[21,41],color:'#facc15'}]
    }]});""")
    shim = """<script>
      window.d3={json:async()=>({objects:{countries:{}}})};
      window.topojson={feature:()=>({type:'FeatureCollection',features:[]})};
      window.echarts={registerMap:()=>{},getMap:()=>({}),init:node=>{
        node.append(document.createElement('canvas'));
        return {setOption:option=>window.__mapOption=option,resize:()=>{},dispose:()=>{}};
      }};
      const append=document.head.append.bind(document.head);
      document.head.append=(...nodes)=>{for(const node of nodes){
        if(node.tagName==='SCRIPT'&&node.src.startsWith('https://cdn.jsdelivr.net/'))
          setTimeout(()=>node.onload(),0);
        else append(node);
      }};
    </script>"""
    html = html.replace("<title>Artifact</title>", "<title>Artifact</title>" + shim)
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(reduced_motion="reduce" if reduced else "no-preference")
            page.set_content(html)
            page.wait_for_function("window.__mapOption !== undefined")
            option = page.evaluate("window.__mapOption")
            scatter, lines = option["series"]
            assert scatter["labelLayout"] == {"hideOverlap": True}
            assert scatter["data"][0]["itemStyle"]["color"] == "#e11d48"
            assert lines["data"][0]["lineStyle"]["color"] == "#facc15"
            assert lines["effect"]["show"] is not reduced
        finally:
            browser.close()


def test_points_and_routes_named_by_place_are_placed_and_unknown_ones_listed():
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content('<html lang="es"><body><section id="geo"></section></body></html>')
            page.add_script_tag(path=str(JS))
            result = page.evaluate("""() => {
              window.ARTIFACT_PLACES = {'Lyon': {lat: 45.75, lon: 4.85}, 'Porto': {lat: 41.15, lon: -8.61}};
              const placed = CoworkPlaces({points: [{name: 'a', place: 'Lyon'}, {name: 'b', place: 'Nowhere'},
                                                    {name: 'c', lat: 1, lon: 2}],
                                           routes: [{from: 'Lyon', to: 'Porto'}, {from: 'Lyon', to: 'Nowhere'},
                                                    {from: [0, 0], to: 'Porto'}]});
              CoworkPlaceNotes(document.getElementById('geo'), placed, {});
              return {points: placed.section.points.map(p => [p.name, p.lon, p.lat]),
                      routes: placed.section.routes.map(r => [r.from, r.to]),
                      notes: [...document.querySelectorAll('#geo p')].map(p => p.textContent),
                      plain: CoworkPlaces({points: [{lat: 1, lon: 2}]}).looked};
            }""")
            assert result["points"] == [["a", 4.85, 45.75], ["c", 2, 1]]
            assert result["routes"] == [[[4.85, 45.75], [-8.61, 41.15]], [[0, 0], [-8.61, 41.15]]]
            assert result["notes"][0].startswith("Fuera del mapa") and result["notes"][0].endswith(": Nowhere")
            assert result["notes"][1] == "GeoNames (CC BY 4.0)"
            assert result["plain"] is False  # coordinates alone: no note, no credit
        finally:
            browser.close()
