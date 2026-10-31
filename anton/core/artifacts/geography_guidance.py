"""Shared geography requirements for artifact planning, generation and repair."""

GEOGRAPHY_GUIDANCE = """\
- MAPS AND GLOBES (Earth geography only): when the request includes a geographic
  map or Earth globe, render real, recognizable coastlines and country geometry.
  Never invent or hand-transcribe coastlines, continent/country outlines, SVG
  paths or coordinate arrays as geography. Do not substitute a blank sphere,
  decorative blobs, or a schematic for requested Earth geography. SVG itself is
  allowed: draw paths from authentic geographic data with a projection.
  Authentic user-supplied GeoJSON or other supplied geographic data is allowed;
  preserve and use it rather than replacing it with invented coordinates.
  These rules do not turn a non-geographic map (a mind map, network or floor
  plan) or an explicitly requested schematic into an Earth map.
  If authentic geographic data is not already supplied, use the tested browser
  recipe below. Load the atlas in the page's JavaScript, not in Python or a
  scratchpad: an atlas download is not a prerequisite data-gathering step.
  ECharts ships no world outline. Pin the exact library versions:
  `<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>`
  `<script src="https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/dist/topojson-client.min.js"></script>`
  `const WORLD_ATLAS = 'https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/countries-110m.json';`
  `async function loadCountries() { const world = await d3.json(WORLD_ATLAS); return topojson.feature(world, world.objects.countries); }`
  Load it with `d3.json`, never an absolute URL in `fetch()`. Await
  `loadCountries()` inside the view's async setup to obtain `countries`, then
  draw it there. Keep both declarations inside that setup. Await or return
  every promise the setup starts so a failed
  atlas load reaches the view's error handling. Draw `countries` in one of
  three ways:
  - d3 into an `<svg>` (`d3.geoNaturalEarth1().fitSize([w, h], countries)`,
    `d3.geoPath`);
  - ECharts 5.5.1
    (`<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>`):
    `echarts.registerMap('world', countries)`, then a `map` series with
    `map: 'world'`;
  - a globe: globe.gl@2.34.2
    (`<script src="https://cdn.jsdelivr.net/npm/globe.gl@2.34.2/dist/globe.gl.min.js"></script>`)
    with `.polygonsData(countries.features)`. It is heavy, so the browser
    check may run out of time on it and leave the page unverified.
  Leaflet, raster map tiles, MapLibre (it needs a web worker) and echarts-gl
  (it needs `eval`) break the web app preview policy.
  When repairing a map or globe, preserve its real geography, requested views
  and interactions; fix the loading/rendering failure instead of deleting the
  geography or replacing it with hand-drawn shapes or a blank sphere. This
  guidance applies during repairs without recalling a skill.
  Keep these library names, CDN URLs and loading instructions out of the
  user-facing brief; describe the map's behavior there in plain language.\
"""
