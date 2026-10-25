"""Compile a small presentation program into an ordinary self-contained HTML artifact."""
import re
from pathlib import Path

HERE = Path(__file__).parent
BEGIN = "/* cowork-presentation-begin */"
END = "/* cowork-presentation-end */"
FALLBACK = "USE_HTML_WRITER"

SYSTEM = """Build the user's complete static artifact using the provided presentation runtime.
Return only JavaScript, without HTML, markdown or commentary. The host supplies all CSS, library
loading, sizing, phone reflow and the exact source JSON at window.ARTIFACT_DATA. Do not reproduce
source arrays or invent figures. The source contract below shows actual paths and runtime values.
If the request needs an interface, library or feature this API cannot express faithfully, return
exactly USE_HTML_WRITER. The ordinary HTML writer will handle it. Never omit a requirement to fit.

Use `await CoworkPage(config)`. API:
config = {title, lang:'en' (the user's language code), subtitle?, emptyText?, loadingText?, errorText?, controls?:[
  {id, label, type:'tabs' or 'select', value, options:[{value,label}]}],
  render: selection => ({lead?,metrics?,sections})}.
Without controls, omit render and put lead, metrics, sections directly on config.
lead = {text: exact scoped finding, value?:formatted value, detail?:context, tone?:'warning'|'danger'|'good'}.
metrics = [{label,value,unit?,detail?,tone?}]. Use up to four supporting metrics; put additional
requested measures in readable details. All source values and units must remain correct.
CoworkNumber(value,unit?,languageCode?) formats exact numeric values; currency units are ISO
codes only when supplied by the source; '%' expects a percentage value such as 23.5. Null is unknown.
sections are any of:
- {type:'chart',title,caption?,option: standard Apache ECharts option object}. The host initializes
  ECharts and resizes it. Supply real series and axis names/units; never mix money, counts and
  percentages on one axis. Prefer horizontal bars for long labels. Tooltips must show exact values.
  Graph/sankey/scatter/line/bar/custom series work. Use proper category/value axes and stable series.
- {type:'table',title,caption?,columns:[{key,label,unit?,format?:(value,row)=>string}],rows:[objects]}.
  Keys are literal keys (including dotted field names). All columns remain visible on phones.
- {type:'map'|'globe',title,caption?,points:[{name,lat,lon,label?,color?}],
  routes?:[{from:[lon,lat],to:[lon,lat],label?,color?}],center?:[lon,lat],library?:'d3'|'echarts'}.
  Map defaults to real D3 world geography with zoom/pan. Globe is actual rotatable globe.gl with
  country polygons and routes; it has no external textures. ECharts is an alternative for maps.
  Use only observed or documented/geocoded coordinates. Report unresolved places in text, not fake positions.
  A point's label is short VISIBLE text. Include requested delay/status labels here as well as exact
  details in a table. Coordinates must describe the requested location role: a destination or an
  organization's headquarters is not the current position of an item moving toward it.
- {type:'timeline',title,caption?,items:[{time,label,detail?}]} for actual event milestones.
- {type:'text',title,text} for concise narrative or an explicit source limitation.

Use normal JavaScript to select, join, group, sort and transform the complete bound data into
these views. A filter's render callback must update the requested views consistently. For edits,
preserve existing correct content and interactions. No DOM manipulation, CSS, HTML, imports,
script tags, fetch, storage, backend, eval, new Function or extra library setup.

The page must answer the user's actual question. Lead with the relevant risk or comparison,
not a quiet default scenario. Quantify all requested measures at their requested scope and show
exact values in labels, a table or details as well as tooltips. A comparison needs a real visual
comparison or table, not a set of decorative number cards. Rank the relevant alternatives only
when the available data supports it; distinguish avoided exposure, revenue, cost and profit.
Keep visible text in the user's language. Do not invent currency or translate source identifiers.
"""

GEO_SYSTEM = """\nThe host also provides an exact GeoNames place index, listed in the user context.
For a map point whose requested position is a named place, pass {name,location:the actual place
string,country?:source country code or name,label}. The host resolves that observed place. Prefer
this to substituting an unrelated destination, headquarters, or default coordinate. Coordinates
supplied for the correct entity/position remain preferred when the source actually has them.
CoworkPosition(location,country?) returns [longitude,latitude] for a uniquely resolved observed
place and can be used for route endpoints. Ambiguous or unmatched places throw; resolve their
source country, report them explicitly, or use the HTML writer if the request needs more.
GeoNames attribution is inserted when this lookup is used. Never invent a position.
"""


def existing_program(html):
    match = re.search(re.escape(BEGIN) + r"(.*?)" + re.escape(END), html, re.S)
    return match.group(1).strip() if match else None


def document(program, *, geocoding=None, chart_tables=False, timeline_axis=False, render_contract=False, geo_controls=False):
    program = program.strip()
    fenced = re.fullmatch(r"```(?:javascript|js)?\s*\n(.*?)\n```", program, re.S | re.I)
    if fenced:
        program = fenced.group(1).strip()
    if not re.search(r"\bCoworkPage\s*\(", program) or re.search(r"</script\s*>", program, re.I):
        raise ValueError("Return a complete JavaScript program calling await CoworkPage(config), without HTML or script tags.")
    css = (HERE / "presentation.css").read_text()
    runtime = (HERE / "presentation.js").read_text()
    if chart_tables:
        runtime += "\n" + (HERE / "presentation_data.js").read_text()
    if timeline_axis:
        runtime += "\n" + (HERE / "presentation_timeline.js").read_text()
    if render_contract:
        runtime += "\n" + (HERE / "presentation_contract.js").read_text()
    if geo_controls:
        runtime += "\n" + (HERE / "presentation_geography.js").read_text()
    import json
    place_data = json.dumps(geocoding or {}, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
            "<title>Artifact</title><style>" + css + "</style></head><body>"
            '<main id="cowork-page"></main><script>window.ARTIFACT_GEOCODING=' + place_data + ";" + runtime + "</script><script>"
            "(async()=>{\n" + BEGIN + "\n" + program + "\n" + END + "\n})().catch(error=>{"
            "const box=document.createElement('p');box.className='failure';box.setAttribute('role','alert');box.textContent=error.message;"
            "(document.getElementById('page-content')||document.getElementById('cowork-page')).replaceChildren(box);console.error(error);throw error;});"
            "</script></body></html>")
