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
  Use coordinates from the data when it has them. For a well-known named place (a city, port or
  country) without coordinates, use its approximate coordinates and say on the page that those
  positions are approximate. Never invent a position for an unknown place, code or address; list
  those places in text instead.
  A point's label is short VISIBLE text.
- {type:'timeline',title,caption?,items:[{time,label,detail?}]} for actual event milestones.
- {type:'text',title,text} for concise narrative or an explicit source limitation.

Use normal JavaScript to select, join, group, sort and transform the complete bound data into
these views. A filter's render callback must update the requested views consistently. For edits,
preserve existing correct content and interactions. No DOM manipulation, CSS, HTML, imports,
script tags, fetch, storage, backend, eval, new Function or extra library setup.

The page must answer the user's actual question. Quantify all requested measures at their
requested scope and show exact values in labels, a table or details as well as tooltips.
Keep visible text in the user's language. Do not invent currency or translate source identifiers.
"""

def _compares_selection(program):
    """render(selection) used as a string: every control would show the same view."""
    head = r"render\s*[:=]\s*(?:function\s*)?\(?\s*([A-Za-z_$][\w$]*)\s*\)?\s*(?:=>|\{)"
    for name in {match.group(1) for match in re.finditer(head, program)}:
        arg = re.escape(name)
        heads = len(re.findall(head.replace("([A-Za-z_$][\\w$]*)", arg), program))
        # A name bound again outside render may be compared in another function.
        binds = re.findall(rf"\b{arg}\s*\)?\s*=>|function\s*[\w$]*\s*\(\s*{arg}\b|\b(?:const|let|var)\s+{arg}\b", program)
        if len(binds) <= heads and re.search(
                rf"(?<!typeof )\b{arg}\s*[!=]==?\s*['\"`]|['\"`]\s*[!=]==?\s*{arg}\b|switch\s*\(\s*{arg}\s*\)", program):
            return True
    return False


def existing_program(html):
    match = re.search(re.escape(BEGIN) + r"(.*?)" + re.escape(END), html, re.S)
    return match.group(1).strip() if match else None


def document(program):
    program = program.strip()
    fenced = re.fullmatch(r"```(?:javascript|js)?\s*\n(.*?)\n```", program, re.S | re.I)
    if fenced:
        program = fenced.group(1).strip()
    if not re.search(r"\bCoworkPage\s*\(", program) or re.search(r"</script\s*>", program, re.I):
        raise ValueError("Return a complete JavaScript program calling await CoworkPage(config), without HTML or script tags.")
    css = (HERE / "presentation.css").read_text()
    if _compares_selection(program):
        raise ValueError(
            "render receives an object keyed by control ids, never a string. Read the control's value by "
            "its id, for example render: ({view}) => view === 'orders' ? ... or selection.view === 'orders'."
        )
    runtime = "\n".join((HERE / name).read_text() for name in ("presentation_contract.js", "presentation.js", "presentation_data.js"))
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
            "<title>Artifact</title><style>" + css + "</style></head><body>"
            '<main id="cowork-page"></main><script>' + runtime + "</script><script>"
            "(async()=>{\n" + BEGIN + "\n" + program + "\n" + END + "\n})().catch(error=>{"
            "const box=document.createElement('p');box.className='failure';box.setAttribute('role','alert');box.textContent=error.message;"
            "(document.getElementById('page-content')||document.getElementById('cowork-page')).replaceChildren(box);console.error(error);throw error;});"
            "</script></body></html>")
