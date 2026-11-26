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
config = {title, lang:'en' (the user's language code), subtitle?, overview?:{lead?,metrics?}, emptyText?, loadingText?, errorText?, controls?:[
  {id, label, type:'tabs' or 'select', value, options:[{value,label}]}],
  render: selection => ({lead?,metrics?,sections})}.
Without controls, omit render and put lead, metrics, sections directly on config.
overview stays visible while controls change the sections. Use it for measures that remain true in every control state, including headline totals across requested areas when tabs split the detail views.
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
  Put shared unit or currency caveats in column headers or captions rather than repeating boilerplate in every cell.
- {type:'map'|'globe',title,caption?,points:[{name,lat,lon,label?,color?} or {name,place,label?,color?}],
  routes?:[{from:[lon,lat] or place,to:[lon,lat] or place,label?,color?}],center?:[lon,lat],library?:'d3'|'echarts'}.
  A place is a place name exactly as in the data ("Hong Kong", or a record's town and region joined
  as "Stratford, Ontario"); the runtime places names with one known match.
  Map defaults to real D3 world geography with zoom/pan. Globe is actual rotatable globe.gl with
  country polygons and routes; it has no external textures. ECharts is an alternative for maps.
  Use only observed coordinates or listed places; never type coordinates. Unresolved places are listed
  under the map by the runtime, not given fake positions.
  A point's label is short VISIBLE text.
- {type:'timeline',title,caption?,items:[{time,label,detail?}]} for actual event milestones.
- {type:'text',title,text} for concise narrative or an explicit source limitation.

The host reconciles shown figures with the data: a metric, lead value, table column, chart, map or
timeline that shows only 0 or nothing is returned for repair, and so is a sum that counts a source's
TOTAL ROWS beside their parts. When zero or empty is the true answer, attach zeroReason (a short
reason in the user's language, shown only when it applies) to the exact owner that is zero: on the
lead or metric object for a zero headline/metric; on each individual column object in `columns` for
any table column that is all 0 in any control state (such as zero delays, cancellations, overtime or
extra spend in a baseline scenario); and on the section object (or zero rows) for an empty table/timeline/map, an all-zero chart, or
any table row whose numeric columns are all 0 (such as unaffected entities or baseline scenarios).

Use normal JavaScript to select, join, group, sort and transform the complete bound data into
these views. A filter's render callback must update the requested views consistently. For edits,
preserve existing correct content and interactions. No DOM manipulation, CSS, HTML, imports,
script tags, fetch, storage, backend, eval, new Function or extra library setup.

The page must answer the user's actual question. Quantify all requested measures at their
requested scope and show exact values in labels, a table or details as well as tooltips.
Keep visible text in the user's language. Do not invent currency or translate source identifiers.
"""

_COWORK_SECTION_TYPES = frozenset({"chart", "table", "map", "globe", "timeline", "text"})
_REGEX_PREV_KEYWORDS = frozenset({
    "return", "case", "throw", "in", "of", "typeof", "void", "delete",
    "new", "yield", "await", "else", "do",
})
_REGEX_PREV_PUNCS = frozenset({
    "(", "[", "{", ",", ";", ":", "=", "!", "?", "&", "|", "^", "+", "-", "*", "%", "<", ">", "~",
})


def _can_precede_regex(prev: tuple[str, str] | None) -> bool:
    if prev is None:
        return True
    kind, val = prev
    if kind == "id" and val in _REGEX_PREV_KEYWORDS:
        return True
    return kind in ("punc", "op") and val in _REGEX_PREV_PUNCS


def _scan_js_range(code: str, start: int, stop_on_rbrace: bool) -> tuple[list[tuple[str, str]], int]:
    """Single-pass O(N) JS lexer handling comments, strings, RegExp literals, and nested `${...}` templates."""
    tokens: list[tuple[str, str]] = []
    n = len(code)
    i = start
    brace_depth = 0
    while i < n:
        ch = code[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "/" and i + 1 < n and code[i + 1] == "/":
            nl = code.find("\n", i + 2)
            i = n if nl < 0 else nl + 1
            continue
        if ch == "/" and i + 1 < n and code[i + 1] == "*":
            end = code.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        if ch == "/" and _can_precede_regex(tokens[-1] if tokens else None):
            j = i + 1
            in_class = False
            while j < n:
                c = code[j]
                if c == "\\" and j + 1 < n:
                    j += 2
                    continue
                if c == "[":
                    in_class = True
                elif c == "]":
                    in_class = False
                elif c == "/" and not in_class:
                    j += 1
                    break
                elif c in ("\n", "\r"):
                    break
                j += 1
            while j < n and code[j].isalpha():
                j += 1
            tokens.append(("regex", code[i:j]))
            i = j
            continue
        if ch in ("'", '"'):
            quote = ch
            j = i + 1
            buf: list[str] = []
            while j < n:
                c = code[j]
                if c == "\\" and j + 1 < n:
                    buf.append(code[j + 1])
                    j += 2
                    continue
                if c == quote:
                    j += 1
                    break
                buf.append(c)
                j += 1
            tokens.append(("str", "".join(buf)))
            i = j
            continue
        if ch == "`":
            j = i + 1
            while j < n:
                c = code[j]
                if c == "\\" and j + 1 < n:
                    j += 2
                    continue
                if c == "$" and j + 1 < n and code[j + 1] == "{":
                    _, j = _scan_js_range(code, j + 2, stop_on_rbrace=True)
                    continue
                if c == "`":
                    j += 1
                    break
                j += 1
            tokens.append(("tpl", ""))
            i = j
            continue
        if ch.isalpha() or ch in ("_", "$"):
            j = i + 1
            while j < n and (code[j].isalnum() or code[j] in ("_", "$")):
                j += 1
            tokens.append(("id", code[i:j]))
            i = j
            continue
        if ch.isdigit():
            j = i + 1
            while j < n and (code[j].isalnum() or code[j] in (".", "_")):
                j += 1
            tokens.append(("num", code[i:j]))
            i = j
            continue
        if ch == "{":
            brace_depth += 1
            tokens.append(("punc", "{"))
            i += 1
            continue
        if ch == "}":
            if stop_on_rbrace and brace_depth == 0:
                return tokens, i + 1
            if brace_depth > 0:
                brace_depth -= 1
            tokens.append(("punc", "}"))
            i += 1
            continue
        if ch in "[]():,.;":
            tokens.append(("punc", ch))
            i += 1
            continue
        if ch in "=!&|?+-*%<>^~/":
            tokens.append(("op", ch))
            i += 1
            continue
        i += 1
    return tokens, i


def _check_section_obj(tokens: list[tuple[str, str]], start_brace: int, invalid: list[str]) -> int:
    """Inspect only the direct (depth-1) `type: '<literal>'` property of one `{...}` section object."""
    n = len(tokens)
    depth = 1
    i = start_brace + 1
    while i < n and depth > 0:
        kind, val = tokens[i]
        if kind == "punc":
            if val in ("{", "[", "("):
                depth += 1
            elif val in ("}", "]", ")"):
                depth -= 1
                if depth == 0:
                    return i + 1
        elif depth == 1 and ((kind == "id" and val == "type") or (kind == "str" and val == "type")):
            prev = tokens[i - 1] if i > start_brace else ("", "")
            if (
                prev in (("punc", "{"), ("punc", ","))
                and i + 2 < n
                and tokens[i + 1] == ("punc", ":")
                and tokens[i + 2][0] == "str"
            ):
                sec_type = tokens[i + 2][1]
                if sec_type not in _COWORK_SECTION_TYPES:
                    invalid.append(sec_type)
                i += 3
                continue
        i += 1
    return i


def _invalid_cowork_section_types(program: str) -> list[str]:
    """Find unsupported literal `type` values on direct section objects in `sections: [...]`,
    `sections = [...]`, or `sections.push({...})`."""
    tokens, _ = _scan_js_range(program, 0, stop_on_rbrace=False)
    n = len(tokens)
    invalid: list[str] = []
    i = 0
    while i < n:
        # Pattern A: `sections: [ {...}, ... ]` or `sections = [ {...}, ... ]`
        if (
            tokens[i] in (("id", "sections"), ("str", "sections"))
            and i + 2 < n
            and tokens[i + 1] in (("punc", ":"), ("op", "="))
            and tokens[i + 2] == ("punc", "[")
        ):
            j = i + 3
            arr_depth = 1
            paren_depth = 0
            while j < n and arr_depth > 0:
                if tokens[j] == ("punc", "["):
                    arr_depth += 1
                elif tokens[j] == ("punc", "]"):
                    arr_depth -= 1
                elif tokens[j] == ("punc", "("):
                    paren_depth += 1
                elif tokens[j] == ("punc", ")") and paren_depth > 0:
                    paren_depth -= 1
                elif arr_depth == 1 and paren_depth == 0 and tokens[j] == ("punc", "{"):
                    j = _check_section_obj(tokens, j, invalid)
                    continue
                j += 1
            i = j
            continue
        # Pattern B: `sections.push({ ... }, ...)`
        if (
            tokens[i] == ("id", "sections")
            and i + 3 < n
            and tokens[i + 1] == ("punc", ".")
            and tokens[i + 2] == ("id", "push")
            and tokens[i + 3] == ("punc", "(")
        ):
            j = i + 4
            paren_depth = 1
            arr_depth = 0
            while j < n and paren_depth > 0:
                if tokens[j] == ("punc", "("):
                    paren_depth += 1
                elif tokens[j] == ("punc", ")"):
                    paren_depth -= 1
                elif tokens[j] == ("punc", "["):
                    arr_depth += 1
                elif tokens[j] == ("punc", "]") and arr_depth > 0:
                    arr_depth -= 1
                elif paren_depth == 1 and arr_depth == 0 and tokens[j] == ("punc", "{"):
                    j = _check_section_obj(tokens, j, invalid)
                    continue
                j += 1
            i = j
            continue
        i += 1
    return list(dict.fromkeys(invalid))


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
    if not re.search(r"\bCoworkPage\s*\(", program) or re.search(r"</script\b", program, re.I):
        raise ValueError("Return a complete JavaScript program calling await CoworkPage(config), without HTML or script tags.")
    css = (HERE / "presentation.css").read_text()
    if _compares_selection(program):
        raise ValueError(
            "render receives an object keyed by control ids, never a string. Read the control's value by "
            "its id, for example render: ({view}) => view === 'orders' ? ... or selection.view === 'orders'."
        )
    bad_types = _invalid_cowork_section_types(program)
    if bad_types:
        raise ValueError(
            f"Unsupported CoworkPage section type {bad_types[0]!r}: supported types are "
            "'chart', 'table', 'map', 'globe', 'timeline', and 'text'. Use one of those or return "
            "USE_HTML_WRITER when custom HTML is required."
        )
    runtime = "\n".join((HERE / name).read_text() for name in ("presentation_contract.js", "presentation.js", "presentation_data.js", "presentation_geography.js",
                                                     "presentation_reconcile.js"))
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
            "<title>Artifact</title><style>" + css + "</style></head><body>"
            '<main id="cowork-page"></main><script>' + runtime + "</script><script>"
            "(async()=>{\n" + BEGIN + "\n" + program + "\n" + END + "\n})().catch(error=>{"
            "const box=document.createElement('p');box.className='failure';box.setAttribute('role','alert');box.textContent=error.message;"
            "(document.getElementById('page-content')||document.getElementById('cowork-page')).replaceChildren(box);console.error(error);throw error;});"
            "</script></body></html>")
