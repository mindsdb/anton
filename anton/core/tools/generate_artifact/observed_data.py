"""Embed observed JSON without making a model transcribe the records again.

Only the artifact's canonical observed-data.json handoff is embedded. Values
remain data: script delimiters are escaped before insertion into the document.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import Counter
from html.parser import HTMLParser
from urllib.parse import urlsplit

from anton.utils.datasources import scrub_output
from anton.utils.nofollow import read_bytes_nofollow

DATA_ID = "cowork-observed-json"
PLACES_ID = "cowork-places-json"
CHECKS_ID = "cowork-figure-checks"
MARKER = "data-cowork-observed"
MAX_DOUBLE_COUNTS = 200
DIFFERENCES_KEY = "scenario_differences"
MAX_DIFFERENCES_CHARS = 5000
MAX_COMPARED_ROWS = 100_000
def _invalid_constant(value):
    raise ValueError(f"Non-JSON constant: {value}")


# Compared after lowercasing and dropping "-" and "_", so apiKey, x-api-key and API_KEY all match.
CREDENTIAL_KEYS = {
    "accesstoken", "refreshtoken", "idtoken", "token", "bearer", "authorization", "password", "passwd",
    "secret", "clientsecret", "apikey", "xapikey", "privatekey", "secretkey", "accesskey",
    "secretaccesskey", "awssecretaccesskey", "sessiontoken", "connectionstring", "dsn",
}
_CREDENTIAL_WORDS = {"password", "passwd", "secret", "token", "credential", "credentials",
                     "authorization", "auth", "bearer", "dsn"}
_KEY_QUALIFIERS = {"api", "auth", "access", "private", "secret", "client", "session",
                   "signing", "encryption", "aws", "ssh"}
_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s<>\"']+", re.I)


def _credential_key(key):
    key = str(key)
    compact = re.sub(r"[-_]", "", key).lower()
    if compact in CREDENTIAL_KEYS:
        return True
    words = re.findall(r"[a-z0-9]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).lower())
    # Only the last word names what the value is: token_count and auth_status are ordinary columns.
    return bool(words) and (words[-1] in _CREDENTIAL_WORDS or
                            words[-1] == "key" and len(words) > 1 and words[-2] in _KEY_QUALIFIERS)


def _credential_url(text):
    for match in _URL.finditer(text):
        try:
            if urlsplit(match.group().rstrip(".,;)]}")).password is not None:
                return True
        except ValueError:
            continue
    return False


def read_handoff(folder):
    """The data channel stays inside this artifact and has an explicit size cap."""
    raw = read_bytes_nofollow(folder, folder / "observed-data.json", limit=4 * 1024 * 1024)
    value = json.loads(raw, parse_constant=_invalid_constant)
    if not isinstance(value, (dict, list)):
        raise ValueError("Structured source data must be an object or array.")
    return value


def _credential_structure(value):
    if isinstance(value, dict):
        return any(_credential_key(k) or _credential_structure(v)
                   for k, v in value.items())
    if isinstance(value, list):
        return any(_credential_structure(v) for v in value)
    if isinstance(value, str):
        return _credential_url(value)
    return False


def credential_bearing(value):
    if _credential_structure(value):
        return True
    try:
        wire = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return True
    return scrub_output(wire) != wire


def identity(value):
    wire = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return "data-" + hashlib.sha256(wire.encode()).hexdigest()[:12]


def contract(value, depth=0):
    """Describe the shape, small values and sample rows without repeating a dataset."""
    if depth > 6:
        return {"type": type(value).__name__}
    if isinstance(value, dict):
        return {k: contract(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) <= 8:
            return [contract(v, depth + 1) for v in value]
        result = {"array_length": len(value), "sample": [contract(v, depth + 1) for v in value[:2]]}
        if all(isinstance(v, dict) for v in value):
            result["record_keys"] = list(dict.fromkeys(k for v in value for k in v))
            # Column descriptors and other small enumerations need their names
            # as well as their schema. Large/free-text fields remain sampled.
            enumerations = {}
            for key in result["record_keys"]:
                entries = [v.get(key) for v in value]
                if all(isinstance(v, (str, int, float, bool, type(None))) for v in entries):
                    distinct = list(dict.fromkeys(entries))
                    if len(distinct) <= 64 and sum(len(str(v)) for v in distinct) < 800:
                        enumerations[key] = distinct
            if enumerations:
                result["distinct_values"] = enumerations
        return result
    if isinstance(value, str) and len(value) > 800:
        return {"string_length": len(value), "sample": value[:160]}
    return value


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _same(a, b, parts):
    # Totals may be rounded to two decimals after summing their parts.
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=0.005 * parts + 1e-9)


def total_rows(rows):
    """Markers of rows that hold the sum of the other rows in their group, such as a
    subtotal row beside its parts: [{field, value, by, measures}]."""
    if len(rows) < 3 or not all(isinstance(r, dict) for r in rows):
        return []
    keys = list(dict.fromkeys(k for r in rows for k in r))
    measures = [k for k in keys if any(_number(r.get(k)) for r in rows)
                and all(_number(r.get(k)) or r.get(k) is None for r in rows)]
    labels = [k for k in keys if k not in measures and all(isinstance(r.get(k), (str, type(None))) for r in rows)]
    found = []
    for field in labels:
        counts = Counter(r.get(field) for r in rows)
        candidates = [v for v, n in counts.items() if isinstance(v, str) and v and n <= len(rows) // 2]
        if len(counts) > 64:
            # A lone total row among many named rows holds the largest value of a measure.
            tops = {max(rows, key=lambda r, m=m: r.get(m) if _number(r.get(m)) else -math.inf).get(field) for m in measures}
            candidates = [v for v in candidates if v in tops]
        for value in candidates:
            marked = [r for r in rows if r.get(field) == value]
            # A label the total rows leave empty spans the whole group.
            by = [k for k in labels if k != field and all(r.get(k) not in (None, "") for r in marked)]
            groups = {}
            for r in rows:
                if r.get(field) != value:
                    groups.setdefault(tuple(r.get(k) for k in by), []).append(r)
            additive, equalities = set(measures), 0
            for r in marked:
                group = groups.get(tuple(r.get(k) for k in by), [])
                if len(group) < (2 if len(marked) > 1 else 3):
                    additive = set()
                    break
                sums = {m: sum(g.get(m) or 0 for g in group) for m in additive}
                additive = {m for m in additive if _number(r.get(m)) and _same(r[m], sums[m], len(group))}
                # One nonzero part equals its "sum" trivially; that is no evidence of a total.
                equalities += sum(1 for m in additive if sum(1 for g in group if g.get(m)) >= 2)
                if not additive:
                    break
            carried = [m for m in measures if all(_number(r.get(m)) for r in marked)]
            # A real total sums most of what it carries; one matching field, such as a line number, is chance.
            if additive and equalities >= 3 and 2 * len(additive) >= len(carried):
                found.append({"field": field, "value": value, "by": by,
                              "measures": [m for m in measures if m in additive]})
    return found


def _record_arrays(value, path, depth=0):
    if depth > 6:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _record_arrays(item, path + "[" + json.dumps(key, ensure_ascii=False) + "]", depth + 1)
    elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        yield path, value


def double_counts(data):
    """Sums that include total rows beside their parts, for the page to reject when shown."""
    found, legitimate = {}, set()
    for key, value in data.items():
        for path, rows in _record_arrays(value, "window.ARTIFACT_DATA[" + json.dumps(key) + "]"):
            for marker in total_rows(rows):
                scopes = [("", rows)] + [(f'{label} = {json.dumps(v, ensure_ascii=False)}', [r for r in rows if r.get(label) == v])
                                         for label in marker["by"] for v in dict.fromkeys(r.get(label) for r in rows)]
                for measure in marker["measures"]:
                    legitimate.update(round(r[measure], 2) for r in rows if _number(r.get(measure)))
                    for scope, subset in scopes:
                        every = sum(r.get(measure) or 0 for r in subset)
                        parts = sum(r.get(measure) or 0 for r in subset if r.get(marker["field"]) != marker["value"])
                        legitimate.add(round(parts, 2))
                        # Small sums match unrelated counts too often to reject.
                        if abs(every) >= 10 and not _same(every, parts, len(subset)):
                            found.setdefault(round(every, 2), {
                                "value": every, "parts": parts, "measure": measure, "path": path, "scope": scope,
                                "field": marker["field"], "marker": marker["value"]})
    return [item for value, item in found.items() if value not in legitimate][:MAX_DOUBLE_COUNTS]


def _tables(value, path=(), depth=0):
    if depth > 6:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _tables(item, path + (str(key),), depth + 1)
    elif isinstance(value, list) and len(value) >= 2 and all(isinstance(v, dict) for v in value):
        yield path, value


def _same_shaped_pairs(tables):
    """Tables with the same fields whose paths differ in one name, such as one per scenario."""
    work = 0
    for i, (path_a, a, fields_a) in enumerate(tables):
        for path_b, b, fields_b in tables[i + 1:]:
            if (fields_a == fields_b and len(path_a) == len(path_b)
                    and sum(x != y for x, y in zip(path_a, path_b)) == 1):
                # Many same-shaped tables would make this quadratic; the page is built without them.
                work += len(a) + len(b)
                if work > MAX_COMPARED_ROWS:
                    return
                yield path_a, a, path_b, b


def _matching_fields(a, b, fields):
    """The fewest fields, up to three, that name each row once in both tables, such as an id and line."""
    def unique(rows, combo):
        keys = [tuple(r.get(f) for f in combo) for r in rows]
        return all(None not in k for k in keys) and len(set(keys)) == len(keys)
    for size in (1, 2, 3):
        for combo in itertools.combinations(fields, size):
            if unique(a, combo) and unique(b, combo):
                return combo
    return None


# Rates, shares, averages and unit prices do not add up across rows.
_NOT_ADDITIVE = {"pct", "percent", "percentage", "rate", "ratio", "share", "average", "avg", "mean", "price"}


def _additive(name):
    words = re.findall(r"[a-z0-9]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower())
    return bool(words) and not (words[0] in _NOT_ADDITIVE or words[-1] in _NOT_ADDITIVE and words[0] != "total")


def _summary(rows, measures, groups):
    # Subtotal rows beside their parts would count those parts twice.
    totals = [(m["field"], m["value"]) for m in total_rows(rows)]
    rows = [r for r in rows if not any(r.get(f) == v for f, v in totals)]

    def sums(subset):
        totals = {m: round(sum(r.get(m) or 0 for r in subset), 2) for m in measures}
        # Huge values can add up to infinity, which JSON cannot hold.
        return {"rows": len(subset), **{m: v for m, v in totals.items() if math.isfinite(v)}}
    result = sums(rows)
    for field, values in groups.items():
        result["by " + field] = {str(v): sums([r for r in rows if type(r.get(field)) is type(v) and r.get(field) == v])
                                 for v in values}
    return result


def _minus(after, before):
    if isinstance(after, dict):
        return {k: _minus(after[k], before[k]) for k in after if k in before}
    return round(after - before, 2)


def scenario_comparisons(value):
    """Counts and sums of the same records saved once per scenario, with their differences,
    so a page can show a gain or a risk between scenarios instead of deriving it."""
    found = {}
    tables = [(path, rows, frozenset(k for r in rows for k in r)) for path, rows in _tables(value)]
    for path_a, a, path_b, b in _same_shaped_pairs(tables):
        keys = list(dict.fromkeys(k for r in a + b for k in r))
        # Text ids first, so a unique number such as an amount is not taken for the record's name.
        candidates = sorted((k for k in keys if all(isinstance(r.get(k), (str, int)) for r in a + b)),
                            key=lambda k: not all(isinstance(r.get(k), str) for r in a + b))
        fields = _matching_fields(a, b, candidates)
        if not fields:
            continue
        rows_a = {tuple(r.get(f) for f in fields): r for r in a}
        rows_b = {tuple(r.get(f) for f in fields): r for r in b}
        shared = rows_a.keys() & rows_b.keys()
        # Mostly different records, such as two periods, are not one set of records under two scenarios.
        if len(shared) < 0.5 * min(len(a), len(b)):
            continue
        changed = [k for k in keys if k not in fields and any(rows_a[s].get(k) != rows_b[s].get(k) for s in shared)]
        # Without a changed value, a smaller table is a filtered subset, such as one site's rows.
        if not changed:
            continue
        at = next(n for n, (x, y) in enumerate(zip(path_a, path_b)) if x != y)
        entry = found.setdefault((path_a[:at], path_a[at + 1:], frozenset(keys)),
                                 {"keys": keys, "fields": fields, "changed": {}, "scenarios": {}, "pairs": []})
        entry["changed"].update(dict.fromkeys(changed))
        entry["scenarios"].setdefault(path_a[at], a)
        entry["scenarios"].setdefault(path_b[at], b)
        entry["pairs"].append((path_a[at], path_b[at]))
    result, size = {}, 0
    for (before, after, _), entry in found.items():
        table = " / ".join(before + ("<scenario>",) + after)
        table += f" ({len(result) + 1})" if table in result else ""
        rows = [r for t in entry["scenarios"].values() for r in t]
        # A field equal on every row, such as line 1, sums to a row count.
        measures = [k for k in entry["keys"] if k not in entry["fields"] and k != "rows" and _additive(k)
                    and all(_number(r.get(k)) or r.get(k) is None for r in rows)
                    and len({r.get(k) for r in rows}) > 1]
        groups, smallest = {}, min(len(t) for t in entry["scenarios"].values())
        for field in entry["changed"]:
            if field in measures or not all(isinstance(r.get(field), (str, int, bool, type(None))) for r in rows):
                continue
            values = list({(type(r.get(field)), r.get(field)): r.get(field) for r in rows}.values())
            # Grouping by a field with a value per few rows, such as a date, is a listing, not a comparison.
            if len(values) <= min(12, smallest / 3) and len({str(v) for v in values}) == len(values):
                groups[field] = values
        summaries = {label: _summary(t, measures, groups) for label, t in entry["scenarios"].items()}
        compared = {"matched on": ", ".join(entry["fields"]), "changed fields": ", ".join(entry["changed"]),
                    "per scenario": summaries,
                    "differences": {f"{later} minus {first}": _minus(summaries[later], summaries[first])
                                    for first, later in entry["pairs"]}}
        cost = len(json.dumps({table: compared}, ensure_ascii=False))
        # The writer reads these as one exact object, so they share a size budget.
        if size + cost <= MAX_DIFFERENCES_CHARS:
            result[table] = compared
            size += cost
    return result


def runtime_contract(value, path, depth=0, *, compact=False):
    """Describe actual access expressions without making arrays look like wrapper objects."""
    if depth > 6:
        return [f"{path}: {type(value).__name__}; nested value retained in full."]
    if isinstance(value, dict):
        if compact:
            # Small summaries remain actual objects, without repeated paths.
            # Arrays retain their explicit runtime type and full-data binding.
            def has_array(item):
                return isinstance(item, list) or isinstance(item, dict) and any(has_array(v) for v in item.values())
            rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            if len(rendered) <= 6000 and not has_array(value):
                return [f"{path} is an OBJECT with these exact observed values: {rendered}"]
        lines = [f"{path} is an object with keys {json.dumps(list(value), ensure_ascii=False)}."]
        for key, item in value.items():
            lines.extend(runtime_contract(item, path + "[" + json.dumps(key, ensure_ascii=False) + "]", depth + 1, compact=compact))
        return lines
    if isinstance(value, list):
        example = value if len(value) <= 8 else value[:2]
        rendered = json.dumps(example, ensure_ascii=False)
        lines = [f"{path} is an ARRAY with {len(value)} elements; iterate this array directly. "
                 f"Example elements (descriptive only): {rendered[:4000]}" +
                 (" [example excerpt truncated]" if len(rendered) > 4000 else "")]
        summary = contract(value)
        if isinstance(summary, dict) and summary.get("record_keys"):
            lines.append("Each object element has source fields " + json.dumps(summary["record_keys"], ensure_ascii=False) + ".")
            if summary.get("distinct_values"):
                lines.append("Observed low-cardinality field values: " + json.dumps(summary["distinct_values"], ensure_ascii=False))
        for marker in total_rows(value):
            group = " with the same " + ", ".join(marker["by"]) if marker["by"] else ""
            lines.append(f"TOTAL ROWS: elements where {json.dumps(marker['field'], ensure_ascii=False)} = "
                         f"{json.dumps(marker['value'], ensure_ascii=False)} hold the sum of the other elements{group} "
                         f"for {', '.join(marker['measures'])}. Count each row once: sum the other elements, or use only these.")
        return lines
    rendered = json.dumps(value, ensure_ascii=False)
    return [f"{path} = {rendered[:1000]}" + (" [value excerpt truncated]" if len(rendered) > 1000 else "")]


def strip_binding(html):
    # A browser ends a script at </script followed by whitespace, "/" or ">", up to the next ">".
    html = re.sub(r'<script\b(?=[^>]*\bdata-cowork-observed\b)[^>]*>.*?</script\b[^>]*>', "", html,
                  flags=re.S | re.I)
    return re.sub(r'<meta\b(?=[^>]*\bdata-cowork-observed\b)[^>]*>', "", html, flags=re.I)


def previous_data(html):
    match = re.search(r'<script\b(?=[^>]*\bid=["\']cowork-observed-json["\'])[^>]*>(.*?)</script\b[^>]*>',
                      html, re.S | re.I)
    if match:
        try:
            value = json.loads(match.group(1), parse_constant=_invalid_constant)
            if isinstance(value, dict):
                return {key: item for key, item in value.items() if key == identity(item) and not credential_bearing(item)}
        except (ValueError, RecursionError):
            pass
    return {}


def collect(handoff, existing=""):
    """Bind a fresh handoff; an edit without one keeps only the page's previous data."""
    data = {identity(handoff): handoff} if handoff is not None else previous_data(existing)
    if handoff is not None and (compared := scenario_comparisons(handoff)):
        differences = {DIFFERENCES_KEY: compared}
        data[identity(differences)] = differences
    schemas = []
    for key, value in data.items():
        path = "window.ARTIFACT_DATA[" + json.dumps(key) + "]"
        schemas.extend(runtime_contract(value, path, compact=True))
        if isinstance(value, dict) and list(value) == [DIFFERENCES_KEY]:
            schemas.append(
                f"SCENARIO DIFFERENCES: {path}[{json.dumps(DIFFERENCES_KEY)}] holds row counts and sums the host "
                "computed from the same records saved once per scenario, or per other group such as a site "
                "(matched on the listed fields), overall and by each changed field, with the differences between "
                "them. When they are scenarios, show a gain, saving or risk between them from these figures "
                "instead of calling it unavailable.")
    return data, "Embedded JSON contracts (actual runtime access paths):\n" + "\n".join(schemas)


def _json_script(script_id, name, value):
    # Escaping '<' also protects against </script>, HTML comments and tags in
    # untrusted data. JSON remains lossless when read through textContent.
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return (f'<script id="{script_id}" type="application/json" {MARKER}>{payload}</script>'
            + (f'<script {MARKER}>window.{name}=JSON.parse(document.getElementById("{script_id}").textContent);</script>'
               if name else ""))


def bind(html, data, places=None, *, checks=False):
    """Insert the observed data before the page's scripts, with the places it names
    (places.index) and, with `checks`, the figure facts the component runtime reconciles
    shown figures with."""
    if not data:
        return html
    scripts = (f'<meta charset="utf-8" {MARKER}>'
               + (_json_script(CHECKS_ID, None, {"doubleCounts": double_counts(data)}) if checks else "")
               + _json_script(DATA_ID, "ARTIFACT_DATA", data)
               + (_json_script(PLACES_ID, "ARTIFACT_PLACES", places) if places else ""))
    html = strip_binding(html)
    line_starts = [0] + [match.end() for match in re.finditer("\n", html)]
    # HTMLParser ignores comments and understands quoted '>' in attributes;
    # a regex can otherwise insert the data inside a comment or an attribute.
    class Locations(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.ends = {}

        def handle_starttag(self, tag, attrs):
            if tag in {"html", "head"} and tag not in self.ends:
                line, column = self.getpos()
                self.ends[tag] = line_starts[line - 1] + column + len(self.get_starttag_text())

    locations = Locations()
    locations.feed(html)
    if end := locations.ends.get("head"):
        return html[:end] + scripts + html[end:]
    if end := locations.ends.get("html"):
        return html[:end] + "<head>" + scripts + "</head>" + html[end:]
    raise ValueError("A complete HTML document is required for data binding.")
