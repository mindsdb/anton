"""Small, exact profiles for inspecting queried records before writing analysis.

No business predicates or totals are inferred. The original records belong in
the caller's cache; a profile describes them without printing that whole cache.
"""
from collections import Counter
from datetime import date
import json

from .observed_data import credential_bearing


def _kind(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return "object" if isinstance(value, dict) else "array"


def profile_tables(tables, *, max_chars=9000):
    """Return JSON text, bounded without cutting off JSON or hiding row counts.

    ``tables`` maps caller-chosen names to complete lists of JSON record objects.
    All columns retain type, missing/null and distinct scalar counts (absent missing/null counts mean zero). Small
    scalar value/count lists and numeric/date ranges are observations, not
    suggested filters. Optional value lists are removed largest-first if the
    budget is exceeded, keeping repeated string categories ahead of constants,
    unique identifiers or numeric enumerations. ``value_lists_omitted`` counts omissions; a
    column without ``values`` is not enumerated. Split unusually wide inputs
    into smaller groups rather than silently dropping a table or column.
    """
    if not isinstance(tables, dict) or not all(isinstance(k, str) for k in tables):
        raise ValueError("Expected a mapping of table names to record lists.")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 256:
        raise ValueError("max_chars must be an integer of at least 256.")
    for rows in tables.values():
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("Every table must be a list of record objects.")
        if not all(isinstance(key, str) for row in rows for key in row):
            raise ValueError("Record field names must be strings.")
    # Reject invalid JSON and credentials before anything can be printed.
    json.dumps(tables, allow_nan=False)
    if credential_bearing(tables):
        raise ValueError("Profiles must not contain credentials.")
    result = {}
    optional = []
    for name, rows in tables.items():
        columns = {}
        for key in dict.fromkeys(key for row in rows for key in row):
            present = [row[key] for row in rows if key in row]
            values = [v for v in present if v is not None]
            facts = {"types": dict(Counter(_kind(v) for v in present))}
            if len(present) < len(rows):
                facts["missing"] = len(rows) - len(present)
            if all(isinstance(v, (str, int, float, bool)) for v in values):
                counts = Counter(json.dumps(v, ensure_ascii=False, allow_nan=False) for v in values)
                facts["distinct"] = len(counts)
                if len(counts) <= 16 and sum(len(v) for v in counts) <= 500:
                    facts["values"] = [[json.loads(v), n] for v, n in counts.items()]
                    repeated_strings = values and all(isinstance(v, str) for v in values) and len(counts) < len(values)
                    priority = 0 if repeated_strings and len(counts) > 1 else 1 if repeated_strings else 2
                    optional.append((priority, len(json.dumps(facts["values"])), facts))
            if values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                facts["range"] = [min(values), max(values)]
            elif values and all(isinstance(v, str) and len(v) == 10 for v in values):
                try:
                    dates = [date.fromisoformat(v) for v in values]
                except ValueError:
                    pass
                else:
                    facts["date_range"] = [min(dates).isoformat(), max(dates).isoformat()]
            columns[key] = facts
        result[name] = {"rows": len(rows), "columns": columns}
    omitted = 0
    def encoded():
        return json.dumps({"tables": result, "value_lists_omitted": omitted},
                          ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    text = encoded()
    for _, _, facts in sorted(optional, key=lambda item: item[:2], reverse=True):
        if len(text) <= max_chars:
            break
        del facts["values"]
        omitted += 1
        text = encoded()
    if len(text) > max_chars:
        raise ValueError("Profile exceeds the output budget; profile fewer tables per call.")
    return text
