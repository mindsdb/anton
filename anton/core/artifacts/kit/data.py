"""Turn the agent's data (DataFrames, lists of dicts, dicts of lists) into JSON-safe rows."""

from __future__ import annotations

import datetime as _dt
import decimal
import math
import re
from typing import Any


def _scalar(v: Any) -> Any:
    """A JSON-safe value: NaN and NaT become None, numpy scalars become Python ones, dates ISO text."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(v, str):
        return v
    if isinstance(v, decimal.Decimal):
        return float(v)
    # pandas / numpy without importing them
    mod = type(v).__module__
    if mod.startswith("numpy"):
        if hasattr(v, "item"):
            return _scalar(v.item())
    if mod.startswith("pandas"):
        name = type(v).__name__
        if name in ("NaTType", "NAType"):
            return None
        if name == "Timestamp":
            return _date_text(v.to_pydatetime())
        if name == "Timedelta":
            return v.total_seconds() / 86400.0
        if name == "Period":
            return str(v)
    if isinstance(v, _dt.datetime):
        return _date_text(v)
    if isinstance(v, _dt.date):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return [_scalar(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _scalar(x) for k, x in v.items()}
    return str(v)


def _date_text(v: _dt.datetime) -> str:
    if v.hour == 0 and v.minute == 0 and v.second == 0 and v.microsecond == 0:
        return v.date().isoformat()
    return v.replace(microsecond=0, tzinfo=None).isoformat(sep=" ")


_DAY_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_WITH_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$")


def _same_shape_times(rows: list[dict]) -> list[dict]:
    """A column of timestamps keeps one shape: when some have a time of day, the midnight ones
    (written as a bare date) get theirs too, so the column reads as times throughout."""
    if not rows:
        return rows
    for f in fields(rows):
        vals = [r.get(f) for r in rows if isinstance(r.get(f), str)]
        if vals and any(_WITH_TIME.match(v) for v in vals) and any(_DAY_ONLY.match(v) for v in vals):
            for r in rows:
                v = r.get(f)
                if isinstance(v, str) and _DAY_ONLY.match(v):
                    r[f] = v + " 00:00:00"
    return rows


def records(data: Any) -> list[dict]:
    """Rows as a list of dicts from a DataFrame, a list of dicts/tuples, or a dict of columns."""
    return _same_shape_times(_records(data))


def _records(data: Any) -> list[dict]:
    if data is None:
        return []
    if hasattr(data, "to_dict") and hasattr(data, "columns"):  # DataFrame
        frame = data
        if any(n is not None for n in (getattr(frame.index, "names", None) or [])):
            frame = frame.reset_index()  # a named index (e.g. after groupby) is a column the page needs
        rows = frame.to_dict(orient="records")
        return [{str(k): _scalar(v) for k, v in r.items()} for r in rows]
    if hasattr(data, "to_dict") and hasattr(data, "index") and not hasattr(data, "columns"):  # Series
        name = getattr(data, "name", None) or "value"
        idx = getattr(data.index, "name", None) or "key"
        return [{str(idx): _scalar(k), str(name): _scalar(v)} for k, v in data.items()]
    if isinstance(data, dict):
        cols = list(data.keys())
        if cols and all(isinstance(data[c], (list, tuple)) for c in cols):
            n = max(len(data[c]) for c in cols)
            return [{str(c): _scalar(data[c][i] if i < len(data[c]) else None) for c in cols} for i in range(n)]
        return [{str(k): _scalar(v) for k, v in data.items()}]
    if isinstance(data, (list, tuple)):
        out = []
        for r in data:
            if isinstance(r, dict):
                out.append({str(k): _scalar(v) for k, v in r.items()})
            elif hasattr(r, "_asdict"):
                out.append({str(k): _scalar(v) for k, v in r._asdict().items()})
            else:
                raise TypeError("rows must be dicts (or a DataFrame); got " + type(r).__name__)
        return out
    raise TypeError("data must be a DataFrame, a list of dicts or a dict of columns; got " + type(data).__name__)


def fields(rows: list[dict]) -> list[str]:
    seen: dict[str, None] = {}
    for r in rows[:500]:
        for k in r:
            seen.setdefault(k, None)
    return list(seen)


def is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def fmt(v: Any, format: str | None = None, currency: str = "USD") -> str:
    """An approximate Python rendering of the page's number formats, for the build summary."""
    if v is None or v == "":
        return "—"
    f = (format or "auto").split(":")[0]
    if not is_number(v):
        return str(v)
    sym = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}.get(currency, currency + " ")
    if f == "percent":
        return f"{v:,.1f}%".replace(".0%", "%")
    if f == "fraction":
        return f"{v * 100:,.1f}%".replace(".0%", "%")
    if f in ("currency_compact", "compact"):
        a = abs(v)
        for div, unit in ((1e9, "B"), (1e6, "M"), (1e4 / 10, "K")):
            if a >= max(div, 1e4 if unit == "K" else div):
                s = f"{v / div:.3g}{unit}"
                return (sym + s) if f == "currency_compact" else s
        return (sym if f == "currency_compact" else "") + f"{v:,.0f}"
    if f == "currency":
        return f"{sym}{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{sym}{v:,.2f}"
    if isinstance(v, float) and not v.is_integer():
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"{int(v):,}"


def profile(data: Any, name: str = "data", max_values: int = 12) -> str:
    """A compact profile of a table: row count, then per column its type, distinct values with counts
    (for few-valued columns) or range and total (for numbers) and span (for dates). Printed and returned.
    Use it before computing, to see which statuses, categories and periods the data really holds."""
    from collections import Counter

    rows = records(data)
    lines = [f"{name}: {len(rows)} rows"]
    # Pages show clock times as written; a timezone-aware column says which zone that is.
    zones = {str(c): str(getattr(t, "tz", None)) for c, t in getattr(data, "dtypes", {}).items()
             if getattr(t, "tz", None) is not None} if hasattr(data, "dtypes") else {}
    for f in fields(rows):
        vals = [r.get(f) for r in rows]
        present = [v for v in vals if v is not None and v != ""]
        missing = len(vals) - len(present)
        tail = f", {missing} empty" if missing else ""
        if present and all(is_number(v) for v in present):
            lines.append(f"  {f}: number, min {min(present):,.6g}, max {max(present):,.6g}, sum {sum(present):,.6g}{tail}")
            continue
        texts = [str(v) for v in present]
        if texts and all(len(t) >= 10 and t[:4].isdigit() and t[4] == "-" for t in texts):
            timed = any(_WITH_TIME.match(t) and t[11:16] != "00:00" for t in texts)
            kind, cut = ("datetime (with times of day)", 16) if timed else ("date", 10)
            zone = f", clock times in {zones[f]}" if f in zones else ""
            lines.append(f"  {f}: {kind}, {min(texts)[:cut]} .. {max(texts)[:cut]}, {len(set(texts))} distinct{zone}{tail}")
            continue
        counts = Counter(texts)
        if len(counts) <= max_values:
            shown = ", ".join(f"{k!r} x{n}" for k, n in counts.most_common())
            lines.append(f"  {f}: {len(counts)} values: {shown}{tail}")
        else:
            sample = ", ".join(repr(k) for k, _ in counts.most_common(4))
            lines.append(f"  {f}: {len(counts)} distinct, e.g. {sample}{tail}")
    out = "\n".join(lines)
    print(out)
    return out


def _num(v: float) -> str:
    return f"{v:,.6g}" if abs(v) < 1e6 else f"{v:,.0f}"


def _delta(a: float, b: float) -> str:
    d = b - a
    return f"{_num(a)} -> {_num(b)} ({'+' if d > 0 else ''}{_num(d)})"


def diff(before: Any, after: Any, names: tuple[str, str] = ("before", "after"), *, max_values: int = 12) -> str:
    """What changes between two versions of the same data: two scenarios, two periods, a plan with
    and without an option. Pass two tables, or two dicts of tables {name: table} with the same names.
    For each table it prints only what differs: the row count; each number column's total; for each
    few-valued column, the count of each value that changed and, for that value, the totals of the
    number columns (e.g. Status 'Late': 0 -> 40 rows, Value 0 -> 1,570,600); and, when the tables share
    an id column, how many rows changed in which columns. Printed and returned. Use it to find what an
    option or scenario changes before choosing the figures; the effect is often in a table downstream
    of the one the request names."""
    from collections import Counter, defaultdict

    a_tabs = before if isinstance(before, dict) and not hasattr(before, "columns") else {"data": before}
    b_tabs = after if isinstance(after, dict) and not hasattr(after, "columns") else {"data": after}
    lines = [f"diff {names[0]} -> {names[1]}"]
    for name in list(dict.fromkeys(list(a_tabs) + list(b_tabs))):
        ra, rb = records(a_tabs.get(name)), records(b_tabs.get(name))
        out = []
        if len(ra) != len(rb):
            out.append(f"  rows: {_delta(len(ra), len(rb))}")
        cols = list(dict.fromkeys(fields(ra) + fields(rb)))
        numeric = [f for f in cols if any(is_number(r.get(f)) for r in ra + rb)
                   and all(is_number(r.get(f)) or r.get(f) in (None, "") for r in ra + rb)]
        for f in numeric:
            sa = sum(r.get(f) or 0 for r in ra)
            sb = sum(r.get(f) or 0 for r in rb)
            if abs(sa - sb) > 1e-9 * max(1.0, abs(sa)):
                out.append(f"  {f} total: {_delta(sa, sb)}")
        for f in cols:
            if f in numeric:
                continue
            ca = Counter(str(r.get(f)) for r in ra if r.get(f) not in (None, ""))
            cb = Counter(str(r.get(f)) for r in rb if r.get(f) not in (None, ""))
            if len(set(ca) | set(cb)) > max_values:
                continue
            for v in sorted(set(ca) | set(cb), key=lambda x: -(ca[x] + cb[x])):
                if ca[v] == cb[v]:
                    continue
                out.append(f"  {f} = {v!r}: rows {_delta(ca[v], cb[v])}")
                for g in numeric:
                    sa = sum(r.get(g) or 0 for r in ra if str(r.get(f)) == v)
                    sb = sum(r.get(g) or 0 for r in rb if str(r.get(f)) == v)
                    if abs(sa - sb) > 1e-9 * max(1.0, abs(sa)):
                        out.append(f"    {g} where {f} = {v!r}: {_delta(sa, sb)}")
        # Rows matched by a shared id: which columns changed, and how many rows.
        ids = [f for f in cols if (f.lower() == "id" or f.lower().endswith(("_id", ".id")) or f.endswith("Id"))
               and len({r.get(f) for r in ra}) == len(ra) and len({r.get(f) for r in rb}) == len(rb) and ra and rb]
        if ids:
            key = ids[0]
            by_b = {r.get(key): r for r in rb}
            changed: dict[str, int] = defaultdict(int)
            n_rows = 0
            for r in ra:
                other = by_b.get(r.get(key))
                if other is None:
                    continue
                diffs = [f for f in cols if f != key and r.get(f) != other.get(f)]
                if diffs:
                    n_rows += 1
                    for f in diffs:
                        changed[f] += 1
            if n_rows:
                detail = ", ".join(f"{f} {n}" for f, n in sorted(changed.items(), key=lambda x: -x[1])[:8])
                out.append(f"  {n_rows} of {len(ra)} rows changed (by {key}): {detail}")
        lines.append(f"{name}: " + ("no changes" if not out else ""))
        lines += out
    text = "\n".join(lines)
    print(text)
    return text
